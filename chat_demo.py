"""
3 Modeli Yan Yana Karşılaştıran Canlı Türkçe Chatbot CLI Demosu
(Model 1: Full Fine-Tuned GPT-2 vs. Model 2: LoRA/PEFT GPT-2 vs. Baseline: Seq2Seq Bi-LSTM)
Proje: Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme
       ve Seq2Seq Karşılaştırmalı Başarım Analizi
Öğrenci: Seyit Ali Arslan (240711024)

Kullanım:
  1) Canlı Sohbet Modu (Sürekli soru-cevap):
     python chat_demo.py

  2) Tek Soru Modu:
     python chat_demo.py --soru "Merhaba, faturamda ödeme hatası görünüyor ne yapmalıyım?"
"""

import os
import sys
import time
import argparse
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from models.baseline_seq2seq_lstm import Seq2SeqBiLSTM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")


def load_all_three_models(device: torch.device):
    print("=" * 80)
    print(" TÜRKÇE CHATBOT 3'LÜ CANLI DEMO (Full FT GPT-2 | LoRA GPT-2 | Baseline Bi-LSTM)")
    print("=" * 80)
    print(f"-> Çalışma Cihazı: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Model 1: Full Fine-Tuned Turkish-GPT-2
    gpt2_dir = "checkpoints/finetuned_gpt2_turkish"
    print("-> [1/3] Model 1 (Full Fine-Tuned Turkish-GPT-2, 124.4M) yükleniyor...")
    gpt2_tok = AutoTokenizer.from_pretrained(gpt2_dir)
    full_mod = AutoModelForCausalLM.from_pretrained(gpt2_dir).to(device)
    if device.type == "cuda":
        full_mod = full_mod.half()
    full_mod.eval()

    # 2. Model 2: LoRA / PEFT Turkish-GPT-2
    lora_dir = "checkpoints/lora_gpt2_turkish"
    lora_mod = None
    if os.path.exists(lora_dir):
        print("-> [2/3] Model 2 (LoRA / PEFT Turkish-GPT-2, 1.62M Eğitilen) yükleniyor...")
        base_for_lora = AutoModelForCausalLM.from_pretrained("redrussianarmy/gpt2-turkish-cased")
        if len(gpt2_tok) > base_for_lora.config.vocab_size:
            base_for_lora.resize_token_embeddings(len(gpt2_tok))
        lora_mod = PeftModel.from_pretrained(base_for_lora, lora_dir).merge_and_unload().to(device)
        if device.type == "cuda":
            lora_mod = lora_mod.half()
        lora_mod.eval()

    # 3. Baseline: 2-Layer Seq2Seq Bi-LSTM + Bahdanau Attention
    bpe_path = "tokenizer/turkish_bpe_12k.json"
    base_ckpt = "checkpoints/baseline_seq2seq_lstm.pt"
    base_tok, base_mod = None, None
    if os.path.exists(bpe_path) and os.path.exists(base_ckpt):
        print("-> [3/3] Baseline Model (2-Katmanlı Seq2Seq Bi-LSTM + Attention, 23.1M) yükleniyor...")
        base_tok = Tokenizer.from_file(bpe_path)
        pad_id = base_tok.token_to_id("[PAD]")
        base_mod = Seq2SeqBiLSTM(base_tok.get_vocab_size(), 256, 256, 2, 0.25, pad_id).to(device)
        base_mod.load_state_dict(torch.load(base_ckpt, map_location=device))
        base_mod.eval()

    # CUDA Warmup
    if device.type == "cuda":
        with torch.no_grad():
            dummy = gpt2_tok("Kullanıcı: Merhaba\nAsistan:", return_tensors="pt").to(device)
            _ = full_mod.generate(**dummy, max_new_tokens=5)
            if lora_mod is not None:
                _ = lora_mod.generate(**dummy, max_new_tokens=5)
        torch.cuda.synchronize()

    print("✔ Her 3 model de GPU belleğine yüklendi ve hazır!\n")
    return gpt2_tok, full_mod, lora_mod, base_tok, base_mod


def generate_gpt2_reply(model, tokenizer, question: str, device: torch.device):
    prompt = f"Kullanıcı: {question}\nAsistan:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=36,
            pad_token_id=0,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.eos_token_id]],
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
            use_cache=True
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat_ms = (time.perf_counter() - t0) * 1000.0
    reply = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    reply = reply.split("\n")[0].split("Kullanıcı:")[0].strip()
    return reply, round(lat_ms, 1)


def ask_all_models(question: str, gpt2_tok, full_mod, lora_mod, base_tok, base_mod, device: torch.device):
    question = question.strip()
    if not question:
        return None

    full_reply, full_ms = generate_gpt2_reply(full_mod, gpt2_tok, question, device)

    lora_reply, lora_ms = "(Model yüklü değil)", 0.0
    if lora_mod is not None:
        lora_reply, lora_ms = generate_gpt2_reply(lora_mod, gpt2_tok, question, device)

    base_reply, base_ms = "(Model yüklü değil)", 0.0
    if base_mod is not None and base_tok is not None:
        bos_id = base_tok.token_to_id("[BOS]")
        eos_id = base_tok.token_to_id("[EOS]")
        enc_ids = [bos_id] + base_tok.encode(question).ids[:38] + [eos_id]
        base_reply, base_ms, _ = base_mod.generate_reply(enc_ids, base_tok, max_len=30, device=device)
        base_ms = round(base_ms, 1)

    print("-" * 80)
    print(f"🧑 Kullanıcı Sorusu           : {question}")
    print(f"🤖 Model 1 (Full FT GPT-2)    : {full_reply}  [{full_ms:.1f} ms | PPL: 32.0]")
    print(f"⚡ Model 2 (LoRA/PEFT GPT-2)  : {lora_reply}  [{lora_ms:.1f} ms | PPL: 56.5 | %1.29 Parametre]")
    print(f"⚙️  Baseline (Seq2Seq Bi-LSTM) : {base_reply}  [{base_ms:.1f} ms | PPL: 1328.0]")
    print("-" * 80)

    return {
        "question": question,
        "full_ft": {"reply": full_reply, "latency_ms": full_ms},
        "lora": {"reply": lora_reply, "latency_ms": lora_ms},
        "baseline": {"reply": base_reply, "latency_ms": base_ms}
    }


def main():
    parser = argparse.ArgumentParser(description="Türkçe Chatbot 3'lü Canlı Soru-Cevap Demosu")
    parser.add_argument("--soru", type=str, default=None, help="Tek bir soru sorup çıkmak için kullanın.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2_tok, full_mod, lora_mod, base_tok, base_mod = load_all_three_models(device)

    if args.soru:
        ask_all_models(args.soru, gpt2_tok, full_mod, lora_mod, base_tok, base_mod, device)
        return

    print("Canlı sohbet başladı! Çıkmak için 'q' veya 'cikis' yazıp Enter'a basın.\n")
    while True:
        try:
            user_in = input("\nSen (Sorunuzu yazın): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSohbet sonlandırıldı.")
            break

        if user_in.lower() in ["q", "quit", "exit", "cikis", "çıkış"]:
            print("Sohbet sonlandırıldı. İyi çalışmalar!")
            break
        if not user_in:
            continue

        ask_all_models(user_in, gpt2_tok, full_mod, lora_mod, base_tok, base_mod, device)


if __name__ == "__main__":
    main()
