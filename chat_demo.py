"""
Canlı Türkçe Chatbot Etkileşim ve Karşılaştırma Demosu (CLI)
Proje: Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme
       ve Seq2Seq Karşılaştırmalı Başarım Analizi
Öğrenci: Seyit Ali Arslan (240711024)

Kullanım:
  1) Canlı Sohbet Modu (Sürekli soru-cevap):
     python chat_demo.py

  2) Tek Soru Modu:
     python chat_demo.py --soru "Merhaba, kargom hala gelmedi ne yapmalıyım?"
"""

import os
import sys
import time
import argparse
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.baseline_seq2seq_lstm import Seq2SeqBiLSTM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")


def load_models(device: torch.device):
    print("=" * 75)
    print(" TÜRKÇE CHATBOT CANLI DEMO YÜKLENİYOR (Baseline Bi-LSTM & Fine-Tuned GPT-2)")
    print("=" * 75)
    print(f"-> Çalışma Cihazı: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Fine-Tuned Turkish-GPT-2 Yükleme
    gpt2_dir = "checkpoints/finetuned_gpt2_turkish"
    if not os.path.exists(gpt2_dir):
        raise FileNotFoundError(f"Fine-Tuned GPT-2 modeli bulunamadı: {gpt2_dir}. Önce 'python train_finetune_gpt2.py' çalıştırın.")

    print("-> [1/2] Model 1 (Fine-Tuned Turkish-GPT-2, 124.4M) yükleniyor...")
    gpt2_tok = AutoTokenizer.from_pretrained(gpt2_dir)
    gpt2_mod = AutoModelForCausalLM.from_pretrained(gpt2_dir).to(device)
    if device.type == "cuda":
        gpt2_mod = gpt2_mod.half()
    gpt2_mod.eval()

    # 2. Baseline Seq2Seq Bi-LSTM Yükleme
    bpe_path = "tokenizer/turkish_bpe_12k.json"
    base_ckpt = "checkpoints/baseline_seq2seq_lstm.pt"
    base_tok, base_mod = None, None
    if os.path.exists(bpe_path) and os.path.exists(base_ckpt):
        print("-> [2/2] Baseline Model (2-Katmanlı Seq2Seq Bi-LSTM + Attention, 23.1M) yükleniyor...")
        base_tok = Tokenizer.from_file(bpe_path)
        pad_id = base_tok.token_to_id("[PAD]")
        base_mod = Seq2SeqBiLSTM(base_tok.get_vocab_size(), 256, 256, 2, 0.25, pad_id).to(device)
        base_mod.load_state_dict(torch.load(base_ckpt, map_location=device))
        base_mod.eval()

    # CUDA Warmup
    if device.type == "cuda":
        with torch.no_grad():
            _ = gpt2_mod.generate(**gpt2_tok("Kullanıcı: Merhaba\nAsistan:", return_tensors="pt").to(device), max_new_tokens=5)
        torch.cuda.synchronize()

    print("✔ Modeller hazır!\n")
    return gpt2_tok, gpt2_mod, base_tok, base_mod


def ask_both_models(question: str, gpt2_tok, gpt2_mod, base_tok, base_mod, device: torch.device):
    question = question.strip()
    if not question:
        return

    # 1. Fine-Tuned Turkish-GPT-2 Yanıtı
    prompt = f"Kullanıcı: {question}\nAsistan:"
    inputs = gpt2_tok(prompt, return_tensors="pt").to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        gen_ids = gpt2_mod.generate(
            **inputs,
            max_new_tokens=40,
            pad_token_id=0,
            eos_token_id=gpt2_tok.eos_token_id,
            bad_words_ids=[[gpt2_tok.eos_token_id]],
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
            use_cache=True
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpt2_ms = (time.perf_counter() - t0) * 1000.0
    gpt2_reply = gpt2_tok.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    gpt2_reply = gpt2_reply.split("\n")[0].split("Kullanıcı:")[0].strip()

    # 2. Baseline Seq2Seq Bi-LSTM Yanıtı
    base_reply, base_ms = "(Model yüklü değil)", 0.0
    if base_mod is not None and base_tok is not None:
        bos_id = base_tok.token_to_id("[BOS]")
        eos_id = base_tok.token_to_id("[EOS]")
        enc_ids = [bos_id] + base_tok.encode(question).ids[:38] + [eos_id]
        base_reply, base_ms, _ = base_mod.generate_reply(enc_ids, base_tok, max_len=30, device=device)

    print("-" * 75)
    print(f"🧑 Kullanıcı             : {question}")
    print(f"🤖 Fine-Tuned GPT-2      : {gpt2_reply}  [{gpt2_ms:.1f} ms]")
    print(f"⚙️  Baseline Bi-LSTM      : {base_reply}  [{base_ms:.1f} ms]")
    print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="Türkçe Chatbot Canlı Soru-Cevap Demosu")
    parser.add_argument("--soru", type=str, default=None, help="Tek bir soru sorup çıkmak için kullanın.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2_tok, gpt2_mod, base_tok, base_mod = load_models(device)

    if args.soru:
        ask_both_models(args.soru, gpt2_tok, gpt2_mod, base_tok, base_mod, device)
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

        ask_both_models(user_in, gpt2_tok, gpt2_mod, base_tok, base_mod, device)


if __name__ == "__main__":
    main()
