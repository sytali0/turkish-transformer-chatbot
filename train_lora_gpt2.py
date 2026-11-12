"""
Model 2: LoRA / PEFT (Düşük Dereceli Adaptasyon) ile Turkish-GPT-2 İnce Ayar Betiği
Proje: Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme
       ve Seq2Seq Karşılaştırmalı Başarım Analizi
Öğrenci: Seyit Ali Arslan (240711024)
Bölüm: Yapay Zeka ve Veri Mühendisliği, Atatürk Üniversitesi
"""

import os
import sys
import math
import time
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from train_finetune_gpt2 import TurkishGPT2DialogueDataset
from train_baseline_lstm import compute_bleu_and_rouge

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def train_and_evaluate_lora_gpt2(
    model_name: str = "redrussianarmy/gpt2-turkish-cased",
    epochs: int = 5,
    batch_size: int = 32,
    max_len: int = 112,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_train_batches: int = 250,
    max_val_batches: int = 50
):
    print("=" * 75)
    print(f"ADIM 1 (FINAL): MODEL 2 — LORA / PEFT TURKISH-GPT-2 İNCE AYARI (r={lora_r}, alpha={lora_alpha})")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Çalışma Cihazı: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    if len(tokenizer) > base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    # LoRA Yapılandırması (GPT-2 Conv1D katmanları için fan_in_fan_out=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        fan_in_fan_out=True
    )

    model = get_peft_model(base_model, lora_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = round((trainable_params / total_params) * 100.0, 2)

    print(f"-> Toplam Parametre Sayısı      : {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"-> Eğitilebilir (LoRA) Parametre: {trainable_params:,} (~{trainable_params/1e6:.2f}M | Sadece %{trainable_pct}!)")

    train_ds = TurkishGPT2DialogueDataset("data/train.csv", tokenizer, max_len=max_len)
    val_ds = TurkishGPT2DialogueDataset("data/val.csv", tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )
    total_steps = epochs * max_train_batches
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
        "epoch_time_sec": []
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for step, batch in enumerate(train_loader):
            if step >= max_train_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            train_steps += 1

        avg_train_loss = total_train_loss / max(1, train_steps)
        train_ppl = math.exp(min(avg_train_loss, 15.0))

        # Doğrulama (Validation)
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if step >= max_val_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                total_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)
        val_ppl = math.exp(min(avg_val_loss, 15.0))
        elapsed = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["train_ppl"].append(round(train_ppl, 2))
        history["val_ppl"].append(round(val_ppl, 2))
        history["epoch_time_sec"].append(round(elapsed, 1))

        print(f"Epoch [{epoch}/{epochs}] | Süre: {elapsed:.1f}s | Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) | Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f})")

    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0
    total_train_time_sec = round(sum(history["epoch_time_sec"]), 1)

    # LoRA adaptör ağırlıklarını kaydet ve çıkarım için temel modelle birleştir (merge_and_unload)
    save_dir = "checkpoints/lora_gpt2_turkish"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n✔ LoRA Adaptör Ağırlıkları Kaydedildi: {save_dir}")

    print("-> LoRA ağırlıkları çıkarım için birleştiriliyor (merge_and_unload) ve değerlendiriliyor...")
    merged_model = model.merge_and_unload().to(device).half()
    merged_model.eval()

    # CUDA Warmup
    with torch.no_grad():
        _ = merged_model.generate(**tokenizer("Kullanıcı: Merhaba\nAsistan:", return_tensors="pt").to(device), max_new_tokens=10)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    val_batch = next(iter(val_loader))
    refs, hyps, latencies, sample_demos = [], [], [], []

    for i in range(min(30, len(val_batch["girdi_text"]))):
        girdi_txt = val_batch["girdi_text"][i]
        gercek_txt = val_batch["yanit_text"][i]
        prompt = f"Kullanıcı: {girdi_txt}\nAsistan:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            gen_ids = merged_model.generate(
                **inputs,
                max_new_tokens=22,
                pad_token_id=0,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[tokenizer.eos_token_id]],
                do_sample=False,
                num_beams=2,
                repetition_penalty=1.3,
                no_repeat_ngram_size=2,
                use_cache=True
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        reply_txt = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).split("\n")[0].split("Kullanıcı:")[0].strip()

        refs.append(gercek_txt)
        hyps.append(reply_txt)
        latencies.append(lat_ms)
        if i < 5:
            sample_demos.append({
                "girdi": girdi_txt,
                "gercek_yanit": gercek_txt,
                "lora_yanit": reply_txt,
                "latency_ms": round(lat_ms, 2)
            })

    text_metrics = compute_bleu_and_rouge(refs, hyps)
    avg_latency_ms = round(sum(latencies) / max(1, len(latencies)), 2)

    lora_results = {
        "model_name": f"Model 2: LoRA / PEFT Fine-Tuned Transformer ({model_name}, r={lora_r}, alpha={lora_alpha})",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": trainable_pct,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "peak_vram_mb": peak_vram_mb,
        "total_train_time_sec": total_train_time_sec,
        "training_history": history,
        "final_val_loss": history["val_loss"][-1],
        "final_val_perplexity": history["val_ppl"][-1],
        "avg_inference_latency_ms": avg_latency_ms,
        "bleu1": text_metrics["bleu1"],
        "bleu2": text_metrics["bleu2"],
        "rouge1": text_metrics["rouge1"],
        "rougeL": text_metrics["rougeL"],
        "sample_outputs": sample_demos
    }

    with open("results/lora_gpt2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(lora_results, f, ensure_ascii=False, indent=2)

    print(f"✔ LoRA Metrik Raporu Kaydedildi: results/lora_gpt2_metrics.json")
    print(f"✔ Tepe VRAM Kullanımı          : {peak_vram_mb} MB | Toplam Eğitim Süresi: {total_train_time_sec} sn")
    print(f"✔ Ortalama Çıkarım Süresi      : {avg_latency_ms} ms")
    print(f"✔ BLEU-1: {text_metrics['bleu1']} | BLEU-2: {text_metrics['bleu2']} | ROUGE-1: {text_metrics['rouge1']} | ROUGE-L: {text_metrics['rougeL']}")
    print("=" * 75)


if __name__ == "__main__":
    train_and_evaluate_lora_gpt2()
