"""
Model 1: Önceden Eğitilmiş Turkish-GPT-2 (~124M) İnce Ayar (Fine-Tuning) ve Karşılaştırmalı Analiz Betiği
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
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from train_baseline_lstm import compute_bleu_and_rouge

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class TurkishGPT2DialogueDataset(Dataset):
    """
    Turkish-GPT-2 resmi BPE Tokenizer yapısı ile diyalog çiftlerini hazırlayan Dataset.
    Yalnızca 'Asistan:' yanıt kısmı üzerinde Loss hesaplanması için girdi kısmını -100 ile maskeler.
    """
    def __init__(self, csv_file: str, tokenizer, max_len: int = 80):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        girdi = str(row["girdi"]).strip()
        yanit = str(row["yanit"]).strip()

        prompt_str = f"Kullanıcı: {girdi}\nAsistan: "
        full_str = f"{prompt_str}{yanit}{self.tokenizer.eos_token}"

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)

        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100  # Kullanıcı sorusu kayba (loss) dahil edilmez

        pad_len = self.max_len - len(full_ids)
        input_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(full_ids) + [0] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "girdi_text": girdi,
            "yanit_text": yanit
        }


def finetune_and_compare_gpt2(
    model_name: str = "redrussianarmy/gpt2-turkish-cased",
    epochs: int = 3,
    batch_size: int = 32,
    max_len: int = 72,
    lr: float = 5e-5,
    max_train_batches: int = 200,
    max_val_batches: int = 40
):
    print("=" * 75)
    print(f"ADIM 4: ÖNCEDEN EĞİTİLMİŞ TURKISH-GPT-2 ({model_name}) FINE-TUNING")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Çalışma Cihazı: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("-> Resmi BPE Tokenizer ve önceden eğitilmiş ağırlıklar yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"-> Turkish-GPT-2 Parametre Sayısı: {total_params:,} (~{total_params/1e6:.1f}M) | Vocab Size: {len(tokenizer):,}")

    train_ds = TurkishGPT2DialogueDataset("data/train.csv", tokenizer, max_len=max_len)
    val_ds = TurkishGPT2DialogueDataset("data/val.csv", tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * max_train_batches
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

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

    # Çıkarım Gecikmesi (Latency) ve BLEU / ROUGE Hesaplama
    print("\n-> Turkish-GPT-2 Çıkarım Gecikmesi ve BLEU/ROUGE hesaplanıyor...")
    val_batch = next(iter(val_loader))
    refs, hyps, latencies = [], [], []
    sample_demos = []

    model.eval()
    for i in range(min(30, len(val_batch["girdi_text"]))):
        girdi_txt = val_batch["girdi_text"][i]
        gercek_txt = val_batch["yanit_text"][i]
        prompt = f"Kullanıcı: {girdi_txt}\nAsistan:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t_start = time.perf_counter()
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.2
            )
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        full_dec = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        reply_txt = full_dec.split("Asistan:")[-1].split("Kullanıcı:")[0].strip()

        refs.append(gercek_txt)
        hyps.append(reply_txt)
        latencies.append(lat_ms)
        if i < 5:
            sample_demos.append({
                "girdi": girdi_txt,
                "gercek_yanit": gercek_txt,
                "gpt2_yanit": reply_txt,
                "latency_ms": round(lat_ms, 2)
            })

    text_metrics = compute_bleu_and_rouge(refs, hyps)
    avg_latency_ms = round(sum(latencies) / max(1, len(latencies)), 2)

    gpt2_results = {
        "model_name": f"Model 1: Fine-Tuned Transformer ({model_name})",
        "parameters": total_params,
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

    with open("results/finetuned_gpt2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(gpt2_results, f, ensure_ascii=False, indent=2)

    print(f"✔ Turkish-GPT-2 Metrik Raporu : results/finetuned_gpt2_metrics.json")
    print(f"✔ Ortalama Çıkarım Süresi     : {avg_latency_ms} ms")
    print(f"✔ BLEU-1: {text_metrics['bleu1']} | BLEU-2: {text_metrics['bleu2']} | ROUGE-L: {text_metrics['rougeL']}")

    # Karşılaştırma Grafiklerini Oluştur
    if os.path.exists("results/baseline_lstm_metrics.json"):
        with open("results/baseline_lstm_metrics.json", "r", encoding="utf-8") as bf:
            base_res = json.load(bf)

        # GRAFİK 5: Baseline LSTM vs Fine-Tuned Transformer (Val Loss ve Perplexity)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        models_lbl = ["Baseline Model\n(Seq2Seq Bi-LSTM)", "Model 1: Fine-Tuned\n(Turkish-GPT-2)"]
        val_losses = [base_res["final_val_loss"], gpt2_results["final_val_loss"]]
        val_ppls = [base_res["final_val_perplexity"], gpt2_results["final_val_perplexity"]]

        b1 = ax1.bar(models_lbl, val_losses, color=["#64748B", "#2563EB"], width=0.5, edgecolor="black")
        ax1.set_title("Validation Loss Karşılaştırması (Düşük = Daha İyi)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax1.set_ylim(0, max(val_losses) * 1.2)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in b1:
            y = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, y + 0.1, f"{y:.2f}", ha="center", fontweight="bold", fontsize=11)

        b2 = ax2.bar(models_lbl, val_ppls, color=["#F59E0B", "#10B981"], width=0.5, edgecolor="black")
        ax2.set_title("Validation Perplexity (PPL) Karşılaştırması (Düşük = Daha İyi)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Perplexity (PPL)", fontsize=11)
        ax2.set_ylim(0, max(val_ppls) * 1.2)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in b2:
            y = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, y + max(val_ppls)*0.03, f"{y:.1f}", ha="center", fontweight="bold", fontsize=11)

        plt.tight_layout()
        fig5_path = "figures/05_model_karsilastirma_loss_ppl.png"
        plt.savefig(fig5_path, dpi=300)
        plt.close()

        # GRAFİK 6: BLEU, ROUGE ve Çıkarım Süresi (Latency) Karşılaştırması
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        metric_names = ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-L"]
        base_scores = [base_res["bleu1"], base_res["bleu2"], base_res["rouge1"], base_res["rougeL"]]
        gpt2_scores = [gpt2_results["bleu1"], gpt2_results["bleu2"], gpt2_results["rouge1"], gpt2_results["rougeL"]]

        x = range(len(metric_names))
        w = 0.35
        r1 = ax1.bar([i - w/2 for i in x], base_scores, width=w, label="Baseline Seq2Seq Bi-LSTM", color="#64748B", edgecolor="black")
        r2 = ax1.bar([i + w/2 for i in x], gpt2_scores, width=w, label="Fine-Tuned Turkish-GPT-2", color="#2563EB", edgecolor="black")
        ax1.set_title("Metin Benzerlik Metrikleri (BLEU & ROUGE - Yüksek = Daha İyi)", fontsize=11, fontweight="bold")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(metric_names, fontsize=10, fontweight="bold")
        ax1.set_ylabel("Skor (%)", fontsize=11)
        ax1.set_ylim(0, max(max(base_scores), max(gpt2_scores)) * 1.25 + 2)
        ax1.legend(fontsize=10)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in list(r1) + list(r2):
            yv = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yv + 0.3, f"{yv:.1f}", ha="center", fontsize=9, fontweight="bold")

        lats = [base_res["avg_inference_latency_ms"], gpt2_results["avg_inference_latency_ms"]]
        b3 = ax2.bar(models_lbl, lats, color=["#10B981", "#6366F1"], width=0.5, edgecolor="black")
        ax2.set_title("Ortalama Çıkarım Gecikmesi (Inference Latency - ms)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Süre (Milisaniye - ms)", fontsize=11)
        ax2.set_ylim(0, max(lats) * 1.22)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in b3:
            yv = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yv + max(lats)*0.03, f"{yv:.1f} ms", ha="center", fontweight="bold", fontsize=11)

        plt.tight_layout()
        fig6_path = "figures/06_metrik_ve_hiz_karsilastirmasi.png"
        plt.savefig(fig6_path, dpi=300)
        plt.close()
        print(f"✔ Karşılaştırma Grafikleri    : {fig5_path}, {fig6_path}")

    print("=" * 75)


if __name__ == "__main__":
    finetune_and_compare_gpt2()
