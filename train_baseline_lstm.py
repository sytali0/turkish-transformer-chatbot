"""
Baseline Seq2Seq Bi-LSTM + Bahdanau Attention Eğitim ve Değerlendirme Betiği
Proje: Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme
       ve Seq2Seq Karşılaştırmalı Başarım Analizi
Öğrenci: Seyit Ali Arslan (240711024)
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from dataset import get_dataloaders
from models.baseline_seq2seq_lstm import Seq2SeqBiLSTM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def compute_bleu_and_rouge(references: list, hypotheses: list):
    """Türkçe sondan eklemeli yapıya uygun kök/ön-ek (F5-stem) normalize edilmiş BLEU-1, BLEU-2 ve ROUGE-1, ROUGE-L skorlarını hesaplar."""
    import re
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

    tr_map = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")
    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    bleu1_scores, bleu2_scores = [], []
    rouge1_scores, rougeL_scores = [], []

    for ref, hyp in zip(references, hypotheses):
        ref_norm = re.sub(r"[^\w\s]", " ", ref.translate(tr_map).lower())
        hyp_norm = re.sub(r"[^\w\s]", " ", hyp.translate(tr_map).lower())
        ref_tokens = [w[:5] for w in ref_norm.split() if len(w) > 1]
        hyp_tokens = [w[:5] for w in hyp_norm.split() if len(w) > 1]
        if not hyp_tokens or not ref_tokens:
            bleu1_scores.append(0.0)
            bleu2_scores.append(0.0)
            rouge1_scores.append(0.0)
            rougeL_scores.append(0.0)
            continue

        b1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
        b2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        bleu1_scores.append(b1 * 100.0)
        bleu2_scores.append(b2 * 100.0)

        r_scores = scorer.score(" ".join(ref_tokens), " ".join(hyp_tokens))
        rouge1_scores.append(r_scores["rouge1"].fmeasure * 100.0)
        rougeL_scores.append(r_scores["rougeL"].fmeasure * 100.0)

    return {
        "bleu1": round(sum(bleu1_scores) / max(1, len(bleu1_scores)), 2),
        "bleu2": round(sum(bleu2_scores) / max(1, len(bleu2_scores)), 2),
        "rouge1": round(sum(rouge1_scores) / max(1, len(rouge1_scores)), 2),
        "rougeL": round(sum(rougeL_scores) / max(1, len(rougeL_scores)), 2)
    }


def train_and_evaluate_baseline(
    epochs: int = 5,
    batch_size: int = 128,
    max_len: int = 40,
    lr: float = 0.0015,
    max_train_batches: int = 150,
    max_val_batches: int = 30
):
    print("=" * 75)
    print("ADIM 3: BASELINE SEQ2SEQ BI-LSTM + BAHDANAU ATTENTION EĞİTİMİ")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Çalışma Cihazı: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    tokenizer = Tokenizer.from_file("tokenizer/turkish_bpe_12k.json")
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    train_loader, val_loader = get_dataloaders(
        train_csv="data/train.csv",
        val_csv="data/val.csv",
        tokenizer_file="tokenizer/turkish_bpe_12k.json",
        batch_size=batch_size,
        max_len=max_len
    )

    model = Seq2SeqBiLSTM(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Model Parametre Sayısı: {total_params:,} (~{total_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

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

        tf_ratio = max(0.3, 0.75 - (epoch - 1) * 0.1)

        for step, batch in enumerate(train_loader):
            if step >= max_train_batches:
                break
            src = batch["encoder_input"].to(device)
            trg_in = batch["decoder_input"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(src, trg_in, teacher_forcing_ratio=tf_ratio)
            # outputs: (batch, seq_len, vocab_size)
            loss = criterion(outputs.reshape(-1, vocab_size), target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, train_steps)
        train_ppl = math.exp(min(avg_train_loss, 15.0))

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if step >= max_val_batches:
                    break
                src = batch["encoder_input"].to(device)
                trg_in = batch["decoder_input"].to(device)
                target = batch["target"].to(device)

                outputs = model(src, trg_in, teacher_forcing_ratio=0.0)
                loss = criterion(outputs.reshape(-1, vocab_size), target.reshape(-1))
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

        print(f"Epoch [{epoch}/{epochs}] | Süre: {elapsed:.1f}s | Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.1f}) | Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.1f})")

    # Modeli kaydet
    ckpt_path = "checkpoints/baseline_seq2seq_lstm.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✔ Baseline Model Ağırlıkları Kaydedildi: {ckpt_path}")

    # Çıkarım Süresi (Inference Latency) ve BLEU / ROUGE Değerlendirmesi
    print("-> Çıkarım Gecikmesi (Latency) ve BLEU/ROUGE hesaplanıyor...")
    val_batch = next(iter(val_loader))
    refs, hyps, latencies = [], [], []
    sample_demos = []

    for i in range(min(30, len(val_batch["src_text"]))):
        src_txt = val_batch["src_text"][i]
        tgt_txt = val_batch["tgt_text"][i]
        enc_ids = [bos_id] + tokenizer.encode(src_txt).ids[:max_len - 2] + [eos_id]
        reply_txt, lat_ms, _ = model.generate_reply(enc_ids, tokenizer, max_len=30, device=device)
        refs.append(tgt_txt)
        hyps.append(reply_txt)
        latencies.append(lat_ms)
        if i < 5:
            sample_demos.append({"girdi": src_txt, "gercek_yanit": tgt_txt, "baseline_yanit": reply_txt, "latency_ms": round(lat_ms, 2)})

    text_metrics = compute_bleu_and_rouge(refs, hyps)
    avg_latency_ms = round(sum(latencies) / max(1, len(latencies)), 2)

    results_data = {
        "model_name": "Baseline: 2-Layer Seq2Seq Bi-LSTM + Bahdanau Attention",
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

    with open("results/baseline_lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    # Eğitim Eğrisi Grafiği Çizdirme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    ep = history["epoch"]

    ax1.plot(ep, history["train_loss"], marker="o", linewidth=2.2, color="#2563EB", label="Train Loss")
    ax1.plot(ep, history["val_loss"], marker="s", linewidth=2.2, color="#DC2626", label="Validation Loss")
    ax1.set_title("Baseline Seq2Seq Bi-LSTM: Epoch / Loss Eğrisi", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_xticks(ep)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(ep, history["train_ppl"], marker="o", linewidth=2.2, color="#10B981", label="Train Perplexity (PPL)")
    ax2.plot(ep, history["val_ppl"], marker="s", linewidth=2.2, color="#F59E0B", label="Validation Perplexity (PPL)")
    ax2.set_title("Baseline Seq2Seq Bi-LSTM: Perplexity (PPL) Değişimi", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Perplexity (PPL)", fontsize=11)
    ax2.set_xticks(ep)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = "figures/04_baseline_lstm_egitim_egrisi.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"✔ Eğitim Grafiği Kaydedildi   : {fig_path}")
    print(f"✔ Metrik Raporu Kaydedildi    : results/baseline_lstm_metrics.json")
    print(f"✔ Ortalama Çıkarım Süresi     : {avg_latency_ms} ms")
    print(f"✔ BLEU-1: {text_metrics['bleu1']} | BLEU-2: {text_metrics['bleu2']} | ROUGE-L: {text_metrics['rougeL']}")
    print("=" * 75)


if __name__ == "__main__":
    train_and_evaluate_baseline()
