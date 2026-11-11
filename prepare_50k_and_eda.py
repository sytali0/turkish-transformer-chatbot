"""
Yeni Proje Önerisine Uygun 50.000 Diyalogluk Veri Seti Hazırlama ve Ara Sunum EDA Grafikleri
Proje: Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme
       ve Seq2Seq Karşılaştırmalı Başarım Analizi
Öğrenci: Seyit Ali Arslan (240711024)
Bölüm: Yapay Zeka ve Veri Mühendisliği, Atatürk Üniversitesi
"""

import os
import sys
import re
import html
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Windows konsolunda Türkçe karakter ayarı
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Matplotlib Türkçe font ve stil ayarları
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def clean_dialogue_text(text: str) -> str:
    """Diyalog metinlerini temizler, markdown kalıntılarını ve sondaki '---' işaretlerini kaldırır."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\ufffd", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Markdown kalın/italik işaretlerini ve sonda kalan '---' ayraçlarını temizle
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"[-—_]{2,}\s*$", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_natural_dialogue(inp: str, tgt: str) -> bool:
    """Doğal ve akıcı diyalog kriterlerini denetler."""
    if not inp or not tgt:
        return False
    if inp.lower() == tgt.lower():
        return False
    w_inp = inp.split()
    w_tgt = tgt.split()
    # Hem Seq2Seq LSTM hem GPT-2 için ideal diyalog uzunluğu (2-45 kelime)
    if not (2 <= len(w_inp) <= 35) or not (2 <= len(w_tgt) <= 45):
        return False
    letters_inp = sum(1 for c in inp if c.isalpha())
    letters_tgt = sum(1 for c in tgt if c.isalpha())
    if letters_inp < 3 or letters_tgt < 3:
        return False
    return True


def build_50k_dataset_and_plots(
    raw_csv="data/turkish_dialogues_100k.csv",
    output_dir="data",
    figures_dir="figures",
    target_size=50000,
    seed=42
):
    print("=" * 75)
    print("ARA SUNUM ADIM 1: 50.000 DİYALOG FİLTRELEME VE GÖRSEL ANALİZ (EDA)")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df_raw = pd.read_csv(raw_csv)
    print(f"-> Ham veri okundu: {len(df_raw):,} satır")

    df_raw["girdi"] = df_raw["kullanici_girdisi"].astype(str).apply(clean_dialogue_text)
    df_raw["yanit"] = df_raw["yanit"].astype(str).apply(clean_dialogue_text)
    df_raw["kaynak"] = df_raw["kaynak"].astype(str)

    # Doğal uzunluk ve kalite filtresi
    mask = [
        is_natural_dialogue(g, y)
        for g, y in zip(df_raw["girdi"], df_raw["yanit"])
    ]
    df_filtered = df_raw[mask].copy()
    df_filtered = df_filtered.drop_duplicates(subset=["girdi", "yanit"]).reset_index(drop=True)
    print(f"-> Doğal diyalog filtresi (2-45 kelime) sonrası: {len(df_filtered):,} satır")

    # Günlük sohbet kaynaklarına öncelik vererek tam 50.000 çift seçelim
    priority_sources = [
        "odmow_turkish_dialogues",
        "3nesdeniz_daily_dialogues",
        "customer_service_dialogues",
        "cisimcik_turkish_chat",
        "kilicai_multi_turn_dialogue",
        "sixfinger_chatbot",
    ]
    df_p1 = df_filtered[df_filtered["kaynak"].isin(priority_sources)]
    df_p2 = df_filtered[~df_filtered["kaynak"].isin(priority_sources)]

    remaining_needed = max(0, target_size - len(df_p1))
    if remaining_needed > 0 and len(df_p2) >= remaining_needed:
        df_p2_sampled = df_p2.sample(n=remaining_needed, random_state=seed)
        df_50k = pd.concat([df_p1, df_p2_sampled], ignore_index=True)
    else:
        df_50k = df_filtered.sample(n=min(target_size, len(df_filtered)), random_state=seed)

    # Karıştır (Shuffle)
    df_50k = df_50k.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total_n = len(df_50k)
    print(f"-> Seçilen ve karıştırılan nihai veri boyutu: {total_n:,} diyalog çifti")

    # Kaynak dağılımını kaydet (Grafik 3 için)
    source_counts = df_50k["kaynak"].value_counts().to_dict()

    # Sadece 'girdi' ve 'yanit' sütunlarını tut
    df_final = df_50k[["girdi", "yanit"]].copy()

    # %80 Train (40.000), %10 Validation (5.000), %10 Test (5.000) Bölme
    train_n = int(total_n * 0.80)
    val_n = int(total_n * 0.10)

    train_df = df_final.iloc[:train_n].reset_index(drop=True)
    val_df = df_final.iloc[train_n:train_n + val_n].reset_index(drop=True)
    test_df = df_final.iloc[train_n + val_n:].reset_index(drop=True)

    print(f"\n[Veri Bölümleme (Split)]")
    print(f"  • Train (%80)      : {len(train_df):,} çift")
    print(f"  • Validation (%10) : {len(val_df):,} çift")
    print(f"  • Test (%10)       : {len(test_df):,} çift")

    # Dosyaları kaydet
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, s_df in splits.items():
        s_df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False, encoding="utf-8-sig")
        s_df.to_json(os.path.join(output_dir, f"{name}.jsonl"), orient="records", lines=True, force_ascii=False)

    # Kelime uzunluklarını hesapla
    inp_lens = df_final["girdi"].apply(lambda x: len(x.split()))
    tgt_lens = df_final["yanit"].apply(lambda x: len(x.split()))

    # -------------------------------------------------------------------------
    # GRAFİK 1: Veri Bölümleme (Train / Validation / Test) Dağılımı
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    labels = ["Eğitim (Train)\n%80", "Doğrulama (Validation)\n%10", "Test\n%10"]
    sizes = [len(train_df), len(val_df), len(test_df)]
    colors = ["#2563EB", "#10B981", "#F59E0B"]

    bars = ax1.bar(labels, sizes, color=colors, width=0.55, edgecolor="black", alpha=0.9)
    ax1.set_title("Veri Kümesi Bölümleme Sayıları (Toplam: 50.000 Çift)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Diyalog Çifti Sayısı", fontsize=11)
    ax1.set_ylim(0, max(sizes) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1000, f"{yval:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax2.pie(
        sizes,
        labels=["Train (40.000)", "Validation (5.000)", "Test (5.000)"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0.04, 0.04, 0.04),
        textprops={"fontsize": 10, "fontweight": "bold"}
    )
    ax2.set_title("Veri Bölme Oranları (%80 / %10 / %10)", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "01_veri_bolumleme_split.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # GRAFİK 2: Girdi ve Yanıt Kelime Uzunluğu Dağılımı Histogramı
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ax1.hist(inp_lens, bins=30, color="#3B82F6", edgecolor="black", alpha=0.8)
    ax1.axvline(inp_lens.mean(), color="#DC2626", linestyle="--", linewidth=2, label=f"Ortalama: {inp_lens.mean():.1f} kelime")
    ax1.set_title("Kullanıcı Girdisi (Input) Kelime Sayısı Dağılımı", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Kelime Sayısı", fontsize=11)
    ax1.set_ylabel("Frekans (Cümle Sayısı)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.hist(tgt_lens, bins=35, color="#10B981", edgecolor="black", alpha=0.8)
    ax2.axvline(tgt_lens.mean(), color="#DC2626", linestyle="--", linewidth=2, label=f"Ortalama: {tgt_lens.mean():.1f} kelime")
    ax2.set_title("Model Yanıtı (Target) Kelime Sayısı Dağılımı", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Kelime Sayısı", fontsize=11)
    ax2.set_ylabel("Frekans (Cümle Sayısı)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "02_kelime_uzunlugu_dagilimi.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # GRAFİK 3: Hugging Face Alt Veri Kaynakları Dağılımı
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    readable_names = {
        "ba2han_turkish_chat": "Ba2han / Turkish_Chat",
        "cisimcik_turkish_chat": "cisimcik / turkish-chat",
        "customer_service_dialogues": "emreseyhan / Customer-Service",
        "3nesdeniz_daily_dialogues": "3nesdeniz / Daily-Dialogues",
        "kilicai_multi_turn_dialogue": "kilicai / Multi-Turn-Dialogue",
        "odmow_turkish_dialogues": "odmow / turkish-dialogues",
        "sixfinger_chatbot": "sixfinger / Chatbot-TR"
    }
    sorted_items = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    src_labels = [readable_names.get(k, k) for k, _ in sorted_items]
    src_vals = [v for _, v in sorted_items]

    bars = ax.barh(src_labels[::-1], src_vals[::-1], color="#6366F1", edgecolor="black", alpha=0.88)
    ax.set_title("Derlenen 50.000 Türkçe Diyalog Çiftinin Hugging Face Kaynak Dağılımı", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Diyalog Çifti Sayısı", fontsize=11)
    ax.set_xlim(0, max(src_vals) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 400, bar.get_y() + bar.get_height() / 2.0, f"{w:,} ({w/total_n*100:.1f}%)", va="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "03_veri_kaynaklari_dagilimi.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # Özet JSON güncelle
    summary = {
        "project": "Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme ve Seq2Seq Karşılaştırmalı Başarım Analizi",
        "student": "Seyit Ali Arslan (240711024)",
        "university": "Atatürk Üniversitesi, Yapay Zeka ve Veri Mühendisliği",
        "total_clean_pairs": total_n,
        "splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "avg_input_words": round(float(inp_lens.mean()), 2),
        "avg_target_words": round(float(tgt_lens.mean()), 2),
        "median_input_words": int(inp_lens.median()),
        "median_target_words": int(tgt_lens.median()),
        "source_distribution": source_counts,
        "generated_figures": [fig1_path, fig2_path, fig3_path]
    }
    summary_path = os.path.join(output_dir, "preprocessing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Üretilen Sunum Grafikleri]")
    print(f"  1. {fig1_path}")
    print(f"  2. {fig2_path}")
    print(f"  3. {fig3_path}")
    print(f"  4. Özet Rapor: {summary_path}")
    print("=" * 75)
    print("ADIM 1 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    build_50k_dataset_and_plots()
