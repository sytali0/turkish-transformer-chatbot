"""
Veri Ön İşleme, Temizleme, Karıştırma (Shuffle) ve Bölme (Split) Boru Hattı
Proje: Turkish Transformer-Based Chatbot
Öğrenci: Seyit Ali Arslan (240711024)

Bu betik:
1. 'kaynak' sütununu kaldırarak yalnızca 'girdi' ve 'yanit' bilgilerini tutar.
2. HTML etiketlerini, bozuk karakterleri, aşırı boşlukları ve anlamsız/boş satırları temizler.
3. Mükerrer (duplicate) çiftleri eler.
4. Verileri rastgele karıştırır (Shuffle - Seed: 42).
5. Proje önerisine uygun olarak %80 Train / %10 Validation / %10 Test oranında böler.
6. Sonuçları hem CSV (Excel ile açılabilir) hem JSONL olarak 'data/' altına kaydeder.
"""

import os
import sys
import re
import html
import json
import pandas as pd

# Windows konsolunda Türkçe karakter ayarı
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean_text(text: str) -> str:
    """Metin temizleme fonksiyonu."""
    if not isinstance(text, str):
        return ""
        
    # 1. HTML varlıklarını çöz (&amp; -> &, &quot; -> ", &#39; -> ' vb.)
    text = html.unescape(text)
    
    # 2. HTML etiketlerini kaldır (<p>, </div>, <br>, <span> vb.)
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 3. Bozuk unicode karakterlerini (\ufffd, kontrol karakterleri vb.) temizle
    text = text.replace("\ufffd", "")
    # Yazdırılamayan görünmez kontrol karakterlerini temizle (sekme ve yeni satır hariç)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    
    # 4. Markdown bağlantı ve resim kalıntılarını sadeleştir
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    
    # 5. Satır sonlarını boşluğa çevir ve aşırı boşlukları normalize et
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def is_valid_line(text: str) -> bool:
    """Cümlenin boş veya anlamsız olup olmadığını denetler."""
    if not text or len(text) < 2:
        return False
        
    # Sadece noktalama veya sayılardan oluşan boş/anlamsız girdileri engelle
    letters = sum(1 for c in text if c.isalpha())
    if letters < 2:
        return False
        
    return True


def run_preprocessing_and_split(
    input_file="data/turkish_dialogues_100k.csv",
    output_dir="data",
    seed=42
):
    print("=" * 70)
    print("VERİ ÖN İŞLEME, TEMİZLEME, SHUFFLE VE BÖLME İŞLEMİ BAŞLATILIYOR")
    print("=" * 70)
    
    if not os.path.exists(input_file):
        print(f"Hata: {input_file} dosyası bulunamadı!")
        return
        
    print(f"-> Veri okunuyor: {input_file}")
    df_raw = pd.read_csv(input_file)
    print(f"   Başlangıçtaki toplam satır sayısı: {len(df_raw):,}")

    # 1. 'kaynak' bölümünü kaldır, yalnızca 'girdi' ve 'yanit' sütunlarını al
    print("\n[Adım 1] 'kaynak' sütunu kaldırılıyor, sütunlar 'girdi' ve 'yanit' olarak adlandırılıyor...")
    df = pd.DataFrame()
    df["girdi"] = df_raw["kullanici_girdisi"].astype(str)
    df["yanit"] = df_raw["yanit"].astype(str)
    
    # 2. Temizleme (HTML, bozuk karakterler, aşırı boşluklar)
    print("[Adım 2] HTML etiketleri, bozuk karakterler ve aşırı boşluklar temizleniyor...")
    df["girdi"] = df["girdi"].apply(clean_text)
    df["yanit"] = df["yanit"].apply(clean_text)
    
    # 3. Boş ve geçersiz satırları filtrele
    print("[Adım 3] Boş, anlamsız veya papağan (girdi==yanıt) satırlar filtreleniyor...")
    valid_mask = (
        df["girdi"].apply(is_valid_line) & 
        df["yanit"].apply(is_valid_line) & 
        (df["girdi"].str.lower() != df["yanit"].str.lower())
    )
    df_clean = df[valid_mask].copy()
    print(f"   Filtreleme sonrası kalan satır sayısı: {len(df_clean):,}")
    
    # 4. Tekilleştirme (Duplicate temizliği)
    print("[Adım 4] Mükerrer (duplicate) çiftler ayıklanıyor...")
    df_clean = df_clean.drop_duplicates(subset=["girdi", "yanit"]).reset_index(drop=True)
    total_clean = len(df_clean)
    print(f"   Tekilleştirme sonrası nihai temiz diyalog sayısı: {total_clean:,}")

    # 5. Karıştırma (Shuffle)
    print(f"\n[Adım 5] Veri kümesi karıştırılıyor (Shuffle - Seed: {seed})...")
    df_shuffled = df_clean.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 6. Bölme (Split: %80 Train / %10 Val / %10 Test)
    print("\n[Adım 6] Veri bölme işlemi uygulanıyor (%80 Train, %10 Val, %10 Test)...")
    train_n = int(total_clean * 0.80)
    val_n = int(total_clean * 0.10)
    
    train_df = df_shuffled.iloc[:train_n].reset_index(drop=True)
    val_df = df_shuffled.iloc[train_n:train_n + val_n].reset_index(drop=True)
    test_df = df_shuffled.iloc[train_n + val_n:].reset_index(drop=True)
    
    print(f"   • Train Seti (Eğitim - %80)     : {len(train_df):,} satır")
    print(f"   • Val Seti   (Doğrulama - %10)  : {len(val_df):,} satır")
    print(f"   • Test Seti  (Test - %10)       : {len(test_df):,} satır")
    print(f"   • Toplam                        : {len(train_df) + len(val_df) + len(test_df):,} satır")

    # 7. Dosyaları kaydetme (Hem CSV hem JSONL)
    print("\n[Adım 7] Dosyalar kaydediliyor...")
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    
    for name, s_df in splits.items():
        csv_path = os.path.join(output_dir, f"{name}.csv")
        jsonl_path = os.path.join(output_dir, f"{name}.jsonl")
        
        # utf-8-sig: Excel'de çift tıklayınca Türkçe karakterlerin tam doğru açılması için
        s_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        s_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        print(f"   ✔ {name:<5}: CSV -> {csv_path} | JSONL -> {jsonl_path}")

    # Özet istatistik dosyası
    summary = {
        "project": "Turkish Transformer-Based Chatbot",
        "student": "Seyit Ali Arslan (240711024)",
        "initial_total": len(df_raw),
        "cleaned_total": total_clean,
        "columns": list(train_df.columns),
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "ratios": "80% Train / 10% Val / 10% Test",
        "sample_dialogue": {
            "girdi": train_df.iloc[0]["girdi"],
            "yanit": train_df.iloc[0]["yanit"]
        }
    }
    
    summary_path = os.path.join(output_dir, "preprocessing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print(f"\n✔ Özet rapor kaydedildi: {summary_path}")
    print("=" * 70)
    print("ÖN İŞLEME VE BÖLME ADIMI BAŞARIYLA TAMAMLANDI!")
    print("=" * 70)


if __name__ == "__main__":
    run_preprocessing_and_split()
