"""
Ham Veri Setlerini İndirme ve Fiziksel Dosya Olarak Kaydetme Betiği
Proje: Turkish Transformer-Based Chatbot
Öğrenci: Seyit Ali Arslan (240711024)

Bu betik iki farklı kaynaktan ham Türkçe diyalog verilerini çeker ve
doğrudan Excel veya metin düzenleyicilerle inceleyebilmeniz için
'data/' klasörüne hem CSV hem de JSONL formatında kaydeder:
  1. Hugging Face Açık Kaynaklı Türkçe Diyaloglar
  2. OPUS OpenSubtitles Türkçe Dizi/Film Altyazı Diyalogları
"""

import os
import sys
import zlib
import json
import pandas as pd
import requests
import certifi
from datasets import load_dataset
from tqdm import tqdm

# Windows konsolunda Türkçe karakterlerin düzgün görünmesi için
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def download_huggingface_raw(output_csv="data/huggingface_dialogues_raw.csv",
                             output_jsonl="data/huggingface_dialogues_raw.jsonl",
                             max_samples=10000):
    """
    Hugging Face üzerindeki hazır Türkçe diyalog veri setlerini indirir.
    Hiçbir karmaşık temizlik yapmadan ham hallerini kaydeder.
    """
    print("\n" + "=" * 60)
    print("1. HUGGING FACE TÜRKÇE DİYALOGLARI İNDİRİLİYOR...")
    print("=" * 60)
    
    rows = []
    
    # Kaynak 1: odmow/turkish-dialogues (Günlük konuşma diyalogları)
    print("-> 'odmow/turkish-dialogues' indiriliyor...")
    try:
        ds1 = load_dataset("odmow/turkish-dialogues", split="train")
        for item in ds1:
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if inp and out:
                rows.append({
                    "kullanici_girdisi": inp,
                    "yanit": out,
                    "kaynak": "hf_odmow_turkish_dialogues"
                })
        print(f"   Alınan diyalog sayısı: {len(rows)}")
    except Exception as e:
        print(f"   Hata oluştu: {e}")

    # Kaynak 2: cisimcik/turkish-chat-max-25k (Kullanıcı-Asistan sohbetleri)
    print("-> 'cisimcik/turkish-chat-max-25k' taranıyor...")
    try:
        ds2 = load_dataset("cisimcik/turkish-chat-max-25k", split="train")
        hf_count_before = len(rows)
        for item in ds2:
            messages = item.get("messages", [])
            for i in range(len(messages) - 1):
                m1 = messages[i]
                m2 = messages[i + 1]
                if m1.get("role") == "user" and m2.get("role") == "assistant":
                    u_text = m1.get("content", "").strip()
                    a_text = m2.get("content", "").strip()
                    # Çok uzun metinleri (makale/kod) basitçe sınırlayalım
                    if 5 < len(u_text) < 300 and 5 < len(a_text) < 300:
                        rows.append({
                            "kullanici_girdisi": u_text,
                            "yanit": a_text,
                            "kaynak": "hf_cisimcik_turkish_chat"
                        })
                        if len(rows) >= max_samples:
                            break
            if len(rows) >= max_samples:
                break
        print(f"   Eklenen sohbet sayısı: {len(rows) - hf_count_before}")
    except Exception as e:
        print(f"   Hata oluştu: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # CSV olarak kaydet (utf-8-sig: Excel'de çift tıklayınca Türkçe karakterler bozulmaz)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    # JSONL olarak kaydet
    df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
    
    print(f"\n✔ Hugging Face verisi kaydedildi:")
    print(f"  - CSV  (Excel ile açılabilir): {output_csv} ({len(df)} satır)")
    print(f"  - JSONL (Kod ile okunabilir) : {output_jsonl}")
    return df


def download_opus_raw(output_csv="data/opus_subtitles_raw.csv",
                      output_jsonl="data/opus_subtitles_raw.jsonl",
                      max_samples=10000):
    """
    OPUS OpenSubtitles arşivinden ham Türkçe altyazı diyaloglarını çeker.
    Ardışık konuşma repliklerini eşleştirerek kaydeder.
    """
    print("\n" + "=" * 60)
    print("2. OPUS OPENSUBTITLES TÜRKÇE ALTYAZILARI İNDİRİLİYOR...")
    print("=" * 60)
    
    url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/tr.txt.gz"
    print(f"-> OPUS akışına bağlanılıyor: {url}")
    
    response = requests.get(url, stream=True, verify=certifi.where(), timeout=30)
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    
    rows = []
    buffer = ""
    prev_line = None
    
    pbar = tqdm(total=max_samples, desc="Altyazı Çiftleri Alınıyor", unit=" satır")
    
    for chunk in response.iter_content(chunk_size=131072):
        try:
            decompressed = d.decompress(chunk).decode("utf-8", errors="ignore")
        except Exception:
            continue
        buffer += decompressed
        
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            
            # Boş veya çok kısa satırları atla
            if not line or len(line) < 3:
                prev_line = None
                continue
                
            # Ardışık iki repliği soru-cevap / diyalog çifti olarak eşle
            if prev_line is not None:
                rows.append({
                    "kullanici_girdisi": prev_line,
                    "yanit": line,
                    "kaynak": "opus_opensubtitles_tr"
                })
                pbar.update(1)
                prev_line = None  # Replik kullanıldı, sıfırla
                if len(rows) >= max_samples:
                    break
            else:
                prev_line = line
                
        if len(rows) >= max_samples:
            break
            
    response.close()
    pbar.close()
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # CSV ve JSONL olarak kaydet
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
    
    print(f"\n✔ OPUS verisi kaydedildi:")
    print(f"  - CSV  (Excel ile açılabilir): {output_csv} ({len(df)} satır)")
    print(f"  - JSONL (Kod ile okunabilir) : {output_jsonl}")
    return df


if __name__ == "__main__":
    print("Veri indirme işlemi başlatılıyor...")
    
    # İlk etapta rahatça açıp fiziksel olarak inceleyebilmeniz için
    # her iki kaynaktan da örnek 10.000'er ham çift indiriyoruz.
    hf_df = download_huggingface_raw(max_samples=10000)
    opus_df = download_opus_raw(max_samples=10000)
    
    print("\n" + "=" * 60)
    print("ÖZET VE İNCELEME")
    print("=" * 60)
    print(f"Toplam HF Diyaloğu  : {len(hf_df)} satır")
    print(f"Toplam OPUS Repliği : {len(opus_df)} satır")
    print("\nMasaüstünüzdeki 'turkish-transformer-chatbot/data/' klasöründen")
    print("şu dosyaları doğrudan çift tıklayarak Excel veya Not Defteri ile açabilirsiniz:")
    print("  1. huggingface_dialogues_raw.csv")
    print("  2. opus_subtitles_raw.csv")
