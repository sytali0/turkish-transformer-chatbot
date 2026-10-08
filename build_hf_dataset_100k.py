"""
Hugging Face 100.000 Türkçe Diyalog Veri Seti Oluşturucu
Proje: Turkish Transformer-Based Chatbot
Öğrenci: Seyit Ali Arslan (240711024)

Bu betik Hugging Face üzerindeki açık kaynaklı Türkçe diyalog, sohbet
ve soru-cevap veri setlerini bir araya getirerek tam 100.000 diyalog çifti üretir.
Çıktıyı Excel'de doğrudan inceleyebilmeniz için CSV ve kod için JSONL olarak kaydeder.
"""

import os
import sys
import json
import re
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Windows konsolunda Türkçe karakter ayarı
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean_line(text: str) -> str:
    """Temel temizlik ve boşluk normalizasyonu."""
    if not text:
        return ""
    # Kod bloklarını ve markdown linklerini sadeleştir
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    # Çoklu boşlukları ve satır sonlarını temizle
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_pair(inp: str, tgt: str) -> bool:
    """Chatbot eğitimi için uygun uzunlukta ve kalitede mi kontrolü."""
    if not inp or not tgt:
        return False
    w_inp = inp.split()
    w_tgt = tgt.split()
    # 2 ila 70 kelime arasındaki doğal diyalogları alalım
    if not (2 <= len(w_inp) <= 50) or not (2 <= len(w_tgt) <= 75):
        return False
    # Papağan yanıtı engelle
    if inp.lower() == tgt.lower():
        return False
    return True


def collect_100k_hf_dialogues(target_count=100000,
                              output_csv="data/turkish_dialogues_100k.csv",
                              output_jsonl="data/turkish_dialogues_100k.jsonl",
                              summary_file="data/dataset_info.json"):
    print("=" * 70)
    print(f"HUGGING FACE TÜRKÇE DİYALOG DERLEME (HEDEF: {target_count:,} ÇİFT)")
    print("=" * 70)
    
    collected = []
    seen = set()
    source_stats = {}

    def add_pair(inp, tgt, src):
        nonlocal collected
        if len(collected) >= target_count:
            return False
        inp_c = clean_line(inp)
        tgt_c = clean_line(tgt)
        if is_valid_pair(inp_c, tgt_c):
            key = (inp_c.lower(), tgt_c.lower())
            if key not in seen:
                seen.add(key)
                collected.append({
                    "kullanici_girdisi": inp_c,
                    "yanit": tgt_c,
                    "kaynak": src
                })
                source_stats[src] = source_stats.get(src, 0) + 1
                return True
        return False

    # 1. odmow/turkish-dialogues (Günlük konuşmalar)
    print("\n[1/7] 'odmow/turkish-dialogues' taranıyor...")
    try:
        ds = load_dataset("odmow/turkish-dialogues", split="train")
        for item in ds:
            add_pair(item.get("input", ""), item.get("output", ""), "odmow_turkish_dialogues")
        print(f"      Toplanan: {source_stats.get('odmow_turkish_dialogues', 0):,} çift")
    except Exception as e:
        print(f"      Hata: {e}")

    # 2. sixfingerdev/chatbot-turkish-dataset-sixfinger-2b
    print("\n[2/7] 'sixfingerdev/chatbot-turkish-dataset' taranıyor...")
    try:
        ds = load_dataset("sixfingerdev/chatbot-turkish-dataset-sixfinger-2b", split="train")
        for item in ds:
            add_pair(item.get("input", ""), item.get("output", ""), "sixfinger_chatbot")
        print(f"      Toplanan: {source_stats.get('sixfinger_chatbot', 0):,} çift")
    except Exception as e:
        print(f"      Hata: {e}")

    # 3. 3nesdeniz/turkish-daily-dialogues-5k (Günlük yaşam diyalogları)
    print("\n[3/7] '3nesdeniz/turkish-daily-dialogues-5k' taranıyor...")
    try:
        for split_name in ["train", "validation", "test"]:
            ds = load_dataset("3nesdeniz/turkish-daily-dialogues-5k", split=split_name)
            for item in ds:
                msgs = item.get("messages", [])
                for i in range(len(msgs) - 1):
                    if msgs[i].get("role") == "user" and msgs[i+1].get("role") == "assistant":
                        add_pair(msgs[i].get("content", ""), msgs[i+1].get("content", ""), "3nesdeniz_daily_dialogues")
        print(f"      Toplanan: {source_stats.get('3nesdeniz_daily_dialogues', 0):,} çift")
    except Exception as e:
        print(f"      Hata: {e}")

    # 4. emreseyhan/Turkish-customer-service-conversations (Müşteri - Temsilci diyalogları)
    print("\n[4/7] 'emreseyhan/Turkish-customer-service' taranıyor...")
    try:
        ds = load_dataset("emreseyhan/Turkish-customer-service-conversations", split="train")
        for item in ds:
            msgs = item.get("messages", [])
            for i in range(len(msgs) - 1):
                if msgs[i].get("role") == "user" and msgs[i+1].get("role") == "assistant":
                    add_pair(msgs[i].get("content", ""), msgs[i+1].get("content", ""), "customer_service_dialogues")
        print(f"      Toplanan: {source_stats.get('customer_service_dialogues', 0):,} çift")
    except Exception as e:
        print(f"      Hata: {e}")

    # 5. kilicai/turkish-sft-multi-turn-dialogue-10k (Soru - Cevap ve Çok Turlu Diyaloglar)
    print("\n[5/7] 'kilicai/turkish-sft-multi-turn-dialogue-10k' taranıyor...")
    try:
        ds = load_dataset("kilicai/turkish-sft-multi-turn-dialogue-10k", split="train")
        for item in ds:
            msgs = item.get("messages", [])
            for i in range(len(msgs) - 1):
                if msgs[i].get("role") == "user" and msgs[i+1].get("role") == "assistant":
                    add_pair(msgs[i].get("content", ""), msgs[i+1].get("content", ""), "kilicai_multi_turn_dialogue")
                    if len(collected) >= target_count:
                        break
            if len(collected) >= target_count:
                break
        print(f"      Toplanan: {source_stats.get('kilicai_multi_turn_dialogue', 0):,} çift")
    except Exception as e:
        print(f"      Hata: {e}")

    # 6. cisimcik/turkish-chat-max-25k (Genel Sohbet Diyalogları)
    print("\n[6/7] 'cisimcik/turkish-chat-max-25k' taranıyor...")
    try:
        ds = load_dataset("cisimcik/turkish-chat-max-25k", split="train")
        for item in ds:
            msgs = item.get("messages", [])
            for i in range(len(msgs) - 1):
                if msgs[i].get("role") == "user" and msgs[i+1].get("role") == "assistant":
                    add_pair(msgs[i].get("content", ""), msgs[i+1].get("content", ""), "cisimcik_turkish_chat")
                    if len(collected) >= target_count:
                        break
            if len(collected) >= target_count:
                break
        print(f"      Toplanan: {source_stats.get('cisimcik_turkish_chat', 0):,} çift")
    except Exception as e:
        print(f"      Hata: {e}")

    # 7. Ba2han/Turkish_Chat-1402 (100.000'e tamamlayıcı geniş Türkçe diyalog havuzu)
    if len(collected) < target_count:
        print(f"\n[7/7] 'Ba2han/Turkish_Chat-1402' taranıyor (Mevcut: {len(collected):,}, Hedef: {target_count:,})...")
        try:
            ds = load_dataset("Ba2han/Turkish_Chat-1402", split="train")
            pbar = tqdm(total=target_count - len(collected), desc="Ba2han Diyalogları Ekleniyor", unit=" çift")
            for item in ds:
                msgs = item.get("messages", [])
                for i in range(len(msgs) - 1):
                    if msgs[i].get("role") == "user" and msgs[i+1].get("role") == "assistant":
                        if add_pair(msgs[i].get("content", ""), msgs[i+1].get("content", ""), "ba2han_turkish_chat"):
                            pbar.update(1)
                            if len(collected) >= target_count:
                                break
                if len(collected) >= target_count:
                    break
            pbar.close()
            print(f"      Toplanan: {source_stats.get('ba2han_turkish_chat', 0):,} çift")
        except Exception as e:
            print(f"      Hata: {e}")

    print("\n" + "=" * 70)
    print(f"TOPLAM TOPLANAN TEMİZ DİYALOG ÇİFTİ: {len(collected):,}")
    print("=" * 70)

    # DataFrame'e dönüştür ve kaydet
    df = pd.DataFrame(collected)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"\nDosyalar kaydediliyor...")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
    
    summary = {
        "project": "Turkish Transformer-Based Chatbot",
        "student": "Seyit Ali Arslan (240711024)",
        "total_pairs": len(df),
        "source_breakdown": source_stats,
        "avg_input_length_words": round(df["kullanici_girdisi"].apply(lambda x: len(x.split())).mean(), 2),
        "avg_target_length_words": round(df["yanit"].apply(lambda x: len(x.split())).mean(), 2),
        "csv_path": output_csv,
        "jsonl_path": output_jsonl
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✔ CSV Kaydedildi   : {output_csv} ({len(df):,} satır)")
    print(f"✔ JSONL Kaydedildi : {output_jsonl}")
    print(f"✔ Özet Bilgi       : {summary_file}")
    return df


if __name__ == "__main__":
    collect_100k_hf_dialogues()
