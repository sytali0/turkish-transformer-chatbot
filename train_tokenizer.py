"""
Özel Türkçe Byte-Pair Encoding (BPE) Tokenizer Eğitimi
Proje: Turkish Transformer-Based Chatbot
Öğrenci: Seyit Ali Arslan (240711024)

Bu betik:
1. 'data/train.csv' dosyasındaki eğitim verilerini (girdi ve yanıt) okur.
2. Türkçe'nin sondan eklemeli yapısına uygun, 12.000 kelime dağarcıklı (Vocab Size)
   özel bir Byte-Pair Encoding (BPE) Tokenizer modelini sıfırdan eğitir.
3. Model için kritik özel belirteçleri tanımlar:
   - [PAD] : 0 (Dizi boyutu eşitleme - Padding)
   - [UNK] : 1 (Bilinmeyen karakterler - Unknown)
   - [BOS] : 2 (Dizi başlangıcı - Beginning of Sequence)
   - [EOS] : 3 (Dizi sonu - End of Sequence)
   - [SEP] : 4 (Girdi ve Yanıt ayracı - Separator)
4. Eğitilen modeli 'tokenizer/turkish_bpe_12k.json' olarak kaydeder.
5. Türkçe morfolojik alt-kelime (subword) ayrıştırma testlerini çalıştırır.
"""

import os
import sys
import json
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Windows konsolunda Türkçe karakter ayarı
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def text_iterator(csv_path: str, chunk_size: int = 5000):
    """
    Büyük veri setini belleği şişirmeden parça parça (chunk) okur.
    Hem 'girdi' hem 'yanit' metinlerini sözlük eğitimine besler.
    """
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        texts = []
        for _, row in chunk.iterrows():
            g = str(row.get("girdi", "")).strip()
            y = str(row.get("yanit", "")).strip()
            if g:
                texts.append(g)
            if y:
                texts.append(y)
        yield texts


def train_bpe_tokenizer(
    train_csv="data/train.csv",
    output_dir="tokenizer",
    vocab_size=12000
):
    print("=" * 70)
    print("TÜRKÇE BYTE-PAIR ENCODING (BPE) TOKENIZER EĞİTİMİ")
    print("=" * 70)
    
    if not os.path.exists(train_csv):
        print(f"Hata: {train_csv} dosyası bulunamadı!")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "turkish_bpe_12k.json")
    
    print(f"-> Eğitim Verisi  : {train_csv}")
    print(f"-> Hedef Vocab Size: {vocab_size:,}")
    print(f"-> Çıktı Dosyası   : {model_path}")
    print("-> Model Tipi      : ByteLevel BPE (Türkçe morfolojiye duyarlı)")

    # 1. BPE Tokenizer Temelini Kur
    # ByteLevel pre-tokenizer: Unicode karakterleri bayt seviyesinde işleyerek OOV (Out-of-Vocabulary) hatasını sıfırlar
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 2. Özel Belirteçler (Special Tokens)
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]

    # 3. Eğitici (Trainer) Yapılandırması
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )

    # 4. Eğitimi Başlat
    print("\n[Adım 1] Tokenizer eğitimi başlatılıyor (Bu işlem 30-60 saniye sürebilir)...")
    tokenizer.train_from_iterator(text_iterator(train_csv), trainer=trainer)
    print("✔ Eğitim başarıyla tamamlandı!")

    # 5. Modeli Kaydet
    print("\n[Adım 2] Model kaydediliyor...")
    tokenizer.save(model_path)
    
    # Ek yapılandırma dosyasını kaydet
    config = {
        "project": "Turkish Transformer-Based Chatbot",
        "student": "Seyit Ali Arslan (240711024)",
        "tokenizer_type": "BPE",
        "vocab_size": vocab_size,
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in special_tokens},
        "model_file": model_path
    }
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        
    print(f"✔ Tokenizer Modeli Kaydedildi : {model_path}")
    print(f"✔ Yapılandırma Dosyası        : {config_path}")

    # 6. Özel Belirteçleri Doğrula
    print("\n[Adım 3] Özel Belirteç ID Doğrulaması:")
    for tok in special_tokens:
        tok_id = tokenizer.token_to_id(tok)
        print(f"   • {tok:<6} -> ID: {tok_id}")

    # 7. Türkçe Sondan Eklemeli Dil Yapısı ve Alt-Kelime (Subword) Testi
    print("\n" + "=" * 70)
    print("TÜRKÇE MORFOLOJİ VE ALT-KELİME AYRIŞTIRMA TESTLERİ")
    print("=" * 70)
    
    test_cases = [
        "Merhaba, nasılsın?",
        "evlerindekilerden",
        "öğrenemeyenlerimizdenmişsinizcesine",
        "Geleneksel sinir ağları yerine Transformer mimarisi kullanıyoruz.",
        "Derin öğrenme modelleri Türkçe ek yapısını başarıyla öğreniyor."
    ]

    for sentence in test_cases:
        encoding = tokenizer.encode(sentence)
        tokens = encoding.tokens
        ids = encoding.ids
        decoded = tokenizer.decode(ids)
        
        print(f"\nCümle : \"{sentence}\"")
        print(f"  • Token Sayısı   : {len(tokens)}")
        print(f"  • Alt-Kelimeler  : {tokens}")
        print(f"  • Token ID'leri  : {ids}")
        print(f"  • Geri Çözülen   : \"{decoded}\"")
        assert decoded == sentence, "Kayıpsız geri çözme başarısız oldu!"

    print("\n" + "=" * 70)
    print("TÜM TESTLER BAŞARIYLA GEÇTİ! TOKENIZER KULLANIMA HAZIR.")
    print("=" * 70)
    return tokenizer


if __name__ == "__main__":
    train_bpe_tokenizer()
