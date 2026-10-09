# Turkish Transformer-Based Chatbot

**Öğrenci:** Seyit Ali Arslan (240711024)  
**Ders:** Deep Learning (2026–2027)  

Bu proje, Türkçe dil yapısına uygun sıfırdan bir Transformer tabanlı diyalog modeli (chatbot) geliştirmeyi ve geleneksel Seq2Seq LSTM baselineları ile karşılaştırmalı başarım analizini amaçlar.

---

## 📁 Proje Klasör Yapısı

```
turkish-transformer-chatbot/
│
├── data/
│   ├── train.csv                        # %80 Eğitim seti (79.967 diyalog çifti)
│   ├── train.jsonl                      # JSONL formatında eğitim seti
│   ├── val.csv                          # %10 Doğrulama seti (9.995 diyalog çifti)
│   ├── val.jsonl                        # JSONL formatında doğrulama seti
│   ├── test.csv                         # %10 Test seti (9.997 diyalog çifti)
│   ├── test.jsonl                       # JSONL formatında test seti
│   ├── preprocessing_summary.json       # Ön işleme ve bölme özet raporu
│   ├── turkish_dialogues_100k.csv       # Ham 100.000 Türkçe diyalog havuzu
│   └── dataset_info.json                # Ham veri seti istatistikleri
│
├── tokenizer/
│   ├── turkish_bpe_12k.json             # 12.000 Vocab Size Türkçe BPE modeli
│   └── tokenizer_config.json            # Özel belirteçler ve yapılandırma
│
├── train_tokenizer.py                   # 12k BPE Tokenizer eğitim betiği
├── preprocess_and_split.py              # Temizleme, shuffle ve %80/%10/%10 bölme betiği
├── build_hf_dataset_100k.py             # 100.000 diyalog derleyici betik
├── download_raw_data.py                 # Ham verileri indiren Python betiği
├── .gitignore                           # Git takip dışı dosyalar
└── README.md
```

---

## 🔤 Özel Türkçe BPE Tokenizer (12.000 Kelime Dağarcığı)

Türkçe sondan eklemeli bir dil olduğundan geleneksel kelime seviyesinde tokenizasyon sözlük patlamasına (Out-Of-Vocabulary) yol açar. Proje teklifine uygun olarak **12.000 Vocab Size** özel **Byte-Pair Encoding (BPE)** modeli sıfırdan eğitilmiştir:

* **Özel Belirteçler (Special Tokens):**
  * `[PAD]`: `0` (Dizi boyutu eşitleme)
  * `[UNK]`: `1` (Bilinmeyen karakter)
  * `[BOS]`: `2` (Dizi başlangıcı)
  * `[EOS]`: `3` (Dizi sonu)
  * `[SEP]`: `4` (Girdi/Yanıt ayracı)
* **Morfolojik Ayrıştırma Testi:**
  * Kelime: `evlerindekilerden` $\rightarrow$ `['ev', 'lerindeki', 'lerden']`
  * Kelime: `öğrenemeyenlerimizdenmişsinizcesine` $\rightarrow$ `['öğ', 'ren', 'e', 'meyen', 'leri', 'mizden', 'miş', 'siniz', 'ce', 'sine']`
  * Cümle: `Merhaba, nasılsın?` $\rightarrow$ `['Merhaba', ',', 'Ġnasıl', 'sın', '?']` (Kayıpsız geri çözme doğrulandı)

---

## 🧹 Veri Ön İşleme ve Bölme (Train / Val / Test)

Uygulanan veri ön işleme boru hattı adımları:
1. **Sütun Sadeleştirme:** `kaynak` sütunu kaldırılarak yalnızca model eğitimi için gerekli olan **`girdi`** ve **`yanit`** sütunları tutuldu.
2. **Gürültü ve HTML Temizliği:** HTML etiketleri (`<p>`, `<div>`, `<br>`), çözülmemiş HTML karakterleri (`&amp;`, `&#39;`), görünmez kontrol karakterleri ve bozuk unicode dizgileri (`\ufffd`) temizlendi.
3. **Boşluk Normalizasyonu:** Çoklu boşluklar, sekme ve satır sonu kalıntıları tek boşluğa normalize edildi.
4. **Geçersiz Satır Filtreleme:** Boş, aşırı kısa veya girdinin yanıta birebir eşit olduğu (papağan yanıtı) satırlar elendi.
5. **Karıştırma (Shuffle):** Tüm veri kümesi rastgele karıştırıldı (`random_state=42`).
6. **Bölme (Split):**
   * **Train (%80):** 79.967 diyalog çifti
   * **Validation (%10):** 9.995 diyalog çifti
   * **Test (%10):** 9.997 diyalog çifti
   * **Toplam Temiz Çift:** 99.959 satır

---

## 🔍 Ham Verileri İnceleme

Masaüstündeki `data/` klasöründe yer alan `.csv` dosyalarını doğrudan **Microsoft Excel**, **Not Defteri** veya **VS Code** ile açarak satır satır inceleyebilirsiniz:

* **`kullanici_girdisi`**: Konuşmayı başlatan kullanıcının veya birinci konuşmacının cümlesi.
* **`yanit`**: Chatbotun öğrenmesini istediğimiz karşı yanıt veya ikinci konuşmacının cümlesi.
* **`kaynak`**: Cümlenin nereden alındığı (Hugging Face diyalogları veya OPUS Açık Altyazıları).
