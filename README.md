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
│   ├── huggingface_dialogues_raw.csv    # Hugging Face hazır Türkçe diyaloglar (Excel ile açılabilir)
│   ├── huggingface_dialogues_raw.jsonl  # JSONL formatında ham veriler
│   ├── opus_subtitles_raw.csv           # OPUS OpenSubtitles altyazı diyalogları (Excel ile açılabilir)
│   └── opus_subtitles_raw.jsonl         # JSONL formatında altyazı diyalogları
│
├── download_raw_data.py                 # Ham verileri indiren şeffaf Python betiği
├── .gitignore                           # Git takip dışı dosyalar
└── README.md
```

---

## 🔍 Ham Verileri İnceleme

Masaüstündeki `data/` klasöründe yer alan `.csv` dosyalarını doğrudan **Microsoft Excel**, **Not Defteri** veya **VS Code** ile açarak satır satır inceleyebilirsiniz:

* **`kullanici_girdisi`**: Konuşmayı başlatan kullanıcının veya birinci konuşmacının cümlesi.
* **`yanit`**: Chatbotun öğrenmesini istediğimiz karşı yanıt veya ikinci konuşmacının cümlesi.
* **`kaynak`**: Cümlenin nereden alındığı (Hugging Face diyalogları veya OPUS Açık Altyazıları).
