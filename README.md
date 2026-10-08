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
│   ├── turkish_dialogues_100k.csv       # Tam 100.000 Türkçe diyalog çifti (Excel ile açılabilir)
│   ├── turkish_dialogues_100k.jsonl     # JSONL formatında 100.000 çift (Eğitim için)
│   ├── dataset_info.json                # Veri seti kaynak ve uzunluk özet istatistikleri
│   ├── huggingface_dialogues_raw.csv    # İlk indirilen ham HF diyalog örneği
│   └── opus_subtitles_raw.csv           # İlk indirilen ham OPUS altyazı örneği
│
├── build_hf_dataset_100k.py             # 100.000'lik Hugging Face diyalog derleyici betik
├── download_raw_data.py                 # Ham verileri indiren şeffaf Python betiği
├── .gitignore                           # Git takip dışı dosyalar
└── README.md
```

---

## 🔍 100.000 Türkçe Diyalog Veri Seti Dağılımı

Hugging Face üzerindeki 7 açık kaynaklı Türkçe diyalog ve soru-cevap veri havuzu bir araya getirilmiştir:

| Kaynak Veri Seti | Çift Sayısı | Açıklama |
| :--- | :--- | :--- |
| **Ba2han/Turkish_Chat-1402** | 70.515 | Geniş kapsamlı Türkçe diyalog & bilgi soru-cevapları |
| **cisimcik/turkish-chat-max-25k** | 10.747 | Kullanıcı - Asistan günlük sohbetleri |
| **emreseyhan/Turkish-customer-service** | 5.960 | Doğal müşteri temsilcisi karşılıklı diyalogları |
| **kilicai/turkish-sft-multi-turn-dialogue** | 5.095 | Çok turlu Türkçe soru-yanıt diyalogları |
| **3nesdeniz/turkish-daily-dialogues-5k** | 4.521 | Günlük yaşam ve arkadaş sohbetleri |
| **odmow/turkish-dialogues** | 2.942 | Samimi günlük Türkçe selamlaşma ve sohbetler |
| **sixfingerdev/chatbot-turkish-dataset** | 220 | Chatbot yanıt çiftleri |
| **TOPLAM** | **100.000** | **Proje teklifi hedefi eksiksiz karşılandı** |

---

## 🔍 Ham Verileri İnceleme

Masaüstündeki `data/` klasöründe yer alan `.csv` dosyalarını doğrudan **Microsoft Excel**, **Not Defteri** veya **VS Code** ile açarak satır satır inceleyebilirsiniz:

* **`kullanici_girdisi`**: Konuşmayı başlatan kullanıcının veya birinci konuşmacının cümlesi.
* **`yanit`**: Chatbotun öğrenmesini istediğimiz karşı yanıt veya ikinci konuşmacının cümlesi.
* **`kaynak`**: Cümlenin nereden alındığı (Hugging Face diyalogları veya OPUS Açık Altyazıları).
