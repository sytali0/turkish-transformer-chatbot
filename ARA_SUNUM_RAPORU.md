# IEEE Formatlı Ara Sunum (Midterm Presentation) Rehberi ve Konuşmacı Notları

- **Proje Başlığı:** Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme ve Seq2Seq Karşılaştırmalı Başarım Analizi
- **Öğrenci:** Seyit Ali Arslan (`240711024`)
- **Bölüm / Kurum:** Yapay Zeka ve Veri Mühendisliği, Atatürk Üniversitesi
- **Ders:** Deep Learning (`2026–2027`)
- **Hazırlanan Resmi IEEE Belgeleri:**
  1. **IEEE Konferans Formatlı Ara Rapor (2 Sütunlu Makale Formatı):** [`ara_rapor_ieee.pdf`](./ara_rapor_ieee.pdf) ([`ara_rapor_ieee.tex`](./ara_rapor_ieee.tex))
  2. **IEEE Akademik Sunum Slaytları (16:9 Beamer Formatı):** [`ara_sunum_ieee_beamer.pdf`](./ara_sunum_ieee_beamer.pdf) ([`ara_sunum_ieee_beamer.tex`](./ara_sunum_ieee_beamer.tex))

---

## Slayt-Slayt Konuşmacı Notları (Sunumda Söylenecek Metinler)

### Slayt 1 & 2 — Kapak ve Sunum Akışı
> *"Sayın hocam ve değerli arkadaşlar, bugün Derin Öğrenme dersi kapsamında yürüttüğüm 'Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme ve Seq2Seq Karşılaştırmalı Başarım Analizi' başlıklı projemin ara sunumunu gerçekleştireceğim. Sunumum IEEE standart bölümlendirmesine uygun olarak; problem tanımı, veri kümesi analizi, sıfırdan eğittiğimiz 2 katmanlı Seq2Seq Bi-LSTM temel modeli, ince ayar yaptığımız Turkish-GPT-2 modeli ve her iki mimarinin deneysel karşılaştırmasından oluşmaktadır."*

### Slayt 3 — I. Giriş ve Problem Tanımı & II. Motivasyon
> *"Türkçe sondan eklemeli (agglutinative) bir dil olduğu için bir kelime köküne gelen yapım ve çekim ekleri yüzeysel kelime dağarcığını üstel olarak büyütmekte ve kelime dışı (OOV) sorununa yol açmaktadır. Geleneksel RNN ve LSTM tabanlı Seq2Seq mimariler ise kelimeleri sıralı işlediği için uzun cümlelerde sabit boyutlu bağlam vektörü darboğazına (bottleneck) takılmaktadır. Sıfırdan devasa bir dil modeli eğitmek yüksek donanım maliyeti gerektirdiğinden, bu çalışmada genel Türkçe metinler üzerinde önceden eğitilmiş Transformer ağırlıklarını transfer öğrenme (Fine-Tuning) ile diyalog alanına adapte ettik ve klasik Seq2Seq Bi-LSTM mimarisiyle aynı veri kümesi üzerinde kıyasladık."*

### Slayt 4 & 5 — III. Veri Kümesi ve Keşifsel Veri Analizi (EDA)
> *"Proje önerimizde belirttiğimiz üzere Hugging Face üzerindeki açık kaynaklı Türkçe diyalog veri kümelerini (`Ba2han/Turkish_Chat`, `AEG/Turkish-Chat-62k`, `yeniguno/turkish-chatbot-data`) birleştirerek kapsamlı bir temizleme boru hattından geçirdik. HTML etiketlerini, Markdown kalıntılarını ve bozuk karakterleri temizledikten sonra 2 ile 45 kelime arasındaki en kaliteli 50.000 Türkçe diyalog çiftini seçtik. Bu veriyi %80 Eğitim (40.000), %10 Doğrulama (5.000) ve %10 Test (5.000) olarak böldük. EDA analizimizde kullanıcı sorularının ortalama 9.73 kelimeden, asistan yanıtlarının ise ortalama 15.49 kelimeden oluştuğunu gözlemledik."*

### Slayt 6 — IV–VI. Metodoloji ve Karşılaştırılan Mimariler
> *"Karşılaştırma adil ve bilimsel olsun diye iki farklı mimari kurduk:*
> *1. **Baseline Modelimiz:** 12.000 alt-kelime kapasiteli Türkçe BPE Tokenizer kullanan, 2 katmanlı Çift Yönlü LSTM (Bi-LSTM) Encoder, Bahdanau Toplamsal Dikkat (Additive Attention) mekanizması ve 2 katmanlı LSTM Decoder'dan oluşan 23.1 Milyon parametreli modeldir. Sıfırdan eğitilmiştir.*
> *2. **Model 1 (Ana Modelimiz):** 124.4 Milyon parametreli, 12 katmanlı ve 12 öz-dikkat (Self-Attention) başlığına sahip önceden eğitilmiş `Turkish-GPT-2` modelidir. Eğitim sırasında kullanıcı sorusuna ait token indekslerini `-100` ile maskeleyerek Cross-Entropy kaybının yalnızca modelin ürettiği `Asistan:` yanıtı üzerinde hesaplanmasını sağladık."*

### Slayt 7 — V. Baseline Seq2Seq Bi-LSTM Eğitim Analizi
> *"RTX 4060 GPU üzerinde 5 Epoch boyunca sıfırdan eğittiğimiz Baseline Seq2Seq Bi-LSTM modelinde Validation Loss değeri 7.69'dan 7.19'a, Validation Perplexity (PPL) değeri ise 2188.9'dan 1328.04'e düşmüştür. Ancak Perplexity değerinin 1328 seviyesinde kalması, sıfırdan eğitilen yinelemeli ağın geniş Türkçe kelime dağarcığında hala yüksek belirsizlik yaşadığını göstermektedir."*

### Slayt 8 & 9 — VII. Deneysel Karşılaştırma (Loss, Perplexity, BLEU, ROUGE ve Latency)
> *"Aynı veri kümesi üzerinde 5 Epoch ince ayar (Fine-Tuning) yaptığımız `Turkish-GPT-2` modelini Baseline Bi-LSTM ile karşılaştırdığımızda çarpıcı sonuçlar elde ettik:*
> - *Doğrulama Kaybı (Validation Loss) **7.1915'ten 3.4666'ya** (%51.8 iyileşme) düştü.*
> - *Doğrulama Perplexity (PPL) skoru **1328.04'ten 32.03'e** inerek **41.5 kat** iyileşme sağladı.*
> - *Metin benzerlik metriklerinde **BLEU-1 %52.3 artışla 10.60'a**, **BLEU-2 %41.8 artışla 5.70'e**, **ROUGE-1 ise %62.8 artışla 15.89'a** yükseldi.*
> - *Çıkarım gecikmesi (Inference Latency) açısından 23M parametreli Bi-LSTM 30.07 ms sürerken, 124.4M parametreli Turkish-GPT-2 modelimiz `FP16 + KV-Cache` optimizasyonu sayesinde ortalama **352.9 ms** (yani 0.35 saniye) ile gerçek zamanlı sohbet sınırları içinde yanıt üretmektedir."*

### Slayt 10 — Niteliksel (Qualitative) Çıktı Analizi
> *"Sayısal metriklerin yanı sıra modellerin ürettiği gerçek yanıtları incelediğimizde; kısa ve kalıplaşmış müşteri hizmetleri sorularında her iki model de anlamlı yanıt verebilirken, uzun ve açık uçlu sorularda (örneğin enerji piyasaları sorusunda) Baseline Bi-LSTM modeli darboğaz nedeniyle '1. ve,: 1. ve,' şeklinde yerel kelime tekrarına düşmektedir. Buna karşın Fine-Tuned Turkish-GPT-2 modelimiz 'Enerji sektörü, enerji verimliliği gibi alanlarda önemli ilerlemeler kaydetmiştir...' şeklinde dilbilgisel olarak tam ve bağlama uygun yanıtlar üretmektedir."*

### Slayt 11 — VIII. Sonuç ve Final Dönemi Yol Haritası
> *"Özetle, ara sunum hedeflerimiz olan 50.000 diyalogluk veri boru hattı, Baseline Seq2Seq Bi-LSTM eğitimi ve Model 1 (Full Fine-Tuned Turkish-GPT-2) karşılaştırma deneylerinin tamamı tamamlanmıştır. Final sunumu aşamasında ise Proposal'daki **Model 2 (LoRA / PEFT ile Parametre Verimli İnce Ayar)** modelini eğiterek 3'lü karşılaştırma tablosunu ve canlı sohbet arayüzünü sunacağız. Dinlediğiniz için teşekkür ederim."*
