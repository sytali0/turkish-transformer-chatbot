# Yarınki Proposal & Ara Sunum Konuşma Metni (Tam Akış)

**Öğrenci:** Seyit Ali Arslan (`240711024`)  
**Proje Başlığı:** Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme ve Seq2Seq Karşılaştırmalı Başarım Analizi  

---

## 1. Açılış ve Giriş (İlk 30–45 Saniye)

> **"Sayın hocam ve değerli arkadaşlar, merhaba.**  
> **Ben Seyit Ali Arslan. Bugün sizlere Derin Öğrenme dersi kapsamında hazırladığım *'Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme ve Seq2Seq Karşılaştırmalı Başarım Analizi'* başlıklı proje önerimi ve şu ana kadar elde ettiğimiz deneysel bulguları sunacağım."**

---

## 2. Bölüm I & II — Problem Tanımı ve Motivasyon (1 Dakika)

> **"İlk olarak hangi problemi çözdüğümüzden bahsetmek istiyorum.**  
> **Bildiğiniz gibi Türkçe, sondan eklemeli (*agglutinative*) ve zengin morfolojik yapıya sahip bir dildir. Bir kelime köküne gelen yapım ve çekim ekleri, kelime dağarcığının üstel olarak büyümesine ve klasik modellerde kelime dışı (*Out-of-Vocabulary - OOV*) sorununa yol açmaktadır.**  
>
> **Geleneksel Yinelemeli Sinir Ağları (RNN ve LSTM tabanlı Seq2Seq mimariler), veriyi kelime kelime sıralı (*sequential*) olarak işler. Bu durum hem cümle uzadıkça sabit boyutlu bağlam vektöründe darboğaza (*bottleneck*) ve bağlam kaybına yol açmakta, hem de GPU üzerinde paralel eğitimi kısıtlamaktadır.**  
>
> **Diğer taraftan, büyük bir Transformer dil modelini sıfırdan eğitmek devasa veri kümeleri ve çok yüksek donanım gücü gerektirir. İşte projemizin temel motivasyonu burada devreye giriyor:**  
> **Sıfırdan model eğitmenin aşırı hesaplama maliyeti yerine, genel Türkçe metinler üzerinde önceden eğitilmiş açık kaynaklı bir Transformer modelini Transfer Öğrenme (*Fine-Tuning*) yöntemiyle Türkçe diyalog alanına adapte ediyoruz. Ardından bu modeli, sıfırdan eğittiğimiz klasik 2 katmanlı Seq2Seq Bi-LSTM + Bahdanau Attention temel (*Baseline*) modeliyle tamamen aynı veri kümesi ve aynı metrikler üzerinde karşılaştırıyoruz."**

---

## 3. Bölüm III — Veri Kümesi (Dataset) ve Ön İşleme (1 Dakika)

> **"Üçüncü bölümde veri kümemizi ele aldık.**  
> **Veri kaynağı olarak Hugging Face üzerinde yer alan açık erişimli Türkçe diyalog veri setlerini (`Ba2han/Turkish_Chat`, `cisimcik/turkish-chat`, `odmow/turkish-dialogues` vb.) birleştirdik.**  
>
> **Ham veriyi doğrudan modele vermek yerine kapsamlı bir veri temizleme boru hattı kurduk:**  
> - **HTML etiketlerini, Markdown kalıntılarını, bozuk karakterleri ve tekrar eden (*duplicate*) satırları temizledik.**  
> - **Çok kısa veya gürültülü cümleleri filtreleyerek 2 ile 45 kelime aralığında, en yüksek kaliteli 50.000 Türkçe diyalog çiftini (*Girdi: Kullanıcı Mesajı, Hedef: Model Yanıtı*) oluşturduk.**  
> - **Son olarak veri setimizi karıştırarak (*shuffle*) %80 Eğitim (40.000 çift), %10 Doğrulama (5.000 çift) ve %10 Test (5.000 çift) olacak şekilde böldük."**

---

## 4. Bölüm IV, V ve VI — Önerilen Metodoloji ve Planlanan Modeller (1.5 Dakika)

> **"Metodoloji ve model mimarisi tarafında projemizi üç aşamalı bir karşılaştırma üzerine kurduk:**  
>
> **1. Temel Modelimiz (Baseline Model):**  
> **Geleneksel mimariler ile Transformer arasındaki farkı somutlaştırmak için; 2 katmanlı Çift Yönlü LSTM (*Bi-LSTM*) Encoder, Bahdanau Dikkat (*Attention*) mekanizması ve 2 katmanlı LSTM Decoder'dan oluşan 23.1 Milyon parametreli bir Seq2Seq modeli sıfırdan kodladık ve eğittik.**  
>
> **2. Model 1 (Standart Full Fine-Tuned Transformer):**  
> **Ana modelimiz olarak yaklaşık 124.4 Milyon parametreli, 12 katmanlı ve 12 öz-dikkat (*Self-Attention*) başlığına sahip `Turkish-GPT-2` modelini kullandık. Sıfırdan tokenizer eğitmek yerine modelin resmi Byte-Pair Encoding (BPE) alt-kelime tokenizer yapısını ve önceden eğitilmiş ağırlıklarını koruduk. Eğitim sürecinde PyTorch ve Hugging Face Transformers kütüphaneleriyle `AdamW` optimizasyon algoritmasını ve `Cosine Learning Rate Scheduler` yapısını kullandık. Ayrıca kayıp fonksiyonunda kullanıcı sorusunu `-100` ile maskeleyerek modelin yalnızca `Asistan:` yanıtlarını üretirken güncellenmesini sağladık.**  
>
> **3. Model 2 (LoRA / PEFT ile Fine-Tuned Transformer):**  
> **Donanım ve bellek verimliliğini analiz etmek amacıyla, düşük dereceli adaptasyon (*Low-Rank Adaptation - LoRA*) yöntemiyle parametrelerin yalnızca ~%1.2'sini güncelleyerek tam ince ayar (*Full Fine-Tuning*) ile parametre verimli ince ayarı kıyaslamayı planladık."**

---

## 5. Bölüm VII — Değerlendirme Metrikleri ve Şu Ana Kadar Elde Ettiğimiz Sonuçlar (1.5 Dakika)

> **"Modellerimizin başarımını 5 temel ve savunulabilir metrikle değerlendiriyoruz:**  
> 1. **Validation Loss ve Perplexity (PPL):** Modelin kelime tahmin belirsizliğini ölçüyor; düşük PPL daha tutarlı bir dil modeli anlamına geliyor.  
> 2. **BLEU Skoru (BLEU-1 ve BLEU-2):** Üretilen yanıt ile referans yanıt arasındaki kelime/alt-kelime örtüşmesini ölçüyor.  
> 3. **ROUGE Skoru (ROUGE-1 ve ROUGE-L):** En uzun ortak alt dizi üzerinden anahtar kavramların korunmasını ölçüyor.  
> 4. **Çıkarım Süresi (Inference Latency - ms):** Tek bir kullanıcı sorusuna modelin ortalama kaç milisaniyede yanıt ürettiğini, yani gerçek zamanlı sohbet hızını ölçüyor.  
> 5. **Niteliksel Değerlendirme (Human Evaluation):** Örnek diyalog çıktıları üzerinde akıcılık ve bağlam koruma analizi.  
>
> **Yerel RTX 4060 GPU üzerinde 5 Epoch boyunca yaptığımız ilk deneylerde çok net sonuçlar elde ettik:**  
> - **Sıfırdan eğittiğimiz Baseline Seq2Seq Bi-LSTM modelinde Validation Loss `7.19`, Perplexity (PPL) ise `1328.04` seviyesinde kalırken; Fine-Tuning yaptığımız `Turkish-GPT-2` modelinde Validation Loss `3.46`'ya, Perplexity ise `32.03`'e düştü. Yani Perplexity'de 41.5 kat iyileşme sağladık.**  
> - **BLEU-1 skorunda %52.3 (`6.96`'dan `10.60`'a), ROUGE-1 skorunda ise %62.8 (`9.76`'dan `15.89`'a) artış elde ettik.**  
> - **Üstelik `FP16` ve `KV-Cache` optimizasyonu sayesinde 124 Milyon parametreli Transformer modelimiz ortalama `352 milisaniyede` (0.35 saniyede) gerçek zamanlı yanıt üretebilmektedir."**

---

## 6. Bölüm VIII — Beklenen Katkı ve Kapanış (30 Saniye)

> **"Sonuç olarak bu çalışmanın üç temel katkısı bulunmaktadır:**  
> 1. **Transfer öğrenmenin (Fine-Tuning), sıfırdan eğitime kıyasla hem hesaplama maliyeti hem de Türkçe diyalog kalitesi üzerindeki üstünlüğünü deneysel olarak kanıtlamak,**  
> 2. **Klasik Seq2Seq Bi-LSTM mimarisi ile güncel Transformer mimarisini kayıp, Perplexity, BLEU/ROUGE ve çıkarım hızı (ms) üzerinden somut biçimde kıyaslamak,**  
> 3. **Tek bir yerel GPU üzerinde tekrarlanabilir (*reproducible*) uçtan uca bir Türkçe chatbot ince ayar boru hattı sunmak.**  
>
> **Dinlediğiniz için teşekkür ederim. Sorularınız varsa memnuniyetle yanıtlayabilirim; ayrıca isterseniz eğittiğimiz her iki modeli şu an terminal üzerinden canlı olarak da test edebiliriz."**

---

## 💡 Hocanın Sorabileceği Muhtemel Sorular ve Hazır Cevapların

1. **Soru:** *"Neden geçen seferki gibi Transformer'ı sıfırdan eğitmek yerine Fine-Tuning (Önceden Eğitilmiş Model) yaklaşımına geçtin?"*  
   **Cevabın:** *"Hocam, Türkçe gibi zengin morfolojili bir dilde Transformer'ı sıfırdan eğittiğimizde modelin genel Türkçe dilbilgisini ve kelime kök-ek ilişkilerini kavraması için milyonlarca cümle ve çok yüksek GPU süresi gerekiyor. Küçük veriyle sıfırdan eğitildiğinde model ezberlemeye (overfitting) veya anlamsız kelimeler üretmeye yatkın oluyor. Buna karşın genel Türkçe derlemlerde önceden eğitilmiş `Turkish-GPT-2` ağırlıklarını transfer ettiğimizde, model Türkçeyi zaten bildiği için 50.000 diyalog çiftiyle sadece 5 Epoch'ta Perplexity değerini 1328'den 32'ye düşürerek çok daha akıcı yanıtlar üretebildi."*

2. **Soru:** *"Baseline olarak neden Seq2Seq Bi-LSTM + Bahdanau Attention seçtin?"*  
   **Cevabın:** *"Çünkü Transformer mimarisi ('Attention Is All You Need') çıkmadan önce doğal dil işlemede altın standart mimari Çift Yönlü LSTM (Bi-LSTM) + Bahdanau Attention mimarisiydi. Öz-dikkat (Self-Attention) mekanizmasının sıralı LSTM darboğazını nasıl aştığını göstermek için en adil ve akademik referans noktası bu modeldir."*

3. **Soru:** *"Perplexity (PPL) tam olarak neyi ifade ediyor, 1328'den 32'ye düşmesi ne demek?"*  
   **Cevabın:** *"Perplexity, Cross-Entropy kaybının üstel değeridir ($\text{PPL} = e^{\text{Loss}}$) ve modelin bir sonraki kelimeyi tahmin ederken kaç farklı seçenek arasında kararsız kaldığını gösterir. Baseline modelimiz her adımda ortalama 1328 alt-kelime arasında kararsız kalırken, Fine-Tuned Turkish-GPT-2 modelimiz bu belirsizliği 32'ye indirmiştir."*
