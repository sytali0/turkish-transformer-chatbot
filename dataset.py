"""
PyTorch Dataset ve DataLoader Boru Hattı
Proje: Turkish Transformer-Based Chatbot
Öğrenci: Seyit Ali Arslan (240711024)

Bu modül:
1. 'data/train.csv' ve 'data/val.csv' dosyalarındaki metinleri okur.
2. BPE Tokenizer ile metinleri sayısal ID dizilerine dönüştürür.
3. [BOS] ve [EOS] özel belirteçlerini ekler.
4. 'max_len' uzunluğuna göre [PAD] (0) ile hizalama (Padding) yapar.
5. Transformer ve Seq2Seq LSTM modelleri için:
   - encoder_input : [BOS] + girdi + [EOS] + [PAD]...
   - decoder_input : [BOS] + yanıt + [PAD]... (Teacher Forcing için)
   - target        : yanıt + [EOS] + [PAD]... (Kayıp/Loss hesabı için)
   - encoder_mask  : Dolgu (PAD) belirteçlerini maskeleyen tensör
   - decoder_mask  : Hem PAD hem de geleceği görmeyi engelleyen nedensel (causal) maske
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import Tokenizer

# Windows konsolunda Türkçe karakter ayarı
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class TurkishChatDataset(Dataset):
    """
    Türkçe diyalog veri kümesini PyTorch tensörlerine dönüştüren Dataset sınıfı.
    """
    def __init__(self, csv_file: str, tokenizer_file: str = "tokenizer/turkish_bpe_12k.json", max_len: int = 64):
        super().__init__()
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {csv_file}")
            
        self.data = pd.read_csv(csv_file)
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.max_len = max_len
        
        # Özel belirteç ID'leri
        self.pad_id = self.tokenizer.token_to_id("[PAD]")  # 0
        self.unk_id = self.tokenizer.token_to_id("[UNK]")  # 1
        self.bos_id = self.tokenizer.token_to_id("[BOS]")  # 2
        self.eos_id = self.tokenizer.token_to_id("[EOS]")  # 3

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        girdi_text = str(row["girdi"]).strip()
        yanit_text = str(row["yanit"]).strip()

        # 1. Tokenize et (Sayısal ID'lere çevir)
        enc_tokens = self.tokenizer.encode(girdi_text).ids
        dec_tokens = self.tokenizer.encode(yanit_text).ids

        # 2. [BOS] ve [EOS] ekle ve max_len sınırına göre kırp (truncate)
        # Giriş için: [BOS] + enc_tokens + [EOS]
        enc_tokens = enc_tokens[: self.max_len - 2]
        encoder_input = [self.bos_id] + enc_tokens + [self.eos_id]

        # Decoder Girdisi (Teacher Forcing): [BOS] + dec_tokens
        dec_tokens_in = dec_tokens[: self.max_len - 1]
        decoder_input = [self.bos_id] + dec_tokens_in

        # Hedef (Target / Loss için): dec_tokens + [EOS]
        target = dec_tokens_in + [self.eos_id]

        # 3. Sabit uzunluğa (max_len) kadar [PAD] ekle
        enc_pad_len = self.max_len - len(encoder_input)
        dec_pad_len = self.max_len - len(decoder_input)
        tgt_pad_len = self.max_len - len(target)

        encoder_input = encoder_input + [self.pad_id] * enc_pad_len
        decoder_input = decoder_input + [self.pad_id] * dec_pad_len
        target = target + [self.pad_id] * tgt_pad_len

        # 4. PyTorch Tensörlerine Dönüştür
        encoder_input_t = torch.tensor(encoder_input, dtype=torch.long)
        decoder_input_t = torch.tensor(decoder_input, dtype=torch.long)
        target_t = torch.tensor(target, dtype=torch.long)

        # 5. Maskeleri Oluştur
        # Encoder Maskesi: PAD olan yerleri (0) maskeler -> (1, max_len)
        encoder_mask = (encoder_input_t != self.pad_id).unsqueeze(0).int()

        # Decoder Padding Maskesi
        decoder_pad_mask = (decoder_input_t != self.pad_id).unsqueeze(0).int()

        # Decoder Causal Maskesi (Geleceği görmeyi engelleyen alt üçgen matris)
        size = self.max_len
        causal_mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.uint8) == 0

        # Decoder nihai maskesi: Hem PAD değil hem de geleceğe bakmıyor
        decoder_mask = decoder_pad_mask & causal_mask

        return {
            "encoder_input": encoder_input_t,      # (max_len)
            "decoder_input": decoder_input_t,      # (max_len)
            "target": target_t,                    # (max_len)
            "encoder_mask": encoder_mask,          # (1, 1, max_len)
            "decoder_mask": decoder_mask,          # (1, max_len, max_len)
            "src_text": girdi_text,
            "tgt_text": yanit_text
        }


def get_dataloaders(
    train_csv: str = "data/train.csv",
    val_csv: str = "data/val.csv",
    tokenizer_file: str = "tokenizer/turkish_bpe_12k.json",
    batch_size: int = 32,
    max_len: int = 64,
    num_workers: int = 0
):
    """
    Train ve Validation DataLoader nesnelerini oluşturur.
    """
    train_dataset = TurkishChatDataset(train_csv, tokenizer_file=tokenizer_file, max_len=max_len)
    val_dataset = TurkishChatDataset(val_csv, tokenizer_file=tokenizer_file, max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Eğitimde her epochta karıştır
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # GPU varsa aktarımı hızlandırır
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # Doğrulamada karıştırmaya gerek yok
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH DATASET VE DATALOADER TESTİ")
    print("=" * 70)

    train_loader, val_loader = get_dataloaders(
        train_csv="data/train.csv",
        val_csv="data/val.csv",
        batch_size=2,       # İncelemek için küçük batch
        max_len=16          # Ekrana rahat sığması için kısa max_len
    )

    print(f"✔ Train Loader Yığın (Batch) Sayısı: {len(train_loader):,}")
    print(f"✔ Val Loader Yığın (Batch) Sayısı  : {len(val_loader):,}")

    # İlk batch'i çekip ekrana basalım
    first_batch = next(iter(train_loader))
    print("\n--- İlk Örnek Batch (Boyut: 2 diyalog) ---")
    print("• Girdi Metni (src)        :", first_batch["src_text"][0])
    print("• Yanıt Metni (tgt)        :", first_batch["tgt_text"][0])
    print("• Encoder Input Tensörü    :", first_batch["encoder_input"][0])
    print("• Decoder Input Tensörü    :", first_batch["decoder_input"][0])
    print("• Target Tensörü           :", first_batch["target"][0])
    print("• Encoder Input Boyutu     :", first_batch["encoder_input"].shape)
    print("• Decoder Input Boyutu     :", first_batch["decoder_input"].shape)
    print("• Encoder Mask Boyutu      :", first_batch["encoder_mask"].shape)
    print("• Decoder Mask Boyutu      :", first_batch["decoder_mask"].shape)
    print("=" * 70)
    print("DATASET MODÜLÜ BAŞARIYLA ÇALIŞIYOR!")
