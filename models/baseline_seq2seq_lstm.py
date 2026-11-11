"""
Baseline Model: Bahdanau Dikkat Mekanizmalı 2 Katmanlı Seq2Seq Bi-LSTM Chatbot
Proje: Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme
       ve Seq2Seq Karşılaştırmalı Başarım Analizi
Öğrenci: Seyit Ali Arslan (240711024)
Bölüm: Yapay Zeka ve Veri Mühendisliği, Atatürk Üniversitesi

Mimari Detayları (Bölüm V & VI):
- Encoder : 2 Katmanlı Çift Yönlü LSTM (2-layer Bi-LSTM)
- Attention: Bahdanau (Additive) Dikkat Mekanizması (Bahdanau et al., 2014)
- Decoder : 2 Katmanlı Tek Yönlü LSTM
"""

import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBiLSTM(nn.Module):
    """
    2 Katmanlı Çift Yönlü (Bidirectional) LSTM Encoder.
    Kullanıcı girdisini ileri ve geri yönde okuyarak bağlam temsillerini üretir.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2, pad_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Çift yönlü (forward + backward) gizli durumları Decoder boyutu olan hidden_dim'e indirgeyen katmanlar
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src: torch.Tensor):
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch_size, src_len, embed_dim)
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: (batch_size, src_len, hidden_dim * 2)
        # hidden, cell: (num_layers * 2, batch_size, hidden_dim)

        # Her katman için ileri (2*i) ve geri (2*i+1) yönleri birleştirip Decoder için (num_layers, batch, hidden_dim) yapalım
        hidden_cat = []
        cell_cat = []
        for i in range(self.num_layers):
            h_f = hidden[2 * i]
            h_b = hidden[2 * i + 1]
            c_f = cell[2 * i]
            c_b = cell[2 * i + 1]
            hidden_cat.append(torch.tanh(self.fc_hidden(torch.cat((h_f, h_b), dim=1))))
            cell_cat.append(torch.tanh(self.fc_cell(torch.cat((c_f, c_b), dim=1))))

        hidden_dec = torch.stack(hidden_cat, dim=0)  # (num_layers, batch_size, hidden_dim)
        cell_dec = torch.stack(cell_cat, dim=0)      # (num_layers, batch_size, hidden_dim)

        return outputs, hidden_dec, cell_dec


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Dikkat Mekanizması:
    e_{ij} = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        # decoder_hidden: (batch_size, hidden_dim) -> son katmanın gizli durumu
        # encoder_outputs: (batch_size, src_len, hidden_dim * 2)
        src_len = encoder_outputs.shape[1]

        dec_rep = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch_size, src_len, hidden_dim)
        energy = torch.tanh(self.W_a(dec_rep) + self.U_a(encoder_outputs))  # (batch_size, src_len, hidden_dim)
        scores = self.v_a(energy).squeeze(2)  # (batch_size, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)  # (batch_size, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_dim * 2)

        return context, attn_weights


class DecoderLSTM(nn.Module):
    """
    Bahdanau Dikkat Destekli 2 Katmanlı LSTM Decoder.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(
            input_size=embed_dim + (hidden_dim * 2),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc_out = nn.Linear(hidden_dim + (hidden_dim * 2) + embed_dim, vocab_size)

    def forward(self, input_step: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        # input_step: (batch_size) -> tek bir zaman adımındaki token ID'leri
        input_step = input_step.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(input_step))  # (batch_size, 1, embed_dim)

        # Son katman gizli durumu ile dikkat hesapla
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, mask)

        # Gömme vektörü ve bağlam vektörünü birleştirip LSTM'e ver
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embed_dim + hidden_dim*2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Kelime tahmini
        concat_out = torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(concat_out)  # (batch_size, vocab_size)

        return prediction, hidden, cell, attn_weights


class Seq2SeqBiLSTM(nn.Module):
    """
    Uçtan Uca Seq2Seq Bi-LSTM + Bahdanau Attention Modeli.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2, pad_id: int = 0, bos_id: int = 2, eos_id: int = 3):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.vocab_size = vocab_size

        self.encoder = EncoderBiLSTM(vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_id)
        self.decoder = DecoderLSTM(vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_id)

    def forward(self, src: torch.Tensor, trg_in: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        # src: (batch_size, src_len)
        # trg_in: (batch_size, trg_len) -> [BOS] ile başlayan decoder girdisi
        batch_size = src.shape[0]
        trg_len = trg_in.shape[1]

        outputs = torch.zeros(batch_size, trg_len, self.vocab_size, device=src.device)
        mask = (src != self.pad_id)

        encoder_outputs, hidden, cell = self.encoder(src)

        # İlk adım [BOS] belirteci
        input_step = trg_in[:, 0]

        for t in range(trg_len):
            prediction, hidden, cell, _ = self.decoder(input_step, hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = prediction

            if t + 1 < trg_len:
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                top1 = prediction.argmax(1)
                input_step = trg_in[:, t + 1] if use_teacher_forcing else top1

        return outputs

    @torch.no_grad()
    def generate_reply(self, src_ids: list, tokenizer, max_len: int = 40, device: str = "cpu"):
        """
        Tek bir kullanıcı girdisine yanıt üretir ve çıkarım gecikmesini (ms) ölçer.
        """
        self.eval()
        start_time = time.perf_counter()

        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        mask = (src_tensor != self.pad_id)

        encoder_outputs, hidden, cell = self.encoder(src_tensor)
        input_step = torch.tensor([self.bos_id], dtype=torch.long, device=device)

        generated_ids = []
        attn_matrix = []

        for _ in range(max_len):
            prediction, hidden, cell, attn_weights = self.decoder(input_step, hidden, cell, encoder_outputs, mask)
            # Tekrarları önlemek için basit tekrar cezası (Repetition Penalty)
            for prev_id in set(generated_ids[-4:]):
                prediction[0, prev_id] /= 1.4

            top1 = prediction.argmax(1).item()
            if top1 == self.eos_id or top1 == self.pad_id:
                break
            generated_ids.append(top1)
            attn_matrix.append(attn_weights.squeeze(0).cpu().numpy())
            input_step = torch.tensor([top1], dtype=torch.long, device=device)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        reply_text = tokenizer.decode(generated_ids)
        return reply_text, latency_ms, attn_matrix
