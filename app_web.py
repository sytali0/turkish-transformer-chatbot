"""
3 Modeli Yan Yana Kıyaslayan Canlı Web Arayüzü (http://localhost:7860)
Proje: Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot Geliştirme
       ve Seq2Seq Karşılaştırmalı Başarım Analizi
Öğrenci: Seyit Ali Arslan (240711024)

Kullanım:
  python app_web.py
"""

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from chat_demo import load_all_three_models, ask_all_models

HTML_PAGE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Türkçe Chatbot 3'lü Karşılaştırma Arayüzü — Seyit Ali Arslan</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --blue: #3b82f6;
    --green: #10b981;
    --amber: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
  body { background: var(--bg); color: var(--text); min-height: 100vh; padding: 24px; }
  .container { max-width: 1220px; margin: 0 auto; }
  .header {
    background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 20px;
  }
  .header h1 { font-size: 1.35rem; margin-bottom: 6px; color: #fff; }
  .header p { font-size: 0.92rem; color: #cbd5e1; }
  .input-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 20px;
  }
  .row { display: flex; gap: 12px; }
  input[type="text"] {
    flex: 1;
    background: #0f172a;
    border: 1px solid #475569;
    color: #fff;
    padding: 13px 16px;
    border-radius: 10px;
    font-size: 1rem;
  }
  button {
    background: var(--blue);
    color: #fff;
    border: none;
    padding: 0 26px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.98rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.5; cursor: wait; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .chip {
    background: #0f172a;
    border: 1px solid var(--border);
    color: #cbd5e1;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    cursor: pointer;
  }
  .chip:hover { border-color: var(--blue); color: #fff; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
  .model-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    min-height: 250px;
  }
  .model-card.m1 { border-top: 4px solid var(--blue); }
  .model-card.m2 { border-top: 4px solid var(--green); }
  .model-card.m3 { border-top: 4px solid var(--amber); }
  .card-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 8px; }
  .badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
  .badge {
    font-size: 0.75rem;
    padding: 3px 9px;
    border-radius: 6px;
    background: #0f172a;
    border: 1px solid var(--border);
    color: #cbd5e1;
  }
  .reply-box {
    flex: 1;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px;
    font-size: 0.96rem;
    line-height: 1.55;
    color: #f1f5f9;
  }
  .latency {
    margin-top: 12px;
    font-size: 0.85rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Önceden Eğitilmiş Türkçe Modellerle İnce Ayar (Fine-Tuning) Tabanlı Chatbot & Seq2Seq Karşılaştırmalı Analizi</h1>
    <p>Öğrenci: <b>Seyit Ali Arslan (240711024)</b> | Atatürk Üniversitesi, Yapay Zeka ve Veri Mühendisliği | Deep Learning (2026–2027)</p>
  </div>

  <div class="input-box">
    <div class="row">
      <input id="qInput" type="text" placeholder="Türkçe sorunuzu buraya yazın..." value="Faturamda 'ödeme hatası' var. Ben ödeme yaptıktan sonra neden hala bu hatayı alıyorum?" onkeydown="if(event.key==='Enter') askModels()" />
      <button id="askBtn" onclick="askModels()">3 Modele Sor</button>
    </div>
    <div class="chips">
      <span class="chip" onclick="setQ(this.innerText)">Faturamda 'ödeme hatası' var. Ben ödeme yaptıktan sonra neden hala bu hatayı alıyorum?</span>
      <span class="chip" onclick="setQ(this.innerText)">Merhaba, kargom hala teslim edilmedi ne yapmalıyım?</span>
      <span class="chip" onclick="setQ(this.innerText)">Sanırım meditasyon yapmayı deneyebilirim. Başlamak için iyi bir kaynak var mı?</span>
      <span class="chip" onclick="setQ(this.innerText)">Enerji iç piyasalarındaki ilerlemeleri maddeler hâlinde anlat.</span>
    </div>
  </div>

  <div class="grid">
    <div class="model-card m1">
      <div class="card-title">🤖 Model 1: Full Fine-Tuned GPT-2</div>
      <div class="badges">
        <span class="badge">Parametre: 124.4M (%100)</span>
        <span class="badge">Val PPL: 32.03</span>
        <span class="badge">ROUGE-1: %15.89</span>
      </div>
      <div id="r1" class="reply-box">Sorunuzu yazıp "3 Modele Sor" butonuna tıklayın.</div>
      <div class="latency"><span>Çıkarım Süresi (FP16 + KV-Cache):</span><b id="l1">- ms</b></div>
    </div>

    <div class="model-card m2">
      <div class="card-title">⚡ Model 2: LoRA / PEFT GPT-2</div>
      <div class="badges">
        <span class="badge">Eğitilen: 1.62M (Sadece %1.29!)</span>
        <span class="badge">Val PPL: 56.50</span>
        <span class="badge">ROUGE-1: %11.45</span>
      </div>
      <div id="r2" class="reply-box">Sorunuzu yazıp "3 Modele Sor" butonuna tıklayın.</div>
      <div class="latency"><span>Çıkarım Süresi (r=16, alpha=32):</span><b id="l2">- ms</b></div>
    </div>

    <div class="model-card m3">
      <div class="card-title">⚙️ Baseline: Seq2Seq Bi-LSTM + Attention</div>
      <div class="badges">
        <span class="badge">Parametre: 23.1M (Sıfırdan)</span>
        <span class="badge">Val PPL: 1328.04</span>
        <span class="badge">ROUGE-1: %9.76</span>
      </div>
      <div id="r3" class="reply-box">Sorunuzu yazıp "3 Modele Sor" butonuna tıklayın.</div>
      <div class="latency"><span>Çıkarım Süresi (2-Layer Bi-LSTM):</span><b id="l3">- ms</b></div>
    </div>
  </div>
</div>

<script>
function setQ(txt) {
  document.getElementById('qInput').value = txt;
  askModels();
}
async function askModels() {
  const q = document.getElementById('qInput').value.trim();
  if (!q) return;
  const btn = document.getElementById('askBtn');
  btn.disabled = true;
  btn.innerText = 'Üretiliyor...';
  document.getElementById('r1').innerText = 'Yanıt üretiliyor...';
  document.getElementById('r2').innerText = 'Yanıt üretiliyor...';
  document.getElementById('r3').innerText = 'Yanıt üretiliyor...';
  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await res.json();
    document.getElementById('r1').innerText = data.full_ft.reply;
    document.getElementById('l1').innerText = data.full_ft.latency_ms + ' ms';
    document.getElementById('r2').innerText = data.lora.reply;
    document.getElementById('l2').innerText = data.lora.latency_ms + ' ms';
    document.getElementById('r3').innerText = data.baseline.reply;
    document.getElementById('l3').innerText = data.baseline.latency_ms + ' ms';
  } catch (e) {
    document.getElementById('r1').innerText = 'Hata: ' + e;
  } finally {
    btn.disabled = false;
    btn.innerText = '3 Modele Sor';
  }
}
</script>
</body>
</html>
"""


def run_web_server(port: int = 7860):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2_tok, full_mod, lora_mod, base_tok, base_mod = load_all_three_models(device)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        def do_POST(self):
            if self.path == "/api/ask":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                q = body.get("question", "")
                res = ask_all_models(q, gpt2_tok, full_mod, lora_mod, base_tok, base_mod, device)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"🌐 Web Arayüzü Çalışıyor: {url}")
    print("   (Durdurmak için Ctrl+C tuşlarına basın)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    server.serve_forever()


if __name__ == "__main__":
    run_web_server()
