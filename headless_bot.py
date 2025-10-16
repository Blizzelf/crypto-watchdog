# headless_bot.py
import os
import time
import schedule
import requests
from dotenv import load_dotenv
from datetime import datetime

# Projemizdeki diğer modülleri import ediyoruz
from data import get_env_list, fetch_binance_klines
from indicators import add_all_indicators
from strategy import decide_signals_with_model, get_market_regime
import joblib

# --- AYARLAR ---
load_dotenv()
SYMBOLS = get_env_list("SYMBOLS", "BTCUSDT,ETHUSDT,XRPUSDT,DOGEUSDT")
TIMEFRAME = os.getenv("BOT_TIMEFRAME", "5m") # Bot'un hangi zaman dilimini takip edeceği
POLL_MINUTES = int(os.getenv("BOT_POLL_MINUTES", "1")) # Bot'un kaç dakikada bir çalışacağı

# --- Model Yükleme ---
try:
    model = joblib.load("crypto_xgb_model.pkl")
    print("✅ AI Modeli başarıyla yüklendi.")
except FileNotFoundError:
    print("⚠️ KRİTİK HATA: crypto_xgb_model.pkl bulunamadı. Bot başlatılamıyor.")
    exit()

# --- Telegram Fonksiyonları ---
def send_telegram_alert(message):
    """Verilen mesajı Telegram'a gönderir."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ HATA: .env dosyasında TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID bulunamadı.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        if response.status_code == 200:
            print(f"✅ Telegram mesajı başarıyla gönderildi.")
        else:
            print(f"⚠️ Telegram gönderme hatası: {response.text}")
    except Exception as e:
        print(f"⚠️ Telegram bağlantı hatası: {e}")

# --- Ana Görev Fonksiyonu ---
def check_signals_and_alert():
    """Piyasayı kontrol eder ve yeni bir sinyal varsa Telegram'a uyarı gönderir."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Piyasalar kontrol ediliyor...")

    try:
        # 1. Piyasa Rejimini Belirle ve Türkçeleştir
        regime = get_market_regime()
        regime_tr = "BOĞA" if regime == "BULL" else ("AYI" if regime == "BEAR" else "NÖTR")
        regime_emoji = "🐂" if regime == "BULL" else ("🐻" if regime == "BEAR" else "↔️")
        print(f"Piyasa Rejimi: {regime_emoji} {regime_tr}")

        new_signals = []
        # 2. Her sembol için sinyal üret
        for sym in SYMBOLS:
            df_raw = fetch_binance_klines(sym, TIMEFRAME, 500)
            df_enriched = add_all_indicators(df_raw)
            if df_enriched.empty:
                print(f"-> {sym}: Veri yetersiz.")
                continue
            
            # Not: Bu botta "Korku ve Açgözlülük" filtresini kullanmıyoruz (None gönderiyoruz)
            sig, reason, metrics = decide_signals_with_model(df_enriched, model, market_regime=regime, fear_and_greed=None)
            
            sig_tr = "AL" if sig == "BUY" else ("SAT" if sig == "SELL" else "BEKLE")
            print(f"-> {sym}: {sig_tr} (Fiyat: {metrics.get('price', 0):.4f})")
            
            if sig in ["BUY", "SELL"]:
                new_signals.append({
                    "symbol": sym,
                    "signal": sig,
                    "price": metrics.get('price', 0.0)
                })

        # 4. Eğer yeni bir AL/SAT sinyali varsa, toplu bir mesaj oluştur ve gönder
        if new_signals:
            print("🔥 Önemli sinyal(ler) tespit edildi! Bildirim gönderiliyor...")
            
            # ---> YENİLİK: Mesajda artık Türkçeleştirilmiş 'regime_tr' kullanılıyor
            message_lines = [f"🐺 <b>Crypto Watchdog Sinyal Uyarısı!</b>", f"Piyasa Durumu: {regime_emoji} {regime_tr}\n"]
            
            for s in new_signals:
                badge = "📈🟢 AL" if s['signal'] == "BUY" else "📉🔴 SAT"
                message_lines.append(f"<b>{s['symbol']}</b>: {badge} @ <code>{s['price']:.6g}</code>")
            
            send_telegram_alert("\n".join(message_lines))
        else:
            print("Sakin bir piyasa. Yeni AL/SAT sinyali yok.")

    except Exception as e:
        print(f"❌ Ana görev sırasında kritik bir hata oluştu: {e}")
        send_telegram_alert(f"🤖 BOT HATA UYARISI 🤖\nCrypto Watchdog botu çalışırken bir hata oluştu: {e}")


# --- Zamanlayıcıyı Kurma ve Botu Başlatma ---
print(f"🚀 Crypto Watchdog 'Hayalet Mod' Botu başlatıldı.")
print(f"Her {POLL_MINUTES} dakikada bir '{TIMEFRAME}' zaman dilimi kontrol edilecek.")
send_telegram_alert("🤖 <b>Crypto Watchdog 'Hayalet Mod' Botu</b> başlatıldı ve piyasaları izliyor.")

# İlk kontrolü hemen yap, sonra zamanlayıcıya devret
check_signals_and_alert() 

schedule.every(POLL_MINUTES).minutes.do(check_signals_and_alert)

while True:
    schedule.run_pending()
    time.sleep(1)