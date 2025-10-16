import os, time, requests, feedparser
import pandas as pd
from typing import Dict, List, Tuple
import streamlit as st
import requests


def get_env_list(key: str, default: str = "") -> List[str]:
    return [s.strip() for s in os.getenv(key, default).split(",") if s.strip()]

ddef fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
    """
    Binance API'sinden mum verilerini çeker. Hangi API endpoint'ini kullanacağını
    .env veya Streamlit Secrets'tan okur.
    """
    # ---> YENİLİK: API adresini ortam değişkenlerinden oku, bulamazsan varsayılanı kullan
    base_url = os.getenv("BINANCE_API_ENDPOINT", "https://api.binance.com")
    url = f"{base_url}/api/v3/klines"
    
    params = {"symbol": str(symbol).upper(), "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        
        # Fonksiyonun geri kalanı aynı, sadece URL'yi dinamik hale getirdik
        raw = r.json()
        cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(raw, columns=cols)
        # Sütunları sayısal tipe çevir
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Zaman sütunlarını formatla
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        # Sadece gerekli sütunları döndür
        return df[["open_time","open","high","low","close","volume","close_time"]]
        
    except requests.exceptions.HTTPError as err:
        # Hata mesajına URL'yi de ekleyerek daha anlaşılır hale getir
        raise RuntimeError(f"{err.response.status_code} Client Error for url: {err.response.url}") from err
    except Exception as e:
        raise e

# --- Simple RSS ingestion for headlines (optional) ---
DEFAULT_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]

def fetch_headlines(feeds: List[str] = None, max_items: int = 30) -> List[str]:
    feeds = feeds or DEFAULT_FEEDS
    headlines = []
    for url in feeds:
        try:
            fp = feedparser.parse(url)
            for e in fp.entries[: max_items // len(feeds)]:
                title = getattr(e, "title", "")
                if title:
                    headlines.append(title)
        except Exception:
            continue
    return headlines

@st.cache_data(ttl=600) # Bu veriyi 10 dakika boyunca önbellekte tut
def get_fear_and_greed_index():
    """
    Alternative.me API'sini kullanarak en son Korku ve Açgözlülük Endeksi'ni çeker.
    """
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        response.raise_for_status() # Hata varsa istisna fırlat
        data = response.json().get('data', [])
        if data:
            return {
                "score": int(data[0]['value']),
                "classification": data[0]['value_classification']
            }
    except Exception as e:
        print(f"HATA: Korku ve Açgözlülük Endeksi çekilemedi: {e}")
        return None # Hata durumunda None döndür1