import os, time, requests, feedparser
import pandas as pd
from typing import Dict, List, Tuple
import streamlit as st
import requests

BINANCE_API = "https://api.binance.com/api/v3/klines"

def get_env_list(key: str, default: str = "") -> List[str]:
    return [s.strip() for s in os.getenv(key, default).split(",") if s.strip()]

def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_API, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df[["open_time","open","high","low","close","volume","close_time"]]

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