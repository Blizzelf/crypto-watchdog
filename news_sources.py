# news_sources.py (TAM VE EKSİKSİZ VERSİYON)
import requests
from bs4 import BeautifulSoup
import re
from typing import List
import random

# --- Yardımcı Fonksiyonlar ---

def _fetch_with_requests(url: str) -> str:
    """URL içeriğini güvenli bir şekilde çeker."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        # print(f"URL çekilemedi ({url}): {e}") # Hata ayıklama için açılabilir
        return ""

def _clean_headline(text: str) -> str:
    """Haber başlıklarındaki gereksiz boşlukları ve karakterleri temizler."""
    if not text:
        return ""
    # Birden fazla boşluğu tek boşluğa indirge ve kenar boşluklarını al
    return " ".join(text.strip().split())

# --- Bireysel Haber Kaynağı Fonksiyonları ("İşçiler") ---

def _get_cryptopanic(limit: int = 15) -> List[str]:
    """CryptoPanic'ten başlıkları çeker."""
    html_content = _fetch_with_requests("https://cryptopanic.com/")
    if not html_content: return []
    soup = BeautifulSoup(html_content, 'html.parser')
    headlines = [h.text for h in soup.select('span.title-text')]
    return headlines[:limit]

def _get_cc_com(limit: int = 15) -> List[str]:
    """CryptoSlate'den başlıkları çeker (crypto.com'un eski adı)."""
    html_content = _fetch_with_requests("https://cryptoslate.com/")
    if not html_content: return []
    soup = BeautifulSoup(html_content, 'html.parser')
    headlines = [h.text for h in soup.select('div.list-post > a > h2')]
    return headlines[:limit]

def _get_beincrypto_com(limit: int = 15) -> List[str]:
    """BeInCrypto'dan başlıkları çeker."""
    html_content = _fetch_with_requests("https://beincrypto.com/news/")
    if not html_content: return []
    soup = BeautifulSoup(html_content, 'html.parser')
    headlines = [h.text for h in soup.select('h2.post-title')]
    return headlines[:limit]

def _get_u_today(limit: int = 15) -> List[str]:
    """U.Today'den başlıkları çeker."""
    html_content = _fetch_with_requests("https://u.today/latest-cryptocurrency-news")
    if not html_content: return []
    soup = BeautifulSoup(html_content, 'html.parser')
    headlines = [h.text for h in soup.select('div.category-item__title > a')]
    return headlines[:limit]

def _get_cointelegraph_com(limit: int = 15) -> List[str]:
    """Cointelegraph'tan başlıkları çeker."""
    html_content = _fetch_with_requests("https://cointelegraph.com/tags/crypto")
    if not html_content: return []
    soup = BeautifulSoup(html_content, 'html.parser')
    headlines = [h.text for h in soup.select('span.post-card-inline__title')]
    return headlines[:limit]

# --- Ana Fonksiyon ("Ustabaşı") ---

def get_headlines_multi(max_items: int = 40) -> List[str]:
    """
    Tüm haber kaynaklarını GÜVENLİ bir şekilde çağırır.
    Bir kaynak hata verse bile diğerlerinden devam eder.
    """
    all_headlines = set() # Yinelenen haberleri engellemek için set kullanalım
    
    source_functions = [
        _get_cryptopanic,
        _get_cc_com,
        _get_beincrypto_com,
        _get_u_today,
        _get_cointelegraph_com
    ]
    
    for func in source_functions:
        try:
            headlines = func(limit=max_items)
            if headlines:
                for h in headlines:
                    cleaned_h = _clean_headline(h)
                    if cleaned_h:
                        all_headlines.add(cleaned_h)
        except Exception as e:
            print(f"Haber kaynağı hatası ({func.__name__}): {e}")
            continue

    final_list = list(all_headlines)
    random.shuffle(final_list)
    return final_list[:max_items]