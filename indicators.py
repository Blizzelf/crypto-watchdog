# indicators.py (NİHAİ VERSİYON - Tüm Sütun Adlarını Dinamik Olarak Bularak Çalışır)
import pandas as pd
import pandas_ta as ta

def find_col(df_columns, prefix):
    """Verilen ön ek (prefix) ile başlayan ilk sütun adını bulur."""
    for col in df_columns:
        if col.startswith(prefix):
            return col
    return None

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verilen DataFrame'e tüm gerekli teknik indikatörleri ekler.
    Bu versiyon, tüm sütun adlarını dinamik olarak bularak kütüphane sürüm
    farklılıklarından kaynaklanan TÜM KeyError hatalarını engeller.
    """
    df_copy = df.copy()
    
    # --- İndikatörleri hesapla ---
    # Not: append=True kullanarak doğrudan ana DataFrame'e ekliyoruz.
    # Bu, geçici DataFrame'ler oluşturmaktan daha verimlidir.
    df_copy.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    df_copy.ta.bbands(close='close', length=20, std=2, append=True)
    df_copy.ta.rsi(close='close', length=14, append=True)
    df_copy.ta.atr(high='high', low='low', close='close', length=14, append=True)
    df_copy.ta.sma(close='close', length=50, append=True)
    df_copy.ta.sma(close='close', length=200, append=True)

    # --- Sütun adlarını dinamik olarak bul ve standart isimlerimize ata ---
    
    # MACD Histogramı
    macd_hist_col = find_col(df_copy.columns, 'MACDh')
    if macd_hist_col:
        df_copy['MACD_hist'] = df_copy[macd_hist_col]

    # Bollinger Bantları
    bb_low_col = find_col(df_copy.columns, 'BBL')
    bb_high_col = find_col(df_copy.columns, 'BBU')
    if bb_low_col:
        df_copy['BB_low'] = df_copy[bb_low_col]
    if bb_high_col:
        df_copy['BB_high'] = df_copy[bb_high_col]
        
    # RSI
    rsi_col = find_col(df_copy.columns, 'RSI')
    if rsi_col:
        df_copy['RSI'] = df_copy[rsi_col]
        
    # ATR
    atr_col = find_col(df_copy.columns, 'ATRr')
    if atr_col:
        df_copy['ATR'] = df_copy[atr_col]
        
    # SMA50 ve SMA200
    sma50_col = find_col(df_copy.columns, 'SMA_50')
    sma200_col = find_col(df_copy.columns, 'SMA_200')
    if sma50_col:
        df_copy['SMA50'] = df_copy[sma50_col]
    if sma200_col:
        df_copy['SMA200'] = df_copy[sma200_col]

    # --- Modelin ihtiyaç duyduğu son sütun setini seçelim ---
    final_columns = [
        'open', 'high', 'low', 'close', 'volume', # Orijinal veri
        'MACD_hist', 'BB_low', 'BB_high', 'RSI', 'ATR', 'SMA50', 'SMA200' # Bizim özelliklerimiz
    ]
    
    # Var olan sütunları filtrele (eğer bir indikatör hesaplanamadıysa hata vermesin)
    existing_final_columns = [col for col in final_columns if col in df_copy.columns]
    df_final = df_copy[existing_final_columns]
    
    # Hesaplamalardan sonra oluşan ve değeri olmayan (NaN) satırları temizle
    df_final.dropna(inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    
    return df_final