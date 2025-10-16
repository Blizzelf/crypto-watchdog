# strategy.py (Pırıl Pırıl Versiyon)
import pandas as pd
import numpy as np
from indicators import add_all_indicators
from data import fetch_binance_klines

def decide_signals_with_model(df_features: pd.DataFrame, model, market_regime: str, fear_and_greed: dict = None) -> (str, str, dict):
    """
    AI modelini, Piyasa Rejimini ve KORKU/AÇGÖZLÜLÜK FİLTRESİNİ kullanarak karar verir.
    """
    # ... (fonksiyonun başındaki if/else blokları aynı kalacak) ...
    if model is None:
        price = df_features['close'].iloc[-1] if not df_features.empty else 0.0
        return "BEKLE", "AI Modeli Yüklenemedi", {"price": price, "rsi": 50.0, "macd_hist": 0, "atr": 0, "confidence": 0.0, "strength": "N/A"}

    # ... (diğer kontrol blokları aynı kalacak) ...

    last_data = df_features[['RSI', 'MACD_hist', 'BB_low', 'BB_high', 'ATR', 'SMA50', 'SMA200']].iloc[-1:]
    
    probabilities = model.predict_proba(last_data)[0]
    prediction_mapped = np.argmax(probabilities)
    confidence = probabilities[prediction_mapped]

    signal_map = {0: "SELL", 1: "BEKLE", 2: "BUY"}
    model_signal = signal_map.get(prediction_mapped, "BEKLE")

    # --- FİLTRELEME AŞAMALARI ---
    final_signal = model_signal
    filter_reason = ""

    if market_regime == "BULL" and model_signal == "SELL":
        final_signal = "BEKLE"
        filter_reason = " (Trend Yükselişte, SAT Riskli)"
    elif market_regime == "BEAR" and model_signal == "BUY":
        final_signal = "BEKLE"
        filter_reason = " (Trend Düşüşte, AL Riskli)"
        
    if final_signal != "BEKLE" and fear_and_greed:
        score = fear_and_greed.get("score", 50)
        if score > 80 and final_signal == "BUY":
            final_signal = "BEKLE"
            filter_reason = " (Piyasa Aşırı Açgözlü, AL Riskli)"
        elif score < 20 and final_signal == "SELL":
            final_signal = "BEKLE"
            filter_reason = " (Piyasa Aşırı Korkuda, SAT Riskli)"

    # ---> YENİLİK: Gerekçe metni için modelin ham sinyalini Türkçeleştir
    model_signal_tr = "AL" if model_signal == "BUY" else ("SAT" if model_signal == "SELL" else "BEKLE")
    reason = f"AI Model Tahmini: {model_signal_tr}{filter_reason}"
    
    # --- SON ---
    
    if confidence > 0.85: strength = "Çok Güçlü"
    elif confidence > 0.70: strength = "Güçlü"
    elif confidence > 0.55: strength = "Orta"
    else: strength = "Zayıf"
    
    last_row = df_features.iloc[-1]
    metrics = { "price": last_row.get('close', 0.0), "rsi": last_row.get('RSI', 50.0), "macd_hist": last_row.get('MACD_hist', 0.0), "atr": last_row.get('ATR', 0.0), "confidence": confidence, "strength": strength }
    
    # Not: Dönen 'final_signal' hala İngilizce (BUY/SELL) olmalı, çünkü sistemin geri kalanı onu bekliyor.
    # Biz sadece kullanıcıya gösterilen 'reason' metnini düzelttik.
    return final_signal, reason, metrics

def get_market_regime(symbol: str = "BTCUSDT", timeframe: str = "1d", sma_period: int = 200) -> str:
    """
    Piyasanın genel trendini (rejimini) belirler.
    BTC'nin günlük fiyatının 200 günlük ortalamasına göre karar verir.
    """
    try:
        df_regime = fetch_binance_klines(symbol, timeframe, limit=sma_period + 50)
        if df_regime.empty or len(df_regime) < sma_period: return "NEUTRAL"
        df_regime['SMA_REGIME'] = df_regime['close'].rolling(window=sma_period).mean()
        last_price = df_regime['close'].iloc[-1]
        last_sma = df_regime['SMA_REGIME'].iloc[-1]
        return "BULL" if last_price > last_sma else "BEAR"
    except Exception as e:
        print(f"Market rejim filtresi hatası: {e}")
        return "NEUTRAL"