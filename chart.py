# chart.py (YENİ VE TEMİZ YARDIMCI DOSYASI)
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- Diğer dosyaların ihtiyaç duyduğu yardımcı fonksiyonlar ---

def _ema_cross_points(ema_fast: pd.Series, ema_slow: pd.Series, lookback: int = 50):
    """Son lookback bar içinde EMA fast/slow kesişimlerini bul: ('golden' / 'death')."""
    f = ema_fast.tail(lookback + 1)
    s = ema_slow.tail(lookback + 1)
    if len(f) < 2 or len(s) < 2: return []
    
    out = []
    f_vals, s_vals, base_idx = f.values, s.values, len(ema_fast) - len(f.values)
    for i in range(1, len(f_vals)):
        if (f_vals[i-1] - s_vals[i-1]) <= 0 and (f_vals[i] - s_vals[i]) > 0:
            out.append((base_idx + i, "golden"))
        elif (f_vals[i-1] - s_vals[i-1]) >= 0 and (f_vals[i] - s_vals[i]) < 0:
            out.append((base_idx + i, "death"))
    return out

def _fib_levels(high: float, low: float):
    """Klasik fib retracement seviyeleri."""
    diff = high - low
    return {
        "0.618": high - 0.618 * diff,
    }

# chart.py (eski build_chart_caption fonksiyonunu bu yenisiyle değiştirin)

def build_chart_caption(symbol, df, signal: str | None = None, plan: dict | None = None, regime: str = "NEUTRAL") -> str:
    """
    Grafik için AI sinyali, piyasa rejimi ve kritik seviyeleri içeren
    dinamik bir strateji özeti oluşturur.
    """
    s = str(symbol).upper()
    data = df.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce").ffill()

    # 1. Ana Bilgiler: AI Sinyali ve Piyasa Rejimi
    signal_text = f"AI Sinyali: {signal or 'N/A'}"
    regime_text = "Boğa Piyasası" if regime == "BULL" else ("Ayı Piyasası" if regime == "BEAR" else "Nötr Piyasa")

    # 2. Kritik Seviye Tespiti (Öncelik sırasına göre)
    key_level_text = ""
    if plan and plan.get("stop") and plan.get("t1"):
        # Eğer bir işlem planı varsa, en önemli bilgi odur.
        key_level_text = f"STOP: {plan['stop']:.4f} • T1: {plan['t1']:.4f}"
    else:
        # İşlem planı yoksa, Fibonacci seviyesini gösterelim.
        hi = float(pd.to_numeric(data.get("high", data["close"]), errors="coerce").tail(150).max())
        lo = float(pd.to_numeric(data.get("low",  data["close"]), errors="coerce").tail(150).min())
        fib_val = hi - 0.618 * (hi - lo)
        key_level_text = f"Kritik Fib(0.618): {fib_val:.4f}"

    # 3. Tüm parçaları birleştir
    return f"{s} • {signal_text} • {regime_text} • {key_level_text}"

def compute_trade_levels(df, signal: str, rr_t1: float = 1.5, rr_t2: float = 3.0, atr_multiplier: float = 2.0):
    """
    ATR (volatilite) tabanlı dinamik STOP/TARGET hesabı.
    """
    if df.empty: return {}

    data = df.copy()
    for col in ['high', 'low', 'close']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.dropna(subset=['high', 'low', 'close'], inplace=True)
    if data.empty: return {}

    atr_series = ta.atr(high=data["high"], low=data["low"], close=data["close"], length=14)
    if atr_series is None or atr_series.isna().all(): return {}
            
    last_atr = atr_series.iloc[-1]
    entry = data["close"].iloc[-1]
    
    plan = {"direction": signal, "entry": entry, "stop": None, "t1": None, "t2": None}
    if signal == "BUY":
        stop = entry - (last_atr * atr_multiplier)
        risk = entry - stop
        if risk > 0: plan.update(stop=stop, t1=entry + risk * rr_t1, t2=entry + risk * rr_t2)
    elif signal == "SELL":
        stop = entry + (last_atr * atr_multiplier)
        risk = stop - entry
        if risk > 0: plan.update(stop=stop, t1=entry - risk * rr_t1, t2=entry - risk * rr_t2)
            
    return plan