# interactive_chart.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Helper Functions (Eski chart.py'dan taşındı) ---

def _local_extrema_levels(series: pd.Series, win: int = 5, tol: float = 0.003, max_levels: int = 5):
    """Yerel tepe/diplerden seviye kümele; en çok dokunulan ilk max_levels seviyeyi döndür."""
    vals = series.values
    n = len(vals)
    if n < win * 2 + 1: return []
    highs, lows = [], []
    for i in range(win, n - win):
        window = vals[i - win:i + win + 1]
        if vals[i] == window.max(): highs.append(vals[i])
        if vals[i] == window.min(): lows.append(vals[i])
    levels = sorted(highs + lows)
    if not levels: return []
    
    grouped = []
    for lv in levels:
        if not grouped or abs(lv - np.mean(grouped[-1])) / max(np.mean(grouped[-1]), 1e-9) > tol:
            grouped.append([lv])
        else:
            grouped[-1].append(lv)
            
    reps = sorted([(np.mean(g), len(g)) for g in grouped], key=lambda x: x[1], reverse=True)
    return sorted([lvl for lvl, _ in reps[:max_levels]], reverse=True)

def _ema_cross_points(ema_fast: pd.Series, ema_slow: pd.Series):
    """EMA fast/slow kesişimlerini bul: ('golden' / 'death')."""
    cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) < ema_slow.shift(1))
    cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) > ema_slow.shift(1))
    return cross_up, cross_down

def _fib_levels(high: float, low: float):
    """Klasik fib retracement seviyeleri."""
    diff = high - low
    return {
        "0.0": high, "0.236": high - 0.236 * diff, "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff, "0.618": high - 0.618 * diff, "0.786": high - 0.786 * diff,
        "1.0": low,
    }

# --- Ana Plotly Grafik Fonksiyonu ---

def render_plotly_chart(df: pd.DataFrame, plan: dict = None, theme: str = "dark"):
    """
    Tüm indikatörler, seviyeler ve işlem planı ile interaktif bir Plotly grafiği oluşturur.
    """
    df_chart = df.copy()

    # İndikatörleri hesapla (pandas-ta'yı burada da kullanabiliriz)
    if 'RSI' not in df_chart.columns: df_chart.ta.rsi(append=True)
    if 'SMA50' not in df_chart.columns: df_chart['SMA50'] = df_chart['close'].rolling(50).mean()
    df_chart['EMA20'] = df_chart['close'].ewm(span=20, adjust=False).mean()
    df_chart['EMA50'] = df_chart['close'].ewm(span=50, adjust=False).mean()
    df_chart['EMA200'] = df_chart['close'].ewm(span=200, adjust=False).mean()
    
    # Tema ayarları
    template = "plotly_dark" if theme == "dark" else "plotly_white"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.8, 0.2], specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

    # 1. Ana Grafik: Mumlar ve Ortalamalar
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['open'], high=df_chart['high'],
                                low=df_chart['low'], close=df_chart['close'], name='Fiyat'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA20'], mode='lines', name='EMA 20', line=dict(color='#00BFFF', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA50'], mode='lines', name='EMA 50', line=dict(color='#FF7F50', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA200'], mode='lines', name='EMA 200', line=dict(color='#ADFF2F', width=1.5, dash='dot')), row=1, col=1)

    # 2. Alt Grafik: RSI
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], mode='lines', name='RSI', line=dict(color='#FFD700', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)

    # 3. Seviyeler: Fibonacci, Destek/Direnç
    visible_df = df_chart.tail(150) # Sadece son 150 mumun fib seviyelerini çiz
    fib = _fib_levels(visible_df['high'].max(), visible_df['low'].min())
    for level, value in fib.items():
        fig.add_hline(y=value, line_width=1, line_dash="dot", line_color="gray", opacity=0.4,
                      annotation_text=level, annotation_position="bottom right", row=1, col=1)

    for level in _local_extrema_levels(df_chart['close']):
        fig.add_hline(y=level, line_width=1, line_dash="longdash", line_color="rgba(135, 206, 250, 0.4)", row=1, col=1)

    # 4. İşaretler: EMA Kesişimleri (Golden/Death Cross)
    cross_up, cross_down = _ema_cross_points(df_chart['EMA20'], df_chart['EMA50'])
    fig.add_trace(go.Scatter(x=df_chart.index[cross_up], y=df_chart['EMA50'][cross_up], name='Altın Kesişim',
                             mode='markers', marker=dict(symbol='triangle-up', color='gold', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index[cross_down], y=df_chart['EMA50'][cross_down], name='Ölüm Kesişimi',
                             mode='markers', marker=dict(symbol='triangle-down', color='silver', size=10)), row=1, col=1)

    # 5. İşlem Planı: Stop-Loss ve Kâr-Al
    if plan and plan.get("stop") and plan.get("t1"):
        color_stop = 'red'
        color_profit = 'green'
        fig.add_hline(y=plan['stop'], line_width=2, line_dash="solid", line_color=color_stop,
                      annotation_text="ZARAR DURDUR", annotation_position="bottom right", row=1, col=1)
        fig.add_hline(y=plan['t1'], line_width=2, line_dash="dash", line_color=color_profit,
                      annotation_text="HEDEF 1", annotation_position="bottom right", row=1, col=1)
        if plan.get("t2"):
            fig.add_hline(y=plan['t2'], line_width=2, line_dash="dashdot", line_color=color_profit,
                          annotation_text="HEDEF 2", annotation_position="bottom right", row=1, col=1)

    # Genel Görünüm Ayarları
    fig.update_layout(template=template,
                      height=560,
                      margin=dict(l=20, r=20, t=30, b=20),
                      xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      showlegend=True)
    fig.update_yaxes(title_text="Fiyat", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    return fig