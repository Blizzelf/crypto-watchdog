# backtester_engine.py (YENİ VE GÜÇLENDİRİLMİŞ VERSİYON)
import pandas as pd
import plotly.graph_objects as go
from indicators import add_all_indicators
from chart import compute_trade_levels

def run_ai_backtest(model, df_history: pd.DataFrame, market_regime: str, 
                    atr_multiplier: float, rr_t1: float, rr_t2: float):
    df_features = add_all_indicators(df_history)
    features_for_model = ['RSI', 'MACD_hist', 'BB_low', 'BB_high', 'ATR', 'SMA50', 'SMA200']
    trades = []
    in_position = False
    
    for i in range(len(df_features)):
        current_candle = df_features.iloc[i]
        
        if in_position:
            if trades[-1]['direction'] == 'BUY':
                if current_candle['low'] <= trades[-1]['stop_loss']:
                    trades[-1].update({"exit_price": trades[-1]['stop_loss'], "exit_date": current_candle.name, "reason": "Zarar Durdur"})
                    in_position = False
                elif current_candle['high'] >= trades[-1]['take_profit']:
                    trades[-1].update({"exit_price": trades[-1]['take_profit'], "exit_date": current_candle.name, "reason": "Kâr Al"})
                    in_position = False
            elif trades[-1]['direction'] == 'SELL':
                if current_candle['high'] >= trades[-1]['stop_loss']:
                    trades[-1].update({"exit_price": trades[-1]['stop_loss'], "exit_date": current_candle.name, "reason": "Zarar Durdur"})
                    in_position = False
                elif current_candle['low'] <= trades[-1]['take_profit']:
                    trades[-1].update({"exit_price": trades[-1]['take_profit'], "exit_date": current_candle.name, "reason": "Kâr Al"})
                    in_position = False

        if not in_position:
            last_data = df_features[features_for_model].iloc[i:i+1]
            prediction_mapped = model.predict(last_data)[0]
            signal_map = {0: "SELL", 1: "BEKLE", 2: "BUY"}
            model_signal = signal_map.get(prediction_mapped, "BEKLE")
            
            final_signal = model_signal
            if market_regime == "BULL" and model_signal == "SELL": final_signal = "BEKLE"
            elif market_regime == "BEAR" and model_signal == "BUY": final_signal = "BEKLE"
            
            if final_signal in ["BUY", "SELL"]:
                trade_plan = compute_trade_levels(df_features.iloc[:i+1], final_signal, rr_t1, rr_t2, atr_multiplier)
                if trade_plan and trade_plan.get("stop") and trade_plan.get("t1"):
                    in_position = True
                    trades.append({
                        "entry_date": current_candle.name,
                        "entry_price": current_candle['close'],
                        "direction": final_signal,
                        "stop_loss": trade_plan["stop"],
                        "take_profit": trade_plan["t1"]
                    })
    return trades

# ---> YENİLİK: Fonksiyon artık (metrikler, işlem_verisi) döndürüyor
def calculate_metrics(trades: list, initial_capital: float = 10000.0) -> (dict, pd.DataFrame):
    """Performans metriklerini ve sermaye eğrisi verisini hesaplar."""
    if not trades:
        return {"total_trades": 0}, pd.DataFrame()

    df_trades = pd.DataFrame(trades).dropna(subset=['exit_price'])
    if df_trades.empty:
        return {"total_trades": len(trades), "closed_trades": 0}, pd.DataFrame()

    df_trades['pnl'] = (df_trades['exit_price'] - df_trades['entry_price'])
    df_trades.loc[df_trades['direction'] == 'SELL', 'pnl'] *= -1
    df_trades['pnl_pct'] = (df_trades['pnl'] / df_trades['entry_price']) * 100
    
    # ---> YENİLİK: Sermaye Eğrisini Hesapla
    df_trades['equity_curve'] = initial_capital + df_trades['pnl'].cumsum()

    total_trades = len(df_trades)
    wins = df_trades[df_trades['pnl'] > 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    metrics = {
        "total_trades": total_trades, "win_rate_pct": win_rate, "profit_factor": profit_factor,
        "average_pnl_pct": df_trades['pnl_pct'].mean(),
    }
    return metrics, df_trades

# ---> YENİ FONKSİYON: Sermaye Eğrisi Grafiğini Çizer
def render_equity_curve(df_trades: pd.DataFrame, initial_capital: float = 10000.0):
    """Verilen işlem verisinden bir sermaye eğrisi grafiği oluşturur."""
    if df_trades.empty or 'equity_curve' not in df_trades.columns:
        return None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trades['exit_date'], 
        y=df_trades['equity_curve'],
        mode='lines',
        name='Sermaye',
        line=dict(color='#00BFFF', width=2)
    ))
    
    # Başlangıç sermayesi çizgisini ekle
    fig.add_hline(y=initial_capital, line_width=1.5, line_dash="dash", line_color="gray",
                  annotation_text="Başlangıç Sermayesi", annotation_position="bottom right")

    fig.update_layout(
        title="Sermaye Eğrisi (Equity Curve)",
        xaxis_title="Tarih",
        yaxis_title="Toplam Sermaye (USDT)",
        template="plotly_dark",
        height=400
    )
    return fig