# app.py (NİHAİ, TAM VE HATASIZ VERSİYON)
# -*- coding: utf-8 -*-
import os
import time
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib

from data import get_env_list, fetch_binance_klines, get_fear_and_greed_index
from indicators import add_all_indicators
from strategy import decide_signals_with_model, get_market_regime
from backtester_engine import run_ai_backtest, calculate_metrics, render_equity_curve
from interactive_chart import render_plotly_chart
from chart import build_chart_caption, compute_trade_levels
from advanced_news_ai import analyze_headlines_with_finbert
from news_sources import get_headlines_multi

# --- Genel Ayarlar ve Yardımcı Fonksiyonlar ---
load_dotenv()
SYMBOLS = get_env_list("SYMBOLS", "BTCUSDT,ETHUSDT,XRPUSDT,DOGEUSDT")
LANG = os.getenv("LANGUAGE", "tr").lower().strip()
CHART_BARS = int(os.getenv("CHART_BARS", "200"))
CHART_THEME = os.getenv("CHART_THEME", "dark")
def tr_en(tr_text, en_text): return tr_text if LANG == "tr" else en_text
def badge(sig):
    if sig == "BUY": return "📈🟢 AL"
    if sig == "SELL": return "📉🔴 SAT"
    return "⏸️ BEKLE"

# --- Streamlit Sayfa Yapılandırması ---
st.set_page_config(page_title="Crypto Watchdog", layout="wide", initial_sidebar_state="expanded")

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    try:
        st.image(Image.open("logo.png"), width=70)
    except FileNotFoundError:
        pass
    st.markdown("### ⏱️ Canlı Yenileme")
    auto_refresh = st.toggle("Otomatik yenile", value=True)
    refresh_sec = st.slider("Aralık (sn)", 5, 60, 15)
    st.sidebar.subheader("Genel Ayarlar")
    interval = st.sidebar.selectbox("Zaman Dilimi", ["1m","5m","15m","1h","4h","1d"], index=3)
    bars = st.sidebar.slider("Grafik Mum Sayısı", 60, 500, CHART_BARS, 10)
    theme = st.sidebar.selectbox("Tema", ["Dark", "Light"], index=0 if CHART_THEME == "dark" else 1)
    MODEL_FILENAME = "crypto_xgb_model.pkl"
    try:
        model = joblib.load(MODEL_FILENAME)
        st.sidebar.success("✅ AI Modeli yüklendi.")
    except FileNotFoundError:
        model = None
        st.sidebar.error(f"⚠️ AI Modeli bulunamadı.")
    st.sidebar.caption(f"🕒 Son yenileme: {datetime.now().strftime('%H:%M:%S')}")

if auto_refresh:
    st_autorefresh(interval=refresh_sec * 1000, key="auto_refresher")

# --- Ana Başlık ve Piyasa Durumu Paneli ---
st.title("🐺 Crypto Watchdog")

col1, col2 = st.columns(2)

with col1:
    # Piyasa Rejimi Göstergesi
    with st.spinner("Piyasa rejimi analiz ediliyor..."):
        regime = get_market_regime()
    regime_emoji = "🐂" if regime == "BULL" else ("🐻" if regime == "BEAR" else "↔️")
    regime_text = tr_en("BOĞA PİYASASI", "BULL MARKET") if regime == "BULL" else (tr_en("AYI PİYASASI", "BEAR MARKET") if regime == "BEAR" else "NÖTR")
    st.header(f"Genel Trend: {regime_emoji} {regime_text}")
    st.caption("Bitcoin'in 200 günlük ortalamasına göre belirlenir.")

with col2:
    # Korku ve Açgözlülük Göstergesi
    fng_data = get_fear_and_greed_index()
    if fng_data:
        score = fng_data["score"]
        classification = fng_data["classification"]
        if score < 25: fng_emoji = "😱"
        elif score < 46: fng_emoji = "😨"
        elif score < 55: fng_emoji = "😐"
        elif score < 75: fng_emoji = "😀"
        else: fng_emoji = "🤑"
        st.header(f"Piyasa Psikolojisi: {fng_emoji} {classification}")
        st.caption(f"Korku & Açgözlülük Endeksi: {score}/100")
    else:
        st.header("Piyasa Psikolojisi: ❔ Bilinmiyor")
        st.caption("Korku & Açgözlülük Endeksi verisi alınamadı.")

st.caption("Sinyaller bu iki ana göstergeye göre filtrelenmektedir.")

# --- Haber Akışı ---
st.markdown("---")
st.header("🧠 FinBERT Haber Yorumlama")
CACHE_DURATION_SECONDS = 600
if '_news_cache' in st.session_state and time.time() - st.session_state.get('_news_last_fetched_ts', 0) < CACHE_DURATION_SECONDS:
    headlines = st.session_state.get('_news_cache', [])
else:
    with st.spinner("Güncel haber başlıkları çekiliyor..."):
        headlines = get_headlines_multi(max_items=30)
    if headlines:
        st.session_state['_news_cache'] = headlines
        st.session_state['_news_last_fetched_ts'] = time.time()

if headlines:
    with st.spinner("FinBERT modeli haberleri analiz ediyor..."):
        news_analysis_results = analyze_headlines_with_finbert(headlines)
    if news_analysis_results:
        col1, col2 = st.columns(2)
        positive_news = [res for res in news_analysis_results if res['label'] == 'positive']
        negative_news = [res for res in news_analysis_results if res['label'] == 'negative']
        with col1:
            st.markdown("#### ✅ Pozitif Haberler")
            if positive_news:
                for news in positive_news[:5]: st.success(f"{news['headline']} (Skor: {int(news['score']*100)}%)")
            else:
                st.info("Pozitif haber bulunamadı.")
        with col2:
            st.markdown("#### ⚠️ Negatif Haberler")
            if negative_news:
                for news in negative_news[:5]: st.warning(f"{news['headline']} (Skor: {int(news['score']*100)}%)")
            else:
                st.info("Negatif haber bulunamadı.")
else:
    st.info("Güncel haber başlıkları çekilemedi.")

# --- Sinyal Üretimi (Tüm coinler için bir kerede) ---
signals, rsi_vals, prices = {}, {}, {}
for sym in SYMBOLS:
    try:
        df_raw = fetch_binance_klines(sym, interval, 500)
        df_enriched = add_all_indicators(df_raw)
        if not df_enriched.empty:
            sig, reason, metrics = decide_signals_with_model(df_enriched, model, market_regime=regime)
            signals[sym], rsi_vals[sym], prices[sym] = sig, metrics.get("rsi", 50.0), float(metrics.get("price", 0.0))
        else:
            prices[sym] = 0.0
    except Exception:
        prices[sym] = 0.0

# --- Strateji Brifingi (AI Piyasa Özeti) ---
st.markdown("---")
st.header("💡 AI Strateji Brifingi")
def get_ai_summary_data(rsi_vals, signals, regime):
    buy_count = list(signals.values()).count("BUY"); sell_count = list(signals.values()).count("SELL")
    hold_count = len(signals) - buy_count - sell_count
    durum_tespiti_md = f"#### 🧭 Durum Tespiti\n{regime_text}"
    if regime == "BULL": strateji_aciklamasi_text = "Stratejimiz, ana trend yönündeki **AL** fırsatlarına odaklanıyor."
    elif regime == "BEAR": strateji_aciklamasi_text = "Stratejimiz, düşen piyasada riskli AL sinyallerini filtreleyerek **sermayeyi korumaya** odaklanıyor."
    else: strateji_aciklamasi_text = "Stratejimiz, net bir trend oluşana kadar **temkinli kalmayı** tercih ediyor."
    strateji_aciklamasi_md = f"#### 📜 Strateji Açıklaması\n{strateji_aciklamasi_text}"
    if regime == "BULL": note_text = "Model, trendle uyumlu **giriş fırsatları** tespit etti." if buy_count > 0 else "Trend pozitif olsa da, model **düşük riskli bir giriş noktası** bekliyor."
    elif regime == "BEAR": note_text = "Model, trendle uyumlu **zayıflık sinyalleri** tespit etti." if sell_count > 0 else "Düşüş trendi nedeniyle, potansiyel AL sinyalleri filtrelendi. **Nakit pozisyonu** en güvenli taktik."
    else: note_text = "Piyasadaki kararsızlık nedeniyle, model **kenarda beklemeyi** öneriyor."
    ai_note_md = f"#### 💡 AI Stratejistin Notu\n{note_text}"
    return {"durum_tespiti": durum_tespiti_md, "strateji_aciklamasi": strateji_aciklamasi_md, "ai_note": ai_note_md, "buy_count": buy_count, "sell_count": sell_count, "hold_count": hold_count}
brifing_data = get_ai_summary_data(rsi_vals, signals, regime)
col1, col2, col3 = st.columns(3)
with col1: st.markdown(brifing_data["durum_tespiti"])
with col2: st.markdown(brifing_data["strateji_aciklamasi"])
with col3: st.markdown(brifing_data["ai_note"])
st.markdown("---")
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("📈 AL Sinyalleri", brifing_data["buy_count"])
mcol2.metric("📉 SAT Sinyalleri", brifing_data["sell_count"])
mcol3.metric("⏸️ BEKLE Sinyalleri", brifing_data["hold_count"])

# --- Coin Bazında Göstergeler ---
cols = st.columns(2)
for i, sym in enumerate(SYMBOLS):
    with cols[i % 2]:
        st.markdown("---")
        try:
            df_raw = fetch_binance_klines(sym, interval, 500)
            df_enriched = add_all_indicators(df_raw)
            if df_enriched.empty: st.warning(f"{sym} için veri hesaplanamadı."); continue

            sig, reason, metrics = decide_signals_with_model(df_enriched, model, market_regime=regime)
            st.subheader(f"{sym} — {badge(sig)}")

            # --- YENİLENMİŞ "MİNİ DURUM RAPORU" ---
            macd_hist, momentum_emoji = metrics.get("macd_hist", 0.0), "🔼" if metrics.get("macd_hist", 0.0) > 0 else "🔽"
            momentum_text = "Pozitif" if metrics.get("macd_hist", 0.0) > 0 else "Negatif"
            
            price = metrics.get("price", 1.0)
            atr_pct = (metrics.get("atr", 0.0) / price) * 100 if price > 0 else 0
            volatility_emoji = "🔥" if atr_pct > 2.0 else "💧"
            volatility_text = f"Yüksek ({atr_pct:.2f}%)" if atr_pct > 2.0 else f"Düşük ({atr_pct:.2f}%)"

            # ---> YENİLİK: Sinyal Gücü satırını ekliyoruz
            strength = metrics.get("strength", "N/A")
            confidence = metrics.get("confidence", 0.0)
            strength_emoji = "✅" if strength in ["Çok Güçlü", "Güçlü"] else "🤔"
            strength_text = f"{strength} ({confidence:.0%})"

            report = (
                f"- **Fiyat:** `{metrics.get('price', 0.0):.6g}`\n"
                f"- {momentum_emoji} **Momentum:** {momentum_text}\n"
                f"- {volatility_emoji} **Volatilite:** {volatility_text}\n"
                f"- {strength_emoji} **Sinyal Gücü:** {strength_text}\n" # YENİ SATIR
                f"- 🎯 **Gerekçe:** {reason or '-'}"
            )
            st.markdown(report)
            # --- SON ---

            plan = compute_trade_levels(df_enriched, sig)
            fig = render_plotly_chart(df_enriched.tail(bars), plan=plan, theme=theme)
            st.plotly_chart(fig, use_container_width=True)
            caption_text = build_chart_caption(sym, df_enriched, signal=sig, plan=plan, regime=regime)
            st.caption(caption_text)

        except Exception as e:
            st.error(f"{sym} işlenirken hata oluştu: {e}")

# ==================================
# 📊 NİHAİ VARLIK YÖNETİM SİSTEMİ
# ==================================
st.markdown("---")
st.header("📊 Varlık Yönetim Sistemi")

# --- Portföy Yardımcı Fonksiyonları ---
def add_live_data_and_pnl(df_aggregated, prices_dict):
    if df_aggregated.empty: return df_aggregated
    def _calculate_final_row(row):
        sym = str(row["symbol"]).upper()
        row['Birim Fiyat'] = float(prices_dict.get(sym, 0))
        row['Toplam Değer'] = row['Toplam Adet'] * row['Birim Fiyat']
        row['Kar / Zarar'] = row['Toplam Değer'] - (row['Toplam Adet'] * row['Ort. Maliyet'])
        row['cost_basis'] = row['Toplam Adet'] * row['Ort. Maliyet']
        row['pnl_standard'] = row['Toplam Değer'] - row['cost_basis']
        row['pnl_pct_standard'] = (row['pnl_standard'] / row['cost_basis'] * 100) if row['cost_basis'] > 0 else 0.0
        return row
    return df_aggregated.apply(_calculate_final_row, axis=1)

def aggregate_portfolio(df_trans):
    required_columns = ["symbol", "Toplam Adet", "Ort. Maliyet", "İlk Alış Fiyatı", "İlk Alış Tarihi", "Son İşlem Tarihi"]
    if df_trans.empty: return pd.DataFrame(columns=required_columns)
    df_trans_sorted = df_trans.sort_values(by="timestamp")
    aggregated_data = []
    for symbol, group in df_trans_sorted.groupby('symbol'):
        buys, sells = group[group['type'] == 'BUY'], group[group['type'] == 'SELL']
        total_bought, total_sold = buys['amount'].sum(), sells['amount'].sum()
        current_amount = total_bought - total_sold
        if current_amount < 1e-9: continue
        if not buys.empty:
            first_buy = buys.iloc[0]
            initial_price, initial_date = first_buy['price'], first_buy['timestamp']
            last_transaction_date = group.iloc[-1]['timestamp']
            avg_price = np.average(buys['price'], weights=buys['amount']) if total_bought > 0 else 0
            aggregated_data.append({"symbol": symbol, "Toplam Adet": current_amount, "Ort. Maliyet": avg_price, "İlk Alış Fiyatı": initial_price, "İlk Alış Tarihi": initial_date, "Son İşlem Tarihi": last_transaction_date})
    if not aggregated_data: return pd.DataFrame(columns=required_columns)
    return pd.DataFrame(aggregated_data)

def create_portfolio_pie_chart(pf_df, usdt_balance):
    labels, values = [], []
    if not pf_df.empty and 'Toplam Değer' in pf_df.columns:
        df_chart = pf_df[pf_df['Toplam Değer'] > 0.01].copy()
        labels.extend(df_chart['symbol'].tolist()); values.extend(df_chart['Toplam Değer'].tolist())
    if usdt_balance > 0.01:
        labels.append("NAKİT (USDT)"); values.append(usdt_balance)
    if not values: return None
    pull_values = [0.05 if v == max(values) else 0 for v in values]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#2E8B57']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, textinfo='percent', insidetextorientation='radial', pull=pull_values, marker_colors=colors, sort=False)])
    fig.update_layout(showlegend=True, legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)'), height=350, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig
    
def analyze_portfolio(pf_df: pd.DataFrame, signals: dict) -> dict:
    insights = {"good": [], "bad": [], "info": []}
    if pf_df.empty or 'Toplam Değer' not in pf_df.columns or pf_df["Toplam Değer"].sum() == 0:
        insights["info"].append("Analiz için portföyünüzde değeri olan varlık bulunmalıdır."); return insights
    total_value = pf_df["Toplam Değer"].sum()
    for _, row in pf_df.iterrows():
        concentration = (row["Toplam Değer"] / total_value) * 100
        if concentration > 40: insights["bad"].append(f"**Konsantrasyon Riski:** `{row['symbol']}` portföyünüzün **%{concentration:.0f}**'ını oluşturuyor.")
    if 'pnl_pct_standard' in pf_df.columns:
        perf_df = pf_df[pf_df['cost_basis'] > 0].copy()
        if not perf_df.empty:
            best_performer = perf_df.loc[perf_df['pnl_pct_standard'].idxmax()]
            worst_performer = pf_df.loc[perf_df['pnl_pct_standard'].idxmin()]
            insights["good"].append(f"**Portföyün Yıldızı:** `{best_performer['symbol']}` **%{best_performer['pnl_pct_standard']:.2f}** ile parlıyor.")
            if worst_performer['pnl_pct_standard'] < 0: insights["bad"].append(f"**Zayıf Halka:** `{worst_performer['symbol']}` **%{worst_performer['pnl_pct_standard']:.2f}** ile geride kalıyor.")
    held_assets = pf_df[pf_df['Toplam Adet'] > 0]['symbol'].tolist()
    for asset in held_assets:
        if signals.get(asset) == "SELL": insights["bad"].append(f"**Strateji Çatışması:** Elinizde `{asset}` varken model **SAT** diyor.")
    for symbol, signal in signals.items():
        if signal == "BUY" and symbol not in held_assets: insights["info"].append(f"**Kaçan Fırsat?:** Portföyünüzde olmayan `{symbol}` için model **AL** diyor.")
    return insights

# --- Cüzdan ve İşlem Günlüğü ---
WALLET_FILE, TRANSACTIONS_FILE = "wallet.json", "transactions.csv"
def load_wallet(initial=10000.0):
    if not Path(WALLET_FILE).exists(): save_wallet(initial); return initial
    with open(WALLET_FILE, 'r') as f: return json.load(f).get('usdt', initial)
def save_wallet(amount):
    with open(WALLET_FILE, 'w') as f: json.dump({'usdt': amount}, f)
def load_transactions():
    if not Path(TRANSACTIONS_FILE).exists(): return pd.DataFrame(columns=["timestamp", "symbol", "type", "amount", "price"])
    return pd.read_csv(TRANSACTIONS_FILE, parse_dates=["timestamp"])
def save_transactions(df): df.to_csv(TRANSACTIONS_FILE, index=False)
usdt_balance, transactions = load_wallet(), load_transactions()

# --- İşlem Terminali ve Kasa Yönetimi ---
st.subheader("💲 " + tr_en("Nakit Varlık (USDT)", "Cash Balance (USDT)"))
new_usdt = st.number_input("Mevcut USDT Bakiyeniz", min_value=0.0, value=usdt_balance, format="%.2f", label_visibility="collapsed")
if new_usdt != usdt_balance: 
    save_wallet(new_usdt); usdt_balance = new_usdt; st.rerun()

st.subheader("📈 " + tr_en("Alım / Satım Terminali", "Buy / Sell Terminal"))
t_col1, t_col2, t_col3, t_col4 = st.columns([2, 2, 1.2, 0.8])
with t_col1: t_type = st.radio("İşlem Tipi", ["Alım", "Satım"], horizontal=True, label_visibility="collapsed")
with t_col2: t_sym = st.selectbox("Coin Seç", options=SYMBOLS, label_visibility="collapsed")
with t_col3: t_amount = st.number_input("Adet", min_value=0.0, step=0.001, format="%.6f", label_visibility="collapsed")
with t_col4:    
    if st.button("Onayla", use_container_width=True):
        price = prices.get(t_sym, 0.0)
        if t_type == "Alım":
            cost = t_amount * price
            if cost > usdt_balance: st.error(f"Yetersiz Bakiye! {cost:.2f} USDT gerekli.")
            elif t_amount > 0:
                new_trade = pd.DataFrame([{"timestamp": datetime.now(), "symbol": t_sym, "type": "BUY", "amount": t_amount, "price": price}])
                transactions = pd.concat([transactions, new_trade], ignore_index=True)
                save_transactions(transactions); save_wallet(usdt_balance - cost); st.success("Alım başarılı!"); st.rerun()
        elif t_type == "Satım":
            summary = aggregate_portfolio(transactions)
            held = summary[summary['symbol'] == t_sym]['Toplam Adet'].sum() if not summary.empty else 0
            if t_amount > held: st.error(f"Yetersiz Varlık! En fazla {held:.4f} satılabilir.")
            elif t_amount > 0:
                revenue = t_amount * price
                new_trade = pd.DataFrame([{"timestamp": datetime.now(), "symbol": t_sym, "type": "SELL", "amount": t_amount, "price": price}])
                transactions = pd.concat([transactions, new_trade], ignore_index=True)
                save_transactions(transactions); save_wallet(usdt_balance + revenue); st.success("Satım başarılı!"); st.rerun()

# --- Gösterge Tablosu ve Komuta Paneli ---
df_aggregated = aggregate_portfolio(transactions)
all_symbols_df = pd.DataFrame({'symbol': SYMBOLS})
df_pf_display = pd.merge(all_symbols_df, df_aggregated, on='symbol', how='left')
df_with_live_data = add_live_data_and_pnl(df_pf_display, prices)
df_for_view = df_with_live_data.copy()
for col in ["Toplam Adet", "Birim Fiyat", "Toplam Değer", "Ort. Maliyet", "İlk Alış Fiyatı", "Kar / Zarar"]:
    if col in df_for_view.columns: df_for_view[col] = df_for_view[col].fillna(0.0)
for col in ["İlk Alış Tarihi", "Son İşlem Tarihi"]:
    if col in df_for_view.columns:
        df_for_view[col] = pd.to_datetime(df_for_view[col], errors='coerce')
        df_for_view[col] = df_for_view[col].dt.strftime('%d/%m/%Y').replace('NaT', '')
st.dataframe(df_for_view, column_order=["symbol", "Toplam Adet", "Birim Fiyat", "Toplam Değer", "Ort. Maliyet", "İlk Alış Fiyatı", "Kar / Zarar", "İlk Alış Tarihi", "Son İşlem Tarihi"],
    column_config={"symbol": "🪙 Varlık", "Toplam Adet": "📦 Toplam Miktar", "Birim Fiyat": "💲 Anlık Fiyat", "Toplam Değer": "💰 Toplam Değer (USDT)", "Ort. Maliyet": "💲 Ortalama Alış Fiyatı", "İlk Alış Fiyatı": "📉 İlk Alış Fiyatı", "Kar / Zarar": "💸 Kâr / Zarar (USDT)", "İlk Alış Tarihi": st.column_config.TextColumn("🗓️ İlk Alım Tarihi"), "Son İşlem Tarihi": st.column_config.TextColumn("📅 Son İşlem Tarihi")},
    use_container_width=True, hide_index=True)
if st.button("🗑️ Tüm İşlem Geçmişini Sıfırla", type="primary", use_container_width=True):
    save_transactions(pd.DataFrame(columns=["timestamp", "symbol", "type", "amount", "price"])); save_wallet(10000.0); st.rerun()

# --- NİHAİ KOMUTA PANELİ VE PERFORMANS KARNESİ ---
st.markdown("---")
total_coin_value = float(df_with_live_data["Toplam Değer"].sum()) if not df_with_live_data.empty and "Toplam Değer" in df_with_live_data.columns else 0.0
total_equity = total_coin_value + usdt_balance

# ---> YENİLİK: Ortalamak için CSS kodu ekliyoruz
st.markdown("""<style>
div[data-testid="stMetric"] { display: flex; justify-content: center; align-items: center; width: 100%;}
div[data-testid="stMetric"] > div { justify-content: center;}
label[data-testid="stMetricLabel"] { display: flex; justify-content: center; align-items: center; width: 100%;}
div[data-testid="stCaptionContainer"] 
</style>""", unsafe_allow_html=True)

col_dash_1, col_dash_2, col_dash_3 = st.columns([1.3, 2, 1.3])
with col_dash_1:
    st.markdown("<h3 style='text-align: center;'>🚀 Toplam Varlık</h3>", unsafe_allow_html=True)
    st.metric(label="TOPLAM NET DEĞER (USDT)", value=f"{total_equity:,.2f}", label_visibility="collapsed")
    st.caption(f"""
        <div style='text-align: center;'>
            Coin Varlıkları: {total_coin_value:,.2f} USDT<br>
            Nakit Varlık: {usdt_balance:,.2f} USDT
        </div>
    """, unsafe_allow_html=True)
with col_dash_2:
    st.markdown("<h3 style='text-align: center;'>⚡ Performans Karnesi</h3>", unsafe_allow_html=True)
    if not df_with_live_data.empty and 'pnl_standard' in df_with_live_data.columns:
        positions = df_with_live_data[df_with_live_data['Toplam Adet'] > 0]
        if not positions.empty: 
            profitable = positions[positions['pnl_standard'] > 0]
            gross_profit = profitable['pnl_standard'].sum()
            gross_loss = abs(positions[positions['pnl_standard'] <= 0]['pnl_standard'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            profitable_rate = (len(profitable) / len(positions)) * 100
            best_performer = positions.loc[positions['pnl_pct_standard'].idxmax()]
            worst_performer = positions.loc[positions['pnl_pct_standard'].idxmin()]
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                sub_col1.metric("Kâr Faktörü", f"{profit_factor:.2f}", help="Toplam kârın toplam zarara oranı. >1.5 ise iyi.")
                sub_col1.metric("Portföyün Yıldızı ✨", f"{best_performer['symbol']}", f"{best_performer['pnl_pct_standard']:+.2f}%")
            with sub_col2:
                sub_col2.metric("Karlı Pozisyon Oranı", f"{profitable_rate:.1f}%", help="Kârda olan pozisyonların yüzdesi.")
                sub_col2.metric("Zayıf Halka 🔗", f"{worst_performer['symbol']}", f"{worst_performer['pnl_pct_standard']:+.2f}%")
        else:
            with st.container():
                st.info("Performans analizi için açık pozisyon bulunmalıdır.")
                st.markdown("<style>div[data-testid='stInfo'] {text-align: center;}</style>", unsafe_allow_html=True)
with col_dash_3:
    st.markdown("<h3 style='text-align: center;'>🎨 Varlık Dağılımı</h3>", unsafe_allow_html=True)
    pie_fig = create_portfolio_pie_chart(df_with_live_data, usdt_balance)
    if pie_fig: st.plotly_chart(pie_fig, use_container_width=True)

# --- AI Portföy Analisti (Artık panelin altında) ---
st.markdown("---")
st.markdown("#### 🤖 AI Portföy Analisti")
st.caption("AI modelinin portföyünüz hakkındaki anlık yorumları ve uyarıları:")
if not df_with_live_data.empty and 'Toplam Adet' in df_with_live_data.columns and df_with_live_data['Toplam Adet'].sum() > 0:
    portfolio_insights = analyze_portfolio(df_with_live_data, signals)
    for insight in portfolio_insights.get("bad", []): st.warning(insight)
    for insight in portfolio_insights.get("good", []): st.success(insight)
    for insight in portfolio_insights.get("info", []): st.info(insight)
else:
    st.info("Analiz için portföyde varlık bulunmalıdır.")
   
# ================= Backtest Bölümü ==================  
st.markdown("---")
st.header("🧪 Modern Backtest Motoru")
_bt_symbols = sorted(set(SYMBOLS)) or ["BTCUSDT"]
c1, c2, c3 = st.columns(3)
with c1: bt_sym = st.selectbox("Sembol", _bt_symbols, key="bt_sym_new")
with c2: bt_interval = st.selectbox("Zaman Dilimi", ["15m", "1h", "4h", "1d"], index=1, key="bt_interval_new")
with c3: bt_limit = st.number_input("Test Edilecek Bar Sayısı", value=1000, key="bt_limit_new")
st.markdown("<h6>Strateji Ayarları</h6>", unsafe_allow_html=True)
c4, c5, _ = st.columns(3)
with c4: bt_atr_multiplier = st.slider("ATR Çarpanı", 1.0, 5.0, 2.0, 0.1, key="bt_atr")
with c5: bt_rr_t1 = st.slider("Risk/Kazanç Oranı", 1.0, 5.0, 1.5, 0.1, key="bt_rr_t1")

if st.button("▶︎ Backtest Çalıştır", key="bt_run_new", use_container_width=True):
    if model is None: st.error("Backtest için AI modeli yüklenmiş olmalı.")
    else:
        with st.spinner(f"Simülasyon çalıştırılıyor..."):
            try:
                df_history = fetch_binance_klines(bt_sym, bt_interval, int(bt_limit))
                regime_for_test = get_market_regime(symbol=bt_sym, timeframe=bt_interval)
                st.info(f"Test periyodu için piyasa rejimi: **{regime_for_test}**")
                
                trades = run_ai_backtest(model, df_history, regime_for_test, bt_atr_multiplier, bt_rr_t1, bt_rr_t1 * 2)
                
                # ---> YENİLİK: Artık hem metrikleri hem de işlem verisini alıyoruz
                metrics, df_trades = calculate_metrics(trades)
                
                st.success("Backtest tamamlandı!")
                
                # ---> YENİLİK: Sermaye Eğrisi Grafiğini Çizdiriyoruz
                st.markdown("#### Performans Grafiği")
                equity_fig = render_equity_curve(df_trades)
                if equity_fig:
                    st.plotly_chart(equity_fig, use_container_width=True)
                else:
                    st.info("Grafik çizimi için yeterli sayıda kapanmış işlem bulunamadı.")

                st.markdown("#### Performans Metrikleri")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("İşlem Sayısı", f"{metrics.get('total_trades', 0)}")
                m2.metric("Kazanma Oranı", f"{metrics.get('win_rate_pct', 0):.2f}%")
                m3.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                m4.metric("Ort. %P/L", f"{metrics.get('average_pnl_pct', 0):.3f}%")
                
                if not df_trades.empty:
                    st.markdown("#### İşlem Detayları")
                    st.dataframe(df_trades, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Backtest hatası: {e}")

# ================== Telegram Gönderimi ==================
import requests
st.header("📤 Telegram'a Gönder", divider="gray")
def _build_tg_message():
    lines = [f"🐺 <b>Crypto Watchdog Özeti</b> ({datetime.now().strftime('%H:%M')})", f"<b>Piyasa: {regime_text}</b>"]
    for s in SYMBOLS:
        sig, pr = signals.get(s, "HOLD"), prices.get(s, 0.0)
        badge_txt = "📈🟢 AL" if sig == "BUY" else ("📉🔴 SAT" if sig == "SELL" else "⏸️ BEKLE")
        lines.append(f"\n<b>{s}</b> — {badge_txt} @ <code>{pr:.6g}</code>")
    return "\n".join(lines)
def _tg_send(text):
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: raise RuntimeError(".env dosyasında TELEGRAM ayarları eksik.")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
    if resp.status_code != 200: raise RuntimeError(f"Telegram API hatası: {resp.text}")
    return "OK"
if st.button("📤 Özeti Gönder"):
    try:
        msg = _build_tg_message(); _tg_send(msg); st.success("Telegram'a gönderildi ✅")
    except Exception as e:
        st.error(f"Telegram gönderimi başarısız: {e}")