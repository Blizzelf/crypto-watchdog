# train_model.py
import pandas as pd
from data import fetch_binance_klines
from indicators import add_all_indicators
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import os

# --- AYARLAR (Bu değerleri değiştirerek farklı stratejiler deneyebilirsiniz) ---
SYMBOL = "BTCUSDT"          # Hangi coin verisiyle modeli eğiteceğiniz
INTERVAL = "1h"             # Hangi zaman aralığı (1 saatlik mumlar)
LIMIT = 2000                # Ne kadar geçmiş veriyle eğiteceğimiz (2000 saat = ~83 gün)
LOOK_AHEAD_CANDLES = 10     # Gelecekteki kaç muma bakarak AL/SAT kararı vereceğiz (10 saat)
PROFIT_TARGET = 0.015       # %1.5 kar hedefi (AL için)
LOSS_TARGET = -0.01         # %1 kayıp hedefi (SAT için)
MODEL_FILENAME = "crypto_xgb_model.pkl" # Kaydedilecek modelin dosya adı

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gelecekteki fiyat hareketlerine göre veri setini etiketler.
    Bu, yapay zekanın "doğru cevabın" ne olduğunu öğrenmesini sağlar.

    1: AL  (Fiyat gelecekte kar hedefine ulaştı)
   -1: SAT (Fiyat gelecekte zarar hedefine ulaştı)
    0: BEKLE (İkisi de olmadı)
    """
    # Gelecekteki N mum içindeki en yüksek ve en düşük fiyatları bul
    df['future_high'] = df['high'].rolling(window=LOOK_AHEAD_CANDLES).max().shift(-LOOK_AHEAD_CANDLES)
    df['future_low'] = df['low'].rolling(window=LOOK_AHEAD_CANDLES).min().shift(-LOOK_AHEAD_CANDLES)

    # AL koşulu: Gelecekteki en yüksek fiyat, kar hedefimize ulaştı mı?
    buy_condition = (df['future_high'] / df['close']) - 1 >= PROFIT_TARGET
    
    # SAT koşulu: Gelecekteki en düşük fiyat, zarar hedefimize ulaştı mı?
    sell_condition = (df['future_low'] / df['close']) - 1 <= LOSS_TARGET

    # Etiketleri oluştur (önce her şeye BEKLE diyoruz)
    df['label'] = 0
    # Koşulları sağlayanları AL veya SAT olarak güncelliyoruz
    df.loc[buy_condition, 'label'] = 1  # AL
    df.loc[sell_condition, 'label'] = -1 # SAT
    
    # Son mumlar için gelecek verisi olmadığından etiket oluşturulamayan satırları at
    df.dropna(subset=['future_high', 'future_low'], inplace=True)
    
    return df

def train_and_save_model():
    """Ana fonksiyon: Veriyi çeker, hazırlar, modeli eğitir ve kaydeder."""
    
    print(f"'{SYMBOL}' için {LIMIT} adet geçmiş veri çekiliyor...")
    df_raw = fetch_binance_klines(SYMBOL, INTERVAL, LIMIT)

    print("Teknik indikatörler hesaplanıyor...")
    df_features = add_all_indicators(df_raw)

    print("Veri etiketleniyor (AL/SAT/BEKLE kararları oluşturuluyor)...")
    df_labeled = create_labels(df_features)

    # Modelin "soru" olarak kullanacağı özellikler (X) ve "cevap" olarak öğreneceği hedef (y)
    features = ['RSI', 'MACD_hist', 'BB_low', 'BB_high', 'ATR', 'SMA50', 'SMA200']
    
    # Girdi verileri (X) - Fiyat bilgisini doğrudan vermiyoruz, indikatörlerden öğrensin.
    X = df_labeled[features]
    
    # Hedef veri (y) - Bizim oluşturduğumuz etiketler
    y = df_labeled['label']

    if len(X) == 0:
        print("Eğitim için yeterli veri bulunamadı. Lütfen ayarları kontrol edin.")
        return

    # Veriyi, modelin öğrenmesi için bir eğitim setine ve performansını ölçmek için bir test setine ayırıyoruz.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Model {len(X_train)} veri ile eğitiliyor ve {len(X_test)} veri ile test edilecek...")

    # XGBoost Sınıflandırıcı modelini oluşturuyoruz
    model = XGBClassifier(
        objective='multi:softmax', # Problemimiz çok sınıflı (AL, SAT, BEKLE)
        num_class=3,               # 3 sınıfımız var (-1, 0, 1) -> XGBoost bunu 0,1,2 olarak anlar
        use_label_encoder=False,   # Gelecek sürümlerdeki bir uyarıyı engellemek için
        eval_metric='mlogloss'     # Değerlendirme metriği
    )
    
    # XGBoost -1 etiketini kabul etmez, bu yüzden etiketleri 0, 1, 2'ye dönüştürüyoruz
    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
    y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2})

    # Modeli eğitiyoruz!
    model.fit(X_train, y_train_mapped)

    print("\n--- Model Performans Raporu (Test Verisi Üzerinde) ---")
    y_pred_mapped = model.predict(X_test)
    print(classification_report(y_test_mapped, y_pred_mapped, target_names=['SAT', 'BEKLE', 'AL']))

    print(f"Model '{MODEL_FILENAME}' dosyasına kaydediliyor...")
    joblib.dump(model, MODEL_FILENAME)
    print("İşlem başarıyla tamamlandı!")


# Bu script doğrudan komut satırından çalıştırıldığında train_and_save_model fonksiyonunu çağırır
if __name__ == "__main__":
    train_and_save_model()