# Quick Start Guide 🚀

**5 dakikada ML Trading sistemini çalıştırın!**

## 1️⃣ Kurulum (2 dakika)

```bash
cd ml_trading_advanced
pip install -r requirements.txt
```

## 2️⃣ Konfigürasyon (1 dakika)

`config.py` dosyasını düzenleyin:

```python
# DATA_PATH - CSV dosyanızın yolunu girin
DATA_PATH = "/content/AVAXUSDT_5m_ALL_YEARS.csv"

# Diğer ayarlar varsayılan olarak iyidir
```

### CSV Format

CSV dosyanız şu sütunlara sahip olmalı:
```
timestamp, open, high, low, close, volume
```

Veya:
```
timestamp, Open, High, Low, Close, Volume
```

## 3️⃣ Training (2 dakika - 2 saat, data boyutuna göre)

```bash
python train_advanced.py
```

### Beklenen Çıktı:

```
================================================================================
ADVANCED ML TRADING STRATEGY - TRAINING PIPELINE
================================================================================

STEP 1: LOADING DATA
✓ Data loaded: 150,000 bars

STEP 2: FEATURE ENGINEERING
🔧 Engineering advanced features...
   ✓ Time features added
   ✓ Price action features added
   ✓ Volume features added
   ...
✓ Features engineered. Rows: 149,800

STEP 3: CREATING LABELS (TRIPLE BARRIER METHOD)
🏷️  Creating labels using Triple Barrier Method (both)...
   Long labels created: 15,234 signals (10.17%)
   Short labels created: 14,987 signals (10.00%)

STEP 4: PREPARING FEATURES
Total features: 237
Performing feature selection to get TOP 50 features...
✓ Selected 50 features

STEP 5: TRAINING MODELS
================================================================================
TRAINING LONG MODELS
================================================================================

Training XGBOOST Model
Train samples: 74,860 (Class 1: 7,234, 9.7%)
Val samples:   22,470 (Class 1: 2,187, 9.7%)
Test samples:  52,470 (Class 1: 5,813, 11.1%)

Optimizing hyperparameters with Optuna...
[I 2024-XX-XX ...] Trial 0 finished with value: 0.7234
[I 2024-XX-XX ...] Trial 1 finished with value: 0.7456
...
Best AUC: 0.7821
Best params: {'n_estimators': 347, 'max_depth': 7, ...}

Train  | Acc: 0.892 | Prec: 0.847 | Rec: 0.823 | F1: 0.835 | AUC: 0.934
Val    | Acc: 0.743 | Prec: 0.712 | Rec: 0.698 | F1: 0.705 | AUC: 0.782
Test   | Acc: 0.738 | Prec: 0.709 | Rec: 0.691 | F1: 0.700 | AUC: 0.776

🏆 Best LONG model: XGBOOST
   Test F1: 0.700
   Test AUC: 0.776

STEP 6: WALK-FORWARD VALIDATION
...

✅ ALL DONE! Models are ready for deployment.
```

## 4️⃣ Sonuçları İnceleme

```bash
ls models_advanced/

# Çıktı:
# model_long_xgboost.pkl
# model_short_xgboost.pkl
# scaler_long_xgboost.pkl
# scaler_short_xgboost.pkl
# features.txt
# walk_forward_long.csv
# walk_forward_short.csv
# feature_importance_initial.csv
```

### Walk-Forward Sonuçlarına Bakın:

```python
import pandas as pd

wf = pd.read_csv('models_advanced/walk_forward_long.csv')
print(wf[['fold', 'f1', 'precision', 'recall', 'roc_auc']])

#    fold    f1  precision  recall  roc_auc
# 0     1  0.68       0.71    0.66     0.75
# 1     2  0.72       0.74    0.70     0.78
# 2     3  0.65       0.69    0.62     0.72
# ...

print(f"Average F1: {wf['f1'].mean():.3f}")
print(f"Average AUC: {wf['roc_auc'].mean():.3f}")
```

## 5️⃣ Inference (Tahmin Yapma)

```python
from inference import TradingPredictor
import pandas as pd

# Veri yükle
df = pd.read_csv('/content/AVAXUSDT_5m_ALL_YEARS.csv')

# Predictor oluştur
predictor = TradingPredictor(model_dir='models_advanced')

# En son bar için tahmin
signal = predictor.predict_latest(
    df,
    side='both',
    long_threshold=0.65,
    short_threshold=0.65
)

print(f"Signal: {signal['signal']}")  # 1=LONG, -1=SHORT, 0=NEUTRAL
print(f"LONG prob: {signal['long_prob']:.3f}")
print(f"SHORT prob: {signal['short_prob']:.3f}")
```

---

## 🎯 İlk Sonuçlar Kötüyse?

### 1. Barrier'ları Ayarlayın

`config.py`:
```python
# Daha fazla sinyal istiyorsanız:
STATIC_TP_PCT = 1.0  # 1.5'ten 1.0'a düşürün
MIN_RETURN_THRESHOLD = 0.2  # 0.3'ten 0.2'ye düşürün

# Daha kaliteli sinyal istiyorsanız:
STATIC_TP_PCT = 2.0  # 1.5'ten 2.0'a çıkarın
MIN_RETURN_THRESHOLD = 0.5  # 0.3'ten 0.5'e çıkarın
```

### 2. Feature Sayısını Değiştirin

```python
TOP_N_FEATURES = 40  # 50'den 40'a düşürün (overfitting'i azaltır)
# veya
TOP_N_FEATURES = 70  # 50'den 70'e çıkarın (daha fazla bilgi)
```

### 3. Optuna'yı Atlayın (Hızlı Test)

```python
USE_OPTUNA = False  # İlk testler için
# Sonra True yapıp optimize edin
```

### 4. Daha Fazla Data

- En az 50,000 bar kullanın
- Tercihen 100,000+ bar

### 5. Farklı Asset Deneyin

- Volatilite yüksek → BTC, ETH daha iyi
- Volatilite düşük → Stablecoin pair'ler zor

---

## 📊 Benchmark Sonuçlar (AVAXUSDT 5m)

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Test F1 | >0.60 | >0.70 | >0.80 |
| Test AUC | >0.70 | >0.75 | >0.85 |
| Precision | >0.60 | >0.70 | >0.80 |
| Recall | >0.50 | >0.65 | >0.75 |

**Not**: Test ve Train arasında çok fark varsa (örn. Train F1=0.95, Test F1=0.60) → Overfitting var!

---

## 🔥 Pro Tips

### Tip 1: Threshold Optimization
```python
# models_advanced/walk_forward_predictions_long.csv dosyasını kullanarak:
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score

preds = pd.read_csv('models_advanced/walk_forward_predictions_long.csv')

# Farklı threshold'ları test et
for threshold in np.arange(0.45, 0.80, 0.05):
    y_pred = (preds['y_pred_proba'] >= threshold).astype(int)
    f1 = f1_score(preds['y_true'], y_pred)
    prec = precision_score(preds['y_true'], y_pred)
    print(f"Threshold {threshold:.2f}: F1={f1:.3f}, Precision={prec:.3f}")

# En iyi threshold'u seç
```

### Tip 2: Feature Importance
```python
import pandas as pd

imp = pd.read_csv('models_advanced/feature_importance_initial.csv')
print(imp.head(20))

# En önemli feature'ları not edin
# Eğer 'price_vs_sma_50' çok önemliyse → Trend following çalışıyor
# Eğer 'volatility_10' çok önemliyse → Volatility breakout çalışıyor
```

### Tip 3: Model Comparison
```python
# Farklı modelleri karşılaştırın
# config.py:
MODELS = {
    'xgboost': True,
    'lightgbm': True,
    'catboost': True,
}

# Training sonrası en iyi performansı seçin
```

---

## ⚡ Hızlı Test (1 dakika)

Tam training çok uzun sürüyorsa, küçük bir subset ile test edin:

```python
# train_advanced.py'de bu satırı bulun:
df = pd.read_csv(data_path)

# Hemen altına ekleyin:
df = df.tail(20000)  # Son 20K bar ile test

# config.py'de:
USE_OPTUNA = False
OPTUNA_TRIALS = 20  # 100 yerine
```

---

## ❓ Sık Sorulan Sorular

**S: Training ne kadar sürer?**
A:
- 50K bar, Optuna=False: ~2-5 dakika
- 50K bar, Optuna=True, 100 trials: ~20-40 dakika
- 200K bar, Optuna=True: ~1-2 saat

**S: Test F1 0.50 civarında, normal mi?**
A: Hayır, çok düşük. Barrier'ları ve MIN_RETURN_THRESHOLD'u ayarlayın.

**S: Train F1=0.95, Test F1=0.60, sorun ne?**
A: Overfitting. TOP_N_FEATURES azaltın (30-40), regularization artırın.

**S: Optuna gerekli mi?**
A: İlk testlerde hayır. Ancak production için kesinlikle evet.

**S: LONG iyi, SHORT kötü?**
A: Normal. Crypto genelde uptrend. SHORT'u devre dışı bırakabilirsiniz.

---

**Başarılar! 📈🚀**

Sorun olursa README.md'ye bakın veya issue açın.
