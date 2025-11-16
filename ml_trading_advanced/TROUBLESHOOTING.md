# Troubleshooting Guide 🔧

**Yaygın hatalar ve çözümleri**

---

## ❌ "ValueError: Found array with 0 sample(s)"

### Sebep:
Feature engineering sonrası tüm satırlar dropna() ile silindi.

### Çözüm 1: Data Miktarını Kontrol Edin
```bash
# En az 1000-2000 bar olmalı
python test_system.py
```

En az **1000 bar** data olmalı. Tercihen **10,000+**.

### Çözüm 2: Data Path'i Kontrol Edin
```python
# config.py
DATA_PATH = "/path/to/your/data.csv"  # ⚠️  Doğru path?
```

Dosya var mı?
```bash
ls -la /path/to/your/data.csv
```

### Çözüm 3: CSV Format Kontrolü
CSV şu sütunlara sahip olmalı:
```
timestamp, open, high, low, close, volume
```

Veya:
```
timestamp, Open, High, Low, Close, Volume
```

---

## ❌ "FileNotFoundError: [Errno 2] No such file or directory"

### Çözüm:
```python
# config.py'de absolute path kullanın
DATA_PATH = "/home/user/trading/AVAXUSDT_5m.csv"  # ✅ Good
DATA_PATH = "data/AVAXUSDT_5m.csv"  # ❌ Bad (relative)
```

**Windows'ta:**
```python
DATA_PATH = "C:/Users/YourName/data/AVAXUSDT_5m.csv"
# veya
DATA_PATH = r"C:\Users\YourName\data\AVAXUSDT_5m.csv"
```

---

## ❌ "KeyError: 'timestamp'"

### Sebep:
CSV'de timestamp kolonu yok veya farklı isimde.

### Çözüm:
CSV'nizi kontrol edin:
```python
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.columns)
```

Eğer `date` veya başka bir isim varsa:
```python
# CSV'yi düzeltin:
df.rename(columns={'date': 'timestamp'}, inplace=True)
df.to_csv('your_file.csv', index=False)
```

---

## ❌ "No signals generated" (target_long=0, target_short=0)

### Sebep:
Barrier'lar çok sıkı, hiçbir trade TP'ye ulaşamıyor.

### Çözüm:
```python
# config.py - Barrier'ları gevşetin
STATIC_TP_PCT = 1.0  # 1.5'ten düşürün
STATIC_SL_PCT = 0.8  # 1.0'dan düşürün
MIN_RETURN_THRESHOLD = 0.2  # 0.3'ten düşürün
```

Test edin:
```bash
python test_system.py
```

Signals görmelisiniz:
```
LONG signals: 450 (9.0%)
SHORT signals: 430 (8.6%)
```

**Optimal signal oranı: %5-15**

---

## ❌ "Overfitting: Train F1=0.95, Test F1=0.55"

### Sebep:
Model training data'yı ezberliyor.

### Çözüm 1: Feature Sayısını Azaltın
```python
# config.py
TOP_N_FEATURES = 30  # 50'den azaltın
```

### Çözüm 2: Regularization Artırın
Optuna otomatik yapıyor ama manuel de ayarlayabilirsiniz:
```python
# model_training.py
params = {
    'max_depth': 4,  # 6'dan azaltın
    'min_child_weight': 5,  # Artırın
    'gamma': 2.0,  # Artırın
}
```

### Çözüm 3: Daha Fazla Data
En az 50,000 bar kullanın.

---

## ❌ "Train/Test performance çok düşük (F1 < 0.55)"

### Sebep 1: Kötü Labeling
Asset'inizin volatilitesine göre barrier'lar yanlış ayarlanmış.

**Çözüm:**
```python
# Yüksek volatilite (BTC, altcoin) için:
STATIC_TP_PCT = 2.0
STATIC_SL_PCT = 1.5

# Düşük volatilite (major forex, stablecoin) için:
STATIC_TP_PCT = 0.5
STATIC_SL_PCT = 0.3
```

### Sebep 2: Data Quality
NaN, zero volume, duplicate candles?

**Çözüm:**
```python
# Data temizleme
df = df[df['volume'] > 0]  # Sıfır volume'leri at
df = df.drop_duplicates(subset=['timestamp'])  # Duplikatları at
df = df.dropna()  # NaN'ları at
```

---

## ❌ "SMOTE Error: k_neighbors too large"

### Sebep:
Positive class çok az (< 6 sample).

### Çözüm:
```python
# config.py
USE_SMOTE = False  # SMOTE'u devre dışı bırakın

# Veya barrier'ları gevşetin (daha fazla signal)
MIN_RETURN_THRESHOLD = 0.2
```

---

## ❌ "Optuna çok yavaş / dondu"

### Çözüm 1: Trial Sayısını Azaltın
```python
# config.py
OPTUNA_TRIALS = 20  # 100'den azaltın (test için)
```

### Çözüm 2: Optuna'yı Devre Dışı Bırakın (İlk Testler)
```python
# config.py
USE_OPTUNA = False  # Default parameters kullan
```

Sonra production için açın.

---

## ❌ "ImportError: No module named 'optuna'"

### Çözüm:
```bash
pip install -r requirements.txt

# Veya manuel:
pip install optuna xgboost lightgbm catboost ta imbalanced-learn
```

---

## ❌ "Memory Error / Killed"

### Sebep:
Çok fazla data + çok fazla feature = RAM doldu.

### Çözüm 1: Data Azaltın (Test İçin)
```python
# train_advanced.py başında:
df = pd.read_csv(DATA_PATH)
df = df.tail(20000)  # Son 20K bar ile test
```

### Çözüm 2: Feature Azaltın
```python
# config.py
TOP_N_FEATURES = 30  # 50'den azalt
```

### Çözüm 3: Daha Az Model
```python
# config.py
MODELS = {
    'xgboost': True,
    'lightgbm': False,  # Devre dışı
    'catboost': False,  # Devre dışı
}
```

---

## ❌ "Walk-Forward results çok kötü"

### Sebep:
Model regime change'lere adapt olamıyor.

### Çözüm 1: Window Size Azaltın
```python
# config.py
WALK_FORWARD_WINDOW = 5000  # 10000'den azaltın
WALK_FORWARD_STEP = 1000  # 2000'den azaltın
```

Daha sık retrain = daha iyi adaptation.

### Çözüm 2: Regime Features Ekleyin
Zaten var ama ADX, Hurst gibi features'lara extra ağırlık verin.

---

## ❌ "Freqtrade entegrasyon hatası"

### Çözüm:
```python
# freqtrade_strategy_example.py
model_dir = '/FULL/PATH/TO/ml_trading_advanced/models_advanced'

# PATH'e ekleyin:
import sys
sys.path.append('/FULL/PATH/TO/ml_trading_advanced')
```

**Test edin:**
```bash
freqtrade backtesting --strategy MLAdvancedStrategy --timeframe 5m
```

---

## 🔍 Genel Debugging Adımları

### 1. Test System
```bash
python test_system.py
```

Bu script:
- ✅ Config kontrolü
- ✅ Data yükleme
- ✅ Feature engineering test
- ✅ Labeling test
- ✅ Hataları gösterir

### 2. Check Data
```python
import pandas as pd

df = pd.read_csv('your_file.csv')
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"NaNs: {df.isna().sum().sum()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### 3. Küçük Sample Test
```python
# train_advanced.py'de:
df = df.tail(5000)  # Sadece son 5K bar

# config.py'de:
USE_OPTUNA = False
OPTUNA_TRIALS = 10
```

Hızlı test → Hatayı bul → Düzelt → Full training

---

## 💡 Performance İyileştirme

### Test F1 < 0.60 ise:

1. **Barrier'ları ayarlayın**
   ```python
   # Volatiliteyi hesaplayın:
   df['atr'] = df['high'] - df['low']
   avg_atr_pct = (df['atr'] / df['close']).mean() * 100
   print(f"Average ATR: {avg_atr_pct:.2f}%")

   # TP'yi ATR'nin 2-3 katı yapın:
   STATIC_TP_PCT = avg_atr_pct * 2.5
   ```

2. **Feature selection**
   ```bash
   # Training sonrası:
   cat models_advanced/feature_importance_initial.csv
   ```

   En önemli 20 feature'a bakın. Noise var mı?

3. **Threshold optimization**
   ```python
   import pandas as pd
   from sklearn.metrics import f1_score

   preds = pd.read_csv('models_advanced/walk_forward_predictions_long.csv')

   for t in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
       y_pred = (preds['y_pred_proba'] >= t).astype(int)
       f1 = f1_score(preds['y_true'], y_pred)
       print(f"Threshold {t:.2f}: F1 = {f1:.3f}")
   ```

---

## 📞 Hâlâ Çalışmıyor?

### Checklist:
- [ ] Data dosyası var mı? (`ls your_file.csv`)
- [ ] En az 1000 bar var mı? (`wc -l your_file.csv`)
- [ ] CSV formatı doğru mu? (timestamp, OHLCV)
- [ ] `test_system.py` başarılı mı?
- [ ] Dependencies kurulu mu? (`pip list | grep xgboost`)
- [ ] Python 3.8+ mı? (`python --version`)

### Debug Mode:
```python
# train_advanced.py başına ekleyin:
import warnings
warnings.filterwarnings('default')  # Tüm uyarıları göster

import traceback
import sys

try:
    # ... kod ...
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
```

---

## ✅ Başarı Kriterleri

Sistem çalışıyor demektir:
- ✅ `test_system.py` başarılı
- ✅ Training hatasız tamamlanıyor
- ✅ Test F1 > 0.60
- ✅ Train/Test farkı < %20
- ✅ Walk-forward ortalama F1 > 0.55
- ✅ Signals %5-15 arası

Bu değerlere ulaştıysanız → Paper trading!

---

**Başka sorun?** README.md'ye bakın veya issue açın.
