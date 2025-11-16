# Advanced ML Trading Strategy 🚀

**Profesyonel, institutional-grade machine learning trading sistemi**

Bu sistem, geleneksel ML trading yaklaşımlarının ötesine geçer ve aşağıdaki gelişmiş teknikleri kullanır:

## 🌟 Özellikler

### 1. **Gelişmiş Feature Engineering**
- **Zaman özellikleri**: Döngüsel encoding (sin/cos) ile
- **Fiyat aksiyonu**: Returns, momentum, ROC, price position
- **Hacim analizi**: OBV, Force Index, Money Flow, Volume profile
- **Volatility özellikleri**: ATR, Bollinger Bands, Parkinson volatility, GARCH-like clustering
- **Market rejimi**: ADX, trend strength, efficiency ratio, Hurst exponent
- **Mikroyapı**: Order flow imbalance, buy/sell pressure, illiquidity
- **Fraktal özellikleri**: Multi-timeframe fractal detection
- **İstatistiksel özellikler**: Skewness, kurtosis, rolling z-scores

**Toplam ~200+ özellik** otomatik olarak oluşturulur!

### 2. **Triple Barrier Method Labeling**
Marcos Lopez de Prado'nun "Advances in Financial Machine Learning" kitabından:

```
Entry
  |
  |----[Upper Barrier - Take Profit]
  |
  |----[Lower Barrier - Stop Loss]
  |
  └----[Vertical Barrier - Time Limit]
```

- ✅ Volatilite-bazlı dinamik bariyerler (ATR)
- ✅ Risk-reward dengeli labeling
- ✅ Lookahead bias önlenir
- ✅ Gerçekçi trading koşulları

### 3. **Hyperparameter Optimization**
- **Optuna** ile otomatik tuning
- Bayesian optimization (TPE sampler)
- Multi-objective optimization desteği
- 100+ trial ile en iyi parametreler

### 4. **Time Series Cross-Validation**
- Standart random split yerine **TimeSeriesSplit**
- GAP kullanarak data leakage önlenir
- 5-fold validation
- Out-of-sample gerçekçi performans

### 5. **Walk-Forward Validation**
- Gerçek trading koşullarını simüle eder
- Rolling window training
- Regime change'lere adaptasyon
- Production-ready performans metrikleri

### 6. **Class Imbalance Handling**
- SMOTE (opsiyonel - dikkatli kullanılmalı)
- Class weight balancing
- Probability calibration (isotonic/sigmoid)

### 7. **Ensemble Models**
- XGBoost
- LightGBM
- CatBoost
- Calibrated probabilities

---

## 📦 Kurulum

```bash
cd ml_trading_advanced
pip install -r requirements.txt
```

---

## ⚙️ Konfigürasyon

`config.py` dosyasını kendi verilerinize göre düzenleyin:

```python
# Data path
DATA_PATH = "/path/to/your/OHLCV_data.csv"

# Barrier configuration
VERTICAL_BARRIER_HOURS = 2  # Max holding period (bars)
USE_DYNAMIC_BARRIERS = True
VOLATILITY_LOOKBACK = 20

# Model configuration
USE_OPTUNA = True
OPTUNA_TRIALS = 100

# Features
TOP_N_FEATURES = 50
```

### Önemli Parametreler:

| Parametre | Açıklama | Önerilen |
|-----------|----------|----------|
| `VERTICAL_BARRIER_HOURS` | Maksimum holding period (bar sayısı) | 2-4 (5m için) |
| `USE_DYNAMIC_BARRIERS` | ATR-bazlı dinamik TP/SL | `True` |
| `VOLATILITY_LOOKBACK` | Volatilite hesabı için lookback | 14-20 |
| `MIN_RETURN_THRESHOLD` | Minimum karlılık eşiği (%) | 0.3-0.5 |
| `USE_OPTUNA` | Hyperparameter optimization | `True` |
| `TOP_N_FEATURES` | Feature selection | 40-60 |

---

## 🚀 Kullanım

### 1. Model Eğitimi

```bash
python train_advanced.py
```

Bu script:
1. ✅ Veriyi yükler
2. ✅ 200+ feature oluşturur
3. ✅ Triple Barrier Method ile label'lar
4. ✅ Feature selection yapar
5. ✅ Hyperparameter optimization ile model eğitir
6. ✅ Walk-forward validation yapar
7. ✅ Modelleri kaydeder

**Çıktılar** (`models_advanced/` klasörü):
- `model_long_xgboost.pkl` - En iyi LONG model
- `model_short_xgboost.pkl` - En iyi SHORT model
- `scaler_long_xgboost.pkl` - Feature scaler (long)
- `scaler_short_xgboost.pkl` - Feature scaler (short)
- `features.txt` - Kullanılan feature listesi
- `walk_forward_*.csv` - Validation sonuçları
- `feature_importance_*.csv` - Feature importance

### 2. Inference (Tahmin)

```python
from inference import TradingPredictor

# Predictor'ı yükle
predictor = TradingPredictor(model_dir='models_advanced')

# En son bar için tahmin
signal = predictor.predict_latest(
    df,
    side='both',
    long_threshold=0.6,  # Yüksek threshold = az ama kaliteli sinyaller
    short_threshold=0.6
)

print(signal)
# {
#   'signal': 1,  # 1=LONG, -1=SHORT, 0=NEUTRAL
#   'long_prob': 0.73,
#   'short_prob': 0.32,
#   'close': 42.15
# }
```

### 3. Freqtrade Entegrasyonu

```python
# strategies/MLAdvancedStrategy.py

from inference import TradingPredictor
import pandas as pd

class MLAdvancedStrategy(IStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.predictor = TradingPredictor(model_dir='models_advanced')

    def populate_indicators(self, dataframe, metadata):
        # Tahmin yap
        signal = self.predictor.predict_latest(
            dataframe,
            side='long',
            long_threshold=0.65
        )

        dataframe['ml_prob'] = signal['long_prob']
        dataframe['ml_signal'] = signal['signal']

        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe['ml_signal'] == 1),
            'enter_long'
        ] = 1

        return dataframe
```

---

## 📊 Performans Metrikleri

Model performansı şu metriklerle değerlendirilir:

- **Accuracy**: Genel doğruluk
- **Precision**: Sinyallerin ne kadarı doğru? (False positive oranı)
- **Recall**: Fırsatların ne kadarını yakalıyoruz?
- **F1 Score**: Precision ve Recall dengesi
- **ROC-AUC**: Probability quality
- **MCC**: Matthews Correlation Coefficient (balanced metric)

### Walk-Forward Validation Sonuçları

Training sonrası `models_advanced/walk_forward_*.csv` dosyalarını inceleyin:

```python
import pandas as pd

wf = pd.read_csv('models_advanced/walk_forward_long.csv')
print(wf.describe())

# Örnek çıktı:
#              f1    roc_auc  precision    recall
# mean      0.68       0.75       0.71      0.66
# std       0.08       0.06       0.09      0.10
```

---

## 🎯 En İyi Pratikler

### 1. **Data Quality**
- En az 1-2 yıllık veri kullanın
- Missing data'yı kontrol edin
- Volume = 0 olan barları temizleyin

### 2. **Labeling**
- Pair'inize göre barrier'ları ayarlayın
- Volatile asset → Geniş barrier
- Stable asset → Dar barrier
- Backtest yaparak optimal değerleri bulun

### 3. **Feature Selection**
- Çok fazla feature → overfitting
- TOP 40-60 feature optimal
- Feature importance'a bakın

### 4. **Threshold Optimization**
- Yüksek threshold (0.65-0.75) → Az ama kaliteli sinyal
- Düşük threshold (0.45-0.55) → Çok sinyal ama düşük kalite
- Walk-forward sonuçlarıyla optimize edin

### 5. **Model Retraining**
- Haftada 1-2 kez retrain
- Market rejimi değiştiğinde retrain
- Performance düşerse retrain

---

## 🔥 Gelişmiş Teknikler

### Meta-Labeling
"Should I take this signal?" sorusuna cevap:

```python
from labeling import MetaLabeler

meta_labeler = MetaLabeler(config)
df = meta_labeler.create_meta_labels(df, 'primary_signal', 'target_long')

# İki modelli sistem:
# Model 1: Direction (long/short)
# Model 2: Size/confidence (meta-model)
```

### Fractional Differentiation
Stationarity sağlarken memory koruma:

```python
from labeling import fractional_differentiation

df['price_frac_diff'] = fractional_differentiation(df['close'], d=0.5)
```

### Sample Weights
Label uniqueness'e göre weight:

```python
labeler = TripleBarrierLabeler(config)
df = labeler.add_sample_weights(df, 'target_long')

# Model training'de kullan:
model.fit(X, y, sample_weight=df['sample_weight'])
```

---

## 🐛 Troubleshooting

### Problem: "Too few positive samples"
**Çözüm**: Barrier'ları gevşetin veya MIN_RETURN_THRESHOLD'u düşürün

### Problem: "Overfitting (train >> test performance)"
**Çözüm**:
- Feature sayısını azaltın
- Regularization artırın
- Daha fazla data kullanın

### Problem: "Low recall"
**Çözüm**:
- Threshold'u düşürün
- Class weights kullanın
- SMOTE deneyin (dikkatli!)

### Problem: "Models are too slow"
**Çözüm**:
- OPTUNA_TRIALS azaltın
- Feature sayısını düşürün
- LightGBM kullanın (en hızlı)

---

## 📚 Referanslar

1. **Marcos Lopez de Prado** - "Advances in Financial Machine Learning"
2. **Stefan Jansen** - "Machine Learning for Algorithmic Trading"
3. **Optuna Documentation** - https://optuna.org
4. **XGBoost, LightGBM, CatBoost** papers

---

## ⚠️ Disclaimer

Bu sistem **eğitim amaçlıdır**. Gerçek para ile trade yapmadan önce:

1. ✅ Kapsamlı backtest
2. ✅ Paper trading (en az 1-2 ay)
3. ✅ Küçük pozisyonlarla başlayın
4. ✅ Risk yönetimi kullanın
5. ✅ Hiçbir zaman %100 kesin değildir

**Finansal tavsiye değildir. Kendi riskinizle kullanın.**

---

## 📧 Destek

Sorularınız için:
- Issues açın
- Dokumentasyonu okuyun
- Walk-forward sonuçlarını paylaşın

**Happy Trading! 📈🚀**
