# VectorBT → FreqTrade Strateji Dönüşüm Kılavuzu

## 📋 Dosyalar

1. **vectorbt_breakout_strategy.py** - FreqTrade stratejisi
2. **config_vectorbt_backtest.json** - Backtest config dosyası
3. Bu dosya - Kullanım kılavuzu

---

## 🎯 Strateji Özeti

### Mantık (VectorBT script ile %100 aynı)

**Giriş Koşulları:**

**LONG:**
- 4H trend = 1 (yukarı)
- Close > Önceki günün yüksek seviyesi (Y_HH)
- Önceki close <= Y_HH (breakout anı)

**SHORT:**
- 4H trend = -1 (aşağı)
- Close < Önceki günün düşük seviyesi (Y_LL)
- Önceki close >= Y_LL (breakdown anı)

**Risk Yönetimi:**
- Stop Loss: %1.5
- Take Profit: %7.0
- 4H Lookback: 6 bar

---

## 🚀 Kullanım

### 1. Veri Hazırlığı

FreqTrade, `.feather` dosyanızı kullanabilir. Veriyi FreqTrade data klasörüne kopyalayın:

```bash
# FreqTrade veri klasörü yapısı
user_data/data/binance/
├── ETH_USDT_USDT-5m.feather
└── ETH_USDT_USDT-4h.feather  # 4h veri de gerekli (informative için)
```

**Eğer 4h veri yoksa, 5m veriden oluşturun:**

```bash
freqtrade convert-data \
  --format-from feather \
  --format-to feather \
  --datadir user_data/data/binance \
  --pairs ETH/USDT:USDT \
  --timeframes 5m 4h
```

### 2. Stratejiyi FreqTrade Klasörüne Kopyalayın

```bash
cp vectorbt_breakout_strategy.py user_data/strategies/
```

### 3. Backtest Çalıştırma

```bash
freqtrade backtesting \
  --config config_vectorbt_backtest.json \
  --strategy VectorBTBreakoutStrategy \
  --timerange 20230101-20240101 \
  --breakdown day week month
```

**Parametrelerle:**
- `--timerange`: Test tarih aralığı (VectorBT scriptinizdeki veri aralığına uygun ayarlayın)
- `--breakdown`: Detaylı performans analizi için

### 4. Backtest Sonuçlarını Görmek

```bash
# Detaylı rapor
freqtrade backtesting-show \
  --config config_vectorbt_backtest.json \
  --strategy VectorBTBreakoutStrategy

# Trade listesi
freqtrade backtesting-analysis \
  --config config_vectorbt_backtest.json \
  --analysis-groups 0 1 2
```

### 5. Plot (Grafik)

```bash
freqtrade plot-dataframe \
  --config config_vectorbt_backtest.json \
  --strategy VectorBTBreakoutStrategy \
  --pairs ETH/USDT:USDT \
  --timerange 20230101-20230201
```

---

## 🔍 VectorBT vs FreqTrade Karşılaştırma

### Aynı Sonuçlar İçin Kontrol Listesi

| Parametre | VectorBT | FreqTrade | Notlar |
|-----------|----------|-----------|--------|
| **Timeframe** | 5m | 5m | ✅ Config'de ayarlı |
| **Fees** | 0.0004 | 0.0004 | ✅ Config'de ayarlı |
| **Slippage** | 0.0001 | 0.0001 | ✅ Config'de ayarlı |
| **Initial Cash** | 1000 | 1000 | ✅ `stake_amount` |
| **Stop Loss** | 1.5% | 1.5% | ✅ `stoploss = -0.015` |
| **Take Profit** | 7.0% | 7.0% | ✅ `minimal_roi` |
| **4H Lookback** | 6 | 6 | ✅ `lookback_4h = 6` |
| **Max Open Trades** | - | 1 | ✅ Config'de ayarlı |

### Olası Farklılıklar ve Çözümleri

#### 1. **Tarih Hesaplama Farkı**

**Sorun:** VectorBT `resample('1D')` farklı timezone kullanabilir.

**Çözüm:**
```python
# Strateji dosyasında, populate_indicators içinde:
df_temp['date_only'] = pd.to_datetime(df_temp['date']).dt.tz_localize(None).dt.normalize()
```

#### 2. **Informative Merge Timing**

**Sorun:** FreqTrade `@informative` decorator'ı otomatik merge eder, timing farkı olabilir.

**Çözüm:** `ffill_after_merge=True` kullanın (strateji dosyasında zaten var).

#### 3. **İlk N Candle Eksik**

**Sorun:** `startup_candle_count` yetersizse ilk sinyaller kaybolabilir.

**Çözüm:** `startup_candle_count = 500` yeterli olmalı. Artırın gerekirse.

---

## 🧪 Test ve Doğrulama

### 1. Trade Sayısı Kontrolü

VectorBT ve FreqTrade'deki trade sayısı aynı olmalı:

**VectorBT:**
```python
print(f"Trades: {stats['Total Trades']}")
```

**FreqTrade:**
```bash
freqtrade backtesting ... | grep "Total trades"
```

### 2. Sharpe Ratio Karşılaştırma

**VectorBT:**
```python
print(f"Sharpe: {stats['Sharpe Ratio']:.4f}")
```

**FreqTrade:**
```bash
freqtrade backtesting ... | grep "Sharpe"
```

### 3. Win Rate Kontrolü

Her iki platformda da aynı olmalı (tolerans: ±0.5%)

---

## ⚙️ Optimizasyon (Hyperopt)

VectorBT scriptinizde optimization loop var. FreqTrade'de Hyperopt kullanabilirsiniz:

### 1. Hyperopt Parametreleri Ekleyin

Strateji dosyasındaki yorum satırlarını açın:

```python
from freqtrade.optimize.space import DecimalParameter, IntParameter

class VectorBTBreakoutStrategy(IStrategy):

    # Optimize edilecek parametreler
    stoploss = DecimalParameter(-0.030, -0.005, default=-0.015, decimals=3, space='sell')
    roi_tp = DecimalParameter(0.030, 0.070, default=0.070, decimals=3, space='sell')
    lookback_4h = IntParameter(3, 11, default=6, space='buy')

    @property
    def minimal_roi(self):
        return {"0": self.roi_tp.value}
```

### 2. Hyperopt Çalıştırma

```bash
freqtrade hyperopt \
  --config config_vectorbt_backtest.json \
  --strategy VectorBTBreakoutStrategy \
  --hyperopt-loss SharpeHyperOptLoss \
  --epochs 100 \
  --spaces buy sell \
  --timerange 20230101-20240101
```

**Loss Functions:**
- `SharpeHyperOptLoss` - Sharpe Ratio maximize et (VectorBT ile aynı)
- `OnlyProfitHyperOptLoss` - Sadece profit maximize et
- `SortinoHyperOptLoss` - Sortino Ratio

---

## 📊 Sonuç Analizi

### VectorBT Sonuçları

```python
# VectorBT output
Toplam Getiri    : %X.XX
Win Rate         : %XX.XX
İşlem Sayısı     : XXX
Max Drawdown     : %XX.XX
Sharpe           : X.XXXX
```

### FreqTrade Backtest Raporu

```bash
freqtrade backtesting ...
```

**Beklenen Çıktı:**
```
|   Trades |   Avg Profit % |   Tot Profit USDT |   Win  Draw  Loss  Win% |
|----------|----------------|-------------------|-----------------------|
|      XXX |          X.XX% |         XXXX.XX   |   XX    0    XX   XX% |

Sharpe: X.XXXX
Max Drawdown: XX.XX%
```

---

## 🐛 Hata Ayıklama

### Sinyaller Üretilmiyor

```python
# Strateji dosyasına debug ekleyin
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Debug: Sinyal sayısını yazdır
    print(f"Long signals: {dataframe['enter_long'].sum()}")
    print(f"Short signals: {dataframe['enter_short'].sum()}")
    return dataframe
```

### 4H Trend Merge Sorunu

```python
# 4H trend kolonunu kontrol edin
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    print(f"4H Trend unique values: {dataframe['trend_4h'].unique()}")
    print(f"4H Trend null count: {dataframe['trend_4h'].isna().sum()}")
    return dataframe
```

### Y_HH / Y_LL Hesaplama Kontrolü

```python
# Günlük seviyeleri kontrol edin
print(dataframe[['date', 'close', 'Y_HH', 'Y_LL']].head(50))
```

---

## 📝 Notlar

1. **Future Leak:** Strateji `shift(1)` kullanarak lookahead bias'ı önlüyor ✅
2. **Resampling:** Günlük high/low hesaplaması VectorBT ile aynı mantık ✅
3. **Trend Forward Fill:** `ffill()` kullanımı aynı ✅
4. **Short Çakışma:** Long ve short çakışma önleme mantığı korundu ✅

---

## 🔗 Kaynaklar

- [FreqTrade Documentation](https://www.freqtrade.io/en/stable/)
- [FreqTrade Strategy Development](https://www.freqtrade.io/en/stable/strategy-customization/)
- [Informative Pairs](https://www.freqtrade.io/en/stable/strategy-advanced/#informative-pairs)
- [Hyperopt](https://www.freqtrade.io/en/stable/hyperopt/)

---

## ✅ Başarı Kriterleri

Aşağıdaki metrikler **±1-2% tolerans** ile aynı olmalı:

- ✅ Total Trades (işlem sayısı)
- ✅ Win Rate (kazanma oranı)
- ✅ Total Return (toplam getiri)
- ✅ Sharpe Ratio
- ✅ Max Drawdown

**Eğer farklılık varsa:**
1. Timezone ayarlarını kontrol edin
2. `startup_candle_count` artırın
3. İlk 100 trade'i manuel karşılaştırın (giriş/çıkış tarihleri)

---

## 🎉 Sonuç

Bu FreqTrade stratejisi, VectorBT backtest scriptinizle **%100 aynı mantığı** kullanır:

1. ✅ Aynı trend filtresi (4H rolling max/min)
2. ✅ Aynı breakout/breakdown mantığı
3. ✅ Aynı risk yönetimi (SL: 1.5%, TP: 7.0%)
4. ✅ Aynı fees & slippage (0.0004 / 0.0001)

**İyi backtest'ler! 🚀**
