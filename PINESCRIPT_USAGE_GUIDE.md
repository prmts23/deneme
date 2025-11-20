# VectorBT → PineScript Strateji Dönüşüm Kılavuzu

## 📋 Dosya

**vectorbt_breakout_strategy.pine** - TradingView PineScript v5 stratejisi

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

## 🚀 TradingView'da Kullanım

### 1. Stratejiyi Yüklemek

1. **TradingView'ı Açın:** https://www.tradingview.com/
2. **Chart Seçin:** ETH/USDT, 5 dakika timeframe
3. **Pine Editor'ü Açın:** Alt menüden "Pine Editor" sekmesi
4. **Kodu Yapıştırın:** `vectorbt_breakout_strategy.pine` dosyasının içeriğini kopyalayıp yapıştırın
5. **Kaydedin ve Ekleyin:** "Save" → "Add to Chart" butonuna tıklayın

### 2. Backtest Sonuçlarını Görüntüleme

**Strategy Tester** sekmesini açın (chart altında):

```
📊 Overview Tab:
- Net Profit
- Total Closed Trades
- Percent Profitable (Win Rate)
- Profit Factor
- Max Drawdown

📈 Performance Summary:
- Total Return %
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

📋 List of Trades:
- Her trade'in detayları (entry/exit, profit, %)
```

### 3. Parametreleri Değiştirme

**Settings (⚙️) > Inputs:**

| Parametre | Default | Açıklama |
|-----------|---------|----------|
| **4H Trend Lookback** | 6 | 4 saatlik rolling window bar sayısı |
| **Stop Loss %** | 1.5 | Stop loss yüzdesi |
| **Take Profit %** | 7.0 | Take profit yüzdesi |
| **Use Date Filter** | false | Belirli tarih aralığı için backtest |

**Settings (⚙️) > Properties:**

| Parametre | Default | Açıklama |
|-----------|---------|----------|
| **Initial Capital** | 1000 | Başlangıç sermayesi (USDT) |
| **Order Size** | 100% equity | Her işlemde kullanılan sermaye |
| **Commission** | 0.04% | İşlem ücreti (Binance Futures) |
| **Slippage** | 1 tick | Slippage (kayma) |

---

## 🔍 VectorBT vs PineScript Karşılaştırma

### Aynı Sonuçlar İçin Kontrol Listesi

| Parametre | VectorBT | PineScript | Notlar |
|-----------|----------|-----------|--------|
| **Timeframe** | 5m | 5m | ✅ Chart'ı 5m'ye ayarlayın |
| **Fees** | 0.0004 (0.04%) | 0.04% | ✅ Settings > Properties'de |
| **Slippage** | 0.0001 | 1 tick | ✅ Settings > Properties'de |
| **Initial Cash** | 1000 | 1000 | ✅ Settings > Properties'de |
| **Stop Loss** | 1.5% | 1.5% | ✅ Inputs'ta ayarlı |
| **Take Profit** | 7.0% | 7.0% | ✅ Inputs'ta ayarlı |
| **4H Lookback** | 6 | 6 | ✅ Inputs'ta ayarlı |

### PineScript'e Özgü Implementasyon Detayları

#### 1. **Önceki Günün High/Low (Y_HH, Y_LL)**

**VectorBT:**
```python
df_daily = df_1h.resample('1D').agg({'high': 'max', 'low': 'min'})
prev_day = df_daily.shift(1)
```

**PineScript:**
```pinescript
prev_day_high = request.security(syminfo.tickerid, "D", high[1])
prev_day_low = request.security(syminfo.tickerid, "D", low[1])
```

✅ `lookahead=barmerge.lookahead_off` ile future leak önleme

#### 2. **4H Trend Filtresi**

**VectorBT:**
```python
df_4h['roll_max'] = df_4h['high'].rolling(lb).max().shift(1)
df_4h['roll_min'] = df_4h['low'].rolling(lb).min().shift(1)
```

**PineScript:**
```pinescript
roll_max_4h = ta.highest(high_4h[1], lookback_4h)
roll_min_4h = ta.lowest(low_4h[1], lookback_4h)
```

✅ `[1]` shift ile aynı mantık, `request.security()` ile 4H data

#### 3. **Trend Forward Fill**

**VectorBT:**
```python
df_4h['trend'] = df_4h['trend'].replace(0, np.nan).ffill().fillna(0)
```

**PineScript:**
```pinescript
var float trend_4h = 0.0
if close_4h > roll_max_4h
    trend_4h := 1.0
else if close_4h < roll_min_4h
    trend_4h := -1.0
// else: trend değişmez (var kullanımı forward fill sağlar)
```

✅ `var` keyword ile değer korunur (forward fill etkisi)

---

## 📊 Görselleştirme

### Chart Üzerindeki Göstergeler

1. **Kırmızı Çizgi:** Önceki günün high seviyesi (Y_HH)
2. **Yeşil Çizgi:** Önceki günün low seviyesi (Y_LL)
3. **Yeşil Background:** 4H trend yukarı (long bias)
4. **Kırmızı Background:** 4H trend aşağı (short bias)
5. **Yeşil Üçgen ▲:** Long entry sinyali
6. **Kırmızı Üçgen ▼:** Short entry sinyali
7. **Performans Tablosu:** Sağ üst köşede canlı metrikler

### Performans Tablosu (Sağ Üst)

```
┌─────────────────────────────────┐
│ Metric         │ Value          │
├─────────────────────────────────┤
│ Net Profit     │ XXX.XX USDT    │
│ Total Return % │ XX.XX%         │
│ Win Rate       │ XX.XX%         │
│ Total Trades   │ XXX            │
│ Max Drawdown   │ XX.XX USDT     │
│ SL / TP        │ 1.5% / 7.0%    │
│ 4H Lookback    │ 6 bars         │
└─────────────────────────────────┘
```

---

## 🧪 Test ve Doğrulama

### 1. VectorBT ile Karşılaştırma

**VectorBT Sonuçları:**
```python
print(f"Total Trades: {stats['Total Trades']}")
print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
print(f"Total Return: {pf.total_return() * 100:.2f}%")
print(f"Sharpe: {stats['Sharpe Ratio']:.4f}")
```

**PineScript Sonuçları:**
- Strategy Tester > Overview sekmesinden bakın
- Total Closed Trades ≈ VectorBT Total Trades
- Percent Profitable ≈ VectorBT Win Rate
- Net Profit % ≈ VectorBT Total Return

### 2. Beklenen Farklılıklar

#### a) **Veri Farklılıkları**

**Sorun:** TradingView ve Binance veri kaynakları farklı olabilir.

**Çözüm:**
- Binance veri kaynağını kullanın (chart'ta sağ üst)
- Aynı tarih aralığını test edin
- `Date Filter` kullanarak VectorBT tarih aralığını eşleyin

#### b) **Timezone Farkları**

**Sorun:** Günlük high/low hesaplaması timezone'a bağlı.

**Çözüm:**
- TradingView Settings > Chart > Timezone: UTC+0
- VectorBT scriptinde de UTC kullanın

#### c) **Order Execution Model**

**Sorun:** PineScript bar close'da işlem yapar, VectorBT farklı olabilir.

**Çözüm:**
- `calc_on_every_tick=false` (default) → bar close'da işlem
- `process_orders_on_close=false` → VectorBT ile aynı

### 3. Trade Listesi Kontrolü

**İlk 10 trade'i karşılaştırın:**

1. Strategy Tester > List of Trades
2. Entry tarihi, exit tarihi, profit % kontrol edin
3. VectorBT ile eşleşmeli (tolerans: ±1 bar)

---

## 🎨 Özelleştirme

### 1. Görsel Ayarlar

**Settings (⚙️) > Style:**

- Entry/Exit markerları değiştir
- Y_HH/Y_LL çizgilerinin rengini ayarla
- Background transparency değiştir
- Performans tablosunu gizle/göster

### 2. Alert Kurulumu

**TradingView Alert Oluşturma:**

1. **Alert butonu** (⏰) tıklayın
2. **Condition:** VectorBT Breakout Strategy
3. **Alert name:** Long Entry Signal / Short Entry Signal
4. **Message:** Webhook için JSON format:
   ```json
   {
     "symbol": "{{ticker}}",
     "side": "{{strategy.order.action}}",
     "price": "{{close}}",
     "time": "{{timenow}}"
   }
   ```
5. **Webhook URL:** (Binance API / 3Commas / vs.)

### 3. Optimizasyon (TradingView Premium)

**Deep Backtesting** özelliği ile:

1. Strategy Tester > ⚙️ (Settings)
2. **Deep Backtesting** checkbox'ı aktif edin
3. Daha fazla geçmiş veri ile test edin (1-2 yıl)

**Strategy Optimization:**

1. Settings > Inputs > ⚙️ (Optimize)
2. Stop Loss: 0.5% - 3.0% (adım: 0.5%)
3. Take Profit: 3.0% - 10.0% (adım: 0.5%)
4. 4H Lookback: 3 - 11 (adım: 2)
5. **Run** → En iyi kombinasyonu bulur

---

## 🐛 Hata Ayıklama

### Sinyaller Üretilmiyor

**Kontrol Edin:**
```pinescript
// Debug plot ekleyin
plot(trend_4h, "4H Trend", color=color.blue)
plot(Y_HH, "Y_HH", color=color.red)
plot(Y_LL, "Y_LL", color=color.green)
```

**Olası Nedenler:**
- Chart timeframe 5m değil
- Yeterli geçmiş veri yok (minimum 1-2 gün)
- 4H trend hiç değişmemiş (sideways market)

### Trade Sayısı Çok Az

**Sebep:** Çok sıkı filtreler (trend + breakout birlikte nadir)

**Çözüm:**
- Farklı market koşullarını test edin (trending vs. ranging)
- Lookback parametresini azaltın (3-4 bar)

### SL/TP Çalışmıyor

**Kontrol:**
```pinescript
// Debug: Pozisyon açık mı?
bgcolor(strategy.position_size > 0 ? color.new(color.green, 80) : na)
bgcolor(strategy.position_size < 0 ? color.new(color.red, 80) : na)
```

**Çözüm:**
- `strategy.exit()` her bar'da çağrılmalı (if bloğu içinde)
- Stop/limit fiyatları doğru hesaplanmalı

---

## 📱 Mobil Kullanım

TradingView mobil uygulamasında:

1. **Chart'ı Açın:** ETH/USDT 5m
2. **Indicators:** Sağ üst menü > Indicators
3. **Favorites:** Masaüstünde eklediğiniz strateji favorites'te görünür
4. **Strategy Tester:** Mobil'de kısıtlı (detaylı analiz masaüstünde)

---

## 🔗 Kaynaklar

### TradingView Dokümantasyonu

- [Pine Script v5 User Manual](https://www.tradingview.com/pine-script-docs/en/v5/Introduction.html)
- [Strategy() Function](https://www.tradingview.com/pine-script-reference/v5/#fun_strategy)
- [request.security()](https://www.tradingview.com/pine-script-reference/v5/#fun_request{dot}security)
- [strategy.entry()](https://www.tradingview.com/pine-script-reference/v5/#fun_strategy{dot}entry)
- [strategy.exit()](https://www.tradingview.com/pine-script-reference/v5/#fun_strategy{dot}exit)

### TradingView Topluluk

- [Pine Script Forum](https://www.tradingview.com/scripts/)
- [Pine Coders](https://www.tradingview.com/u/PineCoders/)

---

## ✅ VectorBT Karşılaştırma Checklist

Aşağıdaki metrikler **±2-5% tolerans** ile aynı olmalı:

- ✅ **Total Trades** (işlem sayısı)
- ✅ **Win Rate** (kazanma oranı)
- ⚠️ **Total Return** (veri farkından dolayı değişebilir)
- ⚠️ **Sharpe Ratio** (hesaplama yöntemi farklı olabilir)
- ✅ **Max Drawdown** (yaklaşık aynı olmalı)

### Farklılık Varsa:

1. **Veri Kaynağı:** Binance veri kaynağı seçili mi?
2. **Timezone:** UTC+0 mı?
3. **Tarih Aralığı:** VectorBT ile aynı mı?
4. **Fees/Slippage:** Settings'de doğru ayarlı mı?
5. **İlk 10 Trade:** Karşılaştırın, hangi trade farklı?

---

## 🎯 Ek Özellikler (PineScript'e Özgü)

### 1. **Multi-Timeframe Dashboard**

```pinescript
// 1H, 4H, 1D trend'lerini aynı anda göster
trend_1h = request.security(syminfo.tickerid, "60", trend_4h)
trend_1d = request.security(syminfo.tickerid, "D", trend_4h)

// Tablo oluştur
var table mtf_table = table.new(position.top_left, 3, 2)
table.cell(mtf_table, 0, 0, "1H", bgcolor=trend_1h == 1 ? color.green : color.red)
table.cell(mtf_table, 1, 0, "4H", bgcolor=trend_4h == 1 ? color.green : color.red)
table.cell(mtf_table, 2, 0, "1D", bgcolor=trend_1d == 1 ? color.green : color.red)
```

### 2. **Dinamik Position Sizing**

```pinescript
// Risk bazlı position size
risk_per_trade = 0.01  // %1 risk
stop_distance = close * stop_loss_pct
position_size = (strategy.equity * risk_per_trade) / stop_distance

strategy.entry("Long", strategy.long, qty=position_size)
```

### 3. **Trailing Stop**

```pinescript
// Trailing stop ekle
trailing_pct = input.float(2.0, "Trailing Stop %") / 100

if strategy.position_size > 0
    trail_price = close * (1 - trailing_pct)
    strategy.exit("Long Exit", "Long", trail_price=trail_price, trail_offset=trailing_pct)
```

---

## 🎉 Sonuç

Bu PineScript stratejisi, VectorBT backtest scriptinizle **%100 aynı mantığı** kullanır:

1. ✅ Aynı trend filtresi (4H rolling max/min)
2. ✅ Aynı breakout/breakdown mantığı
3. ✅ Aynı risk yönetimi (SL: 1.5%, TP: 7.0%)
4. ✅ Future leak önleme (`lookahead_off`, `[1]` shift)
5. ✅ Çakışma önleme (long/short conflict)

**TradingView'da canlı test'e hazır! 📈**

---

## 💡 Pro Tips

1. **Paper Trading:** TradingView Paper Trading ile canlı piyasada risk almadan test edin
2. **Alert Webhook:** Otomatik trade için 3Commas/Binance webhook'ları kurun
3. **Multi-Pair:** Aynı stratejiyi farklı coin'lerde test edin (BTC, SOL, vs.)
4. **Market Condition Filter:** Volatilite filtresi ekleyin (ATR bazlı)
5. **News Filter:** Önemli haber saatlerinde trade yapmayın

**İyi trade'ler! 🚀**
