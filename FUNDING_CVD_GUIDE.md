# Funding Rate + CVD Analysis System - Kullanım Kılavuzu

## 📋 İçindekiler
1. [Sistem Özeti](#sistem-özeti)
2. [Kurulum](#kurulum)
3. [Konfigürasyon](#konfigürasyon)
4. [Sinyal Teorisi](#sinyal-teorisi)
5. [Risk Yönetimi](#risk-yönetimi)
6. [Gerçek Dünya Kullanımı](#gerçek-dünya-kullanımı)

---

## 🎯 Sistem Özeti

Bu sistem **3 ana veri kaynağı** kullanarak delta neutral arbitrage fırsatları yakalıyor:

### 1. Funding Rate (Ana Gelir Kaynağı)
```
Perpetual futures her 8 saatte funding ödemesi yapar:
- Positive funding (+) → Longs pay shorts → SEN SHORT AC (para kazan)
- Negative funding (-) → Shorts pay longs → SEN LONG AC (para kazan)
```

### 2. CVD (Cumulative Volume Delta) - Order Flow
```
CVD = Aggressive buyer volume - Aggressive seller volume

Yükselen CVD → Institutional buying (bullish)
Düşen CVD → Institutional selling (bearish)
```

### 3. Basis (Perp vs Spot Premium)
```
Basis = (Perp Price / Spot Price) - 1

Positive basis → Perp expensive (funding likely positive)
Negative basis → Perp cheap (funding likely negative)
```

---

## 🛠️ Kurulum

### 1. Gerekli Kütüphaneler:
```bash
pip install websockets pandas numpy binance-connector python-telegram-bot requests
```

### 2. Telegram Bot Oluştur:
1. Telegram'da @BotFather'a git
2. `/newbot` komutunu çalıştır
3. Bot token'ı al
4. Bot'a mesaj at ve chat ID'ni öğren:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```

### 3. Konfigürasyon Dosyasını Düzenle:
```python
# funding_cvd_system.py içinde:

TELEGRAM_BOT_TOKEN = "123456:ABC-DEF..."  # Bot token
TELEGRAM_CHAT_ID = "123456789"            # Chat ID

SYMBOLS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]  # İzlenecek coinler
POSITION_SIZE_USD = 1000  # Pozisyon büyüklüğü
```

### 4. Çalıştır:
```bash
python funding_cvd_system.py
```

---

## ⚙️ Konfigürasyon

### Temel Parametreler:

```python
# Trading Parameters
POSITION_SIZE_USD = 1000  # Her coin için pozisyon büyüklüğü

# Fees (Binance VIP0)
SPOT_MAKER_FEE = 0.001    # 0.10%
FUTURES_TAKER_FEE = 0.0004  # 0.04%
SLIPPAGE = 0.0005          # 0.05%

# Signal Thresholds
FUNDING_Z_THRESHOLD = 2.0      # Funding spike z-score
CVD_CHANGE_THRESHOLD = 10000   # Significant CVD change
```

### Threshold'ları Ayarlama:

**Conservative (Düşük risk, az sinyal):**
```python
FUNDING_Z_THRESHOLD = 2.5
CVD_CHANGE_THRESHOLD = 15000
```

**Moderate (Dengeli):**
```python
FUNDING_Z_THRESHOLD = 2.0  # Default
CVD_CHANGE_THRESHOLD = 10000  # Default
```

**Aggressive (Yüksek risk, çok sinyal):**
```python
FUNDING_Z_THRESHOLD = 1.5
CVD_CHANGE_THRESHOLD = 5000
```

---

## 🧠 Sinyal Teorisi

### Signal Type 1: HIGH FUNDING + POSITIVE CVD → SHORT

**Ne Zaman Oluşur:**
```
✅ Funding Rate > +0.05% per 8h
✅ Funding Z-score > +2.0 (spike)
✅ CVD Change (15m) > +10,000 (aggressive longs)
✅ Basis > 0 (perp premium)
```

**Ne Anlama Gelir:**
- Çok fazla long pozisyon açılmış (retail FOMO)
- Funding rate aşırı yüksek
- Perp spot'tan pahalı
- Institutional para institutional buyers aggressive

**Strateji:**
```
1. SPOT: BTC al ($1,000)
2. FUTURES: BTC short ($1,000)
→ Delta neutral (fiyat riski yok)

Kazanç Kaynakları:
- Funding rate: Long'lar sana ödüyor (her 8 saatte)
- Mean reversion: Funding normalleşince kapat
- Basis compression: Perp-spot gap kapanır
```

**Beklenen Kazanç:**
```
Funding: +0.10% per 8h
Günlük: 0.10% × 3 = 0.30%
Aylık: 0.30% × 30 = 9%
$1,000 pozisyon = $90/month

Masraf: ~$1.80 (round-trip)
Break-even: 0.6 gün
```

---

### Signal Type 2: LOW FUNDING + NEGATIVE CVD → LONG

**Ne Zaman Oluşur:**
```
✅ Funding Rate < -0.05% per 8h
✅ Funding Z-score < -2.0
✅ CVD Change (15m) < -10,000 (aggressive shorts)
✅ Basis < 0 (perp discount)
```

**Ne Anlama Gelir:**
- Çok fazla short pozisyon (fear/panic)
- Funding rate negatif (shorts ödüyor)
- Perp spot'tan ucuz
- Institutional selling pressure

**Strateji:**
```
1. SPOT: BTC al ($1,000)
2. FUTURES: BTC short ($1,000)
→ Ama bu sefer shorts SANA ödüyor!

Kazanç:
- Negatif funding → Sen para alıyorsun
- Short squeeze potential
```

---

### Signal Type 3: EXTREME FUNDING SPIKE → Mean Reversion

**Ne Zaman Oluşur:**
```
✅ |Funding Z-score| > 3.0 (extreme spike)
✅ Funding rate > ±0.15% per 8h
```

**Ne Anlama Gelir:**
- Funding aşırı yüksek/düşük (unsustainable)
- Mean reversion olasılığı yüksek
- Kısa vadeli arbitrage fırsatı

**Strateji:**
```
Extreme positive funding:
→ SHORT (funding normalleşir, sen kazanırsın)

Extreme negative funding:
→ LONG (funding normalize olur)

Hold period: 1-3 gün (funding normalleşene kadar)
```

---

## 💰 Kar/Zarar Hesaplaması

### Örnek Senaryo:

**Setup:**
- Position: $1,000 BTC delta neutral
- Funding: +0.08% per 8h
- Duration: 7 gün

**Gelirler:**
```
Funding income:
- Per 8h: $1,000 × 0.0008 = $0.80
- Per day: $0.80 × 3 = $2.40
- 7 days: $2.40 × 7 = $16.80
```

**Masraflar:**
```
Entry fees:
- Spot buy: $1,000 × 0.001 = $1.00
- Futures short: $1,000 × 0.0004 = $0.40
- Slippage: $1,000 × 0.001 = $1.00
- Total entry: $2.40

Exit fees: $2.40 (same as entry)

Total fees: $4.80
```

**Net Kar:**
```
Gross profit: $16.80
Fees: -$4.80
Net profit: $12.00

ROI: 1.2% in 7 days (63% annualized)
```

---

## ⚠️ Risk Yönetimi

### 1. Funding Rate Riski (EN BÜYÜK!)

**Problem:**
Funding rate çabuk değişir. Pozitif iken negatife dönebilir!

**Örnek:**
```
Day 1: +0.10% (sen kazanıyorsun)
Day 2: +0.05% (hala kazanç ama düştü)
Day 3: -0.02% (NEGATİF! sen ödüyorsun!)
```

**Çözüm:**
- Funding'i her 4 saatte kontrol et
- Negatife dönerse HEMEN KAPAT
- Auto-exit threshold koy: funding < 0
- Sistemde otomatik var, Telegram'dan bildirim gelir

---

### 2. Execution Risk (Slippage)

**Problem:**
Market order kullanırsan slippage olur.

**Çözüm:**
- Limit order kullan (sabırlı ol)
- Spread'i kontrol et (tight spread gerekli)
- Düşük likidite saatlerinden kaçın (gece 2-5)

---

### 3. Liquidation Risk (Leverage varsa)

**Problem:**
Futures'ta leverage kullanırsan liquidation riski var.

**Çözüm:**
- 1x leverage kullan (veya hiç kullanma)
- Isolated margin modu (cross margin kullanma)
- Margin'i sürekli izle

---

### 4. Capital Allocation

**Problem:**
Tüm sermayeni bir coin'e yatırırsan diversify edemezsin.

**Çözüm:**
```
$1,000 sermaye için:
- BTC: $400 (stable funding)
- ETH: $300 (moderate funding)
- ALT: $200 (high funding but volatile)
- Reserve: $100 (emergencies)
```

---

## 🎯 Gerçek Dünya Kullanımı

### Senaryo 1: $500 Sermaye (Başlangıç)

**Strategi:**
```python
POSITION_SIZE_USD = 250  # Per coin
SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # Sadece stable coinler
```

**Beklenti:**
- Günlük: $0.50-1.00
- Aylık: $15-30 (3-6%)
- Risk: Çok düşük

**Öğrenme Süreci:**
- 1 ay boyunca small position
- Funding nasıl değişiyor gözle
- CVD pattern'leri öğren
- Execution practice yap

---

### Senaryo 2: $2,000 Sermaye (Intermediate)

**Strateji:**
```python
POSITION_SIZE_USD = 500  # Per coin
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
```

**Beklenti:**
- Günlük: $4-8
- Aylık: $120-240 (6-12%)
- Risk: Düşük-Orta

**Taktik:**
- 4 coin diversify
- High funding'de aggressive ol
- Low funding'de wait & watch

---

### Senaryo 3: $10,000 Sermaye (Advanced)

**Strateji:**
```python
POSITION_SIZE_USD = 1000  # Per coin
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT"]
```

**Beklenti:**
- Günlük: $20-40
- Aylık: $600-1,200 (6-12%)
- Risk: Orta

**Advanced Taktikler:**
- Dynamic position sizing (high funding = larger size)
- Multi-timeframe analysis
- Correlation hedging

---

## 📊 Dashboard Okuma

### Konsol Çıktısı:

```
================================================================================
📊 DASHBOARD: SOLUSDT | 2025-01-13 15:30:00
================================================================================

LAST 5 BARS (1-minute):
                    funding_rate  funding_z      cvd  cvd_change_15m  mark_price
2025-01-13 15:26:00      0.000850       2.34   45230            8950      123.45
2025-01-13 15:27:00      0.000870       2.45   46180            9120      123.48
2025-01-13 15:28:00      0.000890       2.56   47350           10240      123.52
2025-01-13 15:29:00      0.000910       2.67   48920           11450      123.56
2025-01-13 15:30:00      0.000930       2.78   50450           12680      123.60

📈 CURRENT VALUES:
  Funding Rate: 0.0930% per 8h
  Funding Z-Score: 2.78
  CVD: 50,450
  CVD Change (15m): 12,680
  Mark Price: $123.60

💰 PROFIT ANALYSIS ($1,000 position):
  Daily Funding Profit: $2.79
  Round-trip Fees: $1.80
  Payback Period: 0.6 days
  ✅ PROFITABLE - Break even in 0.6 days
================================================================================
```

**Ne Anlama Gelir:**

1. **Funding Z-Score: 2.78**
   - Threshold 2.0'ın üstünde → SPIKE!
   - Signal condition met

2. **CVD Change (15m): 12,680**
   - Threshold 10,000'in üstünde → AGGRESSIVE LONGS!
   - İnstitutional buying pressure

3. **Payback: 0.6 days**
   - Masrafları <1 günde geri alırsın
   - Çok karlı!

**Sonuç:** 🟢 SHORT SİNYALİ (Telegram'a gider)

---

## 🤖 Telegram Bildirimleri

### 1. Sistem Başlangıç:
```
🟢 Funding CVD System Started

Monitoring funding rates and order flow...
```

### 2. Signal Notification:
```
🔴 SHORT SIGNAL: SOLUSDT

📊 Signal Confidence: 85.6%

💰 Funding Rate: 0.0930%
📈 CVD Change (15m): 12,680
📉 Basis: 0.085%

💵 Expected Daily Profit: $2.79
💸 Total Fees: $1.80
⏱ Payback Period: 0.6 days

⏰ Time: 2025-01-13 15:30:00
```

### 3. Hourly Summary:
```
📊 Funding Rate Summary

Top Opportunities:
• SOLUSDT: 0.0930%
• AVAXUSDT: 0.0750%
• MATICUSDT: 0.0620%

Market Stats:
• Average Funding: 0.0450%
• Max Funding: 0.0930%
• Active Signals: 2

⏰ 2025-01-13 16:00:00
```

---

## 🔍 Troubleshooting

### Problem 1: WebSocket Disconnects

**Çözüm:**
- Sistem otomatik reconnect yapar
- 5 saniye bekler ve yeniden bağlanır
- Log'ları kontrol et

### Problem 2: Telegram Mesaj Gitmiyor

**Kontrol Et:**
```python
# Test message gönder:
from funding_cvd_system import TelegramNotifier

telegram = TelegramNotifier("YOUR_TOKEN", "YOUR_CHAT_ID")
telegram.send_message("Test mesajı")
```

### Problem 3: CVD Hesaplaması Yanlış

**Kontrol Et:**
- `is_buyer_maker` logic'i doğru mu?
- `m == True` → seller aggressor → negative
- `m == False` → buyer aggressor → positive

---

## 📈 Optimization Tips

### 1. Threshold Fine-Tuning:

```python
# Backtest yap (geçmiş data ile):
for z_threshold in [1.5, 2.0, 2.5, 3.0]:
    for cvd_threshold in [5000, 10000, 15000]:
        # Test et
        # En karlısını bul
```

### 2. Multi-Symbol Correlation:

```python
# Eğer BTC funding spike yaparsa:
# - ETH de spike yapıyor mu?
# - Correlation yüksekse → daha güvenli
# - Correlation düşükse → isolated event (riskli)
```

### 3. Time-Based Filters:

```python
# Bazı saatlerde funding daha volatile:
# - Funding time öncesi/sonrası (00:00, 08:00, 16:00 UTC)
# - Bu saatlerde agresif ol
```

---

## 🎓 İleri Seviye Stratejiler

### 1. Funding Rate Pairs Trading:

```
BTC funding: +0.10%
ETH funding: +0.05%

→ Long ETH + Short BTC (relative value)
→ Hem funding arbitrage, hem pair convergence
```

### 2. CVD Divergence:

```
Price ↑ ama CVD ↓ (bearish divergence)
→ Institutional distribution (sell signal)

Price ↓ ama CVD ↑ (bullish divergence)
→ Institutional accumulation (buy signal)
```

### 3. Funding + OI (Open Interest):

```
Funding spike + OI increase → New leveraged longs
→ Liquidation cascade risk (SHORT setup)

Funding spike + OI decrease → Deleveraging
→ Funding normalizes quickly (quick trade)
```

---

## 📝 Son Notlar

1. **Küçük Başla:** İlk ay $200-500 ile test et
2. **Sabırlı Ol:** Funding rate saatte değişmez, günler/haftalar sürer
3. **Risk Yönet:** Asla %100 sermaye yatırma
4. **Öğren:** Her sinyal bir öğrenme fırsatı
5. **Adapt Et:** Market şartları değişir, strateji de adapt etmeli

**Bu bir get-rich-quick scheme DEĞİL!**

Conservative yaklaşımla aylık %5-10 tutarlı kazanç hedefle. Compound ile yıllık %80-150 çok iyi bir hedef.

---

## 📚 Kaynaklar

- Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
- WebSocket Streams: https://binance-docs.github.io/apidocs/futures/en/#websocket-market-streams
- Funding Rate Docs: https://www.binance.com/en/support/faq/funding-rates

---

**Good luck trading! 🚀**
