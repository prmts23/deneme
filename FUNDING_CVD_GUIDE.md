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

# Dinamik Sembol Seçimi (Otomatik)
MAX_SYMBOLS = 5  # Maksimum izlenecek coin sayısı
MIN_FUNDING_RATE = 0.0003  # Minimum funding rate (0.03% per 8h)
UPDATE_SYMBOLS_INTERVAL = 3600  # Sembol listesini güncelleme sıklığı (saniye)

POSITION_SIZE_USD = 1000  # Pozisyon büyüklüğü
```

**NOT:** Artık manuel olarak `SYMBOLS` listesi belirtmeye gerek yok! Sistem otomatik olarak:
- Tüm USDT perpetual kontratları tarar
- En yüksek funding rate'e sahip coin'leri seçer
- Her saat listeyi günceller
- Daha karlı fırsatlar çıkarsa otomatik değiştirir

### 4. Çalıştır:
```bash
python funding_cvd_system.py
```

---

## ⚙️ Konfigürasyon

### Temel Parametreler:

```python
# Dinamik Sembol Seçimi
MAX_SYMBOLS = 5  # Maksimum izlenecek coin sayısı (1-10 arası önerilir)
MIN_FUNDING_RATE = 0.0003  # Minimum funding rate (0.03% per 8h)
UPDATE_SYMBOLS_INTERVAL = 3600  # Güncelleme sıklığı (saniye, 3600 = 1 saat)

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

### Dinamik Sembol Seçimi Ayarları:

**Conservative (Az coin, sadece en iyiler):**
```python
MAX_SYMBOLS = 3  # Sadece top 3
MIN_FUNDING_RATE = 0.0005  # En az 0.05% (yüksek threshold)
UPDATE_SYMBOLS_INTERVAL = 7200  # Her 2 saatte güncelle
```

**Moderate (Dengeli - Varsayılan):**
```python
MAX_SYMBOLS = 5  # Top 5 coin
MIN_FUNDING_RATE = 0.0003  # En az 0.03%
UPDATE_SYMBOLS_INTERVAL = 3600  # Her saat güncelle
```

**Aggressive (Çok coin, daha fazla fırsat):**
```python
MAX_SYMBOLS = 8  # Top 8 coin
MIN_FUNDING_RATE = 0.0002  # En az 0.02% (düşük threshold)
UPDATE_SYMBOLS_INTERVAL = 1800  # Her 30 dakika güncelle
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

## 🔄 Dinamik Sembol Seçimi

### Nasıl Çalışır?

Sistem başlatıldığında ve her saat başı:

1. **Tarama:** Binance'deki tüm USDT perpetual kontratları taranır (~200+ coin)
2. **Filtreleme:** Minimum funding rate threshold'ını geçen coin'ler seçilir
3. **Sıralama:** Absolute funding rate'e göre sıralanır (hem pozitif hem negatif)
4. **Seçim:** En yüksek funding'e sahip top N coin seçilir
5. **Güncelleme:** Liste değiştiyse WebSocket yeniden bağlanır ve Telegram bildirimi gönderilir

### Avantajları:

✅ **Otomatik Optimizasyon:** Manuel olarak coin seçmeye gerek yok
✅ **Fırsat Yakalama:** Yeni yüksek funding fırsatlarını otomatik yakalar
✅ **Risk Azaltma:** Funding düşen coin'lerden otomatik çıkar
✅ **Zaman Tasarrufu:** Sürekli funding rate taraması yapmana gerek yok
✅ **Diversifikasyon:** Her zaman en karlı coin portfolio'su

### Örnek Senaryo:

```
Saat 10:00 - İlk Tarama:
  SOLUSDT: +0.15%
  AVAXUSDT: +0.12%
  ARBUSDT: +0.10%
  → Bu 3 coin izleniyor

Saat 11:00 - Güncelleme:
  PEPEUSDT: +0.18% (YENİ YÜKSEK!)
  SOLUSDT: +0.14% (hala iyi)
  AVAXUSDT: +0.11% (hala iyi)
  ARBUSDT: +0.05% (düştü)

  → ARBUSDT çıkar, PEPEUSDT girer
  → Telegram bildirimi gelir
  → WebSocket yeniden bağlanır
```

### Manuel Mod:

Eğer yine de manuel sembol seçmek istiyorsan:

```python
# update_symbols() metodunu devre dışı bırak
# WebSocketManager.__init__ içinde:
self.symbols = ["BTCUSDT", "ETHUSDT"]  # Manuel liste

# connect_and_listen() içindeki update check'i kaldır:
# if (datetime.now() - self.last_symbol_update).total_seconds() > UPDATE_SYMBOLS_INTERVAL:
#     ...
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
MAX_SYMBOLS = 2  # Sadece top 2 coin
MIN_FUNDING_RATE = 0.0005  # Yüksek funding'leri seç (0.05%+)
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
- Sistem otomatik olarak en iyi 2 coin'i seçecek

---

### Senaryo 2: $2,000 Sermaye (Intermediate)

**Strateji:**
```python
POSITION_SIZE_USD = 500  # Per coin
MAX_SYMBOLS = 4  # Top 4 coin
MIN_FUNDING_RATE = 0.0004  # 0.04%+ funding
UPDATE_SYMBOLS_INTERVAL = 3600  # Her saat güncelle
```

**Beklenti:**
- Günlük: $4-8
- Aylık: $120-240 (6-12%)
- Risk: Düşük-Orta

**Taktik:**
- 4 coin otomatik diversify
- High funding'de aggressive ol
- Sistem en karlı coin'lere otomatik geçer

---

### Senaryo 3: $10,000 Sermaye (Advanced)

**Strateji:**
```python
POSITION_SIZE_USD = 1000  # Per coin
MAX_SYMBOLS = 5  # Top 5 coin
MIN_FUNDING_RATE = 0.0003  # 0.03%+ funding
UPDATE_SYMBOLS_INTERVAL = 1800  # Her 30 dakika güncelle
```

**Beklenti:**
- Günlük: $20-40
- Aylık: $600-1,200 (6-12%)
- Risk: Orta

**Advanced Taktikler:**
- Dynamic symbol rotation (sistem otomatik)
- Multi-symbol correlation tracking
- Frequent updates (30 dakika)
- Telegram'dan güncellemeleri takip et

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

Ardından ilk sembol taraması yapılır ve en iyi fırsatlar bildirilir:
```
📊 Top Funding Opportunities

🔴 SHORT SOLUSDT
  • Funding: +0.0850% per 8h
  • Daily: $2.55
  • Payback: 0.7 days

🔴 SHORT AVAXUSDT
  • Funding: +0.0720% per 8h
  • Daily: $2.16
  • Payback: 0.8 days

🟢 LONG MATICUSDT
  • Funding: -0.0650% per 8h
  • Daily: $1.95
  • Payback: 0.9 days
```

### 2. Sembol Listesi Güncelleme (Her Saat):
```
🔄 Symbol List Updated

❌ Removed (lower funding):
  • ETHUSDT
  • BNBUSDT

✅ Added (higher funding):
  • PEPEUSDT
  • ARBUSDT

Now monitoring: SOLUSDT, BTCUSDT, PEPEUSDT, ARBUSDT, AVAXUSDT
```

**Ne Anlama Gelir:**
- Sistem otomatik olarak daha karlı coin'lere geçiyor
- Eski coin'lerin funding'i düştü
- Yeni coin'lerin funding'i daha yüksek
- WebSocket yeniden bağlanacak (seamless)

### 3. Signal Notification:
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

### 4. Hourly Summary:
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
