"""
TelegramNotifier - File Logging Version
Claude ortamında çalışan ve alertları dosyaya yazan versiyon
"""

import asyncio
from datetime import datetime
from pathlib import Path
import logging
from enum import Enum
from typing import Optional


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    TRADE = "TRADE"
    STATS = "STATS"
    ERROR = "ERROR"


class TelegramNotifier:
    """
    File-based Telegram notifier - Telegram API'ya ulaşılamadığı için
    mesajları dosyaya yazıyor ve manuel/otomatik olarak gönderilebilmeleri sağlıyor
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
        log_dir: str = "alerts"
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Alert dosyaları
        self.trades_log = self.log_dir / "trades.log"
        self.stats_log = self.log_dir / "stats.log"
        self.errors_log = self.log_dir / "errors.log"
        
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║         ✅ TelegramNotifier (File Logging Mode)               ║
╚════════════════════════════════════════════════════════════════╝

📁 Log Directory: {self.log_dir.absolute()}
📄 Trades Log:   {self.trades_log.absolute()}
📄 Stats Log:    {self.stats_log.absolute()}
📄 Errors Log:   {self.errors_log.absolute()}

⚠️  Claude ortamında Telegram API'ya doğrudan erişim yapılamadığı için,
    mesajlar dosyaya kaydediliyor.

ÇÖZÜM YOLLARI:
1. VPS'de Flask gateway kurarak Telegram'a ilet (REKOMENDED)
2. Log dosyalarını manuel olarak Telegram'a gönder
3. Cron job ile otomatik gönder

HEMEN BAŞLAMAK İÇİN:
- python send_alerts_telegram.py

""")
    
    async def send(self, message: str, level: AlertLevel = AlertLevel.TRADE) -> bool:
        """
        Mesajı dosyaya kaydet
        
        Args:
            message: Gönderilecek mesaj (HTML format)
            level: Alert seviyesi (TRADE, STATS, ERROR)
        
        Returns:
            bool: Başarılı mı?
        """
        
        if not self.enabled:
            logger.warning("TelegramNotifier disabled")
            return False
        
        try:
            # Uygun log dosyasını seç
            if level == AlertLevel.TRADE:
                log_file = self.trades_log
            elif level == AlertLevel.STATS:
                log_file = self.stats_log
            else:
                log_file = self.errors_log
            
            # Mesajı formatla
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            separator = "=" * 70
            
            formatted_message = f"""
{separator}
[{timestamp}] - {level.value}
{separator}
{message}

"""
            
            # Dosyaya yaz
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_message)
            
            logger.info(f"✅ Message logged to {log_file.name}")
            
            # Console output
            print(f"\n📝 [{level.value}] Message saved to logs/")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Error writing to log: {e}", exc_info=True)
            return False
    
    async def close(self):
        """Cleanup"""
        logger.info("TelegramNotifier closed")
    
    def print_summary(self):
        """Log dosyalarının özetini göster"""
        print(f"\n{'='*70}")
        print(f"📊 ALERT SUMMARY")
        print(f"{'='*70}")
        
        for log_file, label in [
            (self.trades_log, "🟢 TRADES"),
            (self.stats_log, "📊 STATS"),
            (self.errors_log, "🔴 ERRORS")
        ]:
            if log_file.exists():
                lines = len(log_file.read_text().strip().split('\n'))
                size = log_file.stat().st_size / 1024  # KB
                print(f"{label:20} {lines:4} entries  ({size:.1f} KB)")
            else:
                print(f"{label:20} {'EMPTY':4}")
        
        print(f"\n📁 View logs: ls -la {self.log_dir.absolute()}")
        print(f"{'='*70}\n")


# HELPER: Log dosyalarını Telegram'a gönder
async def send_alerts_via_telegram(
    telegram_token: str,
    chat_id: str,
    log_dir: str = "alerts"
):
    """
    Kaydedilmiş alert'leri Telegram'a gönder
    (Bu fonksiyon VPS'de çalışmalı, Claude ortamında çalışmaz!)
    """
    import aiohttp
    
    log_dir = Path(log_dir)
    
    for log_file in log_dir.glob("*.log"):
        if log_file.stat().st_size == 0:
            continue
        
        print(f"📤 Sending {log_file.name}...")
        
        content = log_file.read_text()
        
        try:
            async with aiohttp.ClientSession() as session:
                # 4096 karakter sınırı olduğu için parçala
                for i in range(0, len(content), 4000):
                    chunk = content[i:i+4000]
                    
                    async with session.post(
                        f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                        json={
                            'chat_id': chat_id,
                            'text': chunk,
                            'parse_mode': 'HTML'
                        }
                    ) as resp:
                        if resp.status == 200:
                            print(f"  ✅ Chunk sent")
                        else:
                            print(f"  ❌ Error: {resp.status}")
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")


# TEST
async def test():
    """Test alert'leri logla"""
    
    notifier = TelegramNotifier(
        enabled=True,
        log_dir="alerts"
    )
    
    # Test trade alert
    trade_message = """
<b>🟢 LONG ENTRY - UNITRY</b>

<b>Price:</b> 100.50
<b>Stop Loss:</b> 95.00
<b>Take Profit:</b> 110.00
<b>Risk/Reward:</b> 1:2.00

<b>Indicators:</b>
├─ RSI: 32.5
├─ ATR: 0.0045
├─ SMA20: 100.25
└─ SMA50: 99.80

<b>Volume Indicators:</b>
├─ CVD Trend: BULLISH
├─ CVD Momentum: +125.50
└─ Buy Ratio: 58.5%

<i>⏰ 14:32:15</i>
"""
    
    await notifier.send(trade_message, AlertLevel.TRADE)
    
    # Test stats alert
    stats_message = """
<b>📊 UNITRY - İSTATİSTİKLER</b>

<b>Trades:</b> 15
<b>Win Rate:</b> 66.7%
<b>Avg Win:</b> +0.85
<b>Total PnL:</b> +12.75
<b>Sharpe:</b> 1.85
<b>Max DD:</b> -3.20
"""
    
    await notifier.send(stats_message, AlertLevel.STATS)
    
    # Print summary
    notifier.print_summary()
    
    await notifier.close()


if __name__ == "__main__":
    asyncio.run(test())
