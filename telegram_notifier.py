"""
Telegram Notifier - Düzeltilmiş versiyon
Async message sending with proper error handling
"""

import asyncio
import aiohttp
from enum import Enum
from typing import Optional
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    TRADE = "TRADE"
    STATS = "STATS"
    ERROR = "ERROR"


class TelegramNotifier:
    """Telegram mesajları göndermek için"""
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.session: Optional[aiohttp.ClientSession] = None
        
        print(f"✅ TelegramNotifier initialized")
        print(f"   Bot Token: {bot_token[:20]}...")
        print(f"   Chat ID: {chat_id}")
        print(f"   Status: {'ENABLED' if enabled else 'DISABLED'}")
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Session'ı lazy initialize et"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def send(self, message: str, level: AlertLevel = AlertLevel.TRADE) -> bool:
        """Telegram mesajı gönder"""
        
        if not self.enabled:
            logger.warning("TelegramNotifier disabled, skipping message")
            return False
        
        try:
            session = await self.get_session()
            
            # Mesaj payload'ı
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            logger.debug(f"Sending Telegram message to {self.chat_id}")
            logger.debug(f"Message preview: {message[:100]}...")
            
            # POST request
            async with session.post(
                self.api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    if result.get('ok'):
                        logger.info(f"✅ Telegram message sent successfully (ID: {result['result']['message_id']})")
                        return True
                    else:
                        logger.error(f"❌ Telegram API error: {result.get('description')}")
                        return False
                else:
                    logger.error(f"❌ HTTP Error {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return False
        
        except asyncio.TimeoutError:
            logger.error("❌ Telegram request timeout")
            return False
        
        except aiohttp.ClientError as e:
            logger.error(f"❌ Network error: {e}")
            return False
        
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}", exc_info=True)
            return False
    
    async def close(self):
        """Session'ı kapat"""
        if self.session:
            await self.session.close()
            logger.info("TelegramNotifier session closed")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Test
async def test_telegram():
    """Test telegram gönderimi"""
    notifier = TelegramNotifier(
        bot_token="8252371895:AAFleOvcPsxmiOh82x2QvAfUuwLcv1eI8Dw",
        chat_id="652342213",
        enabled=True
    )
    
    test_message = """
<b>🟢 TEST MESSAGE</b>

This is a test message from your strategy.
<b>Status:</b> ✅ Working

<i>⏰ Test timestamp</i>
"""
    
    result = await notifier.send(test_message)
    print(f"\nTest result: {result}")
    
    await notifier.close()


if __name__ == "__main__":
    asyncio.run(test_telegram())
