"""
BTCTurk Trade Feed - Clean Version
No ping/pong - just subscribe and listen to trade messages
"""

import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, Callable, List, Dict, Any, Set
import websockets
import logging
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)


WS_URL = "wss://ws-feed-pro.btcturk.com/"
CH_SUBSCRIBE = 151
CH_TRADE_BATCH = 421
CH_TRADE_SINGLE = 422


def ms_to_iso(ms: int) -> str:
    """Milliseconds to ISO 8601"""
    try:
        return (
            datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    except Exception:
        return str(ms)


def side_str(s: int) -> str:
    """0=BUY, 1=SELL"""
    return "BUY" if s == 0 else "SELL"


class BTCTurkTradeFeed:
    """
    BTCTurk WebSocket Trade Feed
    - No ping/pong (BTCTurk doesn't use it)
    - Just subscribe and listen
    - Multiple pairs support
    """

    def __init__(
        self,
        pairs: List[str] = None,
        on_trade: Optional[Callable] = None,
    ):
        self.pairs: Set[str] = set(p.upper() for p in (pairs or ["BTCTRY"]))
        self.on_trade = on_trade
        self._ws = None
        self._connected = False
        self._stop = asyncio.Event()
        self._retries = 0
        self._backoff = 1.0

    async def start(self):
        """Start WebSocket feed"""
        self._stop.clear()
        try:
            await self._run_forever()
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Stop WebSocket feed"""
        self._stop.set()
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass

    async def add_pair(self, pair: str):
        """Add pair at runtime"""
        pair = pair.upper()
        if pair not in self.pairs:
            self.pairs.add(pair)
            await self._subscribe_trades(pair)
            print(f"✅ Pair added: {pair}")

    async def remove_pair(self, pair: str):
        """Remove pair"""
        pair = pair.upper()
        self.pairs.discard(pair)

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def _run_forever(self):
        """Main WebSocket loop - just listen"""
        while not self._stop.is_set():
            try:
                # Connect without ping/pong
                async with websockets.connect(
                    WS_URL,
                    ping_interval=None,    # No ping/pong
                    ping_timeout=None,
                    max_size=10_000_000,
                    compression=None,
                    close_timeout=10,
                ) as websocket:
                    self._ws = websocket
                    self._connected = True
                    self._retries = 0
                    self._backoff = 1.0

                    print(f"✅ WebSocket CONNECTED")

                    # Subscribe to all pairs
                    for pair in self.pairs:
                        await self._subscribe_trades(pair)

                    # Just listen to messages
                    async for message in websocket:
                        if self._stop.is_set():
                            break
                        await self._handle_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ WebSocket error: {e}")
                self._connected = False

            if self._stop.is_set():
                break

            # Reconnect with exponential backoff
            self._retries += 1
            wait_time = min(self._backoff, 30.0)
            print(f"🔄 Reconnecting ({self._retries}. attempt, waiting {wait_time}s)...")
            await asyncio.sleep(wait_time)
            self._backoff = self._backoff * 1.5

        self._connected = False

    async def _subscribe_trades(self, pair: str):
        """Subscribe to trade channel"""
        if not self._connected or not self._ws:
            return

        try:
            msg = [
                CH_SUBSCRIBE,
                {
                    "type": CH_SUBSCRIBE,
                    "channel": "trade",
                    "event": pair.upper(),
                    "join": True,
                },
            ]
            await self._ws.send(json.dumps(msg))
            print(f"📡 Subscribe: {pair}")
        except Exception as e:
            print(f"⚠️ Subscribe error ({pair}): {e}")

    async def _handle_message(self, message: str):
        """Handle trade messages only"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        obj = self._normalize_message(data)
        msg_type = obj.get("type")

        # Batch trade (421)
        if msg_type == CH_TRADE_BATCH and obj.get("event"):
            pair = obj.get("event")
            if pair in self.pairs:
                items = obj.get("items", [])
                for row in items:
                    await self._handle_trade_row(pair, row)

        # Single trade (422)
        elif msg_type == CH_TRADE_SINGLE and obj.get("PS"):
            pair = obj.get("PS")
            if pair in self.pairs:
                await self._handle_trade_row(pair, obj)

    def _normalize_message(self, msg: Any) -> Dict[str, Any]:
        """Normalize message"""
        if isinstance(msg, dict):
            return msg
        if isinstance(msg, list) and len(msg) >= 2 and isinstance(msg[1], dict):
            d = msg[1].copy()
            d.setdefault("type", msg[0])
            return d
        return {"raw": msg}

    async def _handle_trade_row(self, pair: str, row: Dict[str, Any]):
        """Process trade row"""
        try:
            price = float(row.get("P", 0))
        except (ValueError, TypeError):
            price = None

        try:
            amount = float(row.get("A", 0))
        except (ValueError, TypeError):
            amount = None

        side = side_str(int(row.get("S", 0)))
        ts = row.get("D")

        try:
            ts = ms_to_iso(int(ts))
        except (ValueError, TypeError):
            pass

        trade_data = {
            "pair": pair,
            "side": side,
            "price": price,
            "amount": amount,
            "timestamp": ts,
            "id": row.get("I"),
        }

        # Print
        #print(f"[{pair}] {side:<4} | Price: {price:>15} | Amount: {amount:>12}")

        # Callback
        if self.on_trade:
            if asyncio.iscoroutinefunction(self.on_trade):
                await self.on_trade(pair, trade_data)
            else:
                self.on_trade(pair, trade_data)
