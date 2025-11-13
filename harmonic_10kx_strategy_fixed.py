from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict
from comprehensive_harmonic_detection import ComprehensiveHarmonicDetector

logger = logging.getLogger(__name__)


class Harmonic10kxStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '5m'
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 500

    minimal_roi = {"0": 0.30, "60": 0.10, "180": 0}
    stoploss = -0.05

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    sell_rsi_threshold = 70

    # ✅ Pattern detection için sliding window (performans için)
    pattern_lookback_window = 200  # Sadece son 200 candle'a bak

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.detector = ComprehensiveHarmonicDetector(tolerance=0.4)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        ✅ Indicator'ları hesapla - SADECE SON CANDLE İÇİN pattern detection yap.
        Freqtrade backtest sırasında bu fonksiyonu sliding window olarak çağırır.
        """
        # ATR
        dataframe['tr'] = np.maximum(
            dataframe['high'] - dataframe['low'],
            np.maximum(
                (dataframe['high'] - dataframe['close'].shift()).abs(),
                (dataframe['low'] - dataframe['close'].shift()).abs()
            )
        )
        dataframe['atr'] = dataframe['tr'].rolling(14).mean()

        # RSI
        delta = dataframe['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        # ✅ Pattern detection kolonlarını initialize et
        dataframe['has_bullish_pattern'] = False
        dataframe['has_bearish_pattern'] = False
        dataframe['pattern_name'] = ""
        dataframe['pattern_score'] = 0.0
        dataframe['pattern_type'] = ""  # BUY/SELL
        dataframe['pattern_price'] = np.nan

        # ✅ PERFORMANSLI PATTERN DETECTION
        # Sadece yeterli veri varsa çalıştır
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        try:
            # ✅ Sliding window approach - sadece son N candle'a bak
            lookback = min(self.pattern_lookback_window, len(dataframe))
            recent_data = dataframe.tail(lookback).copy()

            # Pattern detection yap (sadece son window için)
            all_patterns = self.detector.detect_all_patterns(recent_data)

            # ✅ CRITICAL: Pass recent_data to get signals, but they will have correct timestamps
            signals = self.detector.get_all_trading_signals(all_patterns, recent_data)

            logger.info(f"[{metadata.get('pair', '')}] Pattern detection complete: {len(signals)} signals found")

            if not signals:
                return dataframe

            # ✅ TÜM sinyalleri kaydet (sadece son 5 değil!)
            # Backtest'te populate_indicators bir kez çalışır, tüm pattern'ları kaydetmeliyiz
            saved_count = 0
            skipped_count = 0

            for signal in signals:
                sig_timestamp = signal['timestamp']

                # ✅ Timestamp'in dataframe'de olduğundan emin ol
                try:
                    # Test if timestamp exists in index
                    _ = dataframe.loc[sig_timestamp]
                except KeyError:
                    logger.warning(
                        f"Signal timestamp {sig_timestamp} not in dataframe index "
                        f"(type: {type(sig_timestamp)}, first df index: {dataframe.index[0]})"
                    )
                    skipped_count += 1
                    continue

                # Duplicate pattern kontrolü - önceki candle ile aynı mı?
                try:
                    sig_loc = dataframe.index.get_loc(sig_timestamp)
                    if sig_loc > 0:
                        prev_idx = dataframe.index[sig_loc - 1]
                        prev_pattern = dataframe.at[prev_idx, 'pattern_name']
                        prev_type = dataframe.at[prev_idx, 'pattern_type']
                        prev_price = dataframe.at[prev_idx, 'pattern_price']

                        curr_pattern = signal.get('pattern', '')
                        curr_type = signal['signal_type']
                        curr_price = signal.get('price', np.nan)

                        # Aynı pattern ve fiyat çok yakınsa skip et
                        if (prev_pattern == curr_pattern and
                            prev_type == curr_type and
                            not np.isnan(prev_price) and
                            not np.isnan(curr_price)):

                            price_diff = abs(curr_price - prev_price) / curr_price
                            if price_diff < 0.001:  # %0.1'den az fark
                                logger.debug(f"Skipping duplicate pattern @ {sig_timestamp}")
                                continue
                except Exception as e:
                    logger.debug(f"Duplicate check error: {e}")

                # Eğer bu timestamp'te daha yüksek skorlu pattern varsa, onu kullan
                current_score = dataframe.at[sig_timestamp, 'pattern_score']
                if current_score > 0 and signal['score'] <= current_score:
                    continue

                # ✅ Pattern bilgilerini DataFrame'e yaz
                dataframe.at[sig_timestamp, 'pattern_name'] = signal.get('pattern', '')
                dataframe.at[sig_timestamp, 'pattern_score'] = signal['score']
                dataframe.at[sig_timestamp, 'pattern_type'] = signal['signal_type']
                dataframe.at[sig_timestamp, 'pattern_price'] = signal.get('price', np.nan)

                if signal['signal_type'] == 'BUY':
                    dataframe.at[sig_timestamp, 'has_bullish_pattern'] = True
                elif signal['signal_type'] == 'SELL':
                    dataframe.at[sig_timestamp, 'has_bearish_pattern'] = True

                saved_count += 1

                logger.debug(
                    f"[{metadata.get('pair', '')}] Pattern saved @ {sig_timestamp}: "
                    f"{signal.get('pattern', 'N/A')} ({signal['signal_type']}) "
                    f"Score: {signal['score']:.2f}"
                )

            # Summary log
            logger.info(
                f"[{metadata.get('pair', '')}] Pattern detection summary: "
                f"{saved_count} patterns saved, {skipped_count} skipped"
            )

        except Exception as e:
            logger.error(f"Pattern detection error: {e}", exc_info=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        ✅ Pattern detection sonuçlarını kullanarak entry sinyalleri üret.
        """
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # Entry pattern bilgisi kolonları
        for col in [
            'entry_pattern_key',
            'entry_pattern_name',
            'entry_pattern_label',
            'entry_pattern_direction',
            'entry_pattern_score',
        ]:
            if col not in dataframe.columns:
                dataframe[col] = ""

        try:
            # ✅ LONG ENTRY CONDITIONS
            long_conditions = (
                (dataframe['has_bullish_pattern'] == True) &
                (dataframe['pattern_type'] == 'BUY') &
                (dataframe['rsi'] < self.sell_rsi_threshold) &
                (dataframe['pattern_score'] > 0) &
                (dataframe['pattern_name'] != "")
            )

            dataframe.loc[long_conditions, 'enter_long'] = 1

            # Pattern bilgilerini kaydet
            if long_conditions.any():
                # Pattern name'i temizle (bullish/bearish kelimelerini kaldır)
                clean_pattern = (
                    dataframe.loc[long_conditions, 'pattern_name']
                    .str.lower()
                    .str.replace('bullish', '', regex=False)
                    .str.replace('bearish', '', regex=False)
                    .str.strip()
                )

                dataframe.loc[long_conditions, 'entry_pattern_key'] = 'bullish_' + clean_pattern
                dataframe.loc[long_conditions, 'entry_pattern_name'] = dataframe.loc[long_conditions, 'pattern_name']
                dataframe.loc[long_conditions, 'entry_pattern_label'] = dataframe.loc[long_conditions, 'pattern_name']
                dataframe.loc[long_conditions, 'entry_pattern_direction'] = 'BULLISH'
                dataframe.loc[long_conditions, 'entry_pattern_score'] = dataframe.loc[long_conditions, 'pattern_score'].astype(str)

                # Log sadece son 3 entry için (spam önlemek için)
                recent_entries = dataframe[long_conditions].tail(3)
                for idx in recent_entries.index:
                    logger.info(
                        f"[{metadata.get('pair', '')}] ✅ LONG ENTRY: {dataframe.at[idx, 'pattern_name']} "
                        f"@ {idx} | Score: {dataframe.at[idx, 'pattern_score']:.2f} | "
                        f"RSI: {dataframe.at[idx, 'rsi']:.1f}"
                    )

            # ✅ SHORT ENTRY CONDITIONS
            short_conditions = (
                (dataframe['has_bearish_pattern'] == True) &
                (dataframe['pattern_type'] == 'SELL') &
                (dataframe['rsi'] > (100 - self.sell_rsi_threshold)) &
                (dataframe['pattern_score'] > 0) &
                (dataframe['pattern_name'] != "")
            )

            dataframe.loc[short_conditions, 'enter_short'] = 1

            # Pattern bilgilerini kaydet
            if short_conditions.any():
                clean_pattern = (
                    dataframe.loc[short_conditions, 'pattern_name']
                    .str.lower()
                    .str.replace('bullish', '', regex=False)
                    .str.replace('bearish', '', regex=False)
                    .str.strip()
                )

                dataframe.loc[short_conditions, 'entry_pattern_key'] = 'bearish_' + clean_pattern
                dataframe.loc[short_conditions, 'entry_pattern_name'] = dataframe.loc[short_conditions, 'pattern_name']
                dataframe.loc[short_conditions, 'entry_pattern_label'] = dataframe.loc[short_conditions, 'pattern_name']
                dataframe.loc[short_conditions, 'entry_pattern_direction'] = 'BEARISH'
                dataframe.loc[short_conditions, 'entry_pattern_score'] = dataframe.loc[short_conditions, 'pattern_score'].astype(str)

                recent_entries = dataframe[short_conditions].tail(3)
                for idx in recent_entries.index:
                    logger.info(
                        f"[{metadata.get('pair', '')}] ✅ SHORT ENTRY: {dataframe.at[idx, 'pattern_name']} "
                        f"@ {idx} | Score: {dataframe.at[idx, 'pattern_score']:.2f} | "
                        f"RSI: {dataframe.at[idx, 'rsi']:.1f}"
                    )

        except Exception as e:
            logger.error(f"Entry trend error: {e}", exc_info=True)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        ✅ Exit sinyalleri üret - RSI ve ters yönlü pattern'lara göre.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # Exit pattern bilgisi kolonları
        for col in [
            'exit_pattern_key',
            'exit_pattern_name',
            'exit_pattern_label',
            'exit_pattern_direction',
            'exit_pattern_score',
        ]:
            if col not in dataframe.columns:
                dataframe[col] = ""

        try:
            # ✅ EXIT LONG CONDITIONS
            # 1) RSI çok yüksek
            # 2) Bearish pattern oluştu
            exit_long_conditions = (
                (
                    (dataframe['rsi'] > self.sell_rsi_threshold) |
                    (
                        (dataframe['has_bearish_pattern'] == True) &
                        (dataframe['pattern_type'] == 'SELL')
                    )
                )
            )

            dataframe.loc[exit_long_conditions, 'exit_long'] = 1

            # Bearish pattern varsa kaydet
            pattern_exit_long = (
                exit_long_conditions &
                (dataframe['has_bearish_pattern'] == True) &
                (dataframe['pattern_name'] != "")
            )

            if pattern_exit_long.any():
                clean_pattern = (
                    dataframe.loc[pattern_exit_long, 'pattern_name']
                    .str.lower()
                    .str.replace('bullish', '', regex=False)
                    .str.replace('bearish', '', regex=False)
                    .str.strip()
                )

                dataframe.loc[pattern_exit_long, 'exit_pattern_key'] = 'exit_bearish_' + clean_pattern
                dataframe.loc[pattern_exit_long, 'exit_pattern_name'] = dataframe.loc[pattern_exit_long, 'pattern_name']
                dataframe.loc[pattern_exit_long, 'exit_pattern_label'] = dataframe.loc[pattern_exit_long, 'pattern_name']
                dataframe.loc[pattern_exit_long, 'exit_pattern_direction'] = 'EXIT_BEARISH'
                dataframe.loc[pattern_exit_long, 'exit_pattern_score'] = dataframe.loc[pattern_exit_long, 'pattern_score'].astype(str)

                recent_exits = dataframe[pattern_exit_long].tail(3)
                for idx in recent_exits.index:
                    logger.info(
                        f"[{metadata.get('pair', '')}] EXIT LONG (Pattern): {dataframe.at[idx, 'pattern_name']} "
                        f"@ {idx} | Score: {dataframe.at[idx, 'pattern_score']:.2f}"
                    )

            # ✅ EXIT SHORT CONDITIONS
            # 1) RSI çok düşük
            # 2) Bullish pattern oluştu
            exit_short_conditions = (
                (
                    (dataframe['rsi'] < (100 - self.sell_rsi_threshold)) |
                    (
                        (dataframe['has_bullish_pattern'] == True) &
                        (dataframe['pattern_type'] == 'BUY')
                    )
                )
            )

            dataframe.loc[exit_short_conditions, 'exit_short'] = 1

            # Bullish pattern varsa kaydet
            pattern_exit_short = (
                exit_short_conditions &
                (dataframe['has_bullish_pattern'] == True) &
                (dataframe['pattern_name'] != "")
            )

            if pattern_exit_short.any():
                clean_pattern = (
                    dataframe.loc[pattern_exit_short, 'pattern_name']
                    .str.lower()
                    .str.replace('bullish', '', regex=False)
                    .str.replace('bearish', '', regex=False)
                    .str.strip()
                )

                dataframe.loc[pattern_exit_short, 'exit_pattern_key'] = 'exit_bullish_' + clean_pattern
                dataframe.loc[pattern_exit_short, 'exit_pattern_name'] = dataframe.loc[pattern_exit_short, 'pattern_name']
                dataframe.loc[pattern_exit_short, 'exit_pattern_label'] = dataframe.loc[pattern_exit_short, 'pattern_name']
                dataframe.loc[pattern_exit_short, 'exit_pattern_direction'] = 'EXIT_BULLISH'
                dataframe.loc[pattern_exit_short, 'exit_pattern_score'] = dataframe.loc[pattern_exit_short, 'pattern_score'].astype(str)

                recent_exits = dataframe[pattern_exit_short].tail(3)
                for idx in recent_exits.index:
                    logger.info(
                        f"[{metadata.get('pair', '')}] EXIT SHORT (Pattern): {dataframe.at[idx, 'pattern_name']} "
                        f"@ {idx} | Score: {dataframe.at[idx, 'pattern_score']:.2f}"
                    )

        except Exception as e:
            logger.error(f"Exit trend error: {e}", exc_info=True)

        return dataframe
