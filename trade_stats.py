"""
İstatistik sınıfı - Kapsamlı performans analizi
"""

from typing import Dict, List, Optional
from collections import deque
from datetime import datetime, timedelta
import numpy as np


class TradeStats:
    """
    Kapsamlı ticaret istatistiklerini takip et
    Risk metrics, performance indicators, vb.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Args:
            max_history: Kaç son işlemi saklasın
        """
        self.trades = deque(maxlen=max_history)
        self.daily_stats = {}  # date -> stats
        self.hourly_stats = {}  # hour -> stats
        
        # Real-time metrics
        self.current_streak = 0  # +win, -loss
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.consecutive_losses = 0
        
        # Drawdown tracking
        self.peak_balance = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.cumulative_pnl = 0
        
    def add_trade(
        self,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        duration_sec: float,
        trade_type: str = "LONG",
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None
    ):
        """İşlem kaydını ekle"""
        exit_time = exit_time or datetime.now()
        entry_time = entry_time or (exit_time - timedelta(seconds=duration_sec))
        
        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration_sec': duration_sec,
            'type': trade_type,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'timestamp': exit_time,
            'date': exit_time.date(),
            'hour': exit_time.hour
        }
        
        self.trades.append(trade)
        
        # Update cumulative stats
        self._update_cumulative_stats(trade)
        
        # Update streak
        self._update_streak(pnl)
        
        # Update daily/hourly stats
        self._update_time_based_stats(trade)
    
    def _update_cumulative_stats(self, trade: Dict):
        """Kümülatif istatistikleri güncelle"""
        pnl = trade['pnl']
        self.cumulative_pnl += pnl
        
        # Drawdown hesabı
        self.peak_balance = max(self.peak_balance, self.cumulative_pnl)
        self.current_drawdown = self.peak_balance - self.cumulative_pnl
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def _update_streak(self, pnl: float):
        """Win/Loss streak'i güncelle"""
        if pnl > 0:
            self.current_streak = max(1, self.current_streak + 1)
            self.consecutive_losses = 0
            self.max_win_streak = max(self.max_win_streak, self.current_streak)
        else:
            self.current_streak = min(-1, self.current_streak - 1)
            self.consecutive_losses += 1
            self.max_loss_streak = max(self.max_loss_streak, abs(self.current_streak))
    
    def _update_time_based_stats(self, trade: Dict):
        """Zaman bazlı istatistikleri güncelle"""
        date = trade['date']
        hour = trade['hour']
        
        if date not in self.daily_stats:
            self.daily_stats[date] = {
                'trades': 0,
                'wins': 0,
                'pnl': 0,
                'max_win': 0,
                'max_loss': float('inf')
            }
        
        if hour not in self.hourly_stats:
            self.hourly_stats[hour] = {
                'trades': 0,
                'wins': 0,
                'pnl': 0
            }
        
        # Daily update
        self.daily_stats[date]['trades'] += 1
        if trade['pnl'] > 0:
            self.daily_stats[date]['wins'] += 1
        self.daily_stats[date]['pnl'] += trade['pnl']
        self.daily_stats[date]['max_win'] = max(
            self.daily_stats[date]['max_win'],
            trade['pnl']
        )
        self.daily_stats[date]['max_loss'] = min(
            self.daily_stats[date]['max_loss'],
            trade['pnl']
        )
        
        # Hourly update
        self.hourly_stats[hour]['trades'] += 1
        if trade['pnl'] > 0:
            self.hourly_stats[hour]['wins'] += 1
        self.hourly_stats[hour]['pnl'] += trade['pnl']
    
    # ============= TEMEL METRİKLER =============
    
    def total_trades(self) -> int:
        """Toplam işlem sayısı"""
        return len(self.trades)
    
    def total_wins(self) -> int:
        """Toplam kazanç işlemi"""
        return sum(1 for t in self.trades if t['pnl'] > 0)
    
    def total_losses(self) -> int:
        """Toplam kayıp işlemi"""
        return sum(1 for t in self.trades if t['pnl'] < 0)
    
    def win_rate(self) -> float:
        """Kazanç oranı (0-1)"""
        total = self.total_trades()
        if total == 0:
            return 0
        return self.total_wins() / total
    
    def total_pnl(self) -> float:
        """Toplam kâr/zarar"""
        return sum(t['pnl'] for t in self.trades)
    
    def avg_pnl(self) -> float:
        """Ortalama işlem kâr/zarı"""
        total = self.total_trades()
        if total == 0:
            return 0
        return self.total_pnl() / total
    
    # ============= İLERİ METRİKLER =============
    
    def avg_win(self) -> float:
        """Ortalama kazanç işlemi"""
        wins = self.total_wins()
        if wins == 0:
            return 0
        return sum(t['pnl'] for t in self.trades if t['pnl'] > 0) / wins
    
    def avg_loss(self) -> float:
        """Ortalama kayıp işlemi"""
        losses = self.total_losses()
        if losses == 0:
            return 0
        return sum(t['pnl'] for t in self.trades if t['pnl'] < 0) / losses
    
    def profit_factor(self) -> float:
        """Kâr faktörü (toplam kazanç / toplam kayıp)"""
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss
    
    def expectancy(self) -> float:
        """Matematiksel beklenti"""
        win_rate = self.win_rate()
        avg_win = self.avg_win()
        avg_loss = self.avg_loss()
        
        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Sharpe oranı"""
        pnl_list = [t['pnl'] for t in self.trades]
        
        if len(pnl_list) < 2:
            return 0
        
        mean_return = np.mean(pnl_list)
        std_return = np.std(pnl_list)
        
        if std_return == 0:
            return 0
        
        return (mean_return - risk_free_rate) / std_return
    
    def sortino_ratio(self, target_return: float = 0.0) -> float:
        """Sortino oranı (sadece aşağı yönlü volatilite)"""
        pnl_list = [t['pnl'] for t in self.trades]
        
        if len(pnl_list) < 2:
            return 0
        
        mean_return = np.mean(pnl_list)
        downside_returns = [r - target_return for r in pnl_list if r < target_return]
        
        if not downside_returns:
            return 0
        
        downside_std = np.sqrt(np.mean(np.array(downside_returns) ** 2))
        
        if downside_std == 0:
            return 0
        
        return (mean_return - target_return) / downside_std
    
    def calmar_ratio(self) -> float:
        """Calmar oranı"""
        annual_return = self.total_pnl() * 252  # 252 trading days
        
        if self.max_drawdown == 0:
            return 0
        
        return annual_return / self.max_drawdown if self.max_drawdown != 0 else 0
    
    def max_consecutive_wins(self) -> int:
        """Maksimum ardışık kazanç"""
        return self.max_win_streak
    
    def max_consecutive_losses(self) -> int:
        """Maksimum ardışık kayıp"""
        return self.max_loss_streak
    
    def recovery_factor(self) -> float:
        """Kurtarma faktörü (Total PnL / Max Drawdown)"""
        if self.max_drawdown == 0:
            return 0
        return self.total_pnl() / self.max_drawdown
    
    # ============= ZAMAN BAZLI ANALİZ =============
    
    def best_trading_hours(self) -> List[tuple]:
        """En iyi ticaret saatlerini döndür (saat, kazanç, oran)"""
        results = []
        for hour, stats in self.hourly_stats.items():
            win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            results.append((hour, stats['pnl'], win_rate, stats['trades']))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def best_trading_days(self) -> List[tuple]:
        """En iyi ticaret günlerini döndür"""
        results = []
        for date, stats in self.daily_stats.items():
            win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            results.append((date, stats['pnl'], win_rate, stats['trades']))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def avg_duration(self) -> float:
        """Ortalama işlem süresi (saniye)"""
        if len(self.trades) == 0:
            return 0
        return np.mean([t['duration_sec'] for t in self.trades])
    
    def avg_win_duration(self) -> float:
        """Kazanç işlemlerinin ortalama süresi"""
        wins = [t['duration_sec'] for t in self.trades if t['pnl'] > 0]
        if not wins:
            return 0
        return np.mean(wins)
    
    def avg_loss_duration(self) -> float:
        """Kayıp işlemlerinin ortalama süresi"""
        losses = [t['duration_sec'] for t in self.trades if t['pnl'] < 0]
        if not losses:
            return 0
        return np.mean(losses)
    
    # ============= RAPOR =============
    
    def get_summary(self) -> Dict:
        """Kısa özet döndür"""
        return {
            'total_trades': self.total_trades(),
            'wins': self.total_wins(),
            'losses': self.total_losses(),
            'win_rate': self.win_rate(),
            'total_pnl': self.total_pnl(),
            'avg_pnl': self.avg_pnl(),
            'avg_win': self.avg_win(),
            'avg_loss': self.avg_loss(),
            'profit_factor': self.profit_factor(),
            'sharpe_ratio': self.sharpe_ratio(),
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'current_streak': self.current_streak,
            'max_win_streak': self.max_win_streak,
            'max_loss_streak': self.max_loss_streak,
        }
    
    def get_detailed_report(self) -> str:
        """Detaylı rapor döndür"""
        summary = self.get_summary()
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║                  📊 TİCARET İSTATİSTİKLERİ                ║
╚════════════════════════════════════════════════════════════╝

📈 TEMEL METRİKLER
├─ Toplam İşlem: {summary['total_trades']}
├─ Kazanç İşlemi: {summary['wins']}
├─ Kayıp İşlemi: {summary['losses']}
├─ Kazanç Oranı: {summary['win_rate']:.2%}
└─ Ortalama İşlem: {summary['avg_pnl']:+.2f}

💰 KÂR/ZARAR
├─ Toplam PnL: {summary['total_pnl']:+.2f}
├─ Ortalama Kazanç: {summary['avg_win']:+.2f}
├─ Ortalama Kayıp: {summary['avg_loss']:+.2f}
├─ Kâr Faktörü: {summary['profit_factor']:.2f}
└─ Beklenti: {self.expectancy():+.2f}

⚠️  RİSK METRİKLERİ
├─ Sharpe Oranı: {summary['sharpe_ratio']:.2f}
├─ Sortino Oranı: {self.sortino_ratio():.2f}
├─ Calmar Oranı: {self.calmar_ratio():.2f}
├─ Max Drawdown: {summary['max_drawdown']:+.2f}
├─ Current Drawdown: {summary['current_drawdown']:+.2f}
└─ Recovery Factor: {self.recovery_factor():.2f}

🔄 STREAKS
├─ Current Streak: {summary['current_streak']:+d}
├─ Max Win Streak: {summary['max_win_streak']}
└─ Max Loss Streak: {summary['max_loss_streak']}

⏱️  ZAMAN
├─ Ortalama Süre: {self.avg_duration():.0f}s
├─ Ort. Kazanç Süresi: {self.avg_win_duration():.0f}s
└─ Ort. Kayıp Süresi: {self.avg_loss_duration():.0f}s
"""
        return report
    
    def print_report(self):
        """Raporu yazdır"""
        print(self.get_detailed_report())
    
    def print_best_hours(self):
        """En iyi saatleri yazdır"""
        hours = self.best_trading_hours()
        print("\n🕐 EN İYİ TİCARET SAATLERİ")
        for hour, pnl, wr, trades in hours[:5]:
            print(f"  {hour:02d}:00 → PnL: {pnl:+.2f} | W/L: {wr:.1%} | İşlem: {trades}")
    
    def print_best_days(self):
        """En iyi günleri yazdır"""
        days = self.best_trading_days()
        print("\n📅 EN İYİ TİCARET GÜNÜ")
        for date, pnl, wr, trades in days[:5]:
            print(f"  {date} → PnL: {pnl:+.2f} | W/L: {wr:.1%} | İşlem: {trades}")
