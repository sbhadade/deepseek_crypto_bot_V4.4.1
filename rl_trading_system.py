"""
ENHANCED RL TRADING SYSTEM V2 - COMPLETE
With improved risk management, Kelly sizing, and multi-asset support
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade record"""
    timestamp: datetime
    asset: str
    action: str
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    holding_period_hours: float
    rsi: float
    macd: float
    volume_ratio: float
    volatility: float
    trend_strength: float
    success: bool
    max_adverse_move: float
    max_favorable_move: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None


@dataclass
class MarketState:
    """Market state for RL"""
    asset: str
    price: float
    rsi: float
    macd: float
    macd_signal: float
    ema_9: float
    ema_20: float
    ema_50: float
    volume_ratio: float
    volatility: float
    trend_strength: float
    order_flow: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to normalized feature vector"""
        return np.array([
            self.rsi / 100.0,
            np.clip((self.macd + 1) / 2, 0, 1),
            np.clip((self.macd_signal + 1) / 2, 0, 1),
            np.clip(self.volume_ratio / 3.0, 0, 2),
            np.clip(self.volatility * 10, 0, 1),
            np.clip((self.trend_strength + 1) / 2, 0, 1),
            np.clip((self.order_flow + 1) / 2, 0, 1),
            1.0 if self.ema_9 > self.ema_20 else 0.0,
            1.0 if self.ema_20 > self.ema_50 else 0.0,
        ])


@dataclass
class VolatilityState:
    """Volatility metrics"""
    symbol: str
    current_volatility: float
    volatility_mean: float
    volatility_std: float
    z_score: float
    is_extreme: bool
    regime: str
    percentile: float


@dataclass
class TrustScore:
    """Trust scoring for AI decisions"""
    decision_type: str
    market_regime: str
    predicted_confidence: float
    actual_accuracy: float
    trust_score: float
    sample_size: int
    adjustment_factor: float
    calibration_error: float


@dataclass
class ProbabilityScore:
    """Probability scoring with Kelly"""
    win_probability: float
    loss_probability: float
    expected_return: float
    expected_return_std: float
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    kelly_fraction: float
    half_kelly_fraction: float
    recommended_size_pct: float
    max_drawdown_estimate: float


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_exposure: float
    exposure_pct: float
    largest_position_pct: float
    correlation_score: float
    diversification_score: float
    heat_score: float
    concentration_risk: float


class VolatilityZScoreGate:
    """Gate #1: Advanced volatility filtering"""
    
    def __init__(self, window_size: int = 100, extreme_z: float = 2.5):
        self.window_size = window_size
        self.extreme_z = extreme_z
        self.volatility_history = {}
        self.load_history()
    
    def update_volatility(self, symbol: str, volatility: float):
        """Add volatility observation"""
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = deque(maxlen=self.window_size)
        
        self.volatility_history[symbol].append({
            'timestamp': datetime.now(),
            'volatility': volatility
        })
        
        if len(self.volatility_history[symbol]) % 10 == 0:
            self.save_history()
    
    def get_volatility_state(self, symbol: str, current_vol: float) -> VolatilityState:
        """Calculate comprehensive volatility metrics"""
        if symbol not in self.volatility_history or len(self.volatility_history[symbol]) < 10:
            return VolatilityState(
                symbol=symbol,
                current_volatility=current_vol,
                volatility_mean=current_vol,
                volatility_std=current_vol * 0.2,
                z_score=0.0,
                is_extreme=False,
                regime="unknown",
                percentile=50.0
            )
        
        vols = [v['volatility'] for v in self.volatility_history[symbol]]
        vol_mean = np.mean(vols)
        vol_std = np.std(vols)
        
        z_score = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0
        percentile = (np.sum(np.array(vols) < current_vol) / len(vols)) * 100
        
        # Enhanced regime classification
        if z_score > 3:
            regime = "crisis"
            is_extreme = True
        elif z_score > 2.5:
            regime = "extreme"
            is_extreme = True
        elif z_score > 1.5:
            regime = "elevated"
            is_extreme = False
        elif z_score < -1:
            regime = "calm"
            is_extreme = False
        else:
            regime = "normal"
            is_extreme = False
        
        return VolatilityState(
            symbol=symbol,
            current_volatility=current_vol,
            volatility_mean=vol_mean,
            volatility_std=vol_std,
            z_score=z_score,
            is_extreme=is_extreme,
            regime=regime,
            percentile=percentile
        )
    
    def save_history(self):
        """Save to disk"""
        try:
            data = {k: list(v) for k, v in self.volatility_history.items()}
            with open('volatility_history.json', 'w') as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save volatility history: {e}")
    
    def load_history(self):
        """Load from disk"""
        try:
            if os.path.exists('volatility_history.json'):
                with open('volatility_history.json', 'r') as f:
                    data = json.load(f)
                    for symbol, vols in data.items():
                        self.volatility_history[symbol] = deque(
                            [{'timestamp': datetime.fromisoformat(v['timestamp']), 
                              'volatility': v['volatility']} 
                             for v in vols],
                            maxlen=self.window_size
                        )
                logger.info(f"✅ Loaded volatility history for {len(self.volatility_history)} assets")
        except Exception as e:
            logger.error(f"Failed to load volatility history: {e}")
    
    def _find_similar_trades(self, state: MarketState, experience) -> List[TradeRecord]:
        """Find similar past trades using euclidean distance"""
        if not hasattr(experience, 'buffer') or not experience.buffer:
            return []
        
        state_vec = state.to_vector()
        similarities = []
        
        for trade in experience.buffer:
            if trade.asset != state.asset:
                continue
            
            trade_vec = np.array([
                trade.rsi / 100.0,
                np.clip((trade.macd + 1) / 2, 0, 1),
                0.5,
                np.clip(trade.volume_ratio / 3.0, 0, 2),
                np.clip(trade.volatility * 10, 0, 1),
                np.clip((trade.trend_strength + 1) / 2, 0, 1),
                0.5, 0.5, 0.5
            ])
            
            distance = np.linalg.norm(state_vec - trade_vec)
            similarities.append((trade, distance))
        
        similarities.sort(key=lambda x: x[1])
        return [trade for trade, _ in similarities[:50]]
    
    def _calculate_max_consecutive_losses(self, trades: List[TradeRecord]) -> int:
        """Calculate maximum consecutive losses"""
        if not trades:
            return 3
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if not trade.success:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max(max_consecutive, 3)


class ExperienceReplay:
    """Enhanced experience replay with analytics"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.save_path = "experience_replay.pkl"
        self.load()
    
    def add(self, trade: TradeRecord):
        """Add trade to buffer"""
        self.buffer.append(trade)
        if len(self.buffer) % 10 == 0:
            self.save()
    
    def save(self):
        try:
            with open(self.save_path, 'wb') as f:
                pickle.dump(list(self.buffer), f)
        except Exception as e:
            logger.error(f"Failed to save experience: {e}")
    
    def load(self):
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'rb') as f:
                    trades = pickle.load(f)
                    self.buffer.extend(trades)
                logger.info(f"✅ Loaded {len(self.buffer)} trades from experience")
        except Exception as e:
            logger.error(f"Failed to load experience: {e}")
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        if not self.buffer:
            return {}
        
        trades = list(self.buffer)
        wins = [t for t in trades if t.success]
        losses = [t for t in trades if not t.success]
        
        recent_trades = [t for t in trades if (datetime.now() - t.timestamp).days < 7]
        recent_wins = [t for t in recent_trades if t.success]
        
        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'total_pnl': sum(t.pnl for t in trades),
            'avg_win': np.mean([t.pnl for t in wins]) if wins else 0,
            'avg_loss': np.mean([t.pnl for t in losses]) if losses else 0,
            'best_trade': max([t.pnl for t in trades]) if trades else 0,
            'worst_trade': min([t.pnl for t in trades]) if trades else 0,
            'avg_holding_hours': np.mean([t.holding_period_hours for t in trades]) if trades else 0,
            'recent_7d_win_rate': len(recent_wins) / len(recent_trades) * 100 if recent_trades else 0,
            'sharpe_estimate': self._calculate_sharpe(trades),
            'max_drawdown': self._calculate_max_drawdown(trades)
        }
    
    def _calculate_sharpe(self, trades: List[TradeRecord]) -> float:
        """Calculate Sharpe ratio from trades"""
        if not trades:
            return 0.0
        
        returns = [t.pnl_pct / 100 for t in trades]
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return / std_return) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, trades: List[TradeRecord]) -> float:
        """Calculate maximum drawdown"""
        if not trades:
            return 0.0
        
        cumulative = 0
        peak = 0
        max_dd = 0
        
        for trade in sorted(trades, key=lambda x: x.timestamp):
            cumulative += trade.pnl_pct / 100
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative)
            max_dd = max(max_dd, drawdown)
        
        return max_dd


class AdaptiveStrategyLearner:
    """Enhanced adaptive strategy learner"""
    
    def __init__(self, asset: str):
        self.asset = asset
        self.experience = ExperienceReplay()
        self.strategy_performance = {}
        self.regime_performance = {}
        self.load_state()
    
    def get_best_action(self, state: MarketState, portfolio: Dict) -> Tuple[str, float]:
        """Get best action with regime awareness"""
        similar_trades = self._find_similar_trades(state)
        
        if len(similar_trades) < 5:
            return 'HOLD', 0.3
        
        # Separate by action
        buy_trades = [t for t in similar_trades if t.action in ['BUY', 'LONG']]
        sell_trades = [t for t in similar_trades if t.action in ['SELL', 'SHORT']]
        
        # Calculate success rates and expected returns
        buy_success = sum(1 for t in buy_trades if t.success)
        sell_success = sum(1 for t in sell_trades if t.success)
        
        buy_rate = buy_success / len(buy_trades) if buy_trades else 0
        sell_rate = sell_success / len(sell_trades) if sell_trades else 0
        
        buy_exp_return = np.mean([t.pnl_pct for t in buy_trades]) if buy_trades else 0
        sell_exp_return = np.mean([t.pnl_pct for t in sell_trades]) if sell_trades else 0
        
        # Decision with expected return consideration
        if buy_rate > 0.55 and buy_exp_return > 0.03:
            return 'BUY', min(buy_rate, 0.85)
        elif sell_rate > 0.55 and sell_exp_return > 0.03:
            return 'SELL', min(sell_rate, 0.85)
        elif buy_rate > sell_rate and buy_rate > 0.5:
            return 'BUY', buy_rate
        elif sell_rate > buy_rate and sell_rate > 0.5:
            return 'SELL', sell_rate
        else:
            return 'HOLD', 0.3
    
    def _find_similar_trades(self, state: MarketState) -> List[TradeRecord]:
        """Find similar historical trades"""
        if not self.experience.buffer:
            return []
        
        state_vec = state.to_vector()
        similarities = []
        
        for trade in self.experience.buffer:
            if trade.asset != state.asset:
                continue
            
            trade_vec = np.array([
                trade.rsi / 100.0,
                np.clip((trade.macd + 1) / 2, 0, 1),
                0.5,
                np.clip(trade.volume_ratio / 3.0, 0, 2),
                np.clip(trade.volatility * 10, 0, 1),
                np.clip((trade.trend_strength + 1) / 2, 0, 1),
                0.5, 0.5, 0.5
            ])
            
            distance = np.linalg.norm(state_vec - trade_vec)
            similarities.append((trade, distance))
        
        similarities.sort(key=lambda x: x[1])
        return [trade for trade, _ in similarities[:30]]
    
    def record_trade(self, trade: TradeRecord):
        """Record trade with regime tracking"""
        self.experience.add(trade)
        
        # Track regime performance
        regime_key = f"vol_{trade.volatility:.3f}"
        if regime_key not in self.regime_performance:
            self.regime_performance[regime_key] = []
        
        self.regime_performance[regime_key].append({
            'success': trade.success,
            'pnl_pct': trade.pnl_pct
        })
        
        self.save_state()
    
    def get_regime_stats(self) -> Dict:
        """Get performance by regime"""
        stats = {}
        for regime, trades in self.regime_performance.items():
            if len(trades) >= 3:
                successes = sum(1 for t in trades if t['success'])
                stats[regime] = {
                    'win_rate': successes / len(trades) * 100,
                    'avg_pnl': np.mean([t['pnl_pct'] for t in trades]),
                    'sample_size': len(trades)
                }
        return stats
    
    def save_state(self):
        """Save state"""
        try:
            state_data = {
                'strategy_performance': self.strategy_performance,
                'regime_performance': self.regime_performance
            }
            with open(f'strategy_learner_{self.asset}.json', 'w') as f:
                json.dump(state_data, f)
        except Exception as e:
            logger.error(f"Failed to save learner state: {e}")
    
    def load_state(self):
        """Load state"""
        try:
            filepath = f'strategy_learner_{self.asset}.json'
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    state_data = json.load(f)
                    self.strategy_performance = state_data.get('strategy_performance', {})
                    self.regime_performance = state_data.get('regime_performance', {})
        except Exception as e:
            logger.error(f"Failed to load learner state: {e}")


class MultiAssetManager:
    """Enhanced multi-asset manager with correlation tracking"""
    
    def __init__(self, assets: List[str]):
        self.assets = assets
        self.learners = {asset: AdaptiveStrategyLearner(asset) for asset in assets}
        self.correlation_matrix = {}
        self.asset_performance = {}
    
    def get_best_action(self, asset: str, state: MarketState, portfolio: Dict) -> Tuple[str, float]:
        """Get action for specific asset"""
        if asset not in self.learners:
            return 'HOLD', 0.3
        
        return self.learners[asset].get_best_action(state, portfolio)
    
    def record_trade(self, asset: str, trade: TradeRecord):
        """Record trade for asset"""
        if asset in self.learners:
            self.learners[asset].record_trade(trade)
            self._update_asset_performance(asset, trade)
    
    def _update_asset_performance(self, asset: str, trade: TradeRecord):
        """Track asset-specific performance"""
        if asset not in self.asset_performance:
            self.asset_performance[asset] = []
        
        self.asset_performance[asset].append({
            'timestamp': trade.timestamp,
            'pnl_pct': trade.pnl_pct,
            'success': trade.success
        })
        
        # Keep last 100 trades per asset
        if len(self.asset_performance[asset]) > 100:
            self.asset_performance[asset] = self.asset_performance[asset][-100:]
    
    def get_portfolio_risk(self, active_positions: Dict) -> PortfolioRisk:
        """Calculate portfolio-level risk"""
        if not active_positions:
            return PortfolioRisk(
                total_exposure=0,
                exposure_pct=0,
                largest_position_pct=0,
                correlation_score=0,
                diversification_score=1.0,
                heat_score=0,
                concentration_risk=0
            )
        
        # Calculate exposure
        position_sizes = [p['size'] * p.get('price', 0) for p in active_positions.values()]
        total_exposure = sum(position_sizes)
        largest_position = max(position_sizes) if position_sizes else 0
        
        # Diversification score (higher is better)
        num_positions = len(active_positions)
        diversification = min(1.0, num_positions / 5.0)
        
        # Concentration risk
        concentration = largest_position / total_exposure if total_exposure > 0 else 0
        
        return PortfolioRisk(
            total_exposure=total_exposure,
            exposure_pct=0,  # Set by caller
            largest_position_pct=concentration * 100,
            correlation_score=0.5,  # Placeholder
            diversification_score=diversification,
            heat_score=total_exposure / 1000,  # Placeholder
            concentration_risk=concentration
        )
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all assets"""
        stats = {}
        for asset, learner in self.learners.items():
            exp_stats = learner.experience.get_stats()
            if exp_stats:
                stats[asset] = exp_stats
        return stats


class MetaRLTrustLayer:
    """Gate #2: Enhanced trust calibration"""
    
    def __init__(self):
        self.decision_history = {}
        self.load_state()
    
    def evaluate_decision(
        self,
        action: str,
        market_regime: str,
        predicted_confidence: float,
        market_state: MarketState
    ) -> TrustScore:
        """Evaluate trust with calibration metrics"""
        
        key = f"{action}_{market_regime}"
        
        if key not in self.decision_history or len(self.decision_history[key]) < 5:
            # Insufficient history - be conservative
            return TrustScore(
                decision_type=action,
                market_regime=market_regime,
                predicted_confidence=predicted_confidence,
                actual_accuracy=0.5,
                trust_score=0.5,
                sample_size=0,
                adjustment_factor=0.7,  # Very conservative
                calibration_error=0.0
            )
        
        history = self.decision_history[key]
        
        # Calculate metrics
        total = len(history)
        successes = sum(1 for h in history if h['success'])
        actual_accuracy = successes / total
        
        # Calculate calibration error
        avg_predicted = np.mean([h['predicted_confidence'] for h in history])
        calibration_error = abs(actual_accuracy - avg_predicted / 100)
        
        # Trust score (well-calibrated = high trust)
        trust_score = max(0, 1 - calibration_error * 2.5)
        
        # Adjustment factor with more granularity
        if actual_accuracy > 0.65:
            adjustment_factor = min(1.3, 1.0 + (actual_accuracy - 0.5) * 1.5)
        elif actual_accuracy > 0.55:
            adjustment_factor = 1.0 + (actual_accuracy - 0.55) * 0.5
        elif actual_accuracy > 0.45:
            adjustment_factor = 0.9 + (actual_accuracy - 0.45)
        elif actual_accuracy > 0.35:
            adjustment_factor = max(0.6, actual_accuracy / 0.5)
        else:
            adjustment_factor = 0.5  # Severe penalty
        
        return TrustScore(
            decision_type=action,
            market_regime=market_regime,
            predicted_confidence=predicted_confidence,
            actual_accuracy=actual_accuracy,
            trust_score=trust_score,
            sample_size=total,
            adjustment_factor=adjustment_factor,
            calibration_error=calibration_error
        )
    
    def record_outcome(
        self,
        action: str,
        market_regime: str,
        predicted_confidence: float,
        actual_success: bool,
        pnl_pct: float = 0.0
    ):
        """Record outcome with P&L"""
        key = f"{action}_{market_regime}"
        
        if key not in self.decision_history:
            self.decision_history[key] = []
        
        self.decision_history[key].append({
            'timestamp': datetime.now(),
            'predicted_confidence': predicted_confidence,
            'success': actual_success,
            'pnl_pct': pnl_pct
        })
        
        # Keep last 150 decisions for better statistics
        if len(self.decision_history[key]) > 150:
            self.decision_history[key] = self.decision_history[key][-150:]
        
        self.save_state()
    
    def get_performance_summary(self) -> Dict:
        """Get performance across all decision types"""
        summary = {}
        for key, history in self.decision_history.items():
            if len(history) >= 5:
                successes = sum(1 for h in history if h['success'])
                summary[key] = {
                    'sample_size': len(history),
                    'win_rate': successes / len(history) * 100,
                    'avg_pnl': np.mean([h['pnl_pct'] for h in history])
                }
        return summary
    
    def save_state(self):
        """Save to disk"""
        try:
            with open('trust_layer_state.json', 'w') as f:
                json.dump(self.decision_history, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save trust state: {e}")
    
    def load_state(self):
        """Load from disk"""
        try:
            if os.path.exists('trust_layer_state.json'):
                with open('trust_layer_state.json', 'r') as f:
                    data = json.load(f)
                    self.decision_history = {
                        k: [{'timestamp': datetime.fromisoformat(d['timestamp']),
                             'predicted_confidence': d['predicted_confidence'],
                             'success': d['success'],
                             'pnl_pct': d.get('pnl_pct', 0.0)}
                            for d in v]
                        for k, v in data.items()
                    }
                logger.info(f"✅ Loaded trust history: {len(self.decision_history)} decision types")
        except Exception as e:
            logger.error(f"Failed to load trust state: {e}")


class ProbabilityScorer:
    """Gate #3: Enhanced probability scoring with Kelly"""
    
    def calculate_probability(
        self,
        market_state: MarketState,
        ai_decision,
        experience_replay
    ) -> ProbabilityScore:
        """Calculate comprehensive probability metrics"""
        
        similar_trades = self._find_similar_trades(market_state, experience_replay)
        
        if len(similar_trades) < 10:
            # Insufficient data - conservative defaults
            return ProbabilityScore(
                win_probability=0.5,
                loss_probability=0.5,
                expected_return=0.0,
                expected_return_std=0.05,
                value_at_risk_95=0.03,
                value_at_risk_99=0.05,
                conditional_var_95=0.04,
                kelly_fraction=0.05,
                half_kelly_fraction=0.025,
                recommended_size_pct=10.0,
                max_drawdown_estimate=0.15
            )
        
        # Calculate statistics
        wins = [t for t in similar_trades if t.success]
        losses = [t for t in similar_trades if not t.success]
        
        win_prob = len(wins) / len(similar_trades)
        loss_prob = 1 - win_prob
        
        avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss_pct = np.mean([abs(t.pnl_pct) for t in losses]) if losses else 0
        
        # Expected return
        expected_return = win_prob * avg_win_pct - loss_prob * avg_loss_pct
        expected_return_std = np.std([t.pnl_pct for t in similar_trades])
        
        # VaR calculations
        all_returns = sorted([t.pnl_pct for t in similar_trades])
        var_95_idx = int(len(all_returns) * 0.05)
        var_99_idx = int(len(all_returns) * 0.01)
        
        var_95 = abs(all_returns[var_95_idx]) if len(all_returns) > 20 else 0.03
        var_99 = abs(all_returns[var_99_idx]) if len(all_returns) > 100 else 0.05
        
        # Conditional VaR (expected loss beyond VaR)
        tail_losses = [abs(r) for r in all_returns[:var_95_idx]] if var_95_idx > 0 else [0.03]
        conditional_var = np.mean(tail_losses) if tail_losses else var_95
        
        # Kelly Criterion (fractional)
        if avg_loss_pct > 0 and win_prob > 0:
            kelly = (win_prob * avg_win_pct - loss_prob * avg_loss_pct) / avg_loss_pct
            kelly = max(0, min(kelly, 0.35))  # Cap at 35%
        else:
            kelly = 0.10
        
        half_kelly = kelly * 0.5
        
        # Recommended size with additional safety
        if win_prob >= 0.55 and expected_return > 0.02:
            recommended_pct = half_kelly * 100
        elif win_prob >= 0.50:
            recommended_pct = half_kelly * 0.7 * 100
        else:
            recommended_pct = 10.0  # Minimum size
        
        recommended_pct = max(10, min(recommended_pct, 30))
        
        # Max drawdown estimate
        consecutive_losses = self._calculate_max_consecutive_losses(similar_trades)
        max_dd_estimate = min(consecutive_losses * avg_loss_pct, 0.30)
        
        return ProbabilityScore(
            win_probability=win_prob,
            loss_probability=loss_prob,
            expected_return=expected_return,
            expected_return_std=expected_return_std,
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            conditional_var_95=conditional_var,
            kelly_fraction=kelly,
            half_kelly_fraction=half_kelly,
            recommended_size_pct=recommended_pct,
            max_drawdown_estimate=max_dd_estimate
        )
    
    def _find_similar_trades(self, state: MarketState, experience) -> List[TradeRecord]:
        """Find similar past trades"""
        if not hasattr(experience, 'buffer') or not experience.buffer:
            return []
        
        state_vec = state.to_vector()
        similarities = []
        
        for trade in experience.buffer:
            if trade.asset != state.asset:
                continue
            
            trade_vec = np.array([
                trade.rsi / 100.0,
                np.clip((trade.macd + 1) / 2, 0, 1),
                0.5,
                np.clip(trade.volume_ratio / 3.0, 0, 2),
                np.clip(trade.volatility * 10, 0, 1),
                np.clip((trade.trend_strength + 1) / 2, 0, 1),
                0.5, 0.5, 0.5
            ])
            
            distance = np.linalg.norm(state_vec - trade_vec)
            similarities.append((trade, distance))
        
        similarities.sort(key=lambda x: x[1])
        return [trade for trade, _ in similarities[:50]]
    
    def _calculate_max_consecutive_losses(self, trades: List[TradeRecord]) -> int:
        """Calculate maximum consecutive losses"""
        if not trades:
            return 3
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if not trade.success:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max(max_consecutive, 3)


class RLTradingSupervisor:
    """Enhanced supervisor with better risk management"""
    
    def __init__(self, initial_balance: float = 25.0, assets: List[str] = None):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Default to expanded asset list
        if assets is None:
            assets = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
        
        self.multi_asset_manager = MultiAssetManager(assets)
        
        self.state = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'daily_pnl': 0.0,
            'weekly_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': initial_balance,
            'consecutive_losses': 0,
            'consecutive_wins': 0,
            'last_reset_date': datetime.now().date()
        }
        self.load_supervisor_state()
    
    def update_balance(self, pnl: float, is_win: bool):
        """Update balance with streak tracking"""
        self.current_balance += pnl
        self.state['daily_pnl'] += pnl
        self.state['weekly_pnl'] += pnl
        self.state['total_trades'] += 1
        
        if is_win:
            self.state['winning_trades'] += 1
            self.state['consecutive_wins'] += 1
            self.state['consecutive_losses'] = 0
        else:
            self.state['losing_trades'] += 1
            self.state['consecutive_losses'] += 1
            self.state['consecutive_wins'] = 0
        
        # Update peak and drawdown
        if self.current_balance > self.state['peak_balance']:
            self.state['peak_balance'] = self.current_balance
        
        drawdown = (self.state['peak_balance'] - self.current_balance) / self.state['peak_balance']
        self.state['max_drawdown'] = max(self.state['max_drawdown'], drawdown)
        
        # Daily reset
        if datetime.now().date() > self.state['last_reset_date']:
            self.state['daily_pnl'] = 0.0
            self.state['last_reset_date'] = datetime.now().date()
        
        self.save_supervisor_state()
    
    def get_position_size_multiplier(self) -> float:
        """Dynamic position sizing based on performance"""
        win_rate = (self.state['winning_trades'] / self.state['total_trades'] 
                   if self.state['total_trades'] > 0 else 0.5)
        
        # Streak adjustments
        if self.state['consecutive_losses'] >= 3:
            return 0.5  # Reduce size after losing streak
        elif self.state['consecutive_wins'] >= 3:
            return 1.2  # Increase size after winning streak
        
        # Win rate adjustments
        if win_rate > 0.65:
            return 1.3
        elif win_rate > 0.55:
            return 1.1
        elif win_rate < 0.40:
            return 0.7
        elif win_rate < 0.35:
            return 0.5
        else:
            return 1.0
    
    def check_safety_limits(self) -> Tuple[bool, str]:
        """Enhanced safety checks"""
        # Balance check
        if self.current_balance < self.initial_balance * 0.5:
            return False, "Balance below 50% of initial - HALT TRADING"
        
        # Drawdown check
        if self.state['max_drawdown'] > 0.30:
            return False, "Max drawdown exceeded 30% - HALT TRADING"
        
        # Daily loss limit
        if self.state['daily_pnl'] < -self.initial_balance * 0.10:
            return False, "Daily loss limit reached (-10%) - HALT TRADING"
        
        # Consecutive losses
        if self.state['consecutive_losses'] >= 5:
            return False, "5 consecutive losses - HALT TRADING"
        
        # Weekly loss limit
        if self.state['weekly_pnl'] < -self.initial_balance * 0.20:
            return False, "Weekly loss limit reached (-20%) - HALT TRADING"
        
        return True, "All safety checks passed"
    
    def get_supervisor_report(self) -> str:
        """Enhanced status report"""
        win_rate = (self.state['winning_trades'] / self.state['total_trades'] * 100 
                   if self.state['total_trades'] > 0 else 0)
        
        roi = (self.current_balance / self.initial_balance - 1) * 100
        
        return f"""
📊 SUPERVISOR STATUS REPORT
{'='*60}
Balance: ${self.current_balance:.2f} ({roi:+.1f}% ROI)
Peak: ${self.state['peak_balance']:.2f}
Max Drawdown: {self.state['max_drawdown']*100:.1f}%

Performance:
  Total Trades: {self.state['total_trades']}
  Wins: {self.state['winning_trades']} | Losses: {self.state['losing_trades']}
  Win Rate: {win_rate:.1f}%
  
P&L:
  Today: ${self.state['daily_pnl']:+.2f}
  Week: ${self.state['weekly_pnl']:+.2f}

Streaks:
  Consecutive Wins: {self.state['consecutive_wins']}
  Consecutive Losses: {self.state['consecutive_losses']}
  
Position Multiplier: {self.get_position_size_multiplier():.2f}x
{'='*60}
"""
    
    def save_supervisor_state(self):
        """Save state"""
        try:
            state_data = {
                'current_balance': self.current_balance,
                'state': {k: (v.isoformat() if isinstance(v, datetime.date) else v) 
                         for k, v in self.state.items()}
            }
            with open('supervisor_state.json', 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save supervisor state: {e}")
    
    def load_supervisor_state(self):
        """Load state"""
        try:
            if os.path.exists('supervisor_state.json'):
                with open('supervisor_state.json', 'r') as f:
                    state_data = json.load(f)
                    self.current_balance = state_data.get('current_balance', self.initial_balance)
                    loaded_state = state_data.get('state', {})
                    
                    # Convert date string back
                    if 'last_reset_date' in loaded_state:
                        loaded_state['last_reset_date'] = datetime.fromisoformat(
                            loaded_state['last_reset_date']
                        ).date()
                    
                    self.state.update(loaded_state)
                logger.info(f"✅ Loaded supervisor: Balance ${self.current_balance:.2f}")
        except Exception as e:
            logger.error(f"Failed to load supervisor state: {e}")