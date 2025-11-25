"""
META-RL ADAPTIVE SUPERVISOR V5 - WITH TIMEFRAME-SPECIFIC LEARNING
✅ FIXED: Added missing parameters (volatility_z_threshold, expected_value_threshold)
✅ FIXED: Proper optimizer initialization for all parameters
✅ FIXED: Safe division in reward calculation
Enhanced with continuous evolutionary training and timeframe specialization
"""

import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
import pickle
import asyncio
import os

logger = logging.getLogger(__name__)


@dataclass
class ParameterState:
    """Tracks a single adaptive parameter"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    performance_history: List[float]
    recent_win_rate: float
    recent_sharpe: float
    last_update: datetime
    exploration_rate: float


@dataclass
class TimeframeSpecificState:
    """Tracks state for each timeframe"""
    timeframe: str
    optimizers: Dict
    regime_configs: Dict
    performance_history: List[float]
    recent_win_rate: float = 0.0
    recent_sharpe: float = 0.0


class AdaptiveParameterOptimizer:
    """Uses contextual bandits to optimize individual parameters"""
    
    def __init__(self, param_name: str, min_val: float, max_val: float, step: float):
        self.param_name = param_name
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        
        self.values = np.arange(min_val, max_val + step, step)
        self.n_arms = len(self.values)
        
        self.arm_counts = np.zeros(self.n_arms)
        self.arm_rewards = np.zeros(self.n_arms)
        self.arm_sharpe = np.zeros(self.n_arms)
        self.arm_trade_counts = np.zeros(self.n_arms)
        
        self.current_arm_idx = self.n_arms // 2
        self.exploration_rate = 0.2
        
        logger.debug(f"⚙️ Optimizer initialized for {param_name}: [{min_val}, {max_val}], step={step}")
    
    def get_value(self, force_exploit: bool = False) -> float:
        """Get current parameter value with epsilon-greedy exploration"""
        
        if not force_exploit and np.random.random() < self.exploration_rate:
            self.current_arm_idx = np.random.randint(0, self.n_arms)
        else:
            if np.any(self.arm_counts == 0):
                self.current_arm_idx = np.where(self.arm_counts == 0)[0][0]
            else:
                avg_rewards = self.arm_rewards / self.arm_counts
                exploration_bonus = np.sqrt(2 * np.log(np.sum(self.arm_counts)) / self.arm_counts)
                ucb_scores = avg_rewards + exploration_bonus
                self.current_arm_idx = np.argmax(ucb_scores)
        
        return self.values[self.current_arm_idx]
    
    def update_reward(self, reward: float, sharpe: float, traded: bool):
        """Update parameter performance based on outcome"""
        self.arm_counts[self.current_arm_idx] += 1
        self.arm_rewards[self.current_arm_idx] += reward
        
        if sharpe is not None and not np.isnan(sharpe):
            alpha = 0.3
            self.arm_sharpe[self.current_arm_idx] = (
                alpha * sharpe + (1 - alpha) * self.arm_sharpe[self.current_arm_idx]
            )
        
        if traded:
            self.arm_trade_counts[self.current_arm_idx] += 1

    def _calculate_arm_rewards(self):
        """✅ FIXED: Calculate average rewards with safe division"""
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_rewards = np.where(
                self.arm_counts > 0, 
                np.divide(self.arm_rewards, self.arm_counts, where=self.arm_counts > 0), 
                0
            )
            avg_rewards = np.nan_to_num(avg_rewards, nan=0.0, posinf=0.0, neginf=0.0)
        return avg_rewards
        
    def get_stats(self) -> Dict:
        """Get statistics about parameter optimization"""
        if np.sum(self.arm_counts) == 0:
            return {}
        
        avg_rewards = self._calculate_arm_rewards()
        best_arm = np.argmax(avg_rewards)
        
        return {
            'param_name': self.param_name,
            'current_value': self.values[self.current_arm_idx],
            'best_value': self.values[best_arm],
            'best_avg_reward': avg_rewards[best_arm],
            'total_trials': int(np.sum(self.arm_counts)),
            'exploration_rate': self.exploration_rate,
            'arms_tried': int(np.sum(self.arm_counts > 0)),
            'total_arms': self.n_arms
        }


class MetaRLSupervisorV5:
    """
    THE BIG BRAIN V5 - Enhanced with timeframe-specific learning
    ✅ FIXED: All missing parameters added
    ✅ FIXED: Proper initialization for all timeframes
    """
    
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        
        # Timeframe-specific learning systems
        self.timeframe_states = {
            'short_term': self._initialize_timeframe_state('short_term'),
            'mid_term': self._initialize_timeframe_state('mid_term'), 
            'long_term': self._initialize_timeframe_state('long_term')
        }
        
        # Global performance tracking
        self.trade_history = deque(maxlen=1000)
        self.parameter_history = deque(maxlen=500)
        
        # Meta-learning state
        self.total_cycles = 0
        self.cycles_since_update = 0
        self.update_frequency = 20
        
        # Performance windows
        self.short_window = deque(maxlen=10)
        self.medium_window = deque(maxlen=50)
        self.long_window = deque(maxlen=200)
        
        # Continuous learning metrics
        self.trades_from_evolutionary = 0
        self.last_evolutionary_update = datetime.now()
        
        self.load_state()
        
        logger.info("🧠 META-RL SUPERVISOR V5 INITIALIZED")
        logger.info(f"   Timeframe Specialization: {list(self.timeframe_states.keys())}")
        logger.info(f"   Continuous Learning: ENABLED")
    
    def _initialize_timeframe_state(self, timeframe: str) -> TimeframeSpecificState:
        """✅ FIXED: Initialize optimizers with ALL required parameters"""
        
        # Timeframe-specific parameter ranges
        ranges = {
            'short_term': {
                'min_confidence': (45.0, 70.0),
                'min_win_prob': (0.40, 0.60),
                'stop_loss_distance': (0.005, 0.025),
                'take_profit_distance': (0.010, 0.050),
                'max_holding_hours': (0.5, 4.0),
                'volatility_z_threshold': (2.5, 4.0),  # ✅ ADDED
                'expected_value_threshold': (0.005, 0.020)  # ✅ ADDED
            },
            'mid_term': {
                'min_confidence': (50.0, 75.0),
                'min_win_prob': (0.45, 0.65), 
                'stop_loss_distance': (0.015, 0.040),
                'take_profit_distance': (0.030, 0.100),
                'max_holding_hours': (4.0, 24.0),
                'volatility_z_threshold': (2.0, 3.5),  # ✅ ADDED
                'expected_value_threshold': (0.010, 0.025)  # ✅ ADDED
            },
            'long_term': {
                'min_confidence': (55.0, 80.0),
                'min_win_prob': (0.50, 0.70),
                'stop_loss_distance': (0.025, 0.080),
                'take_profit_distance': (0.050, 0.200),
                'max_holding_hours': (24.0, 168.0),
                'volatility_z_threshold': (2.0, 3.0),  # ✅ ADDED
                'expected_value_threshold': (0.015, 0.030)  # ✅ ADDED
            }
        }
        
        timeframe_range = ranges[timeframe]
        
        # ✅ FIXED: Create ALL optimizers including missing ones
        optimizers = {
            'min_confidence': AdaptiveParameterOptimizer(
                'min_confidence', 
                timeframe_range['min_confidence'][0],
                timeframe_range['min_confidence'][1], 
                2.5
            ),
            'min_win_prob': AdaptiveParameterOptimizer(
                'min_win_prob',
                timeframe_range['min_win_prob'][0],
                timeframe_range['min_win_prob'][1],
                0.02
            ),
            'stop_loss_distance': AdaptiveParameterOptimizer(
                'stop_loss_distance',
                timeframe_range['stop_loss_distance'][0], 
                timeframe_range['stop_loss_distance'][1],
                0.005
            ),
            'take_profit_distance': AdaptiveParameterOptimizer(
                'take_profit_distance',
                timeframe_range['take_profit_distance'][0],
                timeframe_range['take_profit_distance'][1], 
                0.01
            ),
            'max_holding_hours': AdaptiveParameterOptimizer(
                'max_holding_hours',
                timeframe_range['max_holding_hours'][0],
                timeframe_range['max_holding_hours'][1],
                2.0
            ),
            # ✅ ADDED: Missing optimizers
            'volatility_z_threshold': AdaptiveParameterOptimizer(
                'volatility_z_threshold',
                timeframe_range['volatility_z_threshold'][0],
                timeframe_range['volatility_z_threshold'][1],
                0.25
            ),
            'expected_value_threshold': AdaptiveParameterOptimizer(
                'expected_value_threshold',
                timeframe_range['expected_value_threshold'][0],
                timeframe_range['expected_value_threshold'][1],
                0.005
            )
        }
        
        return TimeframeSpecificState(
            timeframe=timeframe,
            optimizers=optimizers,
            regime_configs=self._default_regime_configs(timeframe),
            performance_history=[]
        )
    
    def _default_regime_configs(self, timeframe: str) -> Dict:
        """Create regime configs for timeframe"""
        base_configs = {
            'bull_strong': {'position_size_multiplier': 1.2, 'risk_tolerance': 1.1},
            'bull_weak': {'position_size_multiplier': 1.0, 'risk_tolerance': 1.0},
            'bear_strong': {'position_size_multiplier': 0.8, 'risk_tolerance': 0.9},
            'bear_weak': {'position_size_multiplier': 0.9, 'risk_tolerance': 0.95},
            'ranging': {'position_size_multiplier': 1.0, 'risk_tolerance': 1.0},
            'high_volatility': {'position_size_multiplier': 0.7, 'risk_tolerance': 0.8},
            'crash': {'position_size_multiplier': 0.5, 'risk_tolerance': 0.6}
        }
        
        timeframe_multipliers = {
            'short_term': 1.0,
            'mid_term': 0.9,
            'long_term': 0.8
        }
        
        multiplier = timeframe_multipliers[timeframe]
        configs = {}
        
        for regime, config in base_configs.items():
            configs[regime] = {
                'position_size_multiplier': config['position_size_multiplier'] * multiplier,
                'risk_tolerance': config['risk_tolerance'] * multiplier
            }
        
        return configs
    
    def get_adaptive_parameters(self, market_regime: str, timeframe: str = 'mid_term') -> Dict:
        """✅ FIXED: Get timeframe-specific parameters with ALL required fields"""
        if timeframe not in self.timeframe_states:
            timeframe = 'mid_term'
        
        state = self.timeframe_states[timeframe]
        regime_config = state.regime_configs.get(market_regime, {})
        
        # Get base parameters
        base_ev_threshold = state.optimizers['expected_value_threshold'].get_value()
        
        # 🔒 PATCH: Adaptive EV threshold (lowers if agents are idle)
        recent_trades = [t for t in self.trade_history if t.get('timeframe') == timeframe][-20:]
        
        if len(recent_trades) < 5:
            # Very few trades - lower threshold significantly
            adjusted_ev_threshold = base_ev_threshold * 0.5
            logger.debug(f"   📉 EV threshold lowered to {adjusted_ev_threshold*100:.2f}% (low activity)")
        elif len(recent_trades) < 10:
            # Some trades - lower threshold moderately
            adjusted_ev_threshold = base_ev_threshold * 0.7
        else:
            adjusted_ev_threshold = base_ev_threshold
        
        params = {
            # Entry thresholds
            'min_confidence': state.optimizers['min_confidence'].get_value(),
            'min_win_prob': state.optimizers['min_win_prob'].get_value(),
            
            # Exit strategy
            'stop_loss_distance': state.optimizers['stop_loss_distance'].get_value(),
            'take_profit_distance': state.optimizers['take_profit_distance'].get_value(),
            'max_holding_hours': state.optimizers['max_holding_hours'].get_value(),
            
            # Risk management with ADAPTIVE threshold
            'volatility_z_threshold': state.optimizers['volatility_z_threshold'].get_value(),
            'expected_value_threshold': adjusted_ev_threshold,  # 🔒 PATCHED
            'risk_reward_threshold': 2.0,
            
            # Position sizing
            'position_size_multiplier': regime_config.get('position_size_multiplier', 1.0),
            'risk_tolerance': regime_config.get('risk_tolerance', 1.0),
            
            # Metadata
            'timeframe': timeframe,
            'source': f'meta_rl_v5_{timeframe}'
        }
        
        self.parameter_history.append({
            'timestamp': datetime.now(),
            'timeframe': timeframe,
            'regime': market_regime,
            'params': params.copy()
        })
        
        return params
    
    def record_trade_outcome(
        self, 
        traded: bool, 
        action: str, 
        market_regime: str,
        pnl: float, 
        pnl_pct: float, 
        success: bool,
        confidence_used: float,
        win_prob_used: float,
        parameters_used: Dict,
        timeframe: str = 'mid_term'
    ):
        """Record trade outcome with timeframe-specific learning"""
        
        trade_record = {
            'timestamp': datetime.now(),
            'traded': traded,
            'action': action,
            'regime': market_regime,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'success': success,
            'confidence': confidence_used,
            'win_prob': win_prob_used,
            'params': parameters_used,
            'timeframe': timeframe
        }
        
        self.trade_history.append(trade_record)
        
        if traded:
            self.trades_from_evolutionary += 1
            self.short_window.append(trade_record)
            self.medium_window.append(trade_record)
            self.long_window.append(trade_record)
        
        # Timeframe-specific learning
        if timeframe in self.timeframe_states:
            reward = self._calculate_reward(trade_record)
            sharpe = self._calculate_recent_sharpe(window_size=20)
            
            state = self.timeframe_states[timeframe]
            for optimizer in state.optimizers.values():
                optimizer.update_reward(reward, sharpe, traded)
            
            state.performance_history.append(reward)
            if len(state.performance_history) > 100:
                state.performance_history.pop(0)
            
            recent_trades = [t for t in self.trade_history if t.get('timeframe') == timeframe][-20:]
            if recent_trades:
                state.recent_win_rate = sum(1 for t in recent_trades if t['success']) / len(recent_trades)
                state.recent_sharpe = self._calculate_timeframe_sharpe(timeframe)
        
        self.total_cycles += 1
        self.cycles_since_update += 1
        
        # More frequent updates during continuous learning
        if self.trades_from_evolutionary > 0 and self.cycles_since_update >= 10:
            self._meta_update()
            self.cycles_since_update = 0
    
    def learn_optimal_exit_strategy(
        self,
        trade_duration: float,
        final_pnl: float,
        max_favorable: float,
        max_adverse: float,
        exit_reason: str,
        parameters_used: Dict,
        timeframe: str = 'mid_term'
    ):
        """Timeframe-specific exit strategy learning"""
        if timeframe not in self.timeframe_states:
            return
        
        exit_reward = self._calculate_exit_reward(
            trade_duration, final_pnl, max_favorable, max_adverse, exit_reason
        )
        
        state = self.timeframe_states[timeframe]
        
        if exit_reason == "stop_loss":
            if max_favorable > 0.05 and final_pnl < 0:
                state.optimizers['stop_loss_distance'].update_reward(-1.0, 0, True)
        elif exit_reason == "take_profit":
            if max_favorable > final_pnl * 1.5:
                state.optimizers['take_profit_distance'].update_reward(-0.5, 0, True)
        
        if max_favorable > parameters_used.get('trailing_stop_activation', 0.03) * 2:
            state.optimizers['take_profit_distance'].update_reward(0.2, 0, True)
    
    def _calculate_timeframe_sharpe(self, timeframe: str, window_size: int = 20) -> float:
        """
        ✅ PATCHED: Timeframe-specific Sharpe with same protections
        """
        
        timeframe_trades = [t for t in self.trade_history 
                           if t.get('timeframe') == timeframe and t['traded']][-window_size:]
        
        if len(timeframe_trades) < 5:
            return 0.0
        
        # ✅ Same validation as _calculate_recent_sharpe
        returns = []
        for t in timeframe_trades:
            pnl_pct = t.get('pnl_pct', 0.0)
            if not np.isnan(pnl_pct) and not np.isinf(pnl_pct):
                returns.append(np.clip(pnl_pct, -100.0, 100.0))
        
        if len(returns) < 5:
            return 0.0
        
        returns_array = np.array(returns)
        lower_bound = np.percentile(returns_array, 2.5)
        upper_bound = np.percentile(returns_array, 97.5)
        returns_winsorized = np.clip(returns_array, lower_bound, upper_bound)
        
        mean_return = np.mean(returns_winsorized)
        std_return = np.std(returns_winsorized)
        
        EPSILON = 1e-8
        
        if std_return < EPSILON:
            if mean_return > 0:
                return 2.0
            elif mean_return < 0:
                return -2.0
            else:
                return 0.0
        
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        if np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        
        return float(np.clip(sharpe, -10.0, 10.0))

    def _calculate_reward(self, trade: Dict) -> float:
        """
        Enhanced reward calculation
        ✅ FIXED: NaN and Inf protection
        """
        if not trade['traded']:
            recent_trades = [t for t in self.short_window if t['traded']]
            if len(recent_trades) < 3:
                return -0.5
            return 0.0
        
        # ✅ FIX #8: Validate pnl_pct before using
        pnl_pct = trade.get('pnl_pct', 0.0)
        
        # Check for invalid values
        if np.isnan(pnl_pct) or np.isinf(pnl_pct):
            logger.warning(f"⚠️ Invalid pnl_pct detected: {pnl_pct}, defaulting to 0")
            pnl_pct = 0.0
        
        # Clip extreme values (prevent reward explosion)
        pnl_pct = np.clip(pnl_pct, -100.0, 100.0)
        
        reward = pnl_pct * 10
        
        # ✅ Additional validation after calculation
        if np.isnan(reward) or np.isinf(reward):
            logger.warning(f"⚠️ Invalid reward calculated: {reward}, defaulting to -1.0")
            return -1.0
        
        if trade['success']:
            reward += 1.0
        else:
            reward -= 0.5
        
        timeframe = trade.get('timeframe', 'mid_term')
        if timeframe == 'short_term':
            if trade['success']:
                reward += 0.5
        elif timeframe == 'long_term':
            if pnl_pct > 5.0:
                reward += 2.0
        
        # ✅ Final validation before return
        reward = float(np.clip(reward, -1000.0, 1000.0))
        
        if np.isnan(reward) or np.isinf(reward):
            logger.error(f"❌ Final reward still invalid, returning -1.0")
            return -1.0
        
        return reward
    
    def _calculate_exit_reward(self, duration: float, final_pnl: float, max_fav: float, max_adv: float, exit_reason: str) -> float:
        """
        Enhanced exit reward
        ✅ FIXED: NaN and Inf protection
        """
        # ✅ FIX #8: Validate all inputs
        duration = float(duration) if not np.isnan(duration) and not np.isinf(duration) else 0.0
        final_pnl = float(final_pnl) if not np.isnan(final_pnl) and not np.isinf(final_pnl) else 0.0
        max_fav = float(max_fav) if not np.isnan(max_fav) and not np.isinf(max_fav) else 0.0
        max_adv = float(max_adv) if not np.isnan(max_adv) and not np.isinf(max_adv) else 0.0
        
        # Clip extreme values
        final_pnl = np.clip(final_pnl, -100.0, 100.0)
        max_fav = np.clip(max_fav, -100.0, 100.0)
        max_adv = np.clip(max_adv, -100.0, 100.0)
        
        reward = final_pnl * 10
        
        if final_pnl > 0 and max_fav > final_pnl * 1.5:
            reward -= (max_fav - final_pnl) * 5
        
        if max_adv < -0.08:
            reward += 0.5
        
        if duration > 12 and final_pnl < 0.02:
            reward -= 0.3
        elif duration < 1 and final_pnl > 0.02:
            reward += 0.5
        
        # ✅ Final validation
        reward = float(np.clip(reward, -1000.0, 1000.0))
        
        if np.isnan(reward) or np.isinf(reward):
            logger.warning(f"⚠️ Invalid exit reward, returning 0.0")
            return 0.0
        
        return reward
    
    def _calculate_recent_sharpe(self, window_size: int = 20) -> float:
        """
        ✅ PATCHED: NaN-safe Sharpe ratio calculation
        
        FIXES:
        1. Minimum data threshold
        2. Outlier filtering (removes extreme P&Ls)
        3. Safe division with epsilon
        4. Windsorization for extreme values
        """
        
        recent_trades = [t for t in self.trade_history if t['traded']][-window_size:]
        
        # ✅ FIX #1: Minimum data requirement
        if len(recent_trades) < 5:
            return 0.0
        
        # ✅ FIX #2: Extract and validate returns
        returns = []
        for t in recent_trades:
            pnl_pct = t.get('pnl_pct', 0.0)
            
            # Skip invalid values
            if np.isnan(pnl_pct) or np.isinf(pnl_pct):
                continue
            
            # Clip extreme outliers (±100%)
            pnl_pct = float(np.clip(pnl_pct, -100.0, 100.0))
            returns.append(pnl_pct)
        
        if len(returns) < 5:
            return 0.0
        
        # ✅ FIX #3: Windsorization (cap extreme values at 95th percentile)
        returns_array = np.array(returns)
        lower_bound = np.percentile(returns_array, 2.5)
        upper_bound = np.percentile(returns_array, 97.5)
        returns_winsorized = np.clip(returns_array, lower_bound, upper_bound)
        
        # ✅ FIX #4: Calculate with safe division
        mean_return = np.mean(returns_winsorized)
        std_return = np.std(returns_winsorized)
        
        # Epsilon for numerical stability
        EPSILON = 1e-8
        
        if std_return < EPSILON:
            # All returns identical - assign based on sign
            if mean_return > 0:
                return 2.0  # Consistent profits
            elif mean_return < 0:
                return -2.0  # Consistent losses
            else:
                return 0.0  # No movement
        
        # ✅ FIX #5: Calculate Sharpe with annualization
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        # ✅ FIX #6: Validate result
        if np.isnan(sharpe) or np.isinf(sharpe):
            logger.warning(f"⚠️ Invalid Sharpe calculated: mean={mean_return:.4f}, std={std_return:.4f}")
            return 0.0
        
        # ✅ FIX #7: Clip to reasonable range
        sharpe = float(np.clip(sharpe, -10.0, 10.0))
        
        return sharpe
    
    def _meta_update(self):
        """Enhanced meta-update with timeframe specialization"""
        logger.info("\n🧠 META-UPDATE V5 TRIGGERED")
        
        recent_trades = [t for t in self.trade_history if t['traded']][-50:]
        
        if len(recent_trades) < 10:
            logger.info("   Insufficient data for meta-update")
            return
        
        total_win_rate = sum(1 for t in recent_trades if t['success']) / len(recent_trades)
        total_pnl = sum(t['pnl'] for t in recent_trades)
        total_sharpe = self._calculate_recent_sharpe(50)
        
        logger.info(f"   Last 50 trades: {len(recent_trades)} trades")
        logger.info(f"   Overall Win Rate: {total_win_rate*100:.1f}%")
        logger.info(f"   Total P&L: ${total_pnl:+.2f}")
        logger.info(f"   Sharpe: {total_sharpe:.2f}")
        logger.info(f"   Evolutionary Trades: {self.trades_from_evolutionary}")
        
        # Timeframe-specific updates
        for timeframe, state in self.timeframe_states.items():
            timeframe_trades = [t for t in recent_trades if t.get('timeframe') == timeframe]
            
            if len(timeframe_trades) >= 5:
                timeframe_win_rate = sum(1 for t in timeframe_trades if t['success']) / len(timeframe_trades)
                timeframe_sharpe = self._calculate_timeframe_sharpe(timeframe)
                
                logger.info(f"   {timeframe.upper()}: {len(timeframe_trades)} trades, "
                           f"{timeframe_win_rate*100:.1f}% WR, Sharpe: {timeframe_sharpe:.2f}")
                
                for optimizer in state.optimizers.values():
                    if timeframe_win_rate > 0.60 and timeframe_sharpe > 1.5:
                        optimizer.exploration_rate *= 0.95
                        optimizer.exploration_rate = max(optimizer.exploration_rate, 0.05)
                    elif timeframe_win_rate < 0.45 or timeframe_sharpe < 0.8:
                        optimizer.exploration_rate *= 1.1
                        optimizer.exploration_rate = min(optimizer.exploration_rate, 0.35)
        
        # Regime configuration updates
        for timeframe, state in self.timeframe_states.items():
            for regime_name, config in state.regime_configs.items():
                regime_trades = [t for t in recent_trades if t.get('timeframe') == timeframe and t['regime'] == regime_name]
                
                if len(regime_trades) >= 5:
                    regime_win_rate = sum(1 for t in regime_trades if t['success']) / len(regime_trades)
                    
                    if regime_win_rate > 0.70:
                        config['risk_tolerance'] = min(config['risk_tolerance'] * 1.1, 1.5)
                        config['position_size_multiplier'] = min(config['position_size_multiplier'] * 1.1, 2.0)
                    elif regime_win_rate < 0.35:
                        config['risk_tolerance'] = max(config['risk_tolerance'] * 0.9, 0.5)
                        config['position_size_multiplier'] = max(config['position_size_multiplier'] * 0.9, 0.5)
        
        insights = self._generate_insights()
        for insight in insights:
            logger.info(f"   💡 {insight}")
        
        self.save_state()
    
    def _generate_insights(self) -> List[str]:
        """Generate insights with timeframe context"""
        insights = []
        
        recent_trades = [t for t in self.trade_history if t['traded']][-50:]
        if len(recent_trades) < 10:
            return insights
        
        for timeframe in self.timeframe_states.keys():
            timeframe_trades = [t for t in recent_trades if t.get('timeframe') == timeframe]
            if len(timeframe_trades) >= 5:
                win_rate = sum(1 for t in timeframe_trades if t['success']) / len(timeframe_trades)
                if win_rate > 0.70:
                    insights.append(f"Strong {timeframe} performance ({win_rate*100:.0f}% WR)")
                elif win_rate < 0.35:
                    insights.append(f"Weak {timeframe} performance ({win_rate*100:.0f}% WR)")
        
        if self.trades_from_evolutionary > 100:
            evo_rate = self.trades_from_evolutionary / ((datetime.now() - self.last_evolutionary_update).total_seconds() / 3600)
            insights.append(f"Evolutionary training: {evo_rate:.1f} trades/hour")
        
        return insights
    
    def get_meta_report(self) -> str:
        """Enhanced report with timeframe breakdown"""
        recent_trades = [t for t in self.trade_history if t['traded']][-100:]
        
        if len(recent_trades) < 5:
            return "Insufficient data for meta-report"
        
        total_win_rate = sum(1 for t in recent_trades if t['success']) / len(recent_trades) * 100
        total_pnl = sum(t['pnl'] for t in recent_trades)
        total_sharpe = self._calculate_recent_sharpe(100)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║             META-RL SUPERVISOR V5 STATUS REPORT              ║
╚══════════════════════════════════════════════════════════════╝

CONTINUOUS LEARNING:
  Total Cycles: {self.total_cycles}
  Evolutionary Trades: {self.trades_from_evolutionary}
  Live Trades: {len([t for t in self.trade_history if t['traded']])}

OVERALL PERFORMANCE (Last 100 trades):
  Trades: {len(recent_trades)}
  Win Rate: {total_win_rate:.1f}%
  Total P&L: ${total_pnl:+.2f}
  Sharpe Ratio: {total_sharpe:.2f}

TIMEFRAME-SPECIFIC PERFORMANCE:
"""
        
        for timeframe, state in self.timeframe_states.items():
            timeframe_trades = [t for t in recent_trades if t.get('timeframe') == timeframe]
            if timeframe_trades:
                win_rate = sum(1 for t in timeframe_trades if t['success']) / len(timeframe_trades) * 100
                pnl = sum(t['pnl'] for t in timeframe_trades)
                
                report += f"  {timeframe.upper():10s}: {len(timeframe_trades):3d} trades, "
                report += f"{win_rate:5.1f}% WR, P&L: ${pnl:+.2f}\n"
        
        report += "\nCURRENT PARAMETERS BY TIMEFRAME:\n"
        
        for timeframe, state in self.timeframe_states.items():
            report += f"  {timeframe.upper()}:\n"
            for param_name, optimizer in state.optimizers.items():
                stats = optimizer.get_stats()
                if stats:
                    report += f"    {param_name:25s}: {stats['current_value']:6.3f}\n"
        
        report += "\n" + "="*66
        
        return report
    
    def save_state(self):
        """Save enhanced state with timeframe data"""
        try:
            state = {
                'timeframe_states': {
                    timeframe: {
                        'timeframe': state.timeframe,
                        'optimizers': {
                            name: {
                                'values': opt.values.tolist(),
                                'arm_counts': opt.arm_counts.tolist(),
                                'arm_rewards': opt.arm_rewards.tolist(),
                                'current_arm_idx': opt.current_arm_idx,
                                'exploration_rate': opt.exploration_rate
                            } for name, opt in state.optimizers.items()
                        },
                        'regime_configs': state.regime_configs,
                        'performance_history': state.performance_history,
                        'recent_win_rate': state.recent_win_rate,
                        'recent_sharpe': state.recent_sharpe
                    } for timeframe, state in self.timeframe_states.items()
                },
                'total_cycles': self.total_cycles,
                'trades_from_evolutionary': self.trades_from_evolutionary,
                'last_evolutionary_update': self.last_evolutionary_update.isoformat()
            }
            
            with open('meta_rl_supervisor_v5_state.pkl', 'wb') as f:
                pickle.dump(state, f)
            
            logger.debug("✅ Meta-RL V5 state saved")
            
        except Exception as e:
            logger.error(f"Failed to save Meta-RL V5 state: {e}")
    
    def load_state(self):
        """Load enhanced state with timeframe data"""
        try:
            if not os.path.exists('meta_rl_supervisor_v5_state.pkl'):
                return
            
            with open('meta_rl_supervisor_v5_state.pkl', 'rb') as f:
                state = pickle.load(f)
            
            if 'timeframe_states' in state:
                for timeframe, timeframe_state in state['timeframe_states'].items():
                    if timeframe in self.timeframe_states:
                        # Load optimizers
                        for name, opt_state in timeframe_state['optimizers'].items():
                            if name in self.timeframe_states[timeframe].optimizers:
                                opt = self.timeframe_states[timeframe].optimizers[name]
                                opt.values = np.array(opt_state['values'])
                                opt.arm_counts = np.array(opt_state['arm_counts'])
                                opt.arm_rewards = np.array(opt_state['arm_rewards'])
                                opt.current_arm_idx = opt_state['current_arm_idx']
                                opt.exploration_rate = opt_state['exploration_rate']
                        
                        # Load other state
                        self.timeframe_states[timeframe].regime_configs = timeframe_state['regime_configs']
                        self.timeframe_states[timeframe].performance_history = timeframe_state['performance_history']
                        self.timeframe_states[timeframe].recent_win_rate = timeframe_state.get('recent_win_rate', 0.0)
                        self.timeframe_states[timeframe].recent_sharpe = timeframe_state.get('recent_sharpe', 0.0)
            
            self.total_cycles = state.get('total_cycles', 0)
            self.trades_from_evolutionary = state.get('trades_from_evolutionary', 0)
            self.last_evolutionary_update = datetime.fromisoformat(
                state.get('last_evolutionary_update', datetime.now().isoformat())
            )
            
            logger.info(f"✅ Loaded Meta-RL V5 state: {self.total_cycles} cycles")
            
        except Exception as e:
            logger.error(f"Failed to load Meta-RL V5 state: {e}")


async def main():
    """Test the enhanced Meta-RL V5"""
    supervisor = MetaRLSupervisorV5(initial_balance=10000.0)
    
    # Simulate some timeframe-specific learning
    for i in range(50):
        supervisor.record_trade_outcome(
            traded=True,
            action='BUY',
            market_regime='bull_strong',
            pnl=np.random.normal(10, 5),
            pnl_pct=np.random.normal(0.5, 0.3),
            success=np.random.random() > 0.4,
            confidence_used=65.0,
            win_prob_used=0.55,
            parameters_used={},
            timeframe=np.random.choice(['short_term', 'mid_term', 'long_term'])
        )
    
    print(supervisor.get_meta_report())


if __name__ == "__main__":
    asyncio.run(main())