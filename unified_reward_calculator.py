"""
UNIFIED REWARD CALCULATOR FOR RL TRAINING
✅ Consistent rewards across DQN/A3C/PPO
✅ Bounded to [-1, 1] range for stable training
✅ Multi-factor: PnL + Sharpe + Risk-adjusted
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class UnifiedRewardCalculator:
    """
    🎯 SINGLE SOURCE OF TRUTH for RL rewards
    
    Ensures DQN, A3C, and PPO receive consistent, normalized rewards
    """
    
    def __init__(self):
        # Reward component weights (must sum to 1.0)
        self.weights = {
            'pnl': 0.40,           # Raw profitability
            'sharpe': 0.30,        # Risk-adjusted returns
            'direction': 0.15,     # Trend alignment bonus
            'efficiency': 0.15     # Parameter quality
        }
        
        # Normalization parameters (learned from data)
        self.pnl_scale = 50.0      # Typical good trade: $50
        self.sharpe_scale = 2.0    # Typical good Sharpe: 2.0
        
        logger.info("🎯 Unified Reward Calculator initialized")
        logger.info(f"   Weights: {self.weights}")
    
    def calculate_reward(self, 
                        pnl: float,
                        sharpe: float,
                        market_structure,
                        was_sell: bool,
                        parameters_used: dict = None) -> float:
        """
        Calculate unified, normalized reward for RL training
        
        Returns: float in [-1, 1] range
        """
        
        # Component 1: Normalized P&L (bounded by tanh)
        pnl_component = np.tanh(pnl / self.pnl_scale)
        
        # Component 2: Normalized Sharpe (bounded by tanh)
        sharpe_component = np.tanh(sharpe / self.sharpe_scale)
        
        # Component 3: Direction alignment bonus
        trend = market_structure.trend_strength
        
        if was_sell and trend < -0.1:
            # Shorting in downtrend
            direction_component = 0.5
        elif not was_sell and trend > 0.1:
            # Longing in uptrend
            direction_component = 0.5
        else:
            # Counter-trend or neutral
            direction_component = -0.2 if abs(trend) > 0.2 else 0.0
        
        # Component 4: Parameter efficiency
        efficiency_component = 0.0
        if parameters_used:
            efficiency_component = self._evaluate_parameter_efficiency(
                parameters_used, pnl, sharpe, was_sell
            )
        
        # Weighted combination
        reward = (
            self.weights['pnl'] * pnl_component +
            self.weights['sharpe'] * sharpe_component +
            self.weights['direction'] * direction_component +
            self.weights['efficiency'] * efficiency_component
        )
        
        # Final clipping (safety)
        reward = float(np.clip(reward, -1.0, 1.0))
        
        # Logging for debugging
        if abs(reward) > 0.5:  # Log significant rewards
            logger.debug(
                f"💰 Reward: {reward:.3f} | "
                f"P&L: {pnl_component:.2f}, Sharpe: {sharpe_component:.2f}, "
                f"Dir: {direction_component:.2f}, Eff: {efficiency_component:.2f}"
            )
        
        return reward
    
    def _evaluate_parameter_efficiency(self, parameters: dict, 
                                       pnl: float, sharpe: float,
                                       was_sell: bool) -> float:
        """
        Evaluate parameter effectiveness [-1, 1]
        """
        efficiency = 0.0
        
        # Good leverage usage
        leverage = parameters.get('leverage', 5.0)
        if pnl > 0 and 7.0 <= leverage <= 10.0:
            efficiency += 0.3  # High leverage on wins
        elif pnl < -20 and leverage > 8.0:
            efficiency -= 0.4  # Over-leveraged losses
        
        # Appropriate position sizing
        position_size = abs(parameters.get('position_size_base', 0.15))
        if sharpe > 1.5 and position_size > 0.2:
            efficiency += 0.2  # Large size with good risk-adjusted returns
        
        # Stop loss effectiveness
        stop_loss = parameters.get('stop_loss_distance', 0.02)
        if pnl > 0 and stop_loss < 0.025:
            efficiency += 0.15  # Tight stops on winners
        elif pnl < -50 and stop_loss > 0.04:
            efficiency -= 0.2  # Wide stops on losers
        
        return float(np.clip(efficiency, -1.0, 1.0))


# Global instance (singleton pattern)
_reward_calculator = None

def get_reward_calculator():
    """Get or create global reward calculator"""
    global _reward_calculator
    if _reward_calculator is None:
        _reward_calculator = UnifiedRewardCalculator()
    return _reward_calculator


def calculate_unified_reward(pnl: float, sharpe: float, 
                            market_structure, was_sell: bool,
                            parameters_used: dict = None) -> float:
    """
    Convenience function for quick reward calculation
    """
    calculator = get_reward_calculator()
    return calculator.calculate_reward(
        pnl, sharpe, market_structure, was_sell, parameters_used
    )


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from rl_agent_generator import MarketStructure
    
    # Test scenarios
    print("🧪 Testing Unified Reward Calculator\n")
    
    # Scenario 1: Big win in uptrend
    market = MarketStructure(
        volatility_regime=1, trend_strength=0.5, mean_reversion_score=0.5,
        liquidity_score=0.8, funding_rate=0.0001, volume_profile=1.0,
        orderbook_imbalance=0.1, time_of_day=14, day_of_week=2
    )
    
    params = {
        'leverage': 8.0,
        'position_size_base': 0.25,
        'stop_loss_distance': 0.02
    }
    
    reward1 = calculate_unified_reward(
        pnl=75.0, sharpe=2.5, market_structure=market,
        was_sell=False, parameters_used=params
    )
    print(f"✅ Big win (long in uptrend): {reward1:.3f}")
    
    # Scenario 2: Successful short in downtrend
    market.trend_strength = -0.6
    reward2 = calculate_unified_reward(
        pnl=50.0, sharpe=1.8, market_structure=market,
        was_sell=True, parameters_used=params
    )
    print(f"✅ Good short (in downtrend): {reward2:.3f}")
    
    # Scenario 3: Over-leveraged loss
    params['leverage'] = 12.0
    reward3 = calculate_unified_reward(
        pnl=-80.0, sharpe=-1.5, market_structure=market,
        was_sell=False, parameters_used=params
    )
    print(f"❌ Over-leveraged loss: {reward3:.3f}")
    
    # Scenario 4: Small win (efficient)
    params['leverage'] = 5.0
    params['stop_loss_distance'] = 0.015
    reward4 = calculate_unified_reward(
        pnl=20.0, sharpe=1.2, market_structure=market,
        was_sell=True, parameters_used=params
    )
    print(f"✅ Small but efficient: {reward4:.3f}")
    
    print("\n🎯 All rewards in [-1, 1] range ✅")