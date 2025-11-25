"""
VOLATILITY FORECASTER - ARCH/GARCH-style volatility prediction
✅ Complete implementation - just copy and paste
✅ No missing methods
✅ Ready to use immediately
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ARCHGARCHForecaster:
    """
    Real-time volatility forecasting for adaptive position sizing
    Implements simplified GARCH(1,1) model for crypto markets
    """
    
    def __init__(self, lookback: int = 100, warmup_period: int = 20):
        self.lookback = lookback
        self.warmup_period = warmup_period
        self.price_history = deque(maxlen=lookback)
        self.volatility_history = deque(maxlen=50)
        self.return_history = deque(maxlen=lookback)
        
        # GARCH(1,1) parameters optimized for crypto
        self.omega = 0.0001
        self.alpha = 0.12
        self.beta = 0.85
        self.current_variance = 0.02
    
    def update(self, price: float) -> float:
        """
        Update model with latest price, return current volatility
        """
        self.price_history.append(price)
        
        if len(self.price_history) < 2:
            return 0.02
        
        # Calculate log returns
        returns = []
        for i in range(1, len(self.price_history)):
            ret = np.log(self.price_history[i] / self.price_history[i-1])
            returns.append(ret)
        
        # Store the latest return
        if returns:
            self.return_history.append(returns[-1])
        
        if len(returns) < self.warmup_period:
            current_vol = np.std(returns) if returns else 0.02
            self.volatility_history.append(current_vol)
            return current_vol
        
        # GARCH(1,1) update
        latest_return = returns[-1]
        
        # Update variance: σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
        self.current_variance = (self.omega + 
                               self.alpha * (latest_return ** 2) + 
                               self.beta * self.current_variance)
        
        # Ensure variance doesn't explode
        self.current_variance = min(self.current_variance, 0.25)
        
        current_vol = np.sqrt(self.current_variance * 252)
        self.volatility_history.append(current_vol)
        
        return current_vol
    
    def forecast_volatility(self, horizon: int = 5) -> float:
        """
        Forecast volatility for next N periods
        Returns annualized volatility estimate
        """
        if not self.volatility_history:
            return 0.02
        
        current_vol = self.volatility_history[-1]
        
        # Simple mean reversion forecast
        long_term_vol = 0.40
        mean_reversion_speed = 0.1
        
        forecast = (current_vol * (1 - mean_reversion_speed) + 
                   long_term_vol * mean_reversion_speed)
        
        # Add some persistence for short horizons
        if horizon <= 5:
            forecast = current_vol * 0.8 + forecast * 0.2
        
        return max(0.01, min(forecast, 2.0))
    
    def get_volatility_regime(self) -> str:
        """
        Classify current volatility regime for adaptive trading
        """
        if not self.volatility_history:
            return "NORMAL"
        
        current_vol = self.volatility_history[-1]
        
        if current_vol > 0.80:
            return "EXTREME"
        elif current_vol > 0.50:
            return "HIGH"
        elif current_vol > 0.25:
            return "ELEVATED"
        elif current_vol < 0.10:
            return "LOW"
        else:
            return "NORMAL"
    
    def get_position_size_multiplier(self, base_size: float = 1.0, strategy_aggression: float = 1.0) -> float:
        """
        Adjust position size based on volatility forecast
        """
        vol_regime = self.get_volatility_regime()
        
        multipliers = {
            "EXTREME": 0.3,
            "HIGH": 0.5,
            "ELEVATED": 0.7,
            "NORMAL": 1.0,
            "LOW": 1.2
        }
        
        multiplier = multipliers.get(vol_regime, 1.0)
        adjusted_multiplier = multiplier * strategy_aggression
        
        return max(0.1, min(adjusted_multiplier, 2.0))
    
    def get_stop_loss_adjustment(self) -> float:
        """
        Adjust stop-loss distance based on volatility
        """
        vol_regime = self.get_volatility_regime()
        
        adjustments = {
            "EXTREME": 2.0,
            "HIGH": 1.5,
            "ELEVATED": 1.2,
            "NORMAL": 1.0,
            "LOW": 0.8
        }
        
        return adjustments.get(vol_regime, 1.0)
    
    def get_leverage_adjustment(self, base_leverage: float = 5.0) -> float:
        """
        Adjust leverage based on volatility regime
        """
        vol_regime = self.get_volatility_regime()
        
        adjustments = {
            "EXTREME": 0.3,
            "HIGH": 0.5,
            "ELEVATED": 0.7,
            "NORMAL": 1.0,
            "LOW": 1.1
        }
        
        adjustment = adjustments.get(vol_regime, 1.0)
        return base_leverage * adjustment
    
    def get_current_volatility(self) -> float:
        """Get the most recent volatility estimate"""
        if not self.volatility_history:
            return 0.02
        return self.volatility_history[-1]
    
    def get_volatility_trend(self) -> str:
        """Get the trend of volatility (increasing/decreasing/stable)"""
        if len(self.volatility_history) < 5:
            return "STABLE"
        
        recent_vols = list(self.volatility_history)[-5:]
        trend = np.polyfit(range(len(recent_vols)), recent_vols, 1)[0]
        
        if trend > 0.001:
            return "INCREASING"
        elif trend < -0.001:
            return "DECREASING"
        else:
            return "STABLE"
    
    def reset(self) -> None:
        """Reset the forecaster to initial state"""
        self.price_history.clear()
        self.volatility_history.clear()
        self.return_history.clear()
        self.current_variance = 0.02
        logger.info("🔄 Volatility forecaster reset")
    
    def get_forecast_confidence(self) -> float:
        """
        Get confidence in current volatility forecast
        Based on data quality and model stability
        """
        if len(self.volatility_history) < 10:
            return 0.3
        
        # Calculate stability of recent forecasts
        recent_vols = list(self.volatility_history)[-10:]
        volatility_of_vol = np.std(recent_vols) / np.mean(recent_vols)
        
        # More stable = higher confidence
        confidence = 1.0 - min(volatility_of_vol, 0.5)
        
        return max(0.1, min(confidence, 1.0))