import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class MarketStructure:
    """
    Quantify market microstructure for RL state representation
    Shared across all RL trading systems
    """
    volatility_regime: int  # 0=low, 1=mid, 2=high
    trend_strength: float   # -1 to 1
    mean_reversion_score: float  # 0 to 1
    liquidity_score: float  # 0 to 1
    funding_rate: float     # Current funding rate
    volume_profile: float   # Volume vs average
    orderbook_imbalance: float  # Bid/ask pressure
    
    # Time-based features
    time_of_day: int  # 0-23 (UTC hour)
    day_of_week: int  # 0-6
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to RL state vector"""
        return torch.tensor([
            self.volatility_regime / 2.0,
            (self.trend_strength + 1) / 2.0,
            self.mean_reversion_score,
            self.liquidity_score,
            np.tanh(self.funding_rate * 100),
            self.volume_profile,
            self.orderbook_imbalance,
            self.time_of_day / 23.0,
            self.day_of_week / 6.0
        ], dtype=torch.float32)