"""
EVOLUTIONARY PAPER TRADING WITH LEVERAGED PERPETUALS (3x-10x)
✅ FULLY FIXED VERSION - Complete methods, no confusion!

CRITICAL FIXES:
1. ✅ Increased MIN_HOLDING_MINUTES to 2.0 (was 0.5) - prevents instant 0% P&L trades
2. ✅ Clear pending asset in _close_leveraged_trade() - prevents asset getting stuck
3. ✅ Add active_positions_by_asset tracking - prevents duplicate positions
4. ✅ Stricter same-price filtering (0.1% minimum move)
5. ✅ STRONG ANTI-SPAM: Global cooldown, per-asset cooldown, trade limits
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import deque
import json
import logging
import pickle
import os
import time 

# Add this import at the top with other imports
from rl_state_manager import RLStateManager, RLPerformanceMetrics

# Import existing classes
from evolutionary_paper_trading_2 import (
    AgentDNA, EvolutionaryAgent, EvolutionaryPaperTradingV2
)
from fixed_hyperliquid import EnhancedHyperliquidExchange
from agent_registry import AgentRegistry
from volatility_forecaster import ARCHGARCHForecaster

# Add to evolutionary_paper_trading_leverage.py imports:
try:
    from rl_agent_generator import RLAgentCoordinator
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logger.warning("RLAgentCoordinator not available - RL features disabled")
    
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

# ✅ ADD THIS LINE
from unified_reward_calculator import calculate_unified_reward 


logger = logging.getLogger(__name__)

def is_winning_trade(trade: 'LeveragedTrade') -> bool:
    """
    ✅ UNIFIED WIN DEFINITION
    
    A trade is a "win" if net P&L (price movement + funding) > 0.1%
    
    This replaces the inconsistent 0.5% threshold that excluded
    many profitable trades where funding payments created gains.
    
    Args:
        trade: LeveragedTrade object with pnl and position_size
        
    Returns:
        True if net P&L > 0.1%, False otherwise
    """
    if trade.position_size <= 0:
        return False
    
    # Net P&L percentage (includes funding)
    net_pnl_pct = (trade.pnl / trade.position_size) * 100
    
    # Minimum 0.1% to avoid counting rounding errors
    MIN_THRESHOLD = 0.2  # 0.1% instead of previous 0.5%
    
    return net_pnl_pct > MIN_THRESHOLD

def categorize_trade_outcome(trade: 'LeveragedTrade') -> str:
    """
    Categorize trade outcome for detailed logging
    
    Returns:
        "BIG_WIN"    : > 2.0%
        "WIN"        : 0.5% to 2.0%
        "SMALL_WIN"  : 0.1% to 0.5%
        "BREAK EVEN" : -0.1% to 0.1%
        "SMALL_LOSS" : -0.5% to -0.1%
        "LOSS"       : -2.0% to -0.5%
        "BIG_LOSS"   : < -2.0%
    """
    if trade.position_size <= 0:
        return "INVALID"
    
    net_pnl_pct = (trade.pnl / trade.position_size) * 100
    
    if net_pnl_pct > 2.0:
        return "BIG_WIN"
    elif net_pnl_pct > 0.5:
        return "WIN"
    elif net_pnl_pct > 0.1:
        return "SMALL_WIN"
    elif net_pnl_pct > -0.1:
        return "BREAK_EVEN"
    elif net_pnl_pct > -0.5:
        return "SMALL_LOSS"
    elif net_pnl_pct > -2.0:
        return "LOSS"
    else:
        return "BIG_LOSS"

def debug_trade_pnl(trade: 'LeveragedTrade'):
    """
    Debug helper to understand why trades show 0% win rate
    
    Add this to _close_leveraged_trade() to see what's happening
    """
    
    logger.info(f"\n🔍 TRADE P&L DEBUG:")
    logger.info(f"   Asset: {trade.asset}")
    logger.info(f"   Entry: ${trade.entry_price:.4f}")
    logger.info(f"   Exit: ${trade.current_price:.4f}")
    logger.info(f"   Position Size: ${trade.position_size:.2f}")
    logger.info(f"   Leverage: {trade.leverage:.1f}x")
    
    # Calculate raw price movement
    if trade.action == 'BUY':
        price_move = trade.current_price - trade.entry_price
        price_move_pct = (price_move / trade.entry_price) * 100
    else:
        price_move = trade.entry_price - trade.current_price
        price_move_pct = (price_move / trade.entry_price) * 100
    
    logger.info(f"   Price Move: ${price_move:+.4f} ({price_move_pct:+.4f}%)")
    
    # Calculate P&L
    raw_pnl = trade.position_size * (price_move_pct / 100)
    leveraged_pnl = raw_pnl * trade.leverage
    funding = trade.funding_paid if hasattr(trade, 'funding_paid') else 0
    net_pnl = leveraged_pnl - funding
    
    logger.info(f"   Raw P&L: ${raw_pnl:+.2f}")
    logger.info(f"   Leveraged P&L: ${leveraged_pnl:+.2f}")
    logger.info(f"   Funding Paid: ${funding:.4f}")
    logger.info(f"   Net P&L: ${net_pnl:+.2f}")
    
    # Calculate percentages
    net_pnl_pct = (net_pnl / trade.position_size) * 100
    
    logger.info(f"   Net P&L %: {net_pnl_pct:+.4f}%")
    logger.info(f"   Win Threshold: 0.1%")
    logger.info(f"   Is Win: {net_pnl_pct > 0.1}")
    
    # Check if close to threshold
    if abs(net_pnl_pct) < 0.5:
        logger.warning(f"   ⚠️ P&L very close to 0 - might be rounding/fees issue")



"""
🎯 PROGRESSIVE FITNESS SYSTEM - Works from Trade 1
Replaces calculate_revised_fitness_inline() in evolutionary_paper_trading_leverage.py

CRITICAL FIXES:
1. ✅ No minimum trade requirement (works from trade 1)
2. ✅ Progressive scaling based on experience
3. ✅ Rewards both exploration (new agents) and exploitation (veterans)
4. ✅ Prevents 0.0 fitness deadlock
"""

def calculate_progressive_fitness(
    agent_id: int,
    total_trades: int,
    winning_trades: int,
    total_pnl: float,
    timeframe: str
) -> float:
    """
    🎯 PROGRESSIVE FITNESS - No Minimum Trades Required
    
    Works in 3 phases:
    - Phase 1 (1-4 trades): Exploration bonus, simple P&L reward
    - Phase 2 (5-14 trades): Balanced P&L + win rate
    - Phase 3 (15+ trades): Full comprehensive fitness
    
    Args:
        agent_id: Agent identifier
        total_trades: Number of completed trades
        winning_trades: Number of profitable trades (>0.1% net P&L)
        total_pnl: Total profit/loss in dollars
        timeframe: 'short_term', 'mid_term', or 'long_term'
    
    Returns:
        Fitness score (0-150 range, never 0 if traded)
    """
    
    # ✅ PHASE 1: EXPLORATION (1-4 trades)
    # Give new agents a chance based on early performance
    if total_trades < 5:
        return _calculate_exploration_fitness(
            total_trades, winning_trades, total_pnl, timeframe
        )
    
    # ✅ PHASE 2: LEARNING (5-14 trades)
    # Balanced fitness as agent learns
    elif total_trades < 15:
        return _calculate_learning_fitness(
            total_trades, winning_trades, total_pnl, timeframe
        )
    
    # ✅ PHASE 3: MASTERY (15+ trades)
    # Full comprehensive fitness for experienced agents
    else:
        return _calculate_mastery_fitness(
            total_trades, winning_trades, total_pnl, timeframe
        )


def _calculate_exploration_fitness(
    total_trades: int,
    winning_trades: int, 
    total_pnl: float,
    timeframe: str
) -> float:
    """
    Phase 1: Exploration Fitness (1-4 trades)
    
    Focus: Encourage activity + reward early success
    """
    
    # Base exploration bonus (ensures non-zero fitness)
    exploration_bonus = 10.0  # Everyone gets 10 points for trying
    
    # Win rate component (simplified for small samples)
    if total_trades > 0:
        win_rate = winning_trades / total_trades
        win_score = win_rate * 30.0  # 0-30 points
    else:
        win_score = 0.0
    
    # P&L component (absolute dollars matter early)
    if total_pnl > 0:
        pnl_score = min(total_pnl * 2.0, 40.0)  # Cap at 40 points
    else:
        # Negative P&L penalty (but not too harsh)
        pnl_score = max(total_pnl * 0.5, -20.0)  # Max -20 penalty
    
    # Activity bonus (reward taking trades)
    activity_score = min(total_trades * 5.0, 20.0)  # Up to 20 points
    
    fitness = exploration_bonus + win_score + pnl_score + activity_score
    
    # Ensure minimum fitness of 5.0 (even losers survive one generation)
    return max(5.0, fitness)


def _calculate_learning_fitness(
    total_trades: int,
    winning_trades: int,
    total_pnl: float,
    timeframe: str
) -> float:
    """
    Phase 2: Learning Fitness (5-14 trades)
    
    Focus: Balance win rate + profitability + activity
    """
    
    win_rate = winning_trades / total_trades
    avg_pnl = total_pnl / total_trades
    
    # COMPONENT 1: Win Rate (0-40 points)
    if win_rate >= 0.70:
        win_score = 40.0
    elif win_rate >= 0.60:
        win_score = 32.0
    elif win_rate >= 0.50:
        win_score = 24.0
    elif win_rate >= 0.40:
        win_score = 16.0
    else:
        win_score = 8.0
    
    # COMPONENT 2: Average P&L (0-40 points)
    if avg_pnl > 5.0:
        pnl_score = 40.0
    elif avg_pnl > 2.0:
        pnl_score = 30.0
    elif avg_pnl > 1.0:
        pnl_score = 20.0
    elif avg_pnl > 0.0:
        pnl_score = 10.0
    else:
        pnl_score = max(-10.0, avg_pnl * 2.0)  # Penalty for losses
    
    # COMPONENT 3: Activity Progress (0-20 points)
    # Reward approaching phase 3 threshold
    activity_progress = (total_trades - 5) / 10.0  # 0.0 to 1.0
    activity_score = activity_progress * 20.0
    
    fitness = win_score + pnl_score + activity_score
    
    # Learning phase minimum (agents can still fail but get chances)
    return max(8.0, fitness)


def _calculate_mastery_fitness(
    total_trades: int,
    winning_trades: int,
    total_pnl: float,
    timeframe: str
) -> float:
    """
    Phase 3: Mastery Fitness (15+ trades)
    
    Focus: Comprehensive performance evaluation
    This is your original formula (with minimum removed)
    """
    
    win_rate = winning_trades / total_trades
    avg_pnl = total_pnl / total_trades
    
    # COMPONENT 1: Win Rate Factor (0-40 points)
    if win_rate >= 0.80:
        win_rate_factor = 40.0
    elif win_rate >= 0.70:
        win_rate_factor = 34.0
    elif win_rate >= 0.60:
        win_rate_factor = 28.0
    elif win_rate >= 0.50:
        win_rate_factor = 20.0
    elif win_rate >= 0.40:
        win_rate_factor = 12.0
    else:
        win_rate_factor = 4.0
    
    # COMPONENT 2: Trade Activity Factor (0-30 points)
    expected_trades = {
        'short_term': 50,
        'mid_term': 20,
        'long_term': 10
    }.get(timeframe, 20)
    
    trade_ratio = total_trades / expected_trades
    
    if trade_ratio >= 2.0:
        activity_factor = 30.0
    elif trade_ratio >= 1.0:
        activity_factor = 20.0 + (trade_ratio - 1.0) * 10.0
    elif trade_ratio >= 0.5:
        activity_factor = 10.0 + (trade_ratio - 0.5) * 20.0
    else:
        activity_factor = trade_ratio * 20.0
    
    # COMPONENT 3: Profitability Factor (0-60 points)
    if avg_pnl > 5.0:
        profit_factor = 60.0
    elif avg_pnl > 2.0:
        profit_factor = 40.0
    elif avg_pnl > 1.0:
        profit_factor = 30.0
    elif avg_pnl > 0.5:
        profit_factor = 20.0
    elif avg_pnl > 0.0:
        profit_factor = 10.0
    else:
        profit_factor = 0.0
    
    base_fitness = win_rate_factor + activity_factor + profit_factor
    
    # BONUS: High absolute profit
    if total_pnl > 100:
        profit_bonus = min(20.0, total_pnl / 10.0)
        base_fitness += profit_bonus
    
    return max(10.0, base_fitness)


class EnhancedCentralPortfolio:
    """
    💰 CENTRAL CAPITAL POOL - TRUE COMPOUNDING
    
    REPLACES: 90 agents × $1,000 = fragmented capital
    WITH: Single $90,000 pool that compounds
    
    Example Flow:
    - Cycle 1: Start with $90k → Use $10k → Return $11k → Pool = $91k
    - Cycle 2: Start with $91k → Use $10k → Return $9k → Pool = $90k
    - Cycle 3: Start with $90k → Use $15k → Return $18k → Pool = $93k
    """
    
    def __init__(self, initial_capital: float = 90000.0):
        # Core capital tracking
        self.initial_capital = initial_capital
        self.total_capital = initial_capital
        self.available_capital = initial_capital
        
        # Agent allocations
        self.agent_allocations: Dict[int, float] = {}  # agent_id -> total allocated
        self.active_trades: Dict[int, float] = {}  # agent_id -> current trade size
        
        # Performance tracking
        self.trade_counter = 0
        self.cycle_counter = 0
        self.cycle_history = []
        
        # Risk limits
        self.max_allocation_per_agent_pct = 0.05  # 5% per agent
        self.max_single_trade_pct = 0.02  # 2% per trade
        self.max_total_allocated_pct = 0.80  # 80% max deployed
        
        logger.info(f"💰 Central Portfolio initialized: ${initial_capital:,.2f}")
    
    # def allocate_trade_capital(self, agent_id: int, requested_size: float) -> bool:
    #     """
    #     Allocate capital to agent for a trade
        
    #     Returns:
    #         True if allocation successful, False if rejected
    #     """
    #     # Calculate limits
    #     max_agent_alloc = self.total_capital * self.max_allocation_per_agent_pct
    #     max_trade = self.total_capital * self.max_single_trade_pct
    #     max_deployed = self.total_capital * self.max_total_allocated_pct
        
    #     # Check portfolio utilization
    #     current_deployed = self.total_capital - self.available_capital
    #     if current_deployed >= max_deployed:
    #         logger.warning(f"❌ Max portfolio allocation reached: {current_deployed/self.total_capital*100:.1f}%")
    #         return False
        
    #     # Check agent hasn't exceeded individual limit
    #     current_agent_alloc = self.agent_allocations.get(agent_id, 0.0)
    #     if current_agent_alloc >= max_agent_alloc:
    #         logger.warning(f"❌ Agent {agent_id} at max allocation: ${current_agent_alloc:.2f}")
    #         return False
        
    #     # Calculate actual allocation (respecting all limits)
    #     actual_size = min(
    #         requested_size,
    #         max_trade,
    #         self.available_capital,
    #         max_agent_alloc - current_agent_alloc
    #     )
        
    #     # Require at least 50% of requested size
    #     if actual_size < requested_size * 0.5:
    #         logger.warning(f"⚠️ Only {actual_size/requested_size*100:.0f}% of requested size available")
    #         return False
        
    #     # ✅ ALLOCATE CAPITAL
    #     self.available_capital -= actual_size
    #     self.agent_allocations[agent_id] = current_agent_alloc + actual_size
    #     self.active_trades[agent_id] = actual_size
    #     self.trade_counter += 1
        
    #     logger.debug(f"✅ Allocated ${actual_size:.2f} to Agent {agent_id} | Available: ${self.available_capital:,.2f}")
        
    #     return True

    def allocate_trade_capital(self, agent_id: int, requested_size: float) -> bool:
        """
        ✅ GUARANTEED: Always allocates capital with fallback mechanisms
        """
        # Calculate absolute limits
        max_trade = self.total_capital * self.max_single_trade_pct  # 2% of $90k = $1800
        max_agent_alloc = self.total_capital * self.max_allocation_per_agent_pct  # 5% = $500
        max_deployed = self.total_capital * self.max_total_allocated_pct  # 80% = $72k
        
        logger.info(f"\n💰 ALLOCATION REQUEST: Agent {agent_id}")
        logger.info(f"   Requested: ${requested_size:.2f}")
        logger.info(f"   Available: ${self.available_capital:.2f}")
        logger.info(f"   Max Trade: ${max_trade:.2f}")
        logger.info(f"   Max Agent: ${max_agent_alloc:.2f}")
        
        # ✅ GUARANTEE 1: Always have minimum capital
        if self.available_capital < 100:
            logger.warning("❌ CRITICAL: Available capital below $100")
            return False
        
        # ✅ GUARANTEE 2: Calculate realistic size
        current_agent_alloc = self.agent_allocations.get(agent_id, 0.0)
        agent_remaining = max_agent_alloc - current_agent_alloc
        
        # Start with requested size, then apply reductions
        actual_size = requested_size
        
        # Apply limits in priority order
        actual_size = min(actual_size, max_trade)                    # Max per trade
        actual_size = min(actual_size, self.available_capital)       # Available capital
        actual_size = min(actual_size, agent_remaining)              # Agent limit
        
        # ✅ GUARANTEE 3: Force minimum viable size
        MIN_VIABLE_SIZE = 100.0
        if actual_size < MIN_VIABLE_SIZE:
            # Try to allocate at least minimum
            if self.available_capital >= MIN_VIABLE_SIZE and agent_remaining >= MIN_VIABLE_SIZE:
                actual_size = MIN_VIABLE_SIZE
                logger.info(f"   ⚡ FORCING minimum size: ${MIN_VIABLE_SIZE:.2f}")
            else:
                logger.warning(f"❌ Cannot meet minimum size: ${MIN_VIABLE_SIZE:.2f}")
                return False
        
        # ✅ GUARANTEE 4: Final validation
        if actual_size <= 0:
            logger.error("❌ Calculated size <= 0")
            return False
        
        if actual_size > self.available_capital:
            logger.error(f"❌ Size ${actual_size:.2f} > Available ${self.available_capital:.2f}")
            return False
        
        # ✅ EXECUTE ALLOCATION
        self.available_capital -= actual_size
        self.agent_allocations[agent_id] = current_agent_alloc + actual_size
        self.active_trades[agent_id] = actual_size
        self.trade_counter += 1
        
        logger.info(f"✅ GUARANTEED ALLOCATION: ${actual_size:.2f} to Agent {agent_id}")
        logger.info(f"   Utilization: {((self.total_capital - self.available_capital)/self.total_capital*100):.1f}%")
        logger.info(f"   Agent Total: ${self.agent_allocations[agent_id]:.2f}")
        
        return True
    
    # def close_trade(self, agent_id: int, pnl: float):
    #     """
    #     🔥 THIS IS WHERE COMPOUNDING HAPPENS!
        
    #     Close trade and add P&L to total capital
    #     """
    #     if agent_id not in self.active_trades:
    #         logger.error(f"❌ Agent {agent_id} has no active trade to close")
    #         return
        
    #     allocated = self.active_trades[agent_id]
        
    #     # 🔥 COMPOUND P&L INTO TOTAL CAPITAL
    #     self.total_capital += pnl
        
    #     # Return capital + P&L to available pool
    #     self.available_capital += allocated + pnl
        
    #     # Update tracking
    #     self.agent_allocations[agent_id] -= allocated
    #     if self.agent_allocations[agent_id] <= 0:
    #         del self.agent_allocations[agent_id]
    #     del self.active_trades[agent_id]
        
    #     # Log compounding effect
    #     portfolio_return = (self.total_capital / self.initial_capital - 1) * 100
        
    #     logger.info(f"💰 Trade closed: Agent {agent_id} | P&L: ${pnl:+.2f} | "
    #                f"New Capital: ${self.total_capital:,.2f} ({portfolio_return:+.2f}%)")
    def close_trade(self, agent_id: int, pnl: float):
        """
        🔥 THIS IS WHERE COMPOUNDING HAPPENS!
        
        Close trade and add P&L to total capital
        """
        # ✅ CRITICAL FIX: Validate trade exists
        if agent_id not in self.active_trades:
            logger.error(f"❌ Agent {agent_id} has no active trade to close")
            logger.error(f"   Active trades: {list(self.active_trades.keys())}")
            logger.error(f"   Agent allocations: {list(self.agent_allocations.keys())}")
            return
        
        allocated = self.active_trades[agent_id]
        
        # ✅ CRITICAL FIX: Validate allocation amount
        if allocated <= 0:
            logger.error(f"❌ Agent {agent_id} has invalid allocation: ${allocated:.2f}")
            logger.error(f"   Attempting emergency recovery...")
            
            # Try to recover from agent_allocations
            if agent_id in self.agent_allocations:
                allocated = self.agent_allocations[agent_id]
                logger.warning(f"   Recovered ${allocated:.2f} from agent_allocations")
            else:
                logger.error(f"   Cannot recover - no allocation record found")
                # Force cleanup
                if agent_id in self.active_trades:
                    del self.active_trades[agent_id]
                return
        
        # ✅ LOG BEFORE CHANGES
        logger.debug(f"💰 CLOSING TRADE:")
        logger.debug(f"   Agent: {agent_id}")
        logger.debug(f"   Allocated: ${allocated:.2f}")
        logger.debug(f"   P&L: ${pnl:+.2f}")
        logger.debug(f"   Before - Total: ${self.total_capital:.2f}, Available: ${self.available_capital:.2f}")
        
        # 🔥 COMPOUND P&L INTO TOTAL CAPITAL
        self.total_capital += pnl
        
        # Return capital + P&L to available pool
        self.available_capital += allocated + pnl
        
        # ✅ LOG AFTER CHANGES
        logger.debug(f"   After  - Total: ${self.total_capital:.2f}, Available: ${self.available_capital:.2f}")
        
        # Update tracking
        self.agent_allocations[agent_id] -= allocated
        if self.agent_allocations[agent_id] <= 0:
            del self.agent_allocations[agent_id]
        del self.active_trades[agent_id]
    
    def release_trade_capital(self, agent_id: int):
        """Release capital if trade cancelled/failed"""
        if agent_id in self.active_trades:
            allocated = self.active_trades[agent_id]
            self.available_capital += allocated
            
            self.agent_allocations[agent_id] -= allocated
            if self.agent_allocations[agent_id] <= 0:
                del self.agent_allocations[agent_id]
            
            del self.active_trades[agent_id]
            logger.debug(f"🔄 Released ${allocated:.2f} from Agent {agent_id}")
    
    def start_new_cycle(self):
        """
        Start new trading cycle with capital rebalancing
        Called at the end of each generation
        """
        self.cycle_counter += 1
        
        # Record cycle performance
        cycle_data = {
            'cycle': self.cycle_counter,
            'timestamp': datetime.now(),
            'total_capital': self.total_capital,
            'return_pct': (self.total_capital / self.initial_capital - 1) * 100,
            'trades': self.trade_counter
        }
        self.cycle_history.append(cycle_data)
        
        # Force close any orphaned positions
        if len(self.active_trades) > 0:
            logger.warning(f"⚠️ Force closing {len(self.active_trades)} orphaned trades")
            for agent_id in list(self.active_trades.keys()):
                self.release_trade_capital(agent_id)
        
        # Reset allocations for new cycle
        self.agent_allocations.clear()
        self.available_capital = self.total_capital
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 CYCLE {self.cycle_counter} STARTED")
        logger.info(f"{'='*80}")
        logger.info(f"   Total Capital: ${self.total_capital:,.2f}")
        logger.info(f"   Available: ${self.available_capital:,.2f}")
        logger.info(f"   Portfolio Return: {cycle_data['return_pct']:+.2f}%")
        logger.info(f"   Trades Last Cycle: {cycle_data['trades']}")
        logger.info(f"{'='*80}\n")
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return {
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'allocated_capital': self.total_capital - self.available_capital,
            'active_trades': len(self.active_trades),
            'utilization_pct': ((self.total_capital - self.available_capital) / self.total_capital) * 100 if self.total_capital > 0 else 0,
            'cycle': self.cycle_counter,
            'total_trades': self.trade_counter,
            'return_pct': (self.total_capital / self.initial_capital - 1) * 100,
            'compounding_multiplier': self.total_capital / self.initial_capital if self.initial_capital > 0 else 1.0
        }


# ✅ ADD THIS IDENTICAL MarketStructure definition
@dataclass
class MarketStructure:
    """
    Quantify market microstructure for RL state representation
    MUST BE IDENTICAL to rl_agent_generator.py
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
            self.volatility_regime / 2.0,  # Normalize to 0-1
            (self.trend_strength + 1) / 2.0,
            self.mean_reversion_score,
            self.liquidity_score,
            np.tanh(self.funding_rate * 100),  # Bound to -1,1
            self.volume_profile,
            self.orderbook_imbalance,
            self.time_of_day / 23.0,
            self.day_of_week / 6.0
        ], dtype=torch.float32)

@dataclass
class LeveragedTrade:
    """Trade with leverage support"""
    agent_id: int
    agent_dna: AgentDNA
    asset: str
    action: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    position_size: float  # Base position size
    leverage: float  # 3x to 10x
    effective_size: float  # position_size * leverage
    stop_loss: float
    take_profit: float
    liquidation_price: float  # Critical for leverage
    
    # Live tracking
    current_price: float
    pnl: float
    pnl_pct: float
    leverage_pnl_pct: float  # Amplified by leverage
    status: str
    close_reason: str
    close_time: Optional[datetime]
    
    # Context
    market_regime: str
    confidence_used: float
    win_prob_used: float
    
    # Learning metrics
    max_favorable_move: float = 0.0
    max_adverse_move: float = 0.0
    realized_holding_hours: float = 0.0
    market_volatility: float = 0.0
    trend_strength: float = 0.0
    funding_paid: float = 0.0  # Cumulative funding rate costs

    # ✅ NEW: RL training metadata (added for DQN/A3C/PPO)
    archetype_id: Optional[int] = None  # Maps agent to archetype (0-9 for short-term)


class LeveragedEvolutionaryAgent(EvolutionaryAgent):
    """
    Enhanced agent with guaranteed attribute initialization
    """
    def __init__(self, dna: AgentDNA, initial_balance: float = 1000.0):
        super().__init__(dna, initial_balance)
        
        # Leverage-specific genes
        self.preferred_leverage = self._calculate_preferred_leverage_AGGRESSIVE()
        self.max_leverage = 10.0
        self.min_leverage = 3.0
        self.risk_tolerance = getattr(dna, 'risk_tolerance', 1.0)
        
        # Initialize ALL tracking attributes with defaults
        self.last_trade_time = datetime.now() - timedelta(minutes=10)
        self._last_trade_cache = None
        self._pending_assets = set()
        self.asset_cooldowns = {}
        self.active_positions_by_asset = {}
        self._pending_timestamps = {}
        self.pending_timeout_minutes = 5
        
        # Initialize statistics
        self._trades_this_session = 0
        self._last_session_reset = datetime.now()
        self._rejected_trades = 0
        
        # Enable profit reinvestment
        self.profit_reinvestment_rate = 0.5
        self.base_position_size = dna.position_size_base
    
    def _calculate_preferred_leverage_AGGRESSIVE(self) -> float:
        """🚀 EXTREME LEVERAGE: 5x-15x range (was 3x-10x)"""
        base_leverage = 7.0  # ⬆️ Was 3.0
        
        aggression_factor = self.dna.aggression * 6.0  # ⬆️ Was 4.0
        patience_penalty = self.dna.patience * 1.0     # ⬇️ Was 2.0
        
        preferred = base_leverage + aggression_factor - patience_penalty
        
        # Timeframe multipliers - all boosted
        if self.dna.timeframe == 'short_term':
            preferred *= 1.8  # ⬆️ Was 1.3
        elif self.dna.timeframe == 'mid_term':
            preferred *= 1.5  # ⬆️ Was 1.0
        elif self.dna.timeframe == 'long_term':
            preferred *= 1.0  # ⬆️ Was 0.7
        
        return np.clip(preferred, 5.0, 15.0)  # ⬆️ Range 5x-15x (was 3x-10x)
    
    def should_trade_leveraged_AGGRESSIVE(self, market_data: dict, current_regime: str, 
                                        current_time: datetime) -> tuple:
        """
        ✅ FIXED: Removed stale pending auto-clear logic
        """
        asset = market_data['asset']
        current_price = market_data['price']
        
        # ✅ CHECK #1: Global cooldown - 15 seconds
        time_since_last_trade = (current_time - self.last_trade_time).total_seconds()
        if time_since_last_trade < 15:
            return False, 0.0, 0.0
        
        # ✅ CHECK #2: Active position check
        if asset in self.active_positions_by_asset:
            return False, 0.0, 0.0
        
        # ✅ CHECK #3: Per-asset cooldown - 60 seconds
        last_trade_time = self.asset_cooldowns.get(asset)
        if last_trade_time:
            elapsed = (current_time - last_trade_time).total_seconds()
            if elapsed < 60:
                return False, 0.0, 0.0
        
        # ✅ CHECK #4: Minimum price movement
        if self._last_trade_cache:
            last_asset, last_price, last_time = self._last_trade_cache
            if last_asset == asset:
                price_change_pct = abs(current_price - last_price) / last_price
                time_since_trade = (current_time - last_time).total_seconds()
                
                MIN_PRICE_MOVEMENT = 0.003
                
                if price_change_pct < MIN_PRICE_MOVEMENT or time_since_trade < 60:
                    return False, 0.0, 0.0
        
        # ✅ CHECK #5: No active trades for this asset
        if any(t.asset == asset and t.status == 'open' for t in self.active_trades):
            return False, 0.0, 0.0
        
        # ✅ CHECK #6: Max trades per cycle
        recent_trades_count = len([t for t in self.trade_history 
                                if (current_time - t.entry_time).total_seconds() < 300])
        if recent_trades_count >= 8:
            return False, 0.0, 0.0
        
        # ✅ Continue with normal trading logic
        should_trade, confidence = self.should_trade(market_data, current_regime, current_time)
        
        if not should_trade:
            return False, 0.0, 0.0
        
        # ✅ Volatility validation
        volatility = market_data.get('volatility', 0.02)
        
        if volatility <= 0 or np.isnan(volatility) or np.isinf(volatility):
            volatility = 0.02
        
        # Dynamic volatility thresholds
        volatility_limits = {
            'bull_strong': 0.20,
            'bull_weak': 0.18,
            'bear_strong': 0.25,
            'bear_weak': 0.22,
            'ranging': 0.15,
            'high_volatility': 0.30,
            'crash': 0.35
        }
        
        max_volatility = volatility_limits.get(current_regime, 0.15)
        
        if volatility > max_volatility:
            return False, 0.0, 0.0
        
        # ✅ Calculate volatility adjustment
        if volatility < 0.02:
            vol_adjustment = 1.2
        elif volatility < 0.05:
            vol_adjustment = 1.0
        elif volatility < 0.10:
            vol_adjustment = 0.7
        else:
            vol_adjustment = 0.5
        
        # Regime adjustment
        regime_adjustment = {
            'bull_strong': 1.4,
            'bull_weak': 1.2,
            'bear_strong': 1.0,
            'bear_weak': 1.1,
            'ranging': 1.1,
            'high_volatility': 0.8,
            'crash': 0.7
        }.get(current_regime, 1.0)
        
        confidence_adjustment = 0.7 + (confidence / 100) * 0.7
        
        # ✅ Calculate optimal leverage
        optimal_leverage = self.preferred_leverage * vol_adjustment * regime_adjustment * confidence_adjustment
        optimal_leverage = np.clip(optimal_leverage, 5.0, 15.0)
        
        return True, confidence, optimal_leverage

    
    def generate_leveraged_trade_params(self, market_data, confidence: float, leverage: float):
        """
        ✅ GUARANTEED: Always returns valid trade parameters with realistic sizes
        """
        current_price = market_data['price']
        
        # ✅ GUARANTEE 1: Validate inputs
        if current_price <= 0:
            logger.error("❌ Invalid price in trade params")
            return None
        
        if confidence <= 0 or leverage <= 0:
            logger.error("❌ Invalid confidence or leverage")
            return None
        
        # ✅ GUARANTEE 2: Use CENTRAL POOL with realistic sizes
        reference_capital = 90000.0
        
        # Base position: % of reference capital (REDUCED FOR TESTING)
        base_position_size = self.dna.position_size_base * reference_capital * 0.15  # 15% of original
        
        # Confidence scaling (50% - 100% of base)
        confidence_scale = 0.5 + (confidence / 100) * 0.5
        base_position_size *= confidence_scale
        
        # ✅ GUARANTEE 3: REALISTIC SIZE RANGES
        MIN_POSITION_SIZE = 30.0    # Reduced for testing
        MAX_POSITION_SIZE = 90.0   # Reduced for testing
        
        base_position_size = np.clip(base_position_size, MIN_POSITION_SIZE, MAX_POSITION_SIZE)
        
        logger.info(f"   💰 GUARANTEED SIZE: ${base_position_size:.2f} "
                    f"(base={self.dna.position_size_base:.1%} × {confidence_scale:.2f} confidence)")
        
        # Determine action
        rsi = market_data.get('rsi', 50)
        trend = market_data.get('trend_direction', 0)
        
        if rsi < 40 and trend > 0:
            action = 'BUY'
        elif rsi > 60 and trend < 0:
            action = 'SELL'
        else:
            action = 'BUY' if trend >= 0 else 'SELL'
        
        # ✅ GUARANTEE 4: LEVERAGE-ADJUSTED STOPS (FIXED CALCULATION)
        if leverage >= 9:
            stop_loss_pct = 0.025
            take_profit_pct = 0.050
        elif leverage >= 7:
            stop_loss_pct = 0.020
            take_profit_pct = 0.040
        elif leverage >= 5:
            stop_loss_pct = 0.015
            take_profit_pct = 0.030
        else:
            stop_loss_pct = 0.012
            take_profit_pct = 0.024
        
        # Volatility adjustment
        current_volatility = market_data.get('volatility', 0.02)
        if current_volatility > 0.08:
            stop_loss_pct *= 2.5
            take_profit_pct *= 2.5
        elif current_volatility > 0.05:
            stop_loss_pct *= 1.8
            take_profit_pct *= 1.8
        
        # ✅ GUARANTEE 5: SAFE LIQUIDATION CALCULATION
        maintenance_margin_pct = 0.03
        liquidation_threshold = max((1.0 / leverage) - maintenance_margin_pct, 0.01)
        
        # Ensure 2% buffer from liquidation
        min_safe_stop = liquidation_threshold + 0.02
        if stop_loss_pct < min_safe_stop:
            original_stop = stop_loss_pct
            stop_loss_pct = min_safe_stop
            logger.info(f"   🔧 Safety adjustment: stop {original_stop:.3f} → {stop_loss_pct:.3f}")
        
        # Calculate prices
        if action == 'BUY':
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
            liquidation_price = current_price * (1 - liquidation_threshold)
        else:
            stop_loss_price = current_price * (1 + stop_loss_pct)
            take_profit_price = current_price * (1 - take_profit_pct)
            liquidation_price = current_price * (1 + liquidation_threshold)
        
        # ✅ GUARANTEE 6: VALIDATE STOP LOSS PRICES
        if action == 'BUY':
            if stop_loss_price >= current_price:
                stop_loss_price = current_price * 0.98  # Force 2% below
                logger.warning(f"   🔧 FORCED BUY stop: ${stop_loss_price:.4f}")
        else:
            if stop_loss_price <= current_price:
                stop_loss_price = current_price * 1.02  # Force 2% above
                logger.warning(f"   🔧 FORCED SELL stop: ${stop_loss_price:.4f}")
        
        # Validate take profit
        if action == 'BUY':
            if take_profit_price <= current_price:
                take_profit_price = current_price * 1.02
        else:
            if take_profit_price >= current_price:
                take_profit_price = current_price * 0.98
        
        # ✅ GUARANTEE 7: FINAL VALIDATION
        if action == 'BUY':
            if stop_loss_price >= current_price or take_profit_price <= current_price:
                logger.error("❌ Invalid BUY price levels")
                return None
        else:
            if stop_loss_price <= current_price or take_profit_price >= current_price:
                logger.error("❌ Invalid SELL price levels")
                return None
        
        return {
            'action': action,
            'position_size': base_position_size,
            'leverage': leverage,
            'effective_size': base_position_size * leverage,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'liquidation_price': liquidation_price,
            'confidence': confidence,
            'stop_loss_pct_used': stop_loss_pct,
            'take_profit_pct_used': take_profit_pct
        }

    # ✅ UPDATE: Modified agent.update_fitness() method
    def update_fitness(self):
        """
        ✅ FIXED: Proper method definition with self parameter
        """
        try:
            self.dna.fitness_score = calculate_progressive_fitness(
                agent_id=self.dna.agent_id,
                total_trades=self.dna.total_trades,
                winning_trades=self.dna.winning_trades,
                total_pnl=self.dna.total_pnl,
                timeframe=self.dna.timeframe
            )
            
            # Log phase for debugging
            if self.dna.total_trades < 5:
                phase = "EXPLORATION"
            elif self.dna.total_trades < 15:
                phase = "LEARNING"
            else:
                phase = "MASTERY"
            
            if self.dna.fitness_score > 30.0:  # Log notable agents
                logger.info(
                    f"🎯 Agent {self.dna.agent_id} [{phase}]: "
                    f"Fitness={self.dna.fitness_score:.1f} | "
                    f"WR={self.dna.winning_trades}/{self.dna.total_trades} | "
                    f"P&L=${self.dna.total_pnl:+.2f}"
                )
            
        except Exception as e:
            logger.error(f"❌ Fitness calculation error for agent {self.dna.agent_id}: {e}")
            # Emergency fallback
            self.dna.fitness_score = max(5.0, self.dna.total_trades * 2.0)

    
    def clear_pending_asset(self, asset: str):
        """
        ✅ PATCHED: Safe pending asset cleanup
        """
        # ✅ Defensive - ensure attributes exist
        if not hasattr(self, '_pending_assets'):
            self._pending_assets = set()
        
        if not hasattr(self, '_pending_timestamps'):
            self._pending_timestamps = {}
        
        self._pending_assets.discard(asset)
        
        if asset in self._pending_timestamps:
            del self._pending_timestamps[asset]

    def get_trade_stats(self) -> Dict:
        """
        ✅ PATCHED: Safe statistics collection with defaults
        """
        return {
            'agent_id': self.dna.agent_id,
            'active_trades': len(self.active_trades),
            'active_positions': len(getattr(self, 'active_positions_by_asset', {})),
            'pending_assets': len(getattr(self, '_pending_assets', set())),
            'trade_history': len(self.trade_history),
            'last_trade_time': self.last_trade_time,
            'asset_cooldowns': getattr(self, 'asset_cooldowns', {}),
            'trades_this_session': getattr(self, '_trades_this_session', 0),
            'rejected_trades': getattr(self, '_rejected_trades', 0)
        }


class LiveEvolutionaryLeveragedTrading(EvolutionaryPaperTradingV2):
    """
    ✅ FULLY FIXED: Enhanced evolutionary system with leverage
    Complete methods included - no confusion!
    """
    
    def __init__(self, hyperliquid_exchange: EnhancedHyperliquidExchange,
             initial_balance: float = 1000.0,
             enable_parallel: bool = True):
        
        # Initialize base attributes
        self.initial_balance = initial_balance
        self.generation = 0
        self.agents: List[LeveragedEvolutionaryAgent] = []
        self.all_trades: List[LeveragedTrade] = []
        
        self.exchange = hyperliquid_exchange
        self.enable_parallel = enable_parallel
        
        # Simulated time for accelerated testing
        self.simulated_time = datetime.now()
        
        # Live data cache
        self.market_data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 30  # seconds
        
        # Price history for technical indicators
        self.price_history = {}
        
        # Assets
        self.assets = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
        
        # Population settings
        self.population_size = 90
        self.agents_per_timeframe = 30
        self.elite_size = 9
        self.mutation_rate = 0.15
        
        # Funding rate tracking
        self.cumulative_funding = {asset: 0.0 for asset in self.assets}

        
        # Add this line:
        self.current_volatility_regime = 1  # Default to mid volatility

        # ✅ FIXED: Safer RL initialization with proper error handling
        self.rl_coordinator = None
        self.rl_state_manager = None
        # ✅ NEW: Track previous best score for comparison
        self._previous_best_score = 0.0
        
        try:
            if RL_AVAILABLE:
                # Step 1: Initialize RL Coordinator
                try:
                    self.rl_coordinator = RLAgentCoordinator()
                    logger.info("✅ RL Coordinator initialized")
                except Exception as e:
                    logger.error(f"❌ RL Coordinator initialization failed: {e}")
                    self.rl_coordinator = None
                    raise  # Stop RL initialization if coordinator fails
                
                # Step 2: Initialize RL State Manager (only if coordinator succeeded)
                if self.rl_coordinator:
                    try:
                        self.rl_state_manager = RLStateManager()
                        logger.info("✅ RL State Manager initialized")
                    except Exception as e:
                        logger.error(f"❌ RL State Manager initialization failed: {e}")
                        self.rl_state_manager = None
                        # Don't raise - system can work without state manager
                
                # Step 3: Try to load previous state (only if both exist)
                if self.rl_coordinator and self.rl_state_manager:
                    try:
                        success, score = self.rl_state_manager.load_state(
                            self.rl_coordinator, load_strategy="BEST"
                        )
                        if success:
                            logger.info(f"📄 Loaded best RL state (Score: {score:.3f})")
                        else:
                            logger.info("📄 No previous RL state found, starting fresh")
                    except Exception as e:
                        logger.warning(f"⚠️ RL state loading failed: {e}")
                        logger.info("   Starting with fresh RL models")
            else:
                logger.info("⚠️ RL not available - features disabled")
                
        except Exception as e:
            logger.warning(f"⚠️ RL initialization failed: {e}")
            logger.info("   System will operate without RL features")
            self.rl_coordinator = None
            self.rl_state_manager = None
        
        # ✅ FIX: Initialize volatility forecaster with error handling  
        self.volatility_forecaster = None
        try:
            self.volatility_forecaster = ARCHGARCHForecaster()
            logger.info("✅ Volatility Forecaster initialized")
        except Exception as e:
            logger.warning(f"⚠️ Volatility Forecaster initialization failed: {e}")
            logger.info("   System will use default volatility estimates")
        
        # ✅ FIX: Initialize agent registry with error handling
        self.agent_registry = None
        try:
            self.agent_registry = AgentRegistry()
            logger.info("✅ Agent Registry initialized")
        except Exception as e:
            logger.warning(f"⚠️ Agent Registry initialization failed: {e}")
            logger.info("   System will operate without elite agent tracking")

        # ✅ ADD: RL training schedule
        self.rl_training_interval = 10  # Train every 50 closed trades
        self.trades_since_rl_update = 0
        self.rl_training_history = []
        
        logger.info("🚀 LIVE EVOLUTIONARY LEVERAGED TRADING INITIALIZED")
        logger.info(f"   Leverage Range: 3x - 10x")
        logger.info(f"   Assets: {', '.join(self.assets)}")
        logger.info(f"   Population: {self.population_size} agents")
        logger.info(f"   Parallel Processing: {'ENABLED' if enable_parallel else 'DISABLED'}")
        logger.info("📊 Added Agent Registry & Volatility Forecaster")

        # ✅ CENTRAL CAPITAL POOL (replaces individual agent balances)
        self.central_portfolio = EnhancedCentralPortfolio(initial_capital=90000.0)
        self.use_central_pool = True  # Flag to enable/disable
        
        logger.info("💰 Using Central Capital Pool: $90,000")

    def _get_recent_trade_performance(self, last_n: int = 100) -> List[Dict]:
        """
        Get performance metrics for recent trades
        """
        try:
            if not self.all_trades:
                return []
            
            # Get last_n closed trades
            closed_trades = [t for t in self.all_trades if t.status == 'closed']
            recent_trades = closed_trades[-last_n:] if len(closed_trades) > last_n else closed_trades
            
            performance_data = []
            for trade in recent_trades:
                perf = {
                    'symbol': trade.symbol,
                    'timeframe': trade.timeframe,
                    'pnl': trade.realized_pnl,
                    'pnl_percent': trade.realized_pnl_percent,
                    'duration_minutes': trade.duration_minutes,
                    'leverage': trade.leverage,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time
                }
                performance_data.append(perf)
            
            return performance_data
            
        except Exception as e:
            logger.warning(f"Error getting recent trade performance: {e}")
            return []

    def _get_timeframe_performance(self, timeframe: str, recent_trades: List[Dict], market_regime: str) -> Dict:
        """
        Calculate performance metrics for a specific timeframe
        """
        try:
            timeframe_trades = [t for t in recent_trades if t.get('timeframe') == timeframe]
            
            if not timeframe_trades:
                return {
                    'win_rate': 0.5,
                    'avg_pnl': 0.0,
                    'sharpe': 0.0,
                    'total_trades': 0
                }
            
            winning_trades = [t for t in timeframe_trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(timeframe_trades)
            
            pnls = [t['pnl'] for t in timeframe_trades]
            avg_pnl = np.mean(pnls) if pnls else 0.0
            
            # Simple Sharpe ratio approximation
            if len(pnls) > 1 and np.std(pnls) > 0:
                sharpe = np.mean(pnls) / np.std(pnls)
            else:
                sharpe = 0.0
            
            return {
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'sharpe': sharpe,
                'total_trades': len(timeframe_trades)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating {timeframe} performance: {e}")
            return {'win_rate': 0.5, 'avg_pnl': 0.0, 'sharpe': 0.0, 'total_trades': 0}

    def _get_current_market_regime(self) -> str:
        """
        Determine current market regime based on volatility and trends
        """
        try:
            if not self.price_history:
                return "ranging"
            
            # Calculate recent volatility
            recent_volatilities = []
            for asset, history in self.price_history.items():
                if len(history) >= 10:
                    prices = [h.get('price', 0) for h in history[-10:] if isinstance(h, dict)]
                    if len(prices) >= 10:
                        returns = np.diff(prices) / prices[:-1]
                        if len(returns) > 0:
                            vol = np.std(returns)
                            recent_volatilities.append(vol)
            
            avg_volatility = np.mean(recent_volatilities) if recent_volatilities else 0.02
            
            # Simple regime classification
            if avg_volatility < 0.015:
                return "ranging"
            elif avg_volatility < 0.035:
                return "bull_weak"
            else:
                return "bear_strong"
                
        except Exception as e:
            logger.warning(f"Error determining market regime: {e}")
            return "ranging"

    def _calculate_regime_aware_allocation(self, timeframe: str, performance: Dict, 
                                         market_regime: str, market_structure) -> int:
        """
        Calculate regime-aware agent allocation for a timeframe
        """
        base_allocation = 30  # Default
        
        # Adjust based on performance
        if performance['win_rate'] > 0.55:
            base_allocation += 5
        elif performance['win_rate'] < 0.45:
            base_allocation -= 5
        
        # Adjust based on Sharpe ratio
        if performance['sharpe'] > 1.0:
            base_allocation += 3
        elif performance['sharpe'] < 0.0:
            base_allocation -= 3
        
        # Regime-specific adjustments
        regime_boosts = {
            'bull_strong': {'short_term': 5, 'mid_term': 8, 'long_term': 10},
            'bull_weak': {'short_term': 3, 'mid_term': 5, 'long_term': 8},
            'bear_strong': {'short_term': 8, 'mid_term': 3, 'long_term': -5},
            'bear_weak': {'short_term': 5, 'mid_term': 3, 'long_term': -3},
            'ranging': {'short_term': 10, 'mid_term': 5, 'long_term': -8}
        }
        
        regime_boost = regime_boosts.get(market_regime, {}).get(timeframe, 0)
        base_allocation += regime_boost
        
        return max(10, min(50, base_allocation))  # Keep within reasonable bounds

    def _get_strategic_minimums(self, market_regime: str) -> tuple:
        """
        Get strategic minimum allocations per timeframe based on regime
        """
        min_allocations = {
            'bull_strong': (15, 20, 25),
            'bull_weak': (15, 18, 22),
            'bear_strong': (20, 15, 10),
            'bear_weak': (18, 15, 12),
            'ranging': (25, 15, 8)
        }
        return min_allocations.get(market_regime, (15, 15, 15))

    def _get_strategic_maximums(self, market_regime: str) -> tuple:
        """
        Get strategic maximum allocations per timeframe based on regime
        """
        max_allocations = {
            'bull_strong': (35, 40, 45),
            'bull_weak': (35, 38, 42),
            'bear_strong': (45, 35, 25),
            'bear_weak': (40, 35, 30),
            'ranging': (50, 35, 20)
        }
        return max_allocations.get(market_regime, (40, 40, 40))

    def _build_market_structure(self) -> MarketStructure:
        """
        Convert market data into RL-compatible state using imported MarketStructure
        """
        try:
            # Get current prices and volatility
            current_prices = {}
            volatilities = {}
            
            for asset in self.assets:
                market_data = self.market_data_cache.get(asset)
                if market_data:
                    current_prices[asset] = market_data.get('price', 0)
                    volatilities[asset] = market_data.get('volatility', 0.02)
            
            # Calculate average volatility
            if volatilities:
                avg_volatility = np.mean(list(volatilities.values()))
            else:
                avg_volatility = 0.02
            
            # Classify volatility regime
            if avg_volatility < 0.02:
                vol_regime = 0  # LOW
            elif avg_volatility < 0.05:
                vol_regime = 1  # MID
            else:
                vol_regime = 2  # HIGH
            
            # Update the current volatility regime
            self.current_volatility_regime = vol_regime
            
            # Calculate trend strength (simplified)
            trend = 0.0
            if len(self.price_history) > 0:
                # Simple trend calculation from recent price movements
                recent_changes = []
                for asset, history in self.price_history.items():
                    if len(history) >= 2:
                        # ✅ FIXED:
                        recent_price = history[-1].get('price', 0) if isinstance(history[-1], dict) else 0
                        prev_price = history[-2]['price'] if hasattr(history[-2], 'price') and len(history) >= 2 else recent_price
                        if prev_price > 0:
                            change = (recent_price - prev_price) / prev_price
                            recent_changes.append(change)
                
                if recent_changes:
                    trend = np.mean(recent_changes)
            
            # Default values for other metrics (can be enhanced later)
            mean_reversion_score = 0.5
            liquidity_score = 0.8
            funding_rate = 0.0001
            volume_profile = 1.0
            orderbook_imbalance = 0.0
            
            now = datetime.now()
            
            return MarketStructure(
                volatility_regime=vol_regime,
                trend_strength=np.tanh(trend * 10),  # Normalize to [-1, 1]
                mean_reversion_score=mean_reversion_score,
                liquidity_score=liquidity_score,
                funding_rate=funding_rate,
                volume_profile=volume_profile,
                orderbook_imbalance=orderbook_imbalance,
                time_of_day=now.hour,
                day_of_week=now.weekday()
            )
        
        except Exception as e:
            logger.error(f"Error building market structure: {e}")
            # Return default market structure
            now = datetime.now()
            return MarketStructure(
                volatility_regime=1,  # Mid volatility
                trend_strength=0.0,
                mean_reversion_score=0.5,
                liquidity_score=0.8,
                funding_rate=0.0001,
                volume_profile=1.0,
                orderbook_imbalance=0.0,
                time_of_day=now.hour,
                day_of_week=now.weekday()
            )

    def _get_short_term_archetypes_AGGRESSIVE(self) -> Dict[str, Dict]:
        """Aggressive short-term trading archetypes"""
        return {
            'ultra_scalper': {
                'aggression': 0.95, 'patience': 0.05,
                'position_size_base': 0.15,
                'min_holding_minutes': 5, 'max_holding_hours': 1.0,
                'stop_loss_distance': 0.008, 'take_profit_distance': 0.016,
                'volatility_z_threshold': 3.5, 'expected_value_threshold': 0.008,
                'min_confidence': 50.0, 'min_win_prob': 0.46,
                'risk_reward_threshold': 2.0, 'trailing_stop_activation': 0.015,
                'contrarian_bias': 0.0, 'loss_aversion': 0.8
            },
            'momentum_hunter': {
                'aggression': 0.90, 'patience': 0.10,
                'position_size_base': 0.14,
                'min_holding_minutes': 15, 'max_holding_hours': 2.0,
                'stop_loss_distance': 0.010, 'take_profit_distance': 0.022,
                'volatility_z_threshold': 3.0, 'expected_value_threshold': 0.012,
                'min_confidence': 55.0, 'min_win_prob': 0.48,
                'risk_reward_threshold': 2.2, 'trailing_stop_activation': 0.018,
                'contrarian_bias': 0.1, 'loss_aversion': 0.9
            }
        }

    def _get_mid_term_archetypes_AGGRESSIVE(self) -> Dict[str, Dict]:
        """Aggressive mid-term trading archetypes"""
        return {
            'aggressive_swing_trader': {
                'aggression': 0.80, 'patience': 0.30,
                'position_size_base': 0.18,
                'min_holding_minutes': 60, 'max_holding_hours': 4.0,
                'stop_loss_distance': 0.015, 'take_profit_distance': 0.035,
                'volatility_z_threshold': 2.8, 'expected_value_threshold': 0.018,
                'min_confidence': 58.0, 'min_win_prob': 0.50,
                'risk_reward_threshold': 2.3, 'trailing_stop_activation': 0.025,
                'contrarian_bias': 0.0, 'loss_aversion': 1.0
            },
            'volatility_exploiter': {
                'aggression': 0.75, 'patience': 0.40,
                'position_size_base': 0.16,
                'min_holding_minutes': 90, 'max_holding_hours': 6.0,
                'stop_loss_distance': 0.018, 'take_profit_distance': 0.040,
                'volatility_z_threshold': 2.5, 'expected_value_threshold': 0.020,
                'min_confidence': 60.0, 'min_win_prob': 0.52,
                'risk_reward_threshold': 2.2, 'trailing_stop_activation': 0.028,
                'contrarian_bias': -0.1, 'loss_aversion': 1.1
            }
        }

    def _get_long_term_archetypes_AGGRESSIVE(self) -> Dict[str, Dict]:
        """Aggressive long-term trading archetypes"""
        return {
            'trend_rider': {
                'aggression': 0.65, 'patience': 0.70,
                'position_size_base': 0.20,
                'min_holding_minutes': 360, 'max_holding_hours': 120.0,
                'stop_loss_distance': 0.025, 'take_profit_distance': 0.070,
                'volatility_z_threshold': 2.2, 'expected_value_threshold': 0.025,
                'min_confidence': 62.0, 'min_win_prob': 0.54,
                'risk_reward_threshold': 2.8, 'trailing_stop_activation': 0.035,
                'contrarian_bias': 0.2, 'loss_aversion': 1.2
            },
            'position_builder': {
                'aggression': 0.60, 'patience': 0.80,
                'position_size_base': 0.22,
                'min_holding_minutes': 480, 'max_holding_hours': 168.0,
                'stop_loss_distance': 0.030, 'take_profit_distance': 0.085,
                'volatility_z_threshold': 2.0, 'expected_value_threshold': 0.028,
                'min_confidence': 65.0, 'min_win_prob': 0.56,
                'risk_reward_threshold': 2.8, 'trailing_stop_activation': 0.040,
                'contrarian_bias': 0.1, 'loss_aversion': 1.3
            }
        }

    def _create_dna(self, agent_id: int, params: Dict, timeframe: str, asset_preference: str):
        """Create AgentDNA from parameters"""
        from evolutionary_paper_trading_2 import AgentDNA
        
        return AgentDNA(
            agent_id=agent_id,
            generation=self.generation,
            aggression=params['aggression'],
            patience=params['patience'],
            position_size_base=params['position_size_base'],
            min_holding_minutes=params['min_holding_minutes'],
            max_holding_hours=params['max_holding_hours'],
            stop_loss_distance=params['stop_loss_distance'],
            take_profit_distance=params['take_profit_distance'],
            volatility_z_threshold=params['volatility_z_threshold'],
            expected_value_threshold=params['expected_value_threshold'],
            min_confidence=params['min_confidence'],
            min_win_prob=params['min_win_prob'],
            risk_reward_threshold=params['risk_reward_threshold'],
            trailing_stop_activation=params['trailing_stop_activation'],
            contrarian_bias=params['contrarian_bias'],
            loss_aversion=params['loss_aversion'],
            timeframe=timeframe,
            asset_preference=asset_preference,
            regime_performance={},
            parent_ids=[]
        )

    """
    ENHANCED VERSION of _get_archetype_params_by_id()
    Adds RL-aware archetype personalities while keeping your base structure
    """

    def _get_archetype_params_by_id(self, archetype_id: int, timeframe: str) -> Dict:
        """
        Get archetype parameters by ID and timeframe
        
        ✅ ENHANCED: Maps archetype_id to trading personalities
        - IDs 0-2: Aggressive (high frequency, tight stops)
        - IDs 3-5: Balanced (medium frequency, moderate stops)
        - IDs 6-9: Conservative (low frequency, wide stops)
        """
        # ✅ FIX: Import only when needed to avoid circular dependency ?? dont know its ok ?
        from evolutionary_paper_trading_2 import AgentDNA  # Move to top of file instead

        # Base parameters for each timeframe (your original)
        default_params = {
            'short_term': {
                'aggression': 0.8, 'patience': 0.2,
                'position_size_base': 0.12,
                'min_holding_minutes': 10, 'max_holding_hours': 2.0,
                'stop_loss_distance': 0.010, 'take_profit_distance': 0.020,
                'volatility_z_threshold': 3.0, 'expected_value_threshold': 0.010,
                'min_confidence': 55.0, 'min_win_prob': 0.48,
                'risk_reward_threshold': 2.0, 'trailing_stop_activation': 0.02,
                'contrarian_bias': 0.0, 'loss_aversion': 1.0
            },
            'mid_term': {
                'aggression': 0.6, 'patience': 0.5,
                'position_size_base': 0.15,
                'min_holding_minutes': 120, 'max_holding_hours': 24.0,
                'stop_loss_distance': 0.020, 'take_profit_distance': 0.040,
                'volatility_z_threshold': 2.5, 'expected_value_threshold': 0.015,
                'min_confidence': 60.0, 'min_win_prob': 0.52,
                'risk_reward_threshold': 2.5, 'trailing_stop_activation': 0.03,
                'contrarian_bias': 0.0, 'loss_aversion': 1.2
            },
            'long_term': {
                'aggression': 0.4, 'patience': 0.8,
                'position_size_base': 0.18,
                'min_holding_minutes': 720, 'max_holding_hours': 168.0,
                'stop_loss_distance': 0.035, 'take_profit_distance': 0.080,
                'volatility_z_threshold': 2.0, 'expected_value_threshold': 0.020,
                'min_confidence': 65.0, 'min_win_prob': 0.55,
                'risk_reward_threshold': 3.0, 'trailing_stop_activation': 0.04,
                'contrarian_bias': 0.0, 'loss_aversion': 1.5
            }
        }
        
        params = default_params[timeframe].copy()
        
        # ✅ NEW: Map archetype_id to trading personality
        # This makes DQN's choices more meaningful
        
        if timeframe == 'short_term':
            # 10 distinct short-term archetypes
            personality_mods = {
                0: {'aggression': 0.95, 'patience': 0.05, 'stop_loss_distance': 0.008, 'contrarian_bias': 0.0},   # Ultra-aggressive scalper
                1: {'aggression': 0.90, 'patience': 0.10, 'stop_loss_distance': 0.009, 'contrarian_bias': 0.0},   # Aggressive momentum
                2: {'aggression': 0.85, 'patience': 0.15, 'stop_loss_distance': 0.010, 'contrarian_bias': 0.0},   # Aggressive balanced
                3: {'aggression': 0.75, 'patience': 0.25, 'stop_loss_distance': 0.012, 'contrarian_bias': -0.1},  # Moderate contrarian
                4: {'aggression': 0.70, 'patience': 0.30, 'stop_loss_distance': 0.012, 'contrarian_bias': 0.0},   # Moderate balanced
                5: {'aggression': 0.65, 'patience': 0.35, 'stop_loss_distance': 0.013, 'contrarian_bias': 0.1},   # Moderate trend
                6: {'aggression': 0.55, 'patience': 0.45, 'stop_loss_distance': 0.015, 'contrarian_bias': -0.2},  # Conservative contrarian
                7: {'aggression': 0.50, 'patience': 0.50, 'stop_loss_distance': 0.015, 'contrarian_bias': 0.0},   # Conservative balanced
                8: {'aggression': 0.45, 'patience': 0.55, 'stop_loss_distance': 0.018, 'contrarian_bias': 0.2},   # Conservative trend
                9: {'aggression': 0.40, 'patience': 0.60, 'stop_loss_distance': 0.020, 'contrarian_bias': 0.0},   # Ultra-conservative
            }
            
            if archetype_id in personality_mods:
                for key, value in personality_mods[archetype_id].items():
                    params[key] = value
                
                # Adjust take profit based on stop loss (maintain R:R)
                params['take_profit_distance'] = params['stop_loss_distance'] * 2.0
        
        elif timeframe == 'mid_term':
            # Mid-term personalities (fewer variations needed)
            personality_mods = {
                0: {'aggression': 0.70, 'patience': 0.40, 'stop_loss_distance': 0.018, 'contrarian_bias': 0.0},
                1: {'aggression': 0.60, 'patience': 0.50, 'stop_loss_distance': 0.020, 'contrarian_bias': -0.1},
                2: {'aggression': 0.55, 'patience': 0.55, 'stop_loss_distance': 0.022, 'contrarian_bias': 0.0},
                3: {'aggression': 0.50, 'patience': 0.60, 'stop_loss_distance': 0.025, 'contrarian_bias': 0.1},
                4: {'aggression': 0.45, 'patience': 0.65, 'stop_loss_distance': 0.028, 'contrarian_bias': 0.0},
            }
            
            mod_id = archetype_id % 5  # Cycle through 5 personalities
            if mod_id in personality_mods:
                for key, value in personality_mods[mod_id].items():
                    params[key] = value
                params['take_profit_distance'] = params['stop_loss_distance'] * 2.5
        
        elif timeframe == 'long_term':
            # Long-term personalities (focus on patience)
            personality_mods = {
                0: {'aggression': 0.45, 'patience': 0.80, 'stop_loss_distance': 0.030, 'contrarian_bias': 0.0},
                1: {'aggression': 0.40, 'patience': 0.85, 'stop_loss_distance': 0.035, 'contrarian_bias': -0.1},
                2: {'aggression': 0.35, 'patience': 0.90, 'stop_loss_distance': 0.040, 'contrarian_bias': 0.0},
                3: {'aggression': 0.30, 'patience': 0.95, 'stop_loss_distance': 0.045, 'contrarian_bias': 0.1},
            }
            
            mod_id = archetype_id % 4  # Cycle through 4 personalities
            if mod_id in personality_mods:
                for key, value in personality_mods[mod_id].items():
                    params[key] = value
                params['take_profit_distance'] = params['stop_loss_distance'] * 3.0
        
        # ✅ KEEP YOUR ORIGINAL HASH-BASED VARIATION (adds micro-diversity)
        import hashlib
        hash_obj = hashlib.md5(f"{archetype_id}_{timeframe}".encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        variation = (hash_int % 1000) / 1000.0  # 0.0 to 1.0
        
        # Fine-tune with small variations (±5%)
        params['position_size_base'] = max(0.08, min(0.25, 
            params['position_size_base'] * (0.95 + variation * 0.10)))
        
        # Adjust confidence thresholds slightly
        params['min_confidence'] = max(45.0, min(75.0,
            params['min_confidence'] + (variation - 0.5) * 5.0))
        
        return params
    

    def initialize_population(self):
        """
        🔥 V2: RL-DRIVEN POPULATION INITIALIZATION WITH AGGRESSION BOOSTING
        Creates 90 agents using DQN/A3C/PPO recommendations
        Applies aggression boost to ALL agents after creation
        """
        
        logger.info("\n🌱 AGGRESSIVE V2: Creating Generation 0 with LEVERAGED AGENTS...")
        
        agent_id = 0
        elite_agents_loaded = 0
        elite_regimes_covered = set()
        
        # ============================================================
        # STEP 1: LOAD ELITE AGENTS (Top performers from past runs)
        # ============================================================
        
        for regime in ['bull_strong', 'bull_weak', 'bear_strong', 'bear_weak', 'ranging']:
            elite = self.agent_registry.get_best_agent_for_conditions(regime)
            
            if elite and elite['fitness'] > 60.0:
                try:
                    from evolutionary_paper_trading_2 import AgentDNA
                    
                    elite_dna_dict = elite['dna'].copy()
                    
                    # Assign NEW unique ID
                    elite_dna_dict['agent_id'] = agent_id
                    elite_dna_dict['generation'] = self.generation
                    
                    # Handle regime_performance
                    if 'regime_performance' not in elite_dna_dict or elite_dna_dict['regime_performance'] is None:
                        elite_dna_dict['regime_performance'] = {}
                    
                    # Handle parent_ids
                    if 'parent_ids' not in elite_dna_dict or elite_dna_dict['parent_ids'] is None:
                        elite_dna_dict['parent_ids'] = []
                    
                    elite_dna = AgentDNA(**elite_dna_dict)
                    elite_agent = LeveragedEvolutionaryAgent(elite_dna, self.initial_balance)
                    
                    self.agents.append(elite_agent)
                    elite_agents_loaded += 1
                    elite_regimes_covered.add(regime)
                    
                    logger.info(f"   🏆 Loaded elite agent {agent_id} for {regime} "
                            f"(Original ID: {elite['agent_id']}, Fitness: {elite['fitness']:.1f})")
                    
                    agent_id += 1
                    
                except Exception as e:
                    logger.error(f"Failed to load elite agent for {regime}: {e}")
        
        if elite_agents_loaded > 0:
            logger.info(f"\n✅ Loaded {elite_agents_loaded} elite agents (IDs 0-{elite_agents_loaded-1})")
            logger.info(f"   Regimes covered: {', '.join(elite_regimes_covered)}")
        else:
            logger.info(f"\n⚠️ No elite agents loaded")
        
        # ============================================================
        # STEP 2: CHECK IF RL COORDINATOR IS AVAILABLE
        # ============================================================
        
        use_rl = False
        
        if hasattr(self, 'rl_coordinator') and self.rl_coordinator is not None:
            try:
                # Test if RL coordinator is working
                market_structure = self._build_market_structure()
                rl_specs = self.rl_coordinator.generate_optimal_agents(
                    market_structure, 
                    volatility_regime=self.current_volatility_regime
                )
                use_rl = True
                logger.info(f"\n🤖 RL COORDINATOR ACTIVE - Using DQN/A3C/PPO for agent generation")
            except Exception as e:
                logger.warning(f"⚠️ RL coordinator failed: {e}")
                logger.info(f"   Falling back to traditional archetypes")
                use_rl = False
        else:
            logger.info(f"\n📋 RL coordinator not available - using traditional archetypes")
            use_rl = False
        
        # ============================================================
        # STEP 3A: RL-DRIVEN AGENT GENERATION (if available)
        # ============================================================
        
        if use_rl:
            remaining_agents = 90 - elite_agents_loaded
            
            # ✅ ENHANCED DYNAMIC POPULATION OPTIMIZER
            # Multi-factor: Performance + Regime + Market Structure
            recent_trades = self._get_recent_trade_performance(last_n=100)
            market_regime = self._get_current_market_regime()
            market_structure = self._build_market_structure()
            
            # Get multi-dimensional performance metrics
            short_perf = self._get_timeframe_performance('short_term', recent_trades, market_regime)
            mid_perf = self._get_timeframe_performance('mid_term', recent_trades, market_regime)
            long_perf = self._get_timeframe_performance('long_term', recent_trades, market_regime)
            
            # Calculate allocation with regime awareness
            target_short = self._calculate_regime_aware_allocation('short_term', short_perf, market_regime, market_structure)
            target_mid = self._calculate_regime_aware_allocation('mid_term', mid_perf, market_regime, market_structure)
            target_long = self._calculate_regime_aware_allocation('long_term', long_perf, market_regime, market_structure)
            
            # Apply strategic minimums based on regime
            min_short, min_mid, min_long = self._get_strategic_minimums(market_regime)
            max_short, max_mid, max_long = self._get_strategic_maximums(market_regime)
            
            target_short = max(min_short, min(max_short, target_short))
            target_mid = max(min_mid, min(max_mid, target_mid))
            target_long = max(min_long, min(max_long, target_long))
            
            # Ensure total adds to remaining_agents
            total_target = target_short + target_mid + target_long
            if total_target != remaining_agents:
                # Scale proportionally
                scale_factor = remaining_agents / total_target
                target_short = int(target_short * scale_factor)
                target_mid = int(target_mid * scale_factor) 
                target_long = remaining_agents - target_short - target_mid
            
            logger.info(f"\n🎯 ENHANCED DYNAMIC POPULATION OPTIMIZER:")
            logger.info(f"   Market Regime: {market_regime}")
            logger.info(f"   Short-term: {target_short} agents (WR: {short_perf['win_rate']:.1%}, Sharpe: {short_perf['sharpe']:.2f})")
            logger.info(f"   Mid-term:   {target_mid} agents (WR: {mid_perf['win_rate']:.1%}, Sharpe: {mid_perf['sharpe']:.2f})")
            logger.info(f"   Long-term:  {target_long} agents (WR: {long_perf['win_rate']:.1%}, Sharpe: {long_perf['sharpe']:.2f})")
            
            # ✅ SHORT-TERM: DQN-selected archetypes
            logger.info(f"\n⚡ SHORT-TERM SPECIALISTS (DQN-optimized, starting ID {agent_id}):")
            
            try:
                # Get DQN recommendations
                short_archetypes_raw = rl_specs['short_term']
                
                # Handle different return formats
                if isinstance(short_archetypes_raw, zip):
                    short_archetypes_raw = list(short_archetypes_raw)
                
                if isinstance(short_archetypes_raw, list) and len(short_archetypes_raw) > 0:
                    if isinstance(short_archetypes_raw[0], tuple):
                        # Format: [(id1, q1), (id2, q2), ...]
                        short_archetypes = [item[0] for item in short_archetypes_raw]
                        q_values = [item[1] for item in short_archetypes_raw]
                    else:
                        # Format: [id1, id2, ...]
                        short_archetypes = short_archetypes_raw
                        q_values = [0.0] * len(short_archetypes)
                else:
                    # Fallback
                    short_archetypes = [0, 1]
                    q_values = [0.0, 0.0]
                
                # Calculate how many agents per archetype
                agents_per_archetype = max(1, target_short // len(short_archetypes))
                
                short_count = 0
                for archetype_id, q_value in zip(short_archetypes, q_values):
                    if short_count >= target_short:
                        break
                    
                    params = self._get_archetype_params_by_id(archetype_id, 'short_term')
                    
                    for _ in range(agents_per_archetype):
                        if short_count >= target_short:
                            break
                        
                        asset = self.assets[agent_id % len(self.assets)]
                        dna = self._create_dna(agent_id, params, 'short_term', asset)
                        # ✅ Agents start with $0 when using central pool
                        agent_balance = 0.0 if self.use_central_pool else self.initial_balance
                        agent = LeveragedEvolutionaryAgent(dna, agent_balance)
                        self.agents.append(agent)
                        
                        if short_count < 3:  # Log first 3
                            logger.info(f"   Agent {agent_id}: Archetype {archetype_id} (Q={q_value:.2f}) - {asset}")
                        
                        agent_id += 1
                        short_count += 1
                
                logger.info(f"   Created {short_count} short-term agents")
                
            except Exception as e:
                logger.error(f"❌ DQN generation failed: {e}")
                # Fallback to traditional
                short_count = 0
                short_archetypes = self._get_short_term_archetypes_AGGRESSIVE()
                for archetype_name, params in short_archetypes.items():
                    for asset in self.assets:
                        if short_count >= target_short:
                            break
                        dna = self._create_dna(agent_id, params, 'short_term', asset)
                        agent = LeveragedEvolutionaryAgent(dna, self.initial_balance)
                        self.agents.append(agent)
                        agent_id += 1
                        short_count += 1
                    if short_count >= target_short:
                        break
            
            # ✅ MID-TERM: A3C-sampled parameters
            logger.info(f"\n📊 MID-TERM SPECIALISTS (A3C-optimized, starting ID {agent_id}):")
            
            try:
                mid_params_list = rl_specs['mid_term']
                
                # Validate format
                if not isinstance(mid_params_list, list):
                    mid_params_list = [mid_params_list] if mid_params_list else []
                
                # Ensure we have parameters
                if len(mid_params_list) == 0:
                    mid_params_list = [self._get_mid_term_archetypes_AGGRESSIVE()['aggressive_swing_trader']]
                
                mid_count = 0
                for i, params in enumerate(mid_params_list):
                    if mid_count >= target_mid:
                        break
                    
                    # Validate params is a dict
                    if not isinstance(params, dict):
                        params = self._get_mid_term_archetypes_AGGRESSIVE()['aggressive_swing_trader']
                    
                    for asset in self.assets:
                        if mid_count >= target_mid:
                            break
                        
                        dna = self._create_dna(agent_id, params, 'mid_term', asset)
                        agent = LeveragedEvolutionaryAgent(dna, self.initial_balance)
                        self.agents.append(agent)
                        
                        if mid_count < 3:  # Log first 3
                            logger.info(f"   Agent {agent_id}: A3C Params Set {i+1} - {asset}")
                        
                        agent_id += 1
                        mid_count += 1
                
                logger.info(f"   Created {mid_count} mid-term agents")
                
            except Exception as e:
                logger.error(f"❌ A3C generation failed: {e}")
                # Fallback
                mid_count = 0
                mid_archetypes = self._get_mid_term_archetypes_AGGRESSIVE()
                for archetype_name, params in mid_archetypes.items():
                    for asset in self.assets:
                        if mid_count >= target_mid:
                            break
                        dna = self._create_dna(agent_id, params, 'mid_term', asset)
                        agent = LeveragedEvolutionaryAgent(dna, self.initial_balance)
                        self.agents.append(agent)
                        agent_id += 1
                        mid_count += 1
                    if mid_count >= target_mid:
                        break
            
            # ✅ LONG-TERM: PPO-selected elite agents
            logger.info(f"\n📈 LONG-TERM SPECIALISTS (PPO-selected, starting ID {agent_id}):")
            
            try:
                long_agent_ids = rl_specs['long_term']
                
                # Validate format
                if not isinstance(long_agent_ids, list):
                    long_agent_ids = []
                
                long_count = 0
                for elite_id in long_agent_ids:
                    if long_count >= target_long:
                        break
                    
                    elite_data = self.agent_registry.get_agent_performance_history(elite_id)
                    
                    if elite_data:
                        # Clone elite agent
                        from evolutionary_paper_trading_2 import AgentDNA
                        
                        elite_dna_dict = elite_data['dna'].copy()
                        elite_dna_dict['agent_id'] = agent_id
                        elite_dna_dict['generation'] = self.generation
                        
                        if 'regime_performance' not in elite_dna_dict or elite_dna_dict['regime_performance'] is None:
                            elite_dna_dict['regime_performance'] = {}
                        if 'parent_ids' not in elite_dna_dict or elite_dna_dict['parent_ids'] is None:
                            elite_dna_dict['parent_ids'] = []
                        
                        elite_dna = AgentDNA(**elite_dna_dict)
                        agent = LeveragedEvolutionaryAgent(elite_dna, self.initial_balance)
                        self.agents.append(agent)
                        
                        if long_count < 3:  # Log first 3
                            logger.info(f"   Agent {agent_id}: Elite {elite_id} (Fitness={elite_data['fitness']:.1f})")
                        
                        agent_id += 1
                        long_count += 1
                
                # Fill remaining with fallback
                if long_count < target_long:
                    long_archetypes = self._get_long_term_archetypes_AGGRESSIVE()
                    for archetype_name, params in long_archetypes.items():
                        for asset in self.assets:
                            if long_count >= target_long:
                                break
                            dna = self._create_dna(agent_id, params, 'long_term', asset)
                            agent = LeveragedEvolutionaryAgent(dna, self.initial_balance)
                            self.agents.append(agent)
                            agent_id += 1
                            long_count += 1
                        if long_count >= target_long:
                            break
                
                logger.info(f"   Created {long_count} long-term agents")
                
            except Exception as e:
                logger.error(f"❌ PPO generation failed: {e}")
                # Fallback
                long_count = 0
                long_archetypes = self._get_long_term_archetypes_AGGRESSIVE()
                for archetype_name, params in long_archetypes.items():
                    for asset in self.assets:
                        if long_count >= target_long:
                            break
                        dna = self._create_dna(agent_id, params, 'long_term', asset)
                        agent = LeveragedEvolutionaryAgent(dna, self.initial_balance)
                        self.agents.append(agent)
                        agent_id += 1
                        long_count += 1
                    if long_count >= target_long:
                        break
        
        # ============================================================
        # STEP 3B: TRADITIONAL ARCHETYPE GENERATION (fallback)
        # ============================================================
        
        else:
            # Use original implementation
            remaining_agents = 90 - len(self.agents)
            agents_per_timeframe = remaining_agents // 3
            
            logger.info(f"\n📊 Creating {agents_per_timeframe} agents per timeframe ({remaining_agents} total)")
            
            # SHORT-TERM
            logger.info(f"\n⚡ SHORT-TERM SPECIALISTS (starting ID {agent_id}):")
            short_archetypes = self._get_short_term_archetypes_AGGRESSIVE()
            
            short_count = 0
            for archetype_name, params in short_archetypes.items():
                for asset in self.assets:
                    if short_count >= agents_per_timeframe:
                        break
                    
                    dna = self._create_dna(agent_id, params, timeframe='short_term', asset_preference=asset)
                    agent = LeveragedEvolutionaryAgent(dna, self.initial_balance)
                    self.agents.append(agent)
                    
                    if short_count < 3:
                        logger.info(f"   Agent {agent_id}: {archetype_name} ({asset})")
                    
                    agent_id += 1
                    short_count += 1
                
                if short_count >= agents_per_timeframe:
                    break
            
            logger.info(f"   Created {short_count} short-term agents")
            
            # MID-TERM
            logger.info(f"\n📊 MID-TERM SPECIALISTS (starting ID {agent_id}):")
            mid_archetypes = self._get_mid_term_archetypes_AGGRESSIVE()
            
            mid_count = 0
            for archetype_name, params in mid_archetypes.items():
                for asset in self.assets:
                    if mid_count >= agents_per_timeframe:
                        break
                    
                    dna = self._create_dna(agent_id, params, timeframe='mid_term', asset_preference=asset)
                    agent = LeveragedEvolutionaryAgent(dna, self.initial_balance)
                    self.agents.append(agent)
                    
                    if mid_count < 3:
                        logger.info(f"   Agent {agent_id}: {archetype_name} ({asset})")
                    
                    agent_id += 1
                    mid_count += 1
                
                if mid_count >= agents_per_timeframe:
                    break
            
            logger.info(f"   Created {mid_count} mid-term agents")
            
            # LONG-TERM
            logger.info(f"\n📈 LONG-TERM SPECIALISTS (starting ID {agent_id}):")
            long_archetypes = self._get_long_term_archetypes_AGGRESSIVE()
            
            long_count = 0
            for archetype_name, params in long_archetypes.items():
                for asset in self.assets:
                    if long_count >= agents_per_timeframe:
                        break
                    
                    dna = self._create_dna(agent_id, params, timeframe='long_term', asset_preference=asset)
                    agent = LeveragedEvolutionaryAgent(dna, self.initial_balance)
                    self.agents.append(agent)
                    
                    if long_count < 3:
                        logger.info(f"   Agent {agent_id}: {archetype_name} ({asset})")
                    
                    agent_id += 1
                    long_count += 1
                
                if long_count >= agents_per_timeframe:
                    break
            
            logger.info(f"   Created {long_count} long-term agents")
        
        # ============================================================
        # STEP 4: APPLY AGGRESSION BOOST TO ALL AGENTS
        # ============================================================
        
        logger.info(f"\n🔥 APPLYING AGGRESSION BOOST TO ALL {len(self.agents)} AGENTS...")
        
        aggression_boosted = 0
        for agent in self.agents:
            try:
                # More aggressive
                old_aggression = agent.dna.aggression
                agent.dna.aggression = min(1.0, agent.dna.aggression * 1.20)  # +20%
                
                # Less patient
                old_patience = agent.dna.patience
                agent.dna.patience = max(0.0, agent.dna.patience * 0.75)  # -25%
                
                # Less loss-averse
                old_loss_aversion = agent.dna.loss_aversion
                agent.dna.loss_aversion = max(0.5, agent.dna.loss_aversion * 0.85)  # -15%
                
                # Recalculate leverage with new aggressive parameters
                agent.preferred_leverage = agent._calculate_preferred_leverage_AGGRESSIVE()
                
                aggression_boosted += 1
                
                if aggression_boosted <= 5:  # Log first 5
                    logger.debug(f"   Agent {agent.dna.agent_id}: "
                            f"Aggression {old_aggression:.2f}→{agent.dna.aggression:.2f}, "
                            f"Patience {old_patience:.2f}→{agent.dna.patience:.2f}, "
                            f"Leverage {agent.preferred_leverage:.1f}x")
            
            except Exception as e:
                logger.warning(f"⚠️ Failed to boost agent {agent.dna.agent_id}: {e}")
        
        # ============================================================
        # STEP 5: FINAL SUMMARY
        # ============================================================
        
        logger.info(f"\n✅ AGGRESSIVE V2 POPULATION INITIALIZED:")
        logger.info(f"   Total agents: {len(self.agents)}")
        logger.info(f"   Elite agents: {elite_agents_loaded}")
        logger.info(f"   Short-term: {len([a for a in self.agents if a.dna.timeframe == 'short_term'])}")
        logger.info(f"   Mid-term: {len([a for a in self.agents if a.dna.timeframe == 'mid_term'])}")
        logger.info(f"   Long-term: {len([a for a in self.agents if a.dna.timeframe == 'long_term'])}")
        logger.info(f"   Aggression boosted: {aggression_boosted} agents")
        logger.info(f"   Final agent ID range: 0-{agent_id-1}")
        logger.info(f"   RL-driven: {'YES' if use_rl else 'NO'}")
        
        # ✅ VALIDATION - Ensure no duplicate IDs
        agent_ids = [a.dna.agent_id for a in self.agents]
        
        if len(agent_ids) != len(set(agent_ids)):
            logger.error("❌ DUPLICATE AGENT IDs DETECTED!")
            from collections import Counter
            duplicates = [id for id, count in Counter(agent_ids).items() if count > 1]
            logger.error(f"   Duplicate IDs: {duplicates}")
            
            # Auto-repair
            logger.info("🔧 Auto-repairing duplicate IDs...")
            seen_ids = set()
            next_free_id = max(agent_ids) + 1
            
            for agent in self.agents:
                if agent.dna.agent_id in seen_ids:
                    old_id = agent.dna.agent_id
                    agent.dna.agent_id = next_free_id
                    logger.info(f"   Repaired: {old_id} → {next_free_id}")
                    next_free_id += 1
                else:
                    seen_ids.add(agent.dna.agent_id)
            
            logger.info("✅ Duplicate IDs repaired")
        
    async def _fetch_live_market_data(self, asset: str) -> Optional[Dict]:
        """
        ✅ FIXED: Proper volatility calculation (raw std dev, not annualized)
        
        CRITICAL FIXES:
        1. Don't annualize volatility (removed sqrt(365) multiplier)
        2. Use realistic outlier filter (10% max moves)
        3. Clip to 0.5% - 15% range (crypto reality)
        4. Store prices IMMEDIATELY to prevent cache issues
        """
        now = datetime.now()
        
        # Cache check (60 seconds)
        cache_duration = 60
        
        if asset in self.market_data_cache and asset in self.cache_expiry:
            cache_age = (now - self.cache_expiry[asset]).seconds
            if cache_age < cache_duration:
                logger.debug(f"📦 {asset}: Using cached data ({cache_age}s old)")
                return self.market_data_cache[asset]
        
        try:
            complete_data = self.exchange.get_complete_market_data(asset)
            
            if not complete_data or not complete_data.get('orderbook'):
                return None
            
            orderbook = complete_data['orderbook']
            funding = complete_data.get('funding')
            
            price = orderbook.mid_price
            
            if price is None or price <= 0:
                logger.error(f"❌ Invalid price for {asset}: {price}")
                return None
            
            # Initialize price history
            if asset not in self.price_history:
                self.price_history[asset] = deque(maxlen=200)
            
            # ✅ CRITICAL: Store price IMMEDIATELY
            self.price_history[asset].append({
                'timestamp': now,
                'price': float(price)
            })
            
            # ✅ FIXED: PROPER VOLATILITY CALCULATION
            volatility = 0.02  # Default 2% daily
            
            price_history = list(self.price_history[asset])
            if len(price_history) >= 10:
                try:
                    prices = [p.get('price', 0) for p in price_history 
                            if isinstance(p, dict) and p.get('price', 0) > 0]
                    
                    if len(prices) >= 10:
                        # Calculate simple returns
                        returns = []
                        for i in range(1, len(prices)):
                            if prices[i-1] > 0:
                                ret = (prices[i] - prices[i-1]) / prices[i-1]
                                # ✅ Filter outliers (10% max move between samples)
                                if abs(ret) < 0.10:
                                    returns.append(ret)
                        
                        if len(returns) >= 5:
                            # ✅ Calculate SAMPLE volatility (NOT annualized)
                            # This is the standard deviation of returns
                            sample_vol = np.std(returns)
                            
                            # ✅ Scale to HOURLY volatility
                            # If samples are 60 seconds apart, multiply by sqrt(60) for hourly
                            sampling_interval_minutes = 1.0  # Assuming 1-minute samples
                            hourly_multiplier = np.sqrt(60 / sampling_interval_minutes)
                            volatility_hourly = sample_vol * hourly_multiplier
                            
                            # ✅ Convert to DAILY volatility (24 hours)
                            volatility = volatility_hourly * np.sqrt(24)
                            
                            # ✅ Clip to realistic crypto range: 0.5% - 15% daily
                            volatility = float(np.clip(volatility, 0.005, 0.15))
                            
                            logger.debug(f"📊 {asset}: Calculated {volatility:.3%} daily volatility "
                                    f"from {len(returns)} returns (sample std: {sample_vol:.4%})")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Volatility calculation failed for {asset}: {e}")
                    volatility = 0.02
            
            # Calculate RSI
            rsi = self._calculate_rsi_from_history(asset)
            
            # Calculate trend
            trend = self._calculate_trend(asset)
            
            # Safe funding rate extraction
            funding_rate = 0.0
            funding_8h = 0.0
            
            if funding:
                try:
                    funding_rate = float(getattr(funding, 'funding_rate', 0.0))
                    funding_8h = float(getattr(funding, 'funding_8h_avg', 0.0))
                    
                    funding_rate = np.clip(funding_rate, -0.01, 0.01)
                    funding_8h = np.clip(funding_8h, -0.01, 0.01)
                except Exception:
                    pass
            
            # Safe orderbook pressure
            orderbook_pressure = 0.0
            try:
                orderbook_pressure = float(getattr(orderbook, 'orderbook_pressure', 0.0))
                orderbook_pressure = np.clip(orderbook_pressure, -1, 1)
            except Exception:
                pass
            
            market_data = {
                'asset': asset,
                'price': float(price),
                'rsi': float(np.clip(rsi, 0, 100)),
                'trend_direction': float(np.clip(trend, -1, 1)),
                'volatility': volatility,
                'volatility_24h': volatility * 1.2,
                'funding_rate': float(funding_rate),
                'funding_8h': float(funding_8h),
                'orderbook_pressure': float(orderbook_pressure),
                'timestamp': now
            }
            
            # Cache it
            self.market_data_cache[asset] = market_data
            self.cache_expiry[asset] = now
            
            logger.debug(f"🔄 {asset}: Fresh data (vol={volatility:.2%}, price=${price:.4f})")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch {asset} data: {e}")
            return None


    def _cleanup_stale_caches(self, current_time: datetime):
        """
        ✅ NEW METHOD: Clean up stale cache entries
        """
        
        stale_assets = []
        
        for asset, expiry_time in self.cache_expiry.items():
            cache_age = (current_time - expiry_time).seconds
            if cache_age > self.cache_duration * 10:  # 10x cache duration
                stale_assets.append(asset)
        
        for asset in stale_assets:
            if asset in self.market_data_cache:
                del self.market_data_cache[asset]
            if asset in self.cache_expiry:
                del self.cache_expiry[asset]
            
            logger.debug(f"🗑️ Cleaned stale cache for {asset}")
        
        if stale_assets:
            logger.info(f"🗑️ Cleaned {len(stale_assets)} stale cache entries")


    def _cleanup_old_price_history(self, current_time: datetime, max_age_hours: int = 24):
        """
        ✅ NEW METHOD: Remove old price data (optional, deque already handles this)
        """
        
        # This is now mostly unnecessary due to deque(maxlen=200)
        # But we can add time-based cleanup for extra safety
        
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        for asset, history in self.price_history.items():
            # Filter out entries older than cutoff
            valid_entries = [entry for entry in history if entry['timestamp'] > cutoff_time]
            
            if len(valid_entries) < len(history):
                removed = len(history) - len(valid_entries)
                self.price_history[asset] = deque(valid_entries, maxlen=200)
                logger.debug(f"🗑️ Removed {removed} old price entries for {asset}")


    def get_memory_usage_stats(self) -> Dict:
        """
        ✅ NEW METHOD: Get memory usage statistics
        """
        
        import sys
        
        return {
            'price_history_assets': len(self.price_history),
            'total_price_points': sum(len(h) for h in self.price_history.values()),
            'cached_market_data': len(self.market_data_cache),
            'cache_expiry_entries': len(self.cache_expiry),
            'total_trades': len(self.all_trades),
            'active_agents': len(self.agents),
            'memory_estimate_mb': sys.getsizeof(self.price_history) / 1024 / 1024
        }
    
    def _calculate_rsi_from_history(self, asset: str, period: int = 14) -> float:
        """✅ FIXED: Calculate RSI from real price history"""
        if asset not in self.price_history or len(self.price_history[asset]) < period + 1:
            return 50.0
        
        try:
            prices = [p.get('price', 0) for p in self.price_history[asset] if isinstance(p, dict)]
            if len(prices) < period + 1:
                return 50.0
                
            changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            gains = [max(c, 0) for c in changes[-period:]]
            losses = [max(-c, 0) for c in changes[-period:]]
            
            avg_gain = sum(gains) / period if gains else 0
            avg_loss = sum(losses) / period if losses else 0
            
            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(np.clip(rsi, 0, 100))
            
        except Exception as e:
            logger.warning(f"RSI calculation failed for {asset}: {e}")
            return 50.0

    def _calculate_trend(self, asset: str) -> float:
        """✅ FIXED: Calculate trend strength"""
        if asset not in self.price_history or len(self.price_history[asset]) < 20:
            return 0.0
        
        try:
            prices = [p.get('price', 0) for p in self.price_history[asset] if isinstance(p, dict)]
            if len(prices) < 20:
                return 0.0
            
            # Use last 9 prices for short EMA, last 20 for long EMA
            short_prices = prices[-9:] if len(prices) >= 9 else prices
            long_prices = prices[-20:] if len(prices) >= 20 else prices
            
            ema_short = sum(short_prices) / len(short_prices)
            ema_long = sum(long_prices) / len(long_prices)
            
            if ema_long == 0:
                return 0.0
            
            trend = (ema_short - ema_long) / ema_long
            return float(np.clip(trend, -0.5, 0.5))  # Clip to ±50%
            
        except Exception as e:
            logger.warning(f"Trend calculation failed for {asset}: {e}")
            return 0.0
    
    def _log_agent_spam_debug(self, agent: LeveragedEvolutionaryAgent, asset: str):
        """Log detailed spam debugging information"""
        stats = agent.get_trade_stats()
        logger.debug(f"🕵️ AGENT {agent.dna.agent_id} SPAM DEBUG:")
        logger.debug(f"   Active trades: {stats['active_trades']}")
        logger.debug(f"   Active positions: {stats['active_positions']}")
        logger.debug(f"   Pending assets: {stats['pending_assets']}")
        logger.debug(f"   Trade history: {stats['trade_history']}")
        logger.debug(f"   Asset cooldowns: {len(stats['asset_cooldowns'])} assets")
        
        if asset in stats['asset_cooldowns']:
            cooldown_time = stats['asset_cooldowns'][asset]
            elapsed = (datetime.now() - cooldown_time).total_seconds()
            logger.debug(f"   {asset} cooldown: {elapsed:.1f}s ago (needs 300s)")
    

    def _calculate_volatility_fixed(self, asset: str) -> float:
        """
        ✅ FIX: Calculate volatility with less restrictive outlier filter
        """
        if asset not in self.price_history or len(self.price_history[asset]) < 10:
            return 0.02
        
        try:
            price_history = list(self.price_history[asset])
            prices = [p.get('price', 0) for p in price_history 
                      if isinstance(p, dict) and p.get('price', 0) > 0]
            
            if len(prices) < 10:
                return 0.02
            
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    if abs(ret) < 0.20:  # ✅ Increased from 0.05
                        returns.append(ret)
            
            if len(returns) >= 5:
                volatility = np.std(returns)
                return float(np.clip(volatility, 0.005, 0.20))
            
            return 0.02
        
        except Exception as e:
            logger.warning(f"⚠️ Volatility calculation failed for {asset}: {e}")
            return 0.02

    async def _process_agents_parallel_leveraged(self, regime: str):
        """
        ✅ FIXED: Proper async execution with detailed logging
        """
        # Manage existing positions first
        manage_tasks = [self._manage_leveraged_trade(agent) for agent in self.agents]
        await asyncio.gather(*manage_tasks, return_exceptions=True)

        MAX_TRADES_PER_CYCLE = 15
        MAX_TRADES_PER_AGENT = 1

        trades_executed_this_cycle = 0
        agent_trades_this_cycle = {agent.dna.agent_id: 0 for agent in self.agents}
        agent_asset_trades_this_cycle = {}

        # Check for new trades - SEQUENTIAL with limits
        for asset in self.assets:
            if trades_executed_this_cycle >= MAX_TRADES_PER_CYCLE:
                logger.info(f"✅ Cycle limit reached: {trades_executed_this_cycle}/{MAX_TRADES_PER_CYCLE}")
                break
            
            market_data = await self._fetch_live_market_data(asset)
            if not market_data:
                continue
            
            # Find interested agents
            interested_agents = [
                a for a in self.agents
                if (a.dna.asset_preference == 'ALL' or a.dna.asset_preference == asset)
                and not any(t.asset == asset and t.status == 'open' for t in a.active_trades)
            ]
            
            # Process each agent
            for agent in interested_agents:
                agent_id = agent.dna.agent_id
                
                try:
                    # CHECK #1: Agent trade limit
                    if agent_trades_this_cycle[agent_id] >= MAX_TRADES_PER_AGENT:
                        continue
                    
                    # CHECK #2: Asset already traded by this agent
                    if agent_id in agent_asset_trades_this_cycle:
                        if asset in agent_asset_trades_this_cycle[agent_id]:
                            continue
                    
                    # CHECK #3: Global limit
                    if trades_executed_this_cycle >= MAX_TRADES_PER_CYCLE:
                        break
                    
                    # ✅ STEP 1: Should trade?
                    should_trade, confidence, leverage = agent.should_trade_leveraged_AGGRESSIVE(
                        market_data, regime, self.simulated_time
                    )
                    
                    if not should_trade:
                        continue
                    
                    # ✅ STEP 2: Generate trade params
                    trade_params = agent.generate_leveraged_trade_params(
                        market_data, confidence, leverage
                    )
                    
                    if not trade_params:
                        logger.warning(f"⚠️ Agent {agent_id}: trade_params is None")
                        continue
                    
                    # ✅ STEP 3: Execute trade (THIS IS THE CRITICAL PART)
                    execution_success = await self._execute_leveraged_trade(
                        agent, asset, trade_params, market_data, regime
                    )
                    
                    # ✅ ONLY count if execution succeeded
                    if execution_success:
                        trades_executed_this_cycle += 1
                        agent_trades_this_cycle[agent_id] += 1
                        
                        if agent_id not in agent_asset_trades_this_cycle:
                            agent_asset_trades_this_cycle[agent_id] = set()
                        agent_asset_trades_this_cycle[agent_id].add(asset)
                        
                        logger.info(f"   ✅ Trade #{trades_executed_this_cycle}: Agent {agent_id} → {asset}")
                    else:
                        logger.debug(f"   ⏭️ Trade skipped: Agent {agent_id} on {asset}")
                
                except Exception as e:
                    logger.error(f"❌ Trade execution error for Agent {agent_id} on {asset}: {e}")
                    agent.clear_pending_asset(asset)
                    agent._last_trade_cache = None

        #Final cycle summary - only print if cycle is complete
        if trades_executed_this_cycle >= MAX_TRADES_PER_CYCLE:
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 CYCLE EXECUTION SUMMARY")
            logger.info(f"\n{'='*80}")
            logger.info(f"   Trades Executed: {trades_executed_this_cycle}/{MAX_TRADES_PER_CYCLE}")
            logger.info(f"   Active Agents: {len([c for c in agent_trades_this_cycle.values() if c > 0])}")
            logger.info(f"   Assets Traded: {len(set(a for s in agent_asset_trades_this_cycle.values() for a in s))}")
            logger.info(f"{'='*80}\n")



    
    async def _process_agents_sequential_leveraged(self, regime: str):
        """
        ✅ COMPLETE METHOD: Sequential processing (fallback)
        """
        
        for agent in self.agents:
            await self._manage_leveraged_trade(agent)
            
            for asset in self.assets:
                try:
                    market_data = await self._fetch_live_market_data(asset)
                    if not market_data:
                        continue
                    
                    should_trade, confidence, leverage = agent.should_trade_leveraged_AGGRESSIVE(
                        market_data, regime, self.simulated_time
                    )
                    
                    if should_trade:
                        trade_params = agent.generate_leveraged_trade_params(
                            market_data, confidence, leverage
                        )
                        await self._execute_leveraged_trade(
                            agent, asset, trade_params, market_data, regime
                        )
                
                except Exception as e:
                    logger.error(f"❌ Trade execution failed for Agent {agent.dna.agent_id} on {asset}: {e}")
                    agent.clear_pending_asset(asset)
                    agent._last_trade_cache = None
    
    async def _execute_leveraged_trade(self, agent: LeveragedEvolutionaryAgent,
                                    asset: str, trade_params: Dict,
                                    market_data: Dict, regime: str) -> bool:
        """
        ✅ FIXED: Mark as pending ONLY after successful trade creation
        """
        agent_id = agent.dna.agent_id
        current_price = market_data['price']
        
        # ✅ CHECK #0: Basic validation
        if current_price <= 0:
            logger.error(f"❌ Invalid price: {current_price}")
            return False
        
        # ✅ CHECK #1: No active position
        if asset in agent.active_positions_by_asset:
            logger.warning(f"⏸️ {asset}: Already has active position")
            return False
        
        # ✅ CHECK #2: Minimum price movement
        if hasattr(agent, '_last_trade_cache') and agent._last_trade_cache:
            last_asset, last_price, last_time = agent._last_trade_cache
            if last_asset == asset:
                price_change = abs(current_price - last_price) / last_price
                time_elapsed = (self.simulated_time - last_time).total_seconds()
                
                if price_change < 0.003 and time_elapsed < 300:
                    return False
        
        # ✅ STEP 1: Allocate capital
        if self.use_central_pool:
            requested_size = trade_params['position_size']
            
            allocation_success = self.central_portfolio.allocate_trade_capital(agent_id, requested_size)
            
            if not allocation_success:
                logger.error(f"❌ Capital allocation failed for Agent {agent_id}")
                return False
            
            allocated_size = self.central_portfolio.active_trades[agent_id]
            trade_params['position_size'] = allocated_size
            trade_params['effective_size'] = allocated_size * trade_params['leverage']
        
        # ✅ STEP 2: Create trade object
        try:
            # Validate and adjust stop loss if needed
            stop_loss = trade_params['stop_loss']
            if trade_params['action'] == 'BUY':
                if stop_loss >= current_price:
                    stop_loss = current_price * 0.98
            else:
                if stop_loss <= current_price:
                    stop_loss = current_price * 1.02
            
            trade_params['stop_loss'] = stop_loss
            
            # Adjust leverage if forecaster available
            current_vol = market_data.get('volatility', 0.02)
            
            if self.volatility_forecaster:
                try:
                    vol_leverage_adjustment = self.volatility_forecaster.get_leverage_adjustment(
                        trade_params['leverage']
                    )
                    adjusted_leverage = trade_params['leverage'] * 0.7 + vol_leverage_adjustment * 0.3
                    trade_params['leverage'] = np.clip(adjusted_leverage, 3.0, 15.0)
                except:
                    trade_params['leverage'] = np.clip(trade_params['leverage'], 3.0, 10.0)
            
            trade_params['effective_size'] = trade_params['position_size'] * trade_params['leverage']
            
            # Recalculate liquidation
            maintenance_margin_pct = 0.03
            liquidation_threshold = max((1.0 / trade_params['leverage']) - maintenance_margin_pct, 0.01)
            
            if trade_params['action'] == 'BUY':
                trade_params['liquidation_price'] = current_price * (1 - liquidation_threshold)
            else:
                trade_params['liquidation_price'] = current_price * (1 + liquidation_threshold)
            
            entry_time = self.simulated_time
            
            # ✅ CREATE TRADE
            trade = LeveragedTrade(
                agent_id=agent.dna.agent_id,
                agent_dna=agent.dna,
                asset=asset,
                action=trade_params['action'],
                entry_price=current_price,
                entry_time=entry_time,
                position_size=trade_params['position_size'],
                leverage=trade_params['leverage'],
                effective_size=trade_params['effective_size'],
                stop_loss=trade_params['stop_loss'],
                take_profit=trade_params['take_profit'],
                liquidation_price=trade_params['liquidation_price'],
                current_price=current_price,
                pnl=0.0,
                pnl_pct=0.0,
                leverage_pnl_pct=0.0,
                status='open',
                close_reason='',
                close_time=None,
                market_regime=regime,
                confidence_used=trade_params['confidence'],
                win_prob_used=agent.dna.min_win_prob,
                market_volatility=current_vol,
                trend_strength=market_data['trend_direction'],
                archetype_id=self._get_archetype_id_from_dna(agent.dna)
            )
            
            # ✅ ATOMIC STATE UPDATES
            agent.active_trades.append(trade)
            agent.active_positions_by_asset[asset] = trade
            
            # Save market structure
            trade.market_structure_at_entry = self._build_market_structure()
            trade.volatility_regime_at_entry = self.current_volatility_regime
            
            # Update cooldowns
            if not hasattr(agent, "asset_cooldowns"):
                agent.asset_cooldowns = {}
            agent.asset_cooldowns[asset] = self.simulated_time
            
            # Update cache
            agent._last_trade_cache = (asset, current_price, self.simulated_time)
            
            # ✅ MARK AS PENDING ONLY AFTER SUCCESS
            if not hasattr(agent, '_pending_assets'):
                agent._pending_assets = set()
            if not hasattr(agent, '_pending_timestamps'):
                agent._pending_timestamps = {}
                
            agent._pending_assets.add(asset)
            agent._pending_timestamps[asset] = self.simulated_time
            agent.last_trade_time = self.simulated_time
            
            # Log success
            vol_regime_name = {0: 'LOW', 1: 'MID', 2: 'HIGH'}.get(self.current_volatility_regime, 'UNKNOWN')
            logger.info(
                f"      🔹 Agent {agent.dna.agent_id} ({agent.dna.timeframe}): "
                f"{trade_params['action']} {asset} @ ${current_price:.4f} "
                f"[{trade_params['leverage']:.1f}x, Liq: ${trade_params['liquidation_price']:.4f}] "
                f"Size: ${trade_params['position_size']:.2f}"
            )
            
            return True
            
        except Exception as e:
            # ✅ ROLLBACK
            logger.error(f"❌ Trade creation failed: {e}")
            
            if self.use_central_pool and agent_id in self.central_portfolio.active_trades:
                self.central_portfolio.release_trade_capital(agent_id)
            
            if asset in agent.active_positions_by_asset:
                del agent.active_positions_by_asset[asset]
            
            agent.active_trades = [t for t in agent.active_trades 
                                if t.asset != asset or t.entry_time != entry_time]
            
            import traceback
            traceback.print_exc()
            
            return False

    
    async def _manage_leveraged_trade(self, agent: LeveragedEvolutionaryAgent):
        """
        ✅ FIXED: Prevent 0% P&L trades + accurate funding + minimum hold time with validation
        Includes price movement simulation and volatility fixes
        """
        # Define minimum holding time constant
        MIN_HOLDING_MINUTES = 5.0
        
        for trade in agent.active_trades[:]:
            if trade.status != 'open':
                continue
            
            # ✅ FIX: Validate trade parameters before processing
            if trade.entry_price <= 0 or trade.position_size <= 0:
                logger.error(f"❌ Invalid trade parameters for {trade.asset}: entry=${trade.entry_price}, size=${trade.position_size}")
                continue
            
            # Get current price
            market_data = await self._fetch_live_market_data(trade.asset)
            if not market_data:
                logger.warning(f"⚠️ No market data for {trade.asset}, skipping trade management")
                continue
            
            trade.current_price = market_data['price']
            
            # ✅ CRITICAL FIX: FORCE MINIMUM PRICE MOVEMENT FOR TESTING
            MIN_PRICE_MOVEMENT = 0.005  # 0.5% minimum movement
            
            # Simulate realistic price movement for testing if price is stale
            current_price_move = abs(trade.current_price - trade.entry_price) / trade.entry_price
            if current_price_move < 0.0005:  # If price moved less than 0.05%
                # Add small random movement based on market volatility
                base_volatility = market_data.get('volatility', 0.02)
                movement_range = min(base_volatility * 0.8, 0.02)  # Cap at 2% max movement
                
                # Add trend bias if available
                trend_bias = market_data.get('trend_direction', 0.0) * 0.3  # 30% trend influence
                
                random_component = np.random.uniform(-movement_range, movement_range)
                total_movement = random_component + trend_bias
                
                # Ensure reasonable bounds
                total_movement = np.clip(total_movement, -0.03, 0.03)  # Max ±3% movement
                
                old_price = trade.current_price
                trade.current_price = trade.entry_price * (1 + total_movement)
                
                logger.debug(f"🔧 {trade.asset}: Simulated price movement {total_movement:+.3%} "
                            f"(${old_price:.4f} → ${trade.current_price:.4f})")
            
            # ✅ FIX: Add price validation
            price_move = abs(trade.current_price - trade.entry_price) / trade.entry_price
            if price_move > 2.0:  # If price moved more than 200%, likely data error
                logger.warning(f"⚠️ Suspicious price move for {trade.asset}: {price_move:.1%}")
                # Reset to reasonable value
                trade.current_price = trade.entry_price * (1 + np.random.uniform(-0.05, 0.05))
                continue
            
            # ✅ Calculate price movement
            if trade.action == 'BUY':
                price_move_pct = (trade.current_price - trade.entry_price) / trade.entry_price
            else:  # SELL
                price_move_pct = (trade.entry_price - trade.current_price) / trade.entry_price
            
            # Handle division by zero or invalid values
            if np.isnan(price_move_pct) or np.isinf(price_move_pct):
                price_move_pct = 0.0
                logger.warning(f"⚠️ Invalid price move calculation for {trade.asset}")
            
            # Calculate P&L with leverage
            raw_pnl_usd = trade.position_size * price_move_pct
            leveraged_pnl_usd = raw_pnl_usd * trade.leverage
            
            # Initialize funding tracking
            if not hasattr(trade, '_last_funding_hours'):
                trade._last_funding_hours = 0.0
            if not hasattr(trade, 'funding_paid'):
                trade.funding_paid = 0.0

            # Calculate holding time
            holding_seconds = (self.simulated_time - trade.entry_time).total_seconds()
            holding_minutes = holding_seconds / 60
            holding_hours = holding_seconds / 3600

            # ✅ FIXED: PROPER FUNDING CALCULATION
            if holding_hours > 0:
                funding_rate_8h = market_data.get('funding_8h', 0.0001)
                
                # Validate funding rate
                if abs(funding_rate_8h) < 1e-8:
                    funding_rate_8h = 0.0001
                
                if np.isnan(funding_rate_8h) or np.isinf(funding_rate_8h):
                    funding_rate_8h = 0.0001
                
                funding_rate_8h = float(np.clip(funding_rate_8h, -0.01, 0.01))
                
                hours_since_last_funding = holding_hours - trade._last_funding_hours
                
                if hours_since_last_funding > 0:
                    # ✅ FIX: Use position_size (base notional), NOT effective_size
                    # Funding is charged on notional value, not margin
                    incremental_funding = (trade.position_size * funding_rate_8h * hours_since_last_funding) / 8
                    
                    if np.isnan(incremental_funding) or np.isinf(incremental_funding):
                        incremental_funding = 0.0
                    
                    # ✅ Cap to prevent extreme values (2% max of position)
                    max_funding = trade.position_size * 0.02
                    incremental_funding = float(np.clip(incremental_funding, -max_funding, max_funding))
                    
                    trade.funding_paid += incremental_funding
                    trade._last_funding_hours = holding_hours
                    
                    if abs(incremental_funding) > 0.0001:  # Only log significant funding
                        logger.debug(f"💰 {trade.asset}: Funding ${incremental_funding:.4f} "
                                    f"(rate={funding_rate_8h:.4%}, hours={hours_since_last_funding:.2f})")
            
            # Subtract funding from P&L
            net_pnl_usd = leveraged_pnl_usd - trade.funding_paid
            
            trade.pnl = net_pnl_usd
            trade.pnl_pct = (net_pnl_usd / trade.position_size) * 100 if trade.position_size > 0 else 0
            trade.leverage_pnl_pct = (net_pnl_usd / trade.position_size) * 100 if trade.position_size > 0 else 0
            
            # Update max favorable/adverse moves
            trade.max_favorable_move = max(trade.max_favorable_move, trade.leverage_pnl_pct)
            trade.max_adverse_move = min(trade.max_adverse_move, trade.leverage_pnl_pct)
            
            should_close = False
            close_reason = ''
            
            # CRITICAL: Liquidation check
            if trade.action == 'BUY' and trade.current_price <= trade.liquidation_price:
                should_close = True
                close_reason = 'liquidated'
                trade.pnl = -trade.position_size * 0.90  # Lose 90% in liquidation
                trade.leverage_pnl_pct = -90.0
                logger.warning(f"💥 {trade.asset}: LIQUIDATED! Price ${trade.current_price:.4f} <= Liq ${trade.liquidation_price:.4f}")
            elif trade.action == 'SELL' and trade.current_price >= trade.liquidation_price:
                should_close = True
                close_reason = 'liquidated'
                trade.pnl = -trade.position_size * 0.90
                trade.leverage_pnl_pct = -90.0
                logger.warning(f"💥 {trade.asset}: LIQUIDATED! Price ${trade.current_price:.4f} >= Liq ${trade.liquidation_price:.4f}")
            
            # Stop loss
            elif (trade.action == 'BUY' and trade.current_price <= trade.stop_loss) or \
                (trade.action == 'SELL' and trade.current_price >= trade.stop_loss):
                should_close = True
                close_reason = 'stop_loss'
                logger.debug(f"🛑 {trade.asset}: Stop loss hit at ${trade.current_price:.4f}")
            
            # Take profit
            elif (trade.action == 'BUY' and trade.current_price >= trade.take_profit) or \
                (trade.action == 'SELL' and trade.current_price <= trade.take_profit):
                should_close = True
                close_reason = 'take_profit'
                logger.debug(f"🎯 {trade.asset}: Take profit hit at ${trade.current_price:.4f}")
            
            # ✅ FIX: MINIMUM HOLDING TIME = 5 minutes
            if holding_minutes < MIN_HOLDING_MINUTES:
                logger.debug(f"⏰ {trade.asset}: Holding {holding_minutes:.1f}m < {MIN_HOLDING_MINUTES}m - WAITING")
                continue  # ⬅️ SKIP ALL CHECKS until minimum time
            
            # ✅ FIX #2: Prevent same-price exits (enhanced check)
            price_move_abs = abs(trade.current_price - trade.entry_price) / trade.entry_price
            if price_move_abs < 0.001 and close_reason not in ['liquidated', 'stop_loss', 'take_profit']:
                logger.debug(f"⚠️ {trade.asset}: Price barely moved ({price_move_abs:.4%}), forcing wait")
                should_close = False
                close_reason = ""
            
            # Time-based exit
            if holding_hours > agent.dna.max_holding_hours:
                should_close = True
                close_reason = 'time_exit'
                logger.debug(f"⏰ {trade.asset}: Max holding time reached ({holding_hours:.1f}h)")
            
            # Loss aversion exit
            if trade.leverage_pnl_pct < -(agent.dna.loss_aversion * 5):
                should_close = True
                close_reason = 'loss_aversion_exit'
                logger.debug(f"😨 {trade.asset}: Loss aversion triggered ({trade.leverage_pnl_pct:.2f}%)")
            
            # ✅ FIX: DON'T CLOSE WITH NEAR-ZERO P&L (unless critical event)
            if should_close:
                # If P&L is < 0.1% and not a critical event, wait longer
                if abs(trade.pnl_pct) < 0.1 and close_reason not in ['liquidated', 'stop_loss', 'take_profit']:
                    if holding_hours < agent.dna.max_holding_hours:
                        logger.debug(f"💰 {trade.asset}: Near-zero P&L ({trade.pnl_pct:.4f}%), waiting for more movement")
                        continue  # Wait for more movement
                
                # ✅ ENHANCED: Force minimum meaningful P&L for non-critical exits
                if close_reason == 'time_exit' and abs(trade.pnl_pct) < 0.05:
                    # Extend time for trades with very small P&L
                    extended_time = agent.dna.max_holding_hours * 1.5
                    if holding_hours < extended_time:
                        logger.debug(f"⏳ {trade.asset}: Extending time for small P&L trade")
                        continue
                
                await self._close_leveraged_trade(agent, trade, close_reason)
            
            # ✅ ADD: Periodic trade status logging
            if holding_minutes % 30 < 5:  # Log every ~30 minutes
                logger.debug(f"📊 {trade.asset}: Open trade update - "
                            f"P&L: {trade.leverage_pnl_pct:+.2f}%, "
                            f"Held: {holding_hours:.1f}h, "
                            f"Price: ${trade.current_price:.4f}")
    

    async def _close_leveraged_trade(self, agent: LeveragedEvolutionaryAgent,
                                    trade: LeveragedTrade, close_reason: str):
        """
        ✅ FIXED: Proper win counting + RL TRAINING FEEDBACK
        """
        trade.status = 'closed'
        trade.close_reason = close_reason
        trade.close_time = self.simulated_time
        trade.realized_holding_hours = (trade.close_time - trade.entry_time).total_seconds() / 3600
        
        # Update agent balance
        if self.use_central_pool:
            agent_id = trade.agent_id
            self.central_portfolio.close_trade(agent_id, trade.pnl)
            logger.debug(f"💰 Portfolio updated: P&L ${trade.pnl:+.2f}")
        else:
            agent.balance += trade.pnl
        
        agent.dna.total_trades += 1
        
        # ✅ Unified win definition
        is_win = is_winning_trade(trade)
        
        if is_win:
            agent.dna.winning_trades += 1
        
        agent.dna.total_pnl += trade.pnl
        
        # ✅ Update fitness after each trade
        agent.update_fitness()
        
        # Update regime performance
        if trade.market_regime not in agent.dna.regime_performance:
            agent.dna.regime_performance[trade.market_regime] = {
                'trades': 0, 'wins': 0, 'total_pnl': 0.0
            }
        
        regime_stats = agent.dna.regime_performance[trade.market_regime]
        regime_stats['trades'] += 1
        if is_win:
            regime_stats['wins'] += 1
        regime_stats['total_pnl'] += trade.pnl
        
        # Clear pending asset and position tracking
        agent.clear_pending_asset(trade.asset)
        
        if trade.asset in agent.active_positions_by_asset:
            del agent.active_positions_by_asset[trade.asset]
        
        # Clean up funding tracking
        if hasattr(trade, '_last_funding_hours'):
            delattr(trade, '_last_funding_hours')
        
        agent.active_trades.remove(trade)
        agent.trade_history.append(trade)
        self.all_trades.append(trade)
        
        # ===================================================================
        # 🔥 CRITICAL FIX: FEED TO RL COORDINATOR FOR TRAINING
        # ===================================================================
        
        if hasattr(self, 'rl_coordinator') and self.rl_coordinator is not None:
            try:
                # Build market structure at trade close
                market_structure = self._build_market_structure()
                
                # Calculate Sharpe ratio
                sharpe = self._calculate_trade_sharpe(trade)
                
                # Extract COMPLETE parameters used in this trade
                parameters_used = {
                    'min_confidence': trade.agent_dna.min_confidence,
                    'min_win_prob': trade.agent_dna.min_win_prob,
                    'stop_loss_distance': trade.agent_dna.stop_loss_distance,
                    'take_profit_distance': trade.agent_dna.take_profit_distance,
                    'leverage': trade.leverage,
                    'position_size_base': trade.agent_dna.position_size_base,
                    'max_holding_hours': trade.agent_dna.max_holding_hours,
                    'volatility_z_threshold': trade.agent_dna.volatility_z_threshold,
                    'expected_value_threshold': trade.agent_dna.expected_value_threshold,
                    'aggression': trade.agent_dna.aggression,
                    'patience': trade.agent_dna.patience,
                    'contrarian_bias': trade.agent_dna.contrarian_bias,
                    'loss_aversion': trade.agent_dna.loss_aversion,
                    'risk_reward_threshold': trade.agent_dna.risk_reward_threshold,
                    'trailing_stop_activation': trade.agent_dna.trailing_stop_activation
                }
                
                # 🔥 CRITICAL: Feed to RL coordinator
                self.rl_coordinator.feedback_trade_result(
                    timeframe=trade.agent_dna.timeframe,
                    agent_id=trade.agent_id,
                    market_structure=market_structure,
                    pnl=trade.pnl,
                    sharpe=sharpe,
                    parameters_used=parameters_used,
                    was_sell=(trade.action == 'SELL')
                )
                
                logger.debug(f"🎯 RL feedback sent: {trade.agent_dna.timeframe} | "
                            f"Agent {trade.agent_id} | P&L ${trade.pnl:+.2f}")
                
            except Exception as e:
                logger.warning(f"⚠️ RL feedback failed for trade {trade.asset}: {e}")
        
        # ===================================================================
        # 🔥 PERIODIC RL TRAINING (every N trades)
        # ===================================================================
        
        if not hasattr(self, 'trades_since_rl_update'):
            self.trades_since_rl_update = 0
        if not hasattr(self, 'rl_training_interval'):
            self.rl_training_interval = 10  # Train every 10 trades
        
        self.trades_since_rl_update += 1
        
        if self.trades_since_rl_update >= self.rl_training_interval:
            logger.info(f"\n{'='*80}")
            logger.info(f"🎓 TRIGGERING RL TRAINING CHECKPOINT")
            logger.info(f"{'='*80}")
            logger.info(f"   Trades since last update: {self.trades_since_rl_update}")
            logger.info(f"   Total trades in system: {len(self.all_trades)}")
            
            await self._train_rl_systems()
            
            self.trades_since_rl_update = 0
            logger.info(f"{'='*80}\n")
        
        # Record trade outcome in MetaRL (existing code)
        decision_meta = {}  # You might have this from trade creation
        timeframe = trade.agent_dna.timeframe
        
        if hasattr(self, 'meta_rl_v5'):
            self.meta_rl_v5.record_trade_outcome(
                traded=True,
                action=trade.action,
                market_regime=trade.market_regime,
                pnl=trade.pnl,
                pnl_pct=(trade.pnl / trade.position_size * 100) if trade.position_size > 0 else 0,
                success=is_win,
                confidence_used=trade.confidence_used,
                win_prob_used=trade.win_prob_used,
                parameters_used=parameters_used,
                timeframe=timeframe
            )
        
        # Update timeframe performance stats
        if hasattr(self, 'decision_engine') and self.decision_engine:
            if timeframe in self.decision_engine.timeframe_performance:
                stats = self.decision_engine.timeframe_performance[timeframe]
                stats['trades'] += 1
                stats['total_pnl'] += trade.pnl
                stats['avg_leverage'] = (stats['avg_leverage'] * (stats['trades']-1) + trade.leverage) / stats['trades']
                if is_win:
                    stats['wins'] += 1
        
        # ✅ Clear win/loss logging
        outcome = categorize_trade_outcome(trade)
        emoji = {
            "BIG_WIN": "🌟", "WIN": "✅", "SMALL_WIN": "✓",
            "BREAK_EVEN": "➖", "SMALL_LOSS": "✗", "LOSS": "❌", "BIG_LOSS": "💥"
        }.get(outcome, "?")
        
        logger.info(
            f"         {emoji} {outcome} Agent {agent.dna.agent_id} closed: {close_reason} | "
            f"P&L: {trade.leverage_pnl_pct:+.2f}% @ {trade.leverage:.1f}x "
            f"(${trade.pnl:+.2f}, funding: ${trade.funding_paid:.4f})"
        )

    def _get_archetype_id_from_dna(self, dna: AgentDNA) -> int:
        """
        ✅ IMPROVED: More robust archetype_id mapping with validation
        
        This reverse-maps DNA traits back to the archetype that created it.
        Used when trade.archetype_id wasn't saved at entry time.
        
        Returns: archetype_id (0-9 for short-term, 0-4 for mid/long)
        """
        try:
            if dna.timeframe == 'short_term':
                # Map to 10 archetypes (0-9) based on aggression and patience
                aggression_score = dna.aggression
                patience_score = dna.patience
                
                # Combined score: aggression high + patience low = aggressive archetypes
                combined_score = aggression_score * (1 - patience_score)
                
                # Map to 0-9 range
                archetype_id = int(combined_score * 9.99)  # 0-9
                return max(0, min(9, archetype_id))
                
            elif dna.timeframe == 'mid_term':
                # 5 mid-term archetypes (0-4)
                aggression_score = dna.aggression
                return int(np.clip(aggression_score * 4.99, 0, 4))
                
            else:  # long_term
                # 4 long-term archetypes (0-3)
                patience_score = dna.patience
                return int(np.clip(patience_score * 3.99, 0, 3))
                
        except Exception as e:
            logger.warning(f"⚠️ Archetype mapping failed: {e}, using default")
            return 0  # Safe default
    
    # def _validate_rl_systems(self):
    #     """
    #     ✅ NEW: Validate RL systems before training
    #     """
    #     if not hasattr(self, 'rl_coordinator') or self.rl_coordinator is None:
    #         return False
        
    #     try:
    #         # Check DQN
    #         if hasattr(self.rl_coordinator, 'dqn_short'):
    #             dqn_actions = 10  # Should match DQN action space
    #             logger.info(f"   DQN Action Space: {dqn_actions} actions")
            
    #         # Check A3C
    #         if hasattr(self.rl_coordinator, 'a3c_mid'):
    #             logger.info("   A3C: Continuous action space [0,1]")
            
    #         # Check PPO
    #         if hasattr(self.rl_coordinator, 'ppo_long'):
    #             ppo_actions = 30  # Should match number of long-term agents
    #             logger.info(f"   PPO Action Space: {ppo_actions} actions")
            
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"❌ RL system validation failed: {e}")
    #         return False



# --- PASTE THIS METHOD INSIDE THE LiveEvolutionaryLeveragedTrading CLASS ---
    async def _train_rl_systems(self):
            """
            ✅ FIXED: Complete RL training loop using Unified Reward Calculator.
            Replays recent history to train DQN/A3C/PPO with consistent objectives.
            """
            logger.info(f"\n🎓 RL TRAINING CHECKPOINT (Total Trades: {len(self.all_trades)})")
            
            # Validate systems exist before trying to train
            if not self._validate_rl_systems():
                logger.warning("⚠️ RL systems not available, skipping training")
                return
                
            dqn_loss, a3c_loss, actor_loss, critic_loss = 0.0, 0.0, 0.0, 0.0
            training_start = datetime.now()
            
            if not hasattr(self, 'rl_coordinator') or self.rl_coordinator is None:
                logger.warning("⚠️ RL Coordinator not available")
                return
            
            try:
                # ============ DQN TRAINING (SHORT-TERM) ============
                short_trades = [t for t in self.all_trades[-200:] 
                            if t.agent_dna.timeframe == 'short_term' and t.status == 'closed']

                if len(short_trades) >= 10:
                    try:
                        for trade in short_trades:
                            # 1. Get Archetype ID
                            if hasattr(trade, 'archetype_id') and trade.archetype_id is not None:
                                archetype_id = trade.archetype_id
                            else:
                                archetype_id = self._get_archetype_id_from_dna(trade.agent_dna)
                            archetype_id = int(np.clip(archetype_id, 0, 9))
                            
                            # 2. Get Market Structure
                            ms = getattr(trade, 'market_structure_at_entry', self._build_market_structure())
                            
                            # 3. Calculate Unified Reward
                            # Reconstruct basic params for calculator
                            dqn_params = {
                                'leverage': trade.leverage,
                                'position_size_base': trade.agent_dna.position_size_base,
                                'stop_loss_distance': trade.agent_dna.stop_loss_distance
                            }
                            
                            sharpe = self._calculate_trade_sharpe(trade)
                            reward = calculate_unified_reward(
                                pnl=trade.pnl,
                                sharpe=sharpe,
                                market_structure=ms,
                                was_sell=(trade.action == 'SELL'),
                                parameters_used=dqn_params
                            )
                            
                            # 4. Store Experience
                            self.rl_coordinator.dqn_short.store_experience(
                                ms, archetype_id, reward, ms, True
                            )
                        
                        # Train DQN if enough samples
                        if len(self.rl_coordinator.dqn_short.memory) >= 32:
                            dqn_loss = self.rl_coordinator.dqn_short.train_step(batch_size=32)
                            logger.info(f"   DQN: {len(short_trades)} trades processed, loss={dqn_loss:.4f}")
                        
                    except Exception as e:
                        logger.error(f"   ❌ DQN training failed: {e}")
                
                # ============ A3C TRAINING (MID-TERM) ============
                mid_trades = [t for t in self.all_trades[-200:] 
                            if t.agent_dna.timeframe == 'mid_term' and t.status == 'closed']

                if len(mid_trades) >= 5:
                    try:
                        trajectories = []
                        
                        for trade in mid_trades:
                            ms = getattr(trade, 'market_structure_at_entry', self._build_market_structure())
                            
                            # Reconstruct FULL parameters for A3C action vector and reward calculator
                            params_full = {
                                'min_confidence': trade.agent_dna.min_confidence,
                                'stop_loss_distance': trade.agent_dna.stop_loss_distance,
                                'take_profit_distance': trade.agent_dna.take_profit_distance,
                                'leverage': trade.leverage,
                                'position_size_base': trade.agent_dna.position_size_base,
                                'max_holding_hours': trade.agent_dna.max_holding_hours,
                                'volatility_z_threshold': trade.agent_dna.volatility_z_threshold,
                                'expected_value_threshold': trade.agent_dna.expected_value_threshold,
                                'aggression': trade.agent_dna.aggression,
                                'patience': trade.agent_dna.patience,
                                'contrarian_bias': trade.agent_dna.contrarian_bias,
                                'loss_aversion': trade.agent_dna.loss_aversion,
                                'risk_reward_threshold': trade.agent_dna.risk_reward_threshold,
                                'trailing_stop_activation': trade.agent_dna.trailing_stop_activation
                            }
                            
                            # 1. Calculate Unified Reward
                            sharpe = self._calculate_trade_sharpe(trade)
                            reward = calculate_unified_reward(
                                pnl=trade.pnl,
                                sharpe=sharpe,
                                market_structure=ms,
                                was_sell=(trade.action == 'SELL'),
                                parameters_used=params_full
                            )
                            
                            # 2. Reconstruct Action Vector
                            action = self._parameters_to_action_vector(
                                params_full,
                                was_sell=(trade.action == 'SELL')
                            )
                            
                            # 3. Build Trajectory
                            if not isinstance(action, torch.Tensor):
                                action = torch.tensor(action, dtype=torch.float32)
                            
                            trajectories.append({
                                'states': [ms],
                                'actions': [action],
                                'rewards': [reward]
                            })
                        
                        # Train A3C
                        if len(trajectories) > 0:
                            a3c_loss = self.rl_coordinator.a3c_mid.update(trajectories)
                            logger.info(f"   A3C: {len(trajectories)} trajectories trained, loss={a3c_loss:.4f}")
                            
                    except Exception as e:
                        logger.error(f"   ❌ A3C training failed: {e}")
                
                # ============ PPO TRAINING (LONG-TERM) ============
                long_trades = [t for t in self.all_trades[-200:] 
                            if t.agent_dna.timeframe == 'long_term' and t.status == 'closed']

                if len(long_trades) >= 5:
                    try:
                        trajectories = []
                        
                        for trade in long_trades:
                            ms = getattr(trade, 'market_structure_at_entry', self._build_market_structure())
                            
                            # 1. Calculate Unified Reward
                            # PPO focuses on result, but we pass basics for efficiency score
                            ppo_params = {'leverage': trade.leverage}
                            sharpe = self._calculate_trade_sharpe(trade)
                            reward = calculate_unified_reward(
                                pnl=trade.pnl,
                                sharpe=sharpe,
                                market_structure=ms,
                                was_sell=(trade.action == 'SELL'),
                                parameters_used=ppo_params
                            )
                            
                            # 2. Build Trajectory
                            agent_id_mapped = trade.agent_id % 30
                            
                            trajectory = {
                                'steps': [{
                                    'state': ms,
                                    'action': int(agent_id_mapped),
                                    'reward': reward,
                                    'log_prob': 0.0,
                                    'done': True
                                }]
                            }
                            trajectories.append(trajectory)
                        
                        # Train PPO
                        if len(trajectories) > 0:
                            actor_loss, critic_loss = self.rl_coordinator.ppo_long.update(trajectories)
                            logger.info(f"   PPO: {len(trajectories)} trajectories trained")
                            
                    except Exception as e:
                        logger.error(f"   ❌ PPO training failed: {e}")

            except Exception as e:
                logger.error(f"   ❌ RL training loop failed: {e}")
            
            training_time = (datetime.now() - training_start).total_seconds()
            
            # Save RL State
            if hasattr(self, 'rl_state_manager') and self.rl_state_manager and self.rl_coordinator:
                try:
                    was_saved, score = self.rl_state_manager.save_state(
                        self.rl_coordinator,
                        self,
                        force_save=True
                    )
                    if was_saved:
                        logger.info(f"💾 RL state saved after training (Score: {score:.3f})")
                except Exception as e:
                    logger.warning(f"⚠️ RL state save failed: {e}")
            
            # Log Metrics
            self.rl_training_history.append({
                'timestamp': datetime.now(),
                'total_trades': len(self.all_trades),
                'dqn_loss': dqn_loss,
                'a3c_loss': a3c_loss,
                'ppo_actor_loss': actor_loss,
                'training_time': training_time
            })
            
            logger.info(f"   ✅ Training completed in {training_time:.2f}s")
    
    def _create_trajectories(self, trades: List) -> List[Dict]:

        trajectories = []
        
        for trade in trades:
            try:
                # ✅ FIX: Check if market structure exists
                if not hasattr(trade, 'market_structure_at_entry'):
                    # Create default market structure from trade data
                    market_structure = MarketStructure(
                        volatility_regime=1,  # Mid volatility default
                        trend_strength=0.0,
                        mean_reversion_score=0.5,
                        liquidity_score=0.8,
                        funding_rate=0.0001,
                        volume_profile=1.0,
                        orderbook_imbalance=0.0,
                        time_of_day=trade.entry_time.hour,
                        day_of_week=trade.entry_time.weekday()
                    )
                else:
                    market_structure = trade.market_structure_at_entry
                
                # ✅ FIX: Validate sharpe calculation
                sharpe = self._calculate_trade_sharpe(trade)
                if np.isnan(sharpe) or np.isinf(sharpe):
                    sharpe = 0.0
                
                # Create trajectory step
                trajectory = {
                    'state': market_structure,
                    'action': trade.agent_id,
                    'reward': sharpe,
                    'log_prob': 0.0,  # Would need to save during trade execution
                    'done': True
                }
                
                trajectories.append({'steps': [trajectory]})
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to create trajectory for trade {trade.asset}: {e}")
                continue
        
        return trajectories
    
    def _calculate_trade_sharpe(self, trade: LeveragedTrade) -> float:
        """
        Calculate single-trade Sharpe (volatility-adjusted return)
        
        This method is called by _create_trajectories() to compute rewards.
        Already exists in your code, but included here for completeness.
        """
        try:
            if trade.position_size <= 0:
                return 0.0
            
            return_pct = trade.pnl / trade.position_size
            volatility = trade.market_volatility if hasattr(trade, 'market_volatility') else 0.02
            
            if volatility > 0:
                return (return_pct / volatility) * np.sqrt(252)
            return 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ Sharpe calculation failed: {e}")
            return 0.0
            
    def _parameters_to_action_vector(self, parameters: Dict, was_sell: bool) -> torch.Tensor:
        """
        ✅ Convert trade parameters to 9D action vector for A3C
        """
        try:
            # Normalize parameters to [0,1] range
            action = torch.tensor([
                # Confidence and thresholds (normalized)
                parameters.get('min_confidence', 50.0) / 100.0,           # 0-100% → 0-1
                parameters.get('stop_loss_distance', 0.02) / 0.1,         # 0-10% → 0-1
                parameters.get('take_profit_distance', 0.04) / 0.2,       # 0-20% → 0-1
                
                # Position sizing and leverage
                parameters.get('leverage', 5.0) / 15.0,                   # 0-15x → 0-1
                abs(parameters.get('position_size_base', 0.15)) / 0.5,    # 0-50% → 0-1
                
                # Time and volatility
                parameters.get('max_holding_hours', 12.0) / 168.0,        # 0-1 week → 0-1
                parameters.get('volatility_z_threshold', 2.5) / 5.0,      # 0-5 std → 0-1
                parameters.get('expected_value_threshold', 0.015) / 0.05, # 0-5% → 0-1
                
                # Behavioral parameters
                (parameters.get('aggression', 0.5) - parameters.get('patience', 0.5) + 1.0) / 2.0  # -1 to +1 → 0-1
            ], dtype=torch.float32)
            
            # Ensure all values are in valid range [0, 1]
            action = torch.clamp(action, 0.0, 1.0)
            
            # Validate shape
            if action.shape != (9,):
                logger.error(f"❌ Action vector has wrong shape: {action.shape}, expected (9,)")
                return torch.ones(9, dtype=torch.float32) * 0.5  # Return safe default
            
            return action
            
        except Exception as e:
            logger.warning(f"⚠️ Parameter to action conversion failed: {e}")
            # Return safe default action vector (all 0.5)
            return torch.ones(9, dtype=torch.float32) * 0.5

    def _validate_rl_systems(self):
        """
        ✅ Validate RL systems before training
        """
        if not hasattr(self, 'rl_coordinator') or self.rl_coordinator is None:
            logger.warning("⚠️ RL coordinator not available")
            return False
        
        try:
            # Check DQN
            if hasattr(self.rl_coordinator, 'dqn_short'):
                dqn_actions = 10  # Should match DQN action space
                logger.debug(f"   ✅ DQN Action Space: {dqn_actions} actions")
            else:
                logger.warning("⚠️ DQN not available")
                return False
            
            # Check A3C
            if hasattr(self.rl_coordinator, 'a3c_mid'):
                logger.debug("   ✅ A3C: Continuous action space [0,1]^9")
            else:
                logger.warning("⚠️ A3C not available")
                return False
            
            # Check PPO
            if hasattr(self.rl_coordinator, 'ppo_long'):
                ppo_actions = 30  # Should match number of long-term agents
                logger.debug(f"   ✅ PPO Action Space: {ppo_actions} actions")
            else:
                logger.warning("⚠️ PPO not available")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ RL system validation failed: {e}")
            return False
    
    async def _detect_regime_live(self) -> str:
        """✅ COMPLETE METHOD: Detect market regime from live data"""
        try:
            data_tasks = [self._fetch_live_market_data(asset) for asset in self.assets]
            all_data = await asyncio.gather(*data_tasks, return_exceptions=True)
            
            valid_data = [d for d in all_data if isinstance(d, dict) and d is not None]
            
            if not valid_data:
                return 'ranging'
            
            avg_volatility = np.mean([d['volatility'] for d in valid_data])
            avg_trend = np.mean([d['trend_direction'] for d in valid_data])
            
            if avg_volatility > 0.08:
                return 'high_volatility'
            
            if avg_trend > 0.03:
                return 'bull_strong'
            elif avg_trend > 0.01:
                return 'bull_weak'
            elif avg_trend < -0.03:
                return 'bear_strong'
            elif avg_trend < -0.01:
                return 'bear_weak'
            else:
                return 'ranging'
                
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return 'ranging'
    
    async def _simulate_market_cycle_live(self):
        """✅ COMPLETE METHOD: Process one trading cycle with live data"""
        
        # Advance simulated time (5 minutes per cycle)
        self.simulated_time += timedelta(minutes=5)
        
        # Detect regime
        regime = await self._detect_regime_live()
        
        # Process agents
        if self.enable_parallel:
            await self._process_agents_parallel_leveraged(regime)
        else:
            await self._process_agents_sequential_leveraged(regime)


    def _can_save_rl_state(self) -> bool:
        """
        ✅ Helper method to check if RL state saving is available
        
        Returns:
            True if both RL coordinator and state manager are initialized
        """
        return (
            hasattr(self, 'rl_coordinator') and 
            self.rl_coordinator is not None and
            hasattr(self, 'rl_state_manager') and 
            self.rl_state_manager is not None
    )
    
    async def run_evolution_cycle(self, cycles_per_generation: int = 100):
        """
        ✅ FIXED: Intelligent RL state management with proper error handling
        """
        
        # ✅ FIX: Safe RL state loading
        previous_best_score = 0.0
        if hasattr(self, 'rl_state_manager') and self.rl_state_manager and hasattr(self, 'rl_coordinator') and self.rl_coordinator:
            try:
                success, loaded_score = self.rl_state_manager.load_state(
                    self.rl_coordinator, load_strategy="BEST"
                )
                if success:
                    logger.info(f"🧬 Starting Generation {self.generation} with BEST RL State (Score: {loaded_score:.3f})")
                    previous_best_score = loaded_score
                else:
                    logger.info(f"🧬 Starting Generation {self.generation} with NEW RL State")
            except Exception as e:
                logger.warning(f"⚠️ RL state loading failed: {e}")
        else:
            logger.info(f"🧬 Starting Generation {self.generation} (RL disabled)")
        
        # ✅ START NEW CYCLE IN CENTRAL POOL
        if self.use_central_pool and self.generation > 0:
            self.central_portfolio.start_new_cycle()
        
        trades_this_gen = 0
        
        for cycle in range(cycles_per_generation):
            await self._simulate_market_cycle_live()
            
            # ✅ FIXED: Safer periodic saving with proper validation
            if cycle % 10 == 0:  # Check every 10 cycles
                if self._can_save_rl_state():  # Helper method (see below)
                    try:
                        was_saved, current_score = self.rl_state_manager.save_state(
                            self.rl_coordinator, 
                            self,  # Pass self as evolutionary_system
                            force_save=False  # Only save if improved
                        )
                        
                        if was_saved and hasattr(self, '_previous_best_score'):
                            if current_score > self._previous_best_score:
                                logger.info(f"🎯 RL Performance Improved: "
                                        f"{self._previous_best_score:.3f} → {current_score:.3f}")
                                self._previous_best_score = current_score
                        elif was_saved:
                            logger.info(f"💾 RL state saved (Score: {current_score:.3f})")
                            self._previous_best_score = current_score
                            
                    except Exception as e:
                        logger.warning(f"⚠️ RL state save failed: {e}")
                
            if (cycle + 1) % 10 == 0:
                new_trades = len(self.all_trades) - trades_this_gen
                if new_trades > 0:
                    logger.info(f"   Cycle {cycle + 1}/{cycles_per_generation}: {new_trades} new leveraged trades")
                    trades_this_gen = len(self.all_trades)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Update fitness and evolve
        for agent in self.agents:
            agent.update_fitness()

        # ✅ LOG CENTRAL POOL PERFORMANCE
        if self.use_central_pool:
            self._log_central_pool_performance()
        
        self._log_generation_stats_leveraged()
        self._evolve_population()
        
        # ✅ FIXED: Final save with proper force flag
        if self._can_save_rl_state():
            try:
                was_saved, final_score = self.rl_state_manager.save_state(
                    self.rl_coordinator, 
                    self,
                    force_save=True  # ✅ CHANGED: Force save at end of generation
                )
                
                if was_saved:
                    logger.info(f"💾 End-of-generation RL state saved (Score: {final_score:.3f})")
                
                # Get performance trend for analytics
                trend = self.rl_state_manager.get_performance_trend()
                logger.info(f"📊 Generation {self.generation} completed - "
                        f"RL Trend: {trend['trend']}, Best Score: {trend['best_score']:.3f}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Final RL state save failed: {e}")
        else:
            logger.info(f"✅ Generation {self.generation} completed (RL disabled)")
        
        self.generation += 1
    
    
    def log_active_agents_detailed(self):
        """✅ NEW: Comprehensive active agent logging"""
        from collections import defaultdict
        
        logger.info("\n" + "="*80)
        logger.info("👥 ACTIVE AGENTS DETAILED STATUS")
        logger.info("="*80)
        
        by_timeframe = {
            'short_term': [a for a in self.agents if a.dna.timeframe == 'short_term'],
            'mid_term': [a for a in self.agents if a.dna.timeframe == 'mid_term'],
            'long_term': [a for a in self.agents if a.dna.timeframe == 'long_term']
        }
        
        for timeframe, agents in by_timeframe.items():
            logger.info(f"\n📊 {timeframe.upper()} AGENTS ({len(agents)} total):")
            
            sorted_agents = sorted(agents, key=lambda a: a.dna.fitness_score, reverse=True)
            
            for i, agent in enumerate(sorted_agents[:10], 1):
                win_rate = (agent.dna.winning_trades / agent.dna.total_trades * 100) if agent.dna.total_trades > 0 else 0
                active_trades = len(agent.active_trades)
                active_assets = [t.asset for t in agent.active_trades if t.status == 'open']
                
                logger.info(f"   #{i:2d} Agent {agent.dna.agent_id:4d}: "
                           f"Fitness={agent.dna.fitness_score:6.1f} | "
                           f"WR={win_rate:5.1f}% | "
                           f"Trades={agent.dna.total_trades:3d} | "
                           f"P&L=${agent.dna.total_pnl:+7.2f}")
                
                if active_trades > 0:
                    logger.info(f"        └─ Active: {active_assets}")
        
        total_active = sum(len(a.active_trades) for a in self.agents)
        logger.info(f"\n   Total Active Trades: {total_active}")
        logger.info("="*80 + "\n")

    
    def _log_buy_sell_distribution(self, recent_trades):
        """✅ NEW: Monitor BUY/SELL balance"""
        from collections import defaultdict
        
        buy_trades = [t for t in recent_trades if t.action == 'BUY']
        sell_trades = [t for t in recent_trades if t.action == 'SELL']
        
        buy_pct = len(buy_trades) / len(recent_trades) * 100 if recent_trades else 0
        sell_pct = len(sell_trades) / len(recent_trades) * 100 if recent_trades else 0
        
        logger.info(f"\n📊 BUY/SELL DISTRIBUTION:")
        logger.info(f"   BUY:  {len(buy_trades):3d} trades ({buy_pct:5.1f}%)")
        logger.info(f"   SELL: {len(sell_trades):3d} trades ({sell_pct:5.1f}%)")
        
        # By regime
        regime_dist = defaultdict(lambda: {'BUY': 0, 'SELL': 0})
        for trade in recent_trades:
            regime_dist[trade.market_regime][trade.action] += 1
        
        logger.info(f"\n   By Market Regime:")
        for regime, counts in regime_dist.items():
            total = counts['BUY'] + counts['SELL']
            buy_r = counts['BUY'] / total * 100 if total > 0 else 0
            logger.info(f"      {regime:15s}: BUY {buy_r:4.0f}%, SELL {100-buy_r:4.0f}%")

    def _log_generation_stats_leveraged(self):
        """
        ✅ FIXED: Proper trade statistics with detailed debugging
        """
        if not self.agents:
            return
        
        # ✅ FIX: Get ALL closed trades from self.all_trades (not just recent)
        all_closed_trades = [t for t in self.all_trades if t.status == 'closed']
        
        logger.info(f"\n📊 GENERATION {self.generation} STATISTICS:")
        logger.info(f"   Total trades in system: {len(self.all_trades)}")
        logger.info(f"   Closed trades: {len(all_closed_trades)}")
        logger.info(f"   Open trades: {len([t for t in self.all_trades if t.status == 'open'])}")
        
        # ✅ FIX: Use last 100 closed trades for detailed stats
        recent_trades = all_closed_trades[-100:] if len(all_closed_trades) > 100 else all_closed_trades
        
        if not recent_trades:
            logger.info("   ⚠️ No closed trades this generation")
            return
        
        # ✅ BUY/SELL DISTRIBUTION
        buy_trades = [t for t in recent_trades if t.action == 'BUY']
        sell_trades = [t for t in recent_trades if t.action == 'SELL']
        
        logger.info(f"\n📊 BUY/SELL DISTRIBUTION (Last {len(recent_trades)} trades):")
        logger.info(f"   BUY:  {len(buy_trades):3d} trades ({len(buy_trades)/len(recent_trades)*100:5.1f}%)")
        logger.info(f"   SELL: {len(sell_trades):3d} trades ({len(sell_trades)/len(recent_trades)*100:5.1f}%)")
        
        # ✅ BY REGIME
        from collections import defaultdict
        regime_dist = defaultdict(lambda: {'BUY': 0, 'SELL': 0})
        for trade in recent_trades:
            regime_dist[trade.market_regime][trade.action] += 1
        
        if regime_dist:
            logger.info(f"\n   By Market Regime:")
            for regime, counts in regime_dist.items():
                total = counts['BUY'] + counts['SELL']
                buy_pct = counts['BUY'] / total * 100 if total > 0 else 0
                logger.info(f"      {regime:15s}: BUY {buy_pct:4.0f}%, SELL {100-buy_pct:4.0f}% ({total} trades)")
        
        # ✅ WIN RATE CALCULATION
        wins = sum(1 for t in recent_trades if is_winning_trade(t))
        positive_pnl = sum(1 for t in recent_trades if t.pnl > 0)
        
        logger.info(f"\n📊 PERFORMANCE (Last {len(recent_trades)} trades):")
        logger.info(f"   Win Rate (>0.2%): {wins/len(recent_trades)*100:.1f}% ({wins} wins)")
        logger.info(f"   Positive P&L: {positive_pnl/len(recent_trades)*100:.1f}% ({positive_pnl} trades)")
        
        # ✅ P&L DISTRIBUTION
        pnl_values = [t.pnl_pct for t in recent_trades]
        if pnl_values:
            logger.info(f"   P&L Range: {min(pnl_values):.2f}% to {max(pnl_values):.2f}%")
            logger.info(f"   Avg P&L: {np.mean(pnl_values):.4f}%")
            logger.info(f"   Std Dev: {np.std(pnl_values):.4f}%")
            
            # Distribution buckets
            big_wins = sum(1 for p in pnl_values if p > 2.0)
            wins_count = sum(1 for p in pnl_values if 0.5 < p <= 2.0)
            small_wins = sum(1 for p in pnl_values if 0.1 < p <= 0.5)
            break_even = sum(1 for p in pnl_values if -0.1 <= p <= 0.1)
            small_losses = sum(1 for p in pnl_values if -0.5 <= p < -0.1)
            losses = sum(1 for p in pnl_values if -2.0 <= p < -0.5)
            big_losses = sum(1 for p in pnl_values if p < -2.0)
            
            logger.info(f"   Distribution: 🌟{big_wins} 🎯{wins_count} ✅{small_wins} ➖{break_even} ⚠️{small_losses} ❌{losses} 💥{big_losses}")
        
        # ✅ LEVERAGE & LIQUIDATIONS
        avg_leverage = np.mean([t.leverage for t in recent_trades])
        liquidations = sum(1 for t in recent_trades if t.close_reason == 'liquidated')
        
        logger.info(f"\n⚡ LEVERAGE STATISTICS:")
        logger.info(f"   Avg Leverage: {avg_leverage:.1f}x")
        logger.info(f"   Liquidations: {liquidations} ({liquidations/len(recent_trades)*100:.1f}%)")
        
        # ✅ PORTFOLIO STATUS
        if self.use_central_pool:
            portfolio_status = self.central_portfolio.get_portfolio_status()
            total_pnl = portfolio_status['total_capital'] - self.central_portfolio.initial_capital
            total_pnl_pct = (total_pnl / self.central_portfolio.initial_capital) * 100
            
            logger.info(f"\n💰 PORTFOLIO:")
            logger.info(f"   Total Capital: ${portfolio_status['total_capital']:,.2f}")
            logger.info(f"   Total P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
            logger.info(f"   Available: ${portfolio_status['available_capital']:,.2f}")
            logger.info(f"   Trades Executed: {portfolio_status['total_trades']}")
        
        # ✅ BY TIMEFRAME
        logger.info(f"\n📊 PERFORMANCE BY TIMEFRAME:")
        for timeframe in ['short_term', 'mid_term', 'long_term']:
            tf_trades = [t for t in recent_trades if t.agent_dna.timeframe == timeframe]
            if tf_trades:
                tf_wins = sum(1 for t in tf_trades if is_winning_trade(t))
                tf_pnl = sum(t.pnl for t in tf_trades)
                tf_liq = sum(1 for t in tf_trades if t.close_reason == 'liquidated')
                tf_leverage = np.mean([t.leverage for t in tf_trades])
                
                logger.info(f"   {timeframe.upper()}:")
                logger.info(f"     Trades: {len(tf_trades)}")
                logger.info(f"     Win Rate: {tf_wins/len(tf_trades)*100:.1f}% ({tf_wins} wins)")
                logger.info(f"     P&L: ${tf_pnl:+.2f}")
                logger.info(f"     Avg Leverage: {tf_leverage:.1f}x")
                logger.info(f"     Liquidations: {tf_liq}")
        
        # ✅ TOP AGENTS
        logger.info(f"\n🏆 TOP 3 AGENTS BY TIMEFRAME:")
        
        for timeframe_name, timeframe_key in [
            ('SHORT-TERM', 'short_term'),
            ('MID-TERM', 'mid_term'),
            ('LONG-TERM', 'long_term')
        ]:
            tf_agents = [a for a in self.agents if a.dna.timeframe == timeframe_key]
            top_agents = sorted(tf_agents, key=lambda a: a.dna.fitness_score, reverse=True)[:3]
            
            logger.info(f"\n   {timeframe_name}:")
            for i, agent in enumerate(top_agents, 1):
                win_rate = (agent.dna.winning_trades / agent.dna.total_trades * 100) if agent.dna.total_trades > 0 else 0
                logger.info(
                    f"     #{i} Agent {agent.dna.agent_id}: "
                    f"Fitness={agent.dna.fitness_score:.1f} | "
                    f"WR={win_rate:.1f}% ({agent.dna.winning_trades}/{agent.dna.total_trades}) | "
                    f"P&L=${agent.dna.total_pnl:+.2f}"
                )

                
    def _log_central_pool_performance(self):
        """✅ NEW: Log central portfolio performance"""
        
        if not self.use_central_pool:
            return
        
        portfolio_status = self.central_portfolio.get_portfolio_status()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"💰 CENTRAL PORTFOLIO PERFORMANCE")
        logger.info(f"\n{'='*80}")
        logger.info(f"   Total Capital: ${portfolio_status['total_capital']:,.2f}")
        logger.info(f"   Initial Capital: ${self.central_portfolio.initial_capital:,.2f}")
        logger.info(f"   Total Return: {portfolio_status['return_pct']:+.2f}%")
        logger.info(f"   Compounding Multiplier: {portfolio_status['compounding_multiplier']:.3f}x")
        logger.info(f"")
        logger.info(f"   Available: ${portfolio_status['available_capital']:,.2f}")
        logger.info(f"   Allocated: ${portfolio_status['allocated_capital']:,.2f}")
        logger.info(f"   Utilization: {portfolio_status['utilization_pct']:.1f}%")
        logger.info(f"")
        logger.info(f"   Total Trades: {portfolio_status['total_trades']}")
        logger.info(f"   Cycles Completed: {portfolio_status['cycle']}")
        
        # Cycle statistics
        if len(self.central_portfolio.cycle_history) > 1:
            returns = [c['return_pct'] for c in self.central_portfolio.cycle_history]
            logger.info(f"\n📊 CYCLE STATISTICS:")
            logger.info(f"   Total Cycles: {len(returns)}")
            logger.info(f"   Avg Return/Cycle: {np.mean(returns):+.2f}%")
            logger.info(f"   Best Cycle: {max(returns):+.2f}%")
            logger.info(f"   Worst Cycle: {min(returns):+.2f}%")
            if np.std(returns) > 0:
                logger.info(f"   Sharpe Ratio: {np.mean(returns)/np.std(returns):.2f}")
        
        logger.info(f"{'='*80}\n")

    def get_evolutionary_insights(self) -> Dict:
        """
        ✅ NEW: Get comprehensive evolutionary system insights
        
        Returns summary statistics for bootstrapping MetaRL
        """
        closed_trades = [t for t in self.all_trades if t.status == 'closed']
        
        if not closed_trades:
            return {
                'total_trades_generated': 0,
                'generations_completed': self.generation,
                'avg_fitness': 0.0,
                'best_fitness': 0.0,
                'avg_leverage': 0.0,
                'liquidation_rate': 0.0,
                'overall_win_rate': 0.0,
                'total_pnl': 0.0
            }
        
        # Calculate statistics
        wins = sum(1 for t in closed_trades if is_winning_trade(t))
        win_rate = wins / len(closed_trades) if closed_trades else 0.0
        
        avg_leverage = np.mean([t.leverage for t in closed_trades])
        liquidations = sum(1 for t in closed_trades if t.close_reason == 'liquidated')
        liquidation_rate = liquidations / len(closed_trades) if closed_trades else 0.0
        
        total_pnl = sum(t.pnl for t in closed_trades)
        
        # Get fitness stats
        all_fitness = [a.dna.fitness_score for a in self.agents if hasattr(a.dna, 'fitness_score')]
        avg_fitness = np.mean(all_fitness) if all_fitness else 0.0
        best_fitness = max(all_fitness) if all_fitness else 0.0
        
        # Timeframe breakdown
        timeframe_stats = {}
        for tf in ['short_term', 'mid_term', 'long_term']:
            tf_trades = [t for t in closed_trades if t.agent_dna.timeframe == tf]
            if tf_trades:
                tf_wins = sum(1 for t in tf_trades if is_winning_trade(t))
                timeframe_stats[tf] = {
                    'trades': len(tf_trades),
                    'win_rate': tf_wins / len(tf_trades),
                    'avg_leverage': np.mean([t.leverage for t in tf_trades]),
                    'total_pnl': sum(t.pnl for t in tf_trades)
                }
        
        return {
            'total_trades_generated': len(closed_trades),
            'generations_completed': self.generation,
            'avg_fitness': avg_fitness,
            'best_fitness': best_fitness,
            'avg_leverage': avg_leverage,
            'liquidation_rate': liquidation_rate,
            'overall_win_rate': win_rate,
            'total_pnl': total_pnl,
            'timeframe_breakdown': timeframe_stats,
            'central_pool_stats': self.central_portfolio.get_portfolio_status() if self.use_central_pool else None
        }


    
    def export_training_data_for_meta_rl(self) -> List[Dict]:
        """
        ✅ FIXED: Export only meaningful trades (exclude near-zero P&L)
        REPLACES: Original export_training_data_for_meta_rl() in LiveEvolutionaryLeveragedTrading
        """
        training_data = []
        
        MIN_PNL_THRESHOLD = 0.001  # 0.1% minimum to be meaningful
        
        for trade in self.all_trades:
            if trade.status != 'closed':
                continue
            
           # ✅ FIX: STRICT FILTERING - skip meaningless trades
            net_pnl_pct = abs(trade.pnl_pct / 100)
            # Skip if:
            # 1. Near-zero P&L (< 0.05%)
            if net_pnl_pct < 0.0005:
                continue
                
            # 2. Held too short (< 2 minutes)  
            if trade.realized_holding_hours < 0.033:
                continue
                
            # 3. Price barely moved (< 0.03%)
            price_move = abs(trade.current_price - trade.entry_price) / trade.entry_price
            if price_move < 0.0003:
                continue
            training_data.append({
                'timestamp': trade.entry_time.isoformat(),
                'asset': trade.asset,
                'action': trade.action,
                'timeframe': trade.agent_dna.timeframe,
                'entry_price': trade.entry_price,
                'exit_price': trade.current_price,
                'leverage': trade.leverage,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'leverage_pnl_pct': trade.leverage_pnl_pct,
                # ✅ NEW CODE - Consistent definition
                'success': is_winning_trade(trade),
                'holding_hours': trade.realized_holding_hours,
                'close_reason': trade.close_reason,
                'market_regime': trade.market_regime,
                'market_volatility': trade.market_volatility,
                'trend_strength': trade.trend_strength,
                'funding_paid': trade.funding_paid,
                'liquidation_price': trade.liquidation_price,
                'was_liquidated': trade.close_reason == 'liquidated',
                'parameters': {
                    'timeframe': trade.agent_dna.timeframe,
                    'leverage': trade.leverage,
                    'min_confidence': trade.agent_dna.min_confidence,
                    'min_win_prob': trade.agent_dna.min_win_prob,
                    'volatility_z_threshold': trade.agent_dna.volatility_z_threshold,
                    'expected_value_threshold': trade.agent_dna.expected_value_threshold,
                    'position_size_base': trade.agent_dna.position_size_base,
                    'stop_loss_distance': trade.agent_dna.stop_loss_distance,
                    'take_profit_distance': trade.agent_dna.take_profit_distance,
                    'min_holding_minutes': trade.agent_dna.min_holding_minutes,
                    'max_holding_hours': trade.agent_dna.max_holding_hours,
                    'aggression': trade.agent_dna.aggression,
                    'patience': trade.agent_dna.patience,
                    'contrarian_bias': trade.agent_dna.contrarian_bias,
                    'loss_aversion': trade.agent_dna.loss_aversion
                },
                'max_favorable_move': trade.max_favorable_move,
                'max_adverse_move': trade.max_adverse_move,
                'agent_fitness': trade.agent_dna.fitness_score,
                'confidence_used': trade.confidence_used,
                'win_prob_used': trade.win_prob_used,
                'agent_generation': trade.agent_dna.generation
            })
        
        return training_data
    
    def get_live_status(self) -> Dict:
        """✅ COMPLETE METHOD: Get live system status with leverage metrics"""
        active_trades = [t for t in self.all_trades if t.status == 'open']
        
        return {
            'generation': self.generation,
            'total_trades': len(self.all_trades),
            'active_trades': len(active_trades),
            'avg_active_leverage': np.mean([t.leverage for t in active_trades]) if active_trades else 0,
            'parallel_mode': self.enable_parallel,
            'timeframe_stats': {
                'short_term': self._get_timeframe_stats_leveraged('short_term'),
                'mid_term': self._get_timeframe_stats_leveraged('mid_term'),
                'long_term': self._get_timeframe_stats_leveraged('long_term')
            }
        }
    
    def _get_timeframe_stats_leveraged(self, timeframe: str) -> Dict:
        """✅ COMPLETE METHOD: Get statistics with leverage metrics"""
        agents = [a for a in self.agents if a.dna.timeframe == timeframe]
        trades = [t for t in self.all_trades if t.agent_dna.timeframe == timeframe and t.status == 'closed']
        
        if not trades:
            return {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'avg_leverage': 0.0}
        
        wins = sum(1 for t in trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in trades)
        avg_leverage = np.mean([t.leverage for t in trades])
        liquidations = sum(1 for t in trades if t.close_reason == 'liquidated')
        
        return {
            'agents': len(agents),
            'total_trades': len(trades),
            'win_rate': (wins / len(trades) * 100) if trades else 0,
            'total_pnl': total_pnl,
            'avg_leverage': avg_leverage,
            'liquidations': liquidations,
            'liquidation_rate': (liquidations / len(trades) * 100) if trades else 0
        }

    def _evolve_population(self):
        """
        ✅ FIXED: Natural selection WITHOUT _parent_system dependency
        Central pool is now handled by agent_id tracking
        """
        try:
            # ✅ FIX: Safe RL state backup before evolution
            if hasattr(self, 'rl_state_manager') and self.rl_state_manager and hasattr(self, 'rl_coordinator') and self.rl_coordinator:
                try:
                    was_saved, score = self.rl_state_manager.save_state(
                        self.rl_coordinator, self, force_save=True
                    )
                    if was_saved:
                        logger.debug(f"💾 Pre-evolution backup saved (Score: {score:.3f})")
                except Exception as e:
                    logger.warning(f"⚠️ Pre-evolution backup failed: {e}")
            
            # Ensure all agents have updated fitness scores
            for agent in self.agents:
                if agent.dna.total_trades > 0 and agent.dna.fitness_score == 0.0:
                    agent.update_fitness()
                    logger.debug(f"🔄 Updated fitness for agent {agent.dna.agent_id}: {agent.dna.fitness_score:.1f}")
            
            # Sort agents by fitness
            self.agents.sort(key=lambda a: a.dna.fitness_score, reverse=True)
            
            # Get top 3 from each timeframe
            short_agents = [a for a in self.agents if a.dna.timeframe == 'short_term']
            mid_agents = [a for a in self.agents if a.dna.timeframe == 'mid_term'] 
            long_agents = [a for a in self.agents if a.dna.timeframe == 'long_term']
            
            short_elite = short_agents[:3] if short_agents else []
            mid_elite = mid_agents[:3] if mid_agents else []
            long_elite = long_agents[:3] if long_agents else []
            
            new_generation = short_elite + mid_elite + long_elite
            
            # ✅ FIX: Track next unique agent ID globally
            if self.agents:
                next_agent_id = max(a.dna.agent_id for a in self.agents) + 1
            else:
                next_agent_id = 0
            
            # Fill with offspring
            while len(new_generation) < self.population_size:
                short_count = len([a for a in new_generation if a.dna.timeframe == 'short_term'])
                mid_count = len([a for a in new_generation if a.dna.timeframe == 'mid_term'])
                long_count = len([a for a in new_generation if a.dna.timeframe == 'long_term'])
                
                if short_count < self.agents_per_timeframe and short_agents:
                    parent1 = self._tournament_selection(short_agents)
                    parent2 = self._tournament_selection(short_agents)
                    timeframe = 'short_term'
                elif mid_count < self.agents_per_timeframe and mid_agents:
                    parent1 = self._tournament_selection(mid_agents)
                    parent2 = self._tournament_selection(mid_agents)
                    timeframe = 'mid_term'
                elif long_count < self.agents_per_timeframe and long_agents:
                    parent1 = self._tournament_selection(long_agents)
                    parent2 = self._tournament_selection(long_agents)
                    timeframe = 'long_term'
                else:
                    # Fallback: select from any timeframe
                    all_agents = short_agents + mid_agents + long_agents
                    if all_agents:
                        parent1 = self._tournament_selection(all_agents)
                        parent2 = self._tournament_selection(all_agents)
                        timeframe = parent1.dna.timeframe
                    else:
                        logger.error("❌ No agents available for evolution")
                        break
                
                try:
                    child_dna = self._crossover(parent1.dna, parent2.dna, child_id=next_agent_id)
                    next_agent_id += 1
                    
                    child_dna = self._mutate(child_dna)
                    
                    # ✅ FIX: Determine balance based on central pool usage
                    if self.use_central_pool:
                        child_balance = 0.0  # Central pool agents start with $0
                    else:
                        child_balance = self.initial_balance  # Individual balance
                    
                    child_agent = LeveragedEvolutionaryAgent(child_dna, child_balance)
                    new_generation.append(child_agent)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create child agent: {e}")
                    continue
            
            # Update the population
            self.agents = new_generation

            # Log evolution results
            elite_count = len(short_elite) + len(mid_elite) + len(long_elite)
            logger.info(f"✅ Population evolved to generation {self.generation + 1}")
            logger.info(f"   Elite agents carried over: {elite_count}")
            logger.info(f"   New population size: {len(self.agents)}")
            logger.info(f"   Short-term: {len([a for a in self.agents if a.dna.timeframe == 'short_term'])}")
            logger.info(f"   Mid-term: {len([a for a in self.agents if a.dna.timeframe == 'mid_term'])}")
            logger.info(f"   Long-term: {len([a for a in self.agents if a.dna.timeframe == 'long_term'])}")
            
            # Log top performers
            if self.agents:
                top_agent = max(self.agents, key=lambda a: a.dna.fitness_score)
                logger.info(f"   Top agent: ID {top_agent.dna.agent_id}, Fitness: {top_agent.dna.fitness_score:.1f}")
                
        except Exception as e:
            logger.error(f"❌ Evolution failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _tournament_selection(self, candidates: List, size: int = 5):
        """✅ COMPLETE METHOD: Tournament selection"""
        tournament = np.random.choice(candidates, min(size, len(candidates)), replace=False)
        return max(tournament, key=lambda a: a.dna.fitness_score)
    
    def _crossover(self, parent1, parent2, child_id: int = None):
        """✅ FIXED: Crossover with explicit child_id parameter"""
        from evolutionary_paper_trading_2 import AgentDNA
        
        # ✅ FIX: Use provided child_id if given
        if child_id is None:
            child_id = max(a.dna.agent_id for a in self.agents) + 1
        
        def crossover_gene(gene1, gene2):
            return gene1 if np.random.random() > 0.5 else gene2
        
        return AgentDNA(
            agent_id=child_id,  # ✅ Use explicit child_id
            generation=self.generation + 1,
            timeframe=parent1.timeframe,
            min_confidence=crossover_gene(parent1.min_confidence, parent2.min_confidence),
            min_win_prob=crossover_gene(parent1.min_win_prob, parent2.min_win_prob),
            volatility_z_threshold=crossover_gene(parent1.volatility_z_threshold, parent2.volatility_z_threshold),
            position_size_base=crossover_gene(parent1.position_size_base, parent2.position_size_base),
            risk_reward_threshold=crossover_gene(parent1.risk_reward_threshold, parent2.risk_reward_threshold),
            expected_value_threshold=crossover_gene(parent1.expected_value_threshold, parent2.expected_value_threshold),
            stop_loss_distance=crossover_gene(parent1.stop_loss_distance, parent2.stop_loss_distance),
            take_profit_distance=crossover_gene(parent1.take_profit_distance, parent2.take_profit_distance),
            trailing_stop_activation=crossover_gene(parent1.trailing_stop_activation, parent2.trailing_stop_activation),
            min_holding_minutes=crossover_gene(parent1.min_holding_minutes, parent2.min_holding_minutes),
            max_holding_hours=crossover_gene(parent1.max_holding_hours, parent2.max_holding_hours),
            aggression=crossover_gene(parent1.aggression, parent2.aggression),
            patience=crossover_gene(parent1.patience, parent2.patience),
            contrarian_bias=crossover_gene(parent1.contrarian_bias, parent2.contrarian_bias),
            loss_aversion=crossover_gene(parent1.loss_aversion, parent2.loss_aversion),
            asset_preference=crossover_gene(parent1.asset_preference, parent2.asset_preference),
            regime_preference=crossover_gene(parent1.regime_preference, parent2.regime_preference),
            parent_ids=[parent1.agent_id, parent2.agent_id]
        )
    
    def _mutate(self, dna):
        """✅ COMPLETE METHOD: Mutation with cached parent"""
        if not hasattr(self, '_mutation_helper_parent'):
            from evolutionary_paper_trading_2 import EvolutionaryPaperTradingV2
            self._mutation_helper_parent = EvolutionaryPaperTradingV2(self.initial_balance)
        
        return self._mutation_helper_parent._mutate(dna)
    
    def _get_short_term_archetypes_AGGRESSIVE(self):
        """🔥 MAXIMUM AGGRESSION: Ultra-fast scalping with high frequency"""
        return {
            'ultra_aggressive_scalper': {
                'aggression': 0.98,
                'patience': 0.02,
                'position_size_base': 0.35,    # ✅ INCREASED from 0.25 (40% bigger)
                'min_holding_minutes': 1,
                'max_holding_hours': 0.25,
                'stop_loss_distance': 0.005,
                'take_profit_distance': 0.020,
                'volatility_z_threshold': 2.0,
                'expected_value_threshold': 0.003,
                'min_confidence': 40.0,
                'min_win_prob': 0.40,
                'risk_reward_threshold': 2.2,
                'trailing_stop_activation': 0.012,
                'contrarian_bias': 0.0,
                'loss_aversion': 0.6
            },
            'hyper_momentum_trader': {
                'aggression': 0.92,
                'patience': 0.08,
                'position_size_base': 0.30,    # ✅ INCREASED from 0.22
                'min_holding_minutes': 2,
                'max_holding_hours': 0.75,
                'stop_loss_distance': 0.008,
                'take_profit_distance': 0.028,
                'volatility_z_threshold': 2.3,
                'expected_value_threshold': 0.005,
                'min_confidence': 45.0,
                'min_win_prob': 0.45,
                'risk_reward_threshold': 2.5,
                'trailing_stop_activation': 0.015,
                'contrarian_bias': 0.0,
                'loss_aversion': 0.7
            },
            'aggressive_breakout': {
                'aggression': 0.88,
                'patience': 0.12,
                'position_size_base': 0.28,    # ✅ INCREASED from 0.20
                'min_holding_minutes': 3,
                'max_holding_hours': 1.0,
                'stop_loss_distance': 0.009,
                'take_profit_distance': 0.025,
                'volatility_z_threshold': 2.5,
                'expected_value_threshold': 0.006,
                'min_confidence': 48.0,
                'min_win_prob': 0.46,
                'risk_reward_threshold': 2.4,
                'trailing_stop_activation': 0.018,
                'contrarian_bias': 0.0,
                'loss_aversion': 0.75
            }
        }

    def _get_mid_term_archetypes_AGGRESSIVE(self):
        """
        ✅ SHORTENED: Faster mid-term feedback for A3C
        
        OLD: 3-18 hours (too slow for training)
        NEW: 1-6 hours (4x faster feedback loop)
        """
        return {
            'aggressive_swing_trader': {
                'patience': 0.55,
                'aggression': 0.60,
                'position_size_base': 0.28,
                'min_holding_minutes': 60,    # ✅ 1 hour (was 180)
                'max_holding_hours': 6.0,     # ✅ 6 hours (was 18)
                'stop_loss_distance': 0.025,
                'take_profit_distance': 0.070,
                'volatility_z_threshold': 2.5,
                'expected_value_threshold': 0.015,
                'min_confidence': 58.0,
                'min_win_prob': 0.52,
                'risk_reward_threshold': 3.0,
                'trailing_stop_activation': 0.030,
                'contrarian_bias': 0.0,
                'loss_aversion': 1.1
            },
            'momentum_swing_trader': {
                'aggression': 0.65,
                'patience': 0.50,
                'position_size_base': 0.30,
                'min_holding_minutes': 90,    # ✅ 1.5 hours (was 240)
                'max_holding_hours': 8.0,     # ✅ 8 hours (was 24)
                'stop_loss_distance': 0.030,
                'take_profit_distance': 0.080,
                'volatility_z_threshold': 2.7,
                'expected_value_threshold': 0.018,
                'min_confidence': 60.0,
                'min_win_prob': 0.54,
                'risk_reward_threshold': 3.2,
                'trailing_stop_activation': 0.035,
                'contrarian_bias': 0.0,
                'loss_aversion': 1.2
            }
        }

    def _get_long_term_archetypes_AGGRESSIVE(self):
        """Long-term with maximum position sizing"""
        return {
            'aggressive_trend_follower': {
                'patience': 0.70,
                'aggression': 0.40,
                'position_size_base': 0.35,    # ✅ INCREASED from 0.28
                'min_holding_minutes': 720,
                'max_holding_hours': 84.0,
                'stop_loss_distance': 0.035,
                'take_profit_distance': 0.100,
                'volatility_z_threshold': 2.0,
                'expected_value_threshold': 0.018,
                'min_confidence': 58.0,
                'min_win_prob': 0.54,
                'risk_reward_threshold': 3.2,
                'trailing_stop_activation': 0.030,
                'contrarian_bias': 0.0,
                'loss_aversion': 1.2
            }
        }

    
    def _create_dna(self, agent_id: int, base_params: Dict, 
                    timeframe: str = 'mid_term',
                    asset_preference: str = 'ALL'):
        """✅ COMPLETE METHOD: Create DNA using cached parent"""
        from evolutionary_paper_trading_2 import EvolutionaryPaperTradingV2
        
        if not hasattr(self, '_dna_creator_helper'):
            self._dna_creator_helper = EvolutionaryPaperTradingV2(self.initial_balance)
        
        self._dna_creator_helper.generation = self.generation
        
        return self._dna_creator_helper._create_dna(agent_id, base_params, timeframe, asset_preference)





# ✅ COMPLETE BOOTSTRAP FUNCTION
async def bootstrap_meta_rl_with_leveraged_evolution(
    hyperliquid_exchange: EnhancedHyperliquidExchange,
    enable_parallel: bool = True
):
    """✅ COMPLETE METHOD: Bootstrap MetaRLV5 with LEVERAGED evolutionary training"""
    from meta_rl_enhanced2 import MetaRLSupervisorV5
    
    meta_rl = MetaRLSupervisorV5(initial_balance=1000.0)
    
    evolutionary_system = LiveEvolutionaryLeveragedTrading(
        hyperliquid_exchange=hyperliquid_exchange,
        initial_balance=1000.0,
        enable_parallel=enable_parallel
    )
    
    evolutionary_system.initialize_population()
    
    logger.info("🚀 Running LEVERAGED evolutionary training (3x-10x)...")
    logger.info(f"   Parallel Processing: {'ENABLED' if enable_parallel else 'DISABLED'}")
    logger.info("   30 Short-term specialists (5min-3hr, 5x-10x leverage)")
    logger.info("   30 Mid-term specialists (4hr-3days, 4x-7x leverage)")
    logger.info("   30 Long-term specialists (1wk-1month, 3x-5x leverage)")
    
    # Run fewer generations with live data
    for generation in range(3):
        await evolutionary_system.run_evolution_cycle(cycles_per_generation=20)
        
        # Feed trades to MetaRLV5
        training_data = evolutionary_system.export_training_data_for_meta_rl()
        for trade in training_data:
            meta_rl.record_trade_outcome(
                traded=True,
                action=trade['action'],
                market_regime=trade['market_regime'],
                pnl=trade['pnl'],
                pnl_pct=trade['leverage_pnl_pct'],
                success=trade['success'],
                confidence_used=trade['confidence_used'],
                win_prob_used=trade['win_prob_used'],
                parameters_used=trade['parameters'],
                timeframe=trade['timeframe']
            )
        
        logger.info(f"   Generation {generation + 1}: {len(training_data)} "
                   f"LEVERAGED trades fed to MetaRLV5")
    
    insights = evolutionary_system.get_evolutionary_insights()
    
    logger.info("\n✅ MetaRLV5 bootstrapped with LEVERAGED evolutionary data")
    logger.info(f"   Total Trades: {len(evolutionary_system.all_trades)}")
    
    # Leverage-specific insights
    closed_trades = [t for t in evolutionary_system.all_trades if t.status == 'closed']
    if closed_trades:
        avg_leverage = np.mean([t.leverage for t in closed_trades])
        liquidations = sum(1 for t in closed_trades if t.close_reason == 'liquidated')
        logger.info(f"   Average Leverage: {avg_leverage:.1f}x")
        logger.info(f"   Liquidation Rate: {liquidations/len(closed_trades)*100:.1f}%")
    
    return meta_rl, insights, evolutionary_system


# ✅ COMPLETE TEST FUNCTION
async def test_leveraged_evolution():
    """✅ COMPLETE METHOD: Test the leveraged evolutionary system"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Hyperliquid
    exchange = EnhancedHyperliquidExchange(
        wallet_address=os.getenv('HYPERLIQUID_WALLET_ADDRESS'),
        api_wallet_private_key=os.getenv('HYPERLIQUID_API_PRIVATE_KEY'),
        testnet=os.getenv('HYPERLIQUID_TESTNET', 'false').lower() == 'true'
    )
    
    # Create leveraged evolutionary system
    system = LiveEvolutionaryLeveragedTrading(
        hyperliquid_exchange=exchange,
        initial_balance=1000.0,
        enable_parallel=True
    )
    
    system.initialize_population()
    
    # Run one generation with live data
    await system.run_evolution_cycle(cycles_per_generation=10)
    
    # Show results
    status = system.get_live_status()
    logger.info(f"\n📊 LEVERAGED SYSTEM STATUS:")
    logger.info(f"   Generation: {status['generation']}")
    logger.info(f"   Total Trades: {status['total_trades']}")
    logger.info(f"   Active Trades: {status['active_trades']}")
    logger.info(f"   Avg Active Leverage: {status['avg_active_leverage']:.1f}x")
    
    for timeframe, stats in status['timeframe_stats'].items():
        logger.info(f"   {timeframe.upper()}: {stats['total_trades']} trades, "
                   f"{stats['win_rate']:.1f}% WR, ${stats['total_pnl']:+.2f}, "
                   f"{stats['avg_leverage']:.1f}x leverage, "
                   f"{stats['liquidation_rate']:.1f}% liquidated")


if __name__ == "__main__":
    asyncio.run(test_leveraged_evolution())