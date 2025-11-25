"""
FIXED IMPORTS FOR main4.4_leverage.py
Place this at the top of your file, replacing the existing imports section
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dotenv import load_dotenv
from collections import deque
import numpy as np
from dataclasses import dataclass
import traceback
import aiohttp
import torch
import torch.nn as nn

# ✅ FIX #1: Import from existing files
from rl_trading_system import (
    RLTradingSupervisor,
    MarketState,
    TradeRecord,
    VolatilityZScoreGate,
    MetaRLTrustLayer,
    ProbabilityScorer
)
from fixed_hyperliquid import EnhancedHyperliquidExchange
from ultimate_hedge_fund import HedgeFundBrain, MarketSnapshot
from meta_rl_enhanced2 import MetaRLSupervisorV5

# ✅ FIX #2: Import CORRECTED classes from evolutionary file
from evolutionary_paper_trading_leverage import (
    LiveEvolutionaryLeveragedTrading,
    bootstrap_meta_rl_with_leveraged_evolution,
    EnhancedCentralPortfolio,  # ✅ Use EnhancedCentralPortfolio (not CentralPortfolio)
    LeveragedTrade,
    is_winning_trade 
    )

from rl_agent_generator import RLAgentCoordinator
from agent_registry import AgentRegistry
from volatility_forecaster import ARCHGARCHForecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

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

class LeveragedDecisionEngine:
    """
    🧠 META-RL V5 CENTRAL BRAIN WITH LEVERAGE AWARENESS
    Makes ALL final decisions with leverage optimization
    """
    
    def __init__(self, meta_rl_v5: MetaRLSupervisorV5, 
            hedge_fund_brain: HedgeFundBrain, 
            rl_supervisor: RLTradingSupervisor,
            price_history: Dict = None):  # ✅ ADD price_history parameter
        self.meta_rl = meta_rl_v5
        self.ai_brain = hedge_fund_brain
        self.rl_supervisor = rl_supervisor
        self.volatility_gate = VolatilityZScoreGate()
        self.trust_layer = MetaRLTrustLayer()
        self.probability_scorer = ProbabilityScorer()
        
        # ✅ FIX: Store price history reference
        self.price_history = price_history if price_history is not None else {}
        
        # Timeframe tracking
        self.current_timeframe = 'mid_term'
        self.timeframe_performance = {
            'short_term': {'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'avg_leverage': 0.0},
            'mid_term': {'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'avg_leverage': 0.0},
            'long_term': {'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'avg_leverage': 0.0}
        }
        
        # Leverage tracking
        self.leverage_history = deque(maxlen=100)
        self.liquidation_count = 0
        
        # ✅ FIX: Initialize supporting systems
        self.agent_registry = AgentRegistry()
        self.volatility_forecaster = ARCHGARCHForecaster()
        self.rl_coordinator = RLAgentCoordinator()
        
        # ✅ FIX: Initialize assets list for market structure building
        self.assets = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
        self.market_data_cache = {}
        self.current_volatility_regime = 1  # Default mid volatility
        
        logger.info("🧠 LEVERAGED META-RL V5 DECISION ENGINE INITIALIZED")
        logger.info("   🎯 RL Systems: DQN (short) + A3C (mid) + PPO (long)")
        logger.info("   📊 Volatility Forecaster: GARCH(1,1)")
        logger.info("   🏆 Elite Registry: Active")
        logger.info("   ✅ Price history access: Enabled")
        logger.info("   Leverage Range: 3x - 10x")

    def _build_market_structure(self) -> MarketStructure:
        """
        ✅ IMPROVED: Build market structure with price history access
        """
        try:
            # Use stored regime if available
            vol_regime = getattr(self, 'current_volatility_regime', 1)
            
            # ✅ FIX: Calculate trend from price history if available
            trend = 0.0
            if self.price_history:
                # Get recent price movements across all assets
                recent_changes = []
                for asset, history in self.price_history.items():
                    if len(history) >= 10:
                        try:
                            prices = [h['price'] for h in list(history)[-10:] if isinstance(h, dict)]
                            if len(prices) >= 10:
                                # Simple trend: compare first and last
                                price_change = (prices[-1] - prices[0]) / prices[0]
                                recent_changes.append(price_change)
                        except Exception:
                            continue
                
                if recent_changes:
                    trend = np.mean(recent_changes)
                    trend = float(np.clip(trend, -0.5, 0.5))
            
            # Default values
            mean_reversion_score = 0.5
            liquidity_score = 0.8
            funding_rate = 0.0001
            volume_profile = 1.0
            orderbook_imbalance = 0.0
            
            now = datetime.now()
            
            return MarketStructure(
                volatility_regime=vol_regime,
                trend_strength=trend,
                mean_reversion_score=mean_reversion_score,
                liquidity_score=liquidity_score,
                funding_rate=funding_rate,
                volume_profile=volume_profile,
                orderbook_imbalance=orderbook_imbalance,
                time_of_day=now.hour,
                day_of_week=now.weekday()
            )
        
        except Exception as e:
            logger.error(f"❌ Market structure building error: {e}")
            # Return safe default
            now = datetime.now()
            return MarketStructure(
                volatility_regime=1,
                trend_strength=0.0,
                mean_reversion_score=0.5,
                liquidity_score=0.8,
                funding_rate=0.0001,
                volume_profile=1.0,
                orderbook_imbalance=0.0,
                time_of_day=now.hour,
                day_of_week=now.weekday()
            )
    
    def _calculate_optimal_leverage_AGGRESSIVE(self, market_data, regime: str,
                                            timeframe: str, confidence: float) -> float:
        """🚀 EXTREME LEVERAGE CALCULATION"""
        
        # Base leverage by timeframe - ALL BOOSTED
        base_leverage = {
            'short_term': 10.0,  # ⬆️ Was 7.0
            'mid_term': 8.0,     # ⬆️ Was 5.0
            'long_term': 6.0     # ⬆️ Was 3.5
        }.get(timeframe, 8.0)
        
        # Volatility adjustment - LESS CONSERVATIVE
        volatility = market_data.volatility_24h
        if volatility > 0.08:
            vol_factor = 0.7   # ⬆️ Was 0.5
        elif volatility > 0.05:
            vol_factor = 0.85  # ⬆️ Was 0.7
        elif volatility < 0.02:
            vol_factor = 1.4   # ⬆️ Was 1.2
        else:
            vol_factor = 1.0
        
        # Regime adjustment - MORE AGGRESSIVE
        regime_factor = {
            'bull_strong': 1.4,      # ⬆️ Was 1.2
            'bull_weak': 1.2,        # ⬆️ Was 1.0
            'bear_strong': 1.0,      # ⬆️ Was 0.8
            'bear_weak': 1.1,        # ⬆️ Was 0.9
            'ranging': 1.1,          # ⬆️ Was 0.9
            'high_volatility': 0.8,  # ⬆️ Was 0.6
            'crash': 0.7             # ⬆️ Was 0.5
        }.get(regime, 1.0)
        
        # Confidence adjustment - MORE AGGRESSIVE
        confidence_factor = 0.75 + (confidence / 100) * 0.7  # ⬆️ Was 0.7/0.6
        
        optimal_leverage = base_leverage * vol_factor * regime_factor * confidence_factor
        
        # ⬆️ Range: 5x-15x (was 3x-10x)
        return np.clip(optimal_leverage, 5.0, 15.0)


    def _calculate_liquidation_price(self, market_data, optimal_leverage: float, action: str) -> float:
        """FIXED: Accurate liquidation price calculation"""
        try:
            current_price = market_data.price
            
            # Input validation
            if optimal_leverage <= 1.0 or np.isnan(optimal_leverage) or np.isinf(optimal_leverage):
                logger.warning(f"Invalid leverage: {optimal_leverage}, using 3.0x")
                optimal_leverage = 3.0
            
            optimal_leverage = float(np.clip(optimal_leverage, 1.1, 10.0))
            
            if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
                logger.error(f"Invalid price: {current_price}")
                return 0.0
            
            # Exchange-specific parameters
            maintenance_margin = 0.03  # 3% for most perpetuals
            initial_margin = 1.0 / optimal_leverage
            
            # Safety buffer (prevent immediate liquidation)
            safety_buffer = 0.005  # 0.5% buffer
            
            if action == 'BUY':
                # LONG: price drops to liquidation
                liquidation_threshold = initial_margin - maintenance_margin + safety_buffer
                liquidation_price = current_price * (1 - liquidation_threshold)
                
                # Ensure liquidation price is reasonable
                if liquidation_price >= current_price * 0.99:  # Too close
                    liquidation_price = current_price * 0.95  # Emergency fallback
                    
            else:  # SELL/SHORT
                # SHORT: price rises to liquidation  
                liquidation_threshold = initial_margin - maintenance_margin + safety_buffer
                liquidation_price = current_price * (1 + liquidation_threshold)
                
                # Ensure liquidation price is reasonable
                if liquidation_price <= current_price * 1.01:  # Too close
                    liquidation_price = current_price * 1.05  # Emergency fallback
            
            # Final validation
            if action == 'BUY' and liquidation_price >= current_price:
                liquidation_price = current_price * 0.90
            elif action == 'SELL' and liquidation_price <= current_price:
                liquidation_price = current_price * 1.10
                
            return max(liquidation_price, 0.001)  # Ensure positive
            
        except Exception as e:
            logger.error(f"Liquidation calculation error: {e}")
            # Fallback: 10% away from current price
            return current_price * (0.9 if action == 'BUY' else 1.1)
    
    async def make_leveraged_decision(
        self, 
        asset: str, 
        market_data: MarketSnapshot,
        portfolio: Dict
    ) -> Dict:
        """
        ✅ COMPLETE FIXED: Handles API failures gracefully
        
        CRITICAL CHANGES:
        1. Try-except around AI analysis
        2. Fallback to simple RSI-based logic if AI fails
        3. Continue with decision flow even if AI unavailable
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🧠 LEVERAGED META-RL V5 ANALYZING: {asset}")
        logger.info(f"{'='*70}")
        
        # PHASE 1 ENHANCEMENT: VOLATILITY ANALYSIS
        current_vol = self.volatility_forecaster.update(market_data.price)
        vol_regime = self.volatility_forecaster.get_volatility_regime()
        vol_trend = self.volatility_forecaster.get_volatility_trend()
        forecast_confidence = self.volatility_forecaster.get_forecast_confidence()
        
        logger.info(f"📊 VOLATILITY ANALYSIS: {current_vol:.1%} ({vol_regime} regime, {vol_trend} trend)")
        logger.info(f"   Forecast Confidence: {forecast_confidence:.1%}")
        
        # STEP 1: Detect market regime
        regime = self._detect_regime(market_data)
        logger.info(f"📊 Market Regime: {regime.upper()}")
        
        # PHASE 1 ENHANCEMENT: CHECK ELITE AGENTS FIRST
        elite_agent = self.agent_registry.get_best_agent_for_conditions(regime)
        elite_ensemble = self.agent_registry.get_ensemble_agents(regime, count=3)
        
        if elite_agent and elite_agent['fitness'] > 60.0:
            logger.info(f"🏆 ELITE AGENT FOUND: ID {elite_agent['agent_id']} (Fitness: {elite_agent['fitness']:.1f})")
            logger.info(f"   Win Rate: {elite_agent['win_rate']:.1%} | Trades: {elite_agent['total_trades']} | P&L: ${elite_agent['total_pnl']:.2f}")
        
        if elite_ensemble:
            logger.info(f"👥 ELITE ENSEMBLE: {len(elite_ensemble)} agents available for {regime} regime")
        
        # STEP 2: Select optimal timeframe
        selected_timeframe = self._select_optimal_timeframe(market_data, regime)
        logger.info(f"⏰ Selected Timeframe: {selected_timeframe.upper()}")
        
        # STEP 3: Get timeframe-specific parameters
        adaptive_params = self.meta_rl.get_adaptive_parameters(regime, selected_timeframe)
        
        # PHASE 1 ENHANCEMENT: VOLATILITY-ADJUSTED PARAMETERS
        vol_position_multiplier = self.volatility_forecaster.get_position_size_multiplier(
            adaptive_params['position_size_multiplier'],
            adaptive_params.get('risk_tolerance', 1.0)
        )
        
        adaptive_params['position_size_multiplier'] *= vol_position_multiplier
        
        vol_stop_adjustment = self.volatility_forecaster.get_stop_loss_adjustment()
        adaptive_params['stop_loss_distance'] *= vol_stop_adjustment
        adaptive_params['take_profit_distance'] *= vol_stop_adjustment
        
        logger.info(f"📈 VOL-ADJUSTED PARAMS: Position ×{vol_position_multiplier:.2f} | Stops ×{vol_stop_adjustment:.2f}")
        
        # STEP 4: Run through all gates
        vol_state = self.volatility_gate.get_volatility_state(asset, market_data.volatility_24h)
        self.volatility_gate.update_volatility(asset, market_data.volatility_24h)
        
        logger.info(f"\n🚪 GATE 1 - Volatility:")
        logger.info(f"   Z-Score: {vol_state.z_score:.2f} | Threshold: {adaptive_params['volatility_z_threshold']:.1f}")
        logger.info(f"   Current Vol: {current_vol:.2%} | Regime: {vol_regime}")
        
        if vol_state.z_score > adaptive_params['volatility_z_threshold']:
            logger.warning(f"   ❌ BLOCKED: Volatility too extreme")
            return self._create_blocked_decision('VOLATILITY', vol_state.z_score, adaptive_params, selected_timeframe)
        
        if vol_regime in ["EXTREME", "HIGH"] and current_vol > 0.60:
            logger.warning(f"   ❌ BLOCKED: Extreme volatility regime ({vol_regime})")
            return self._create_blocked_decision('VOLATILITY_REGIME', current_vol, adaptive_params, selected_timeframe)
            
        logger.info(f"   ✅ PASSED")
        
        # ✅ FIX: Try-except around AI analysis
        risk_limits = {
            'max_position_pct': 30,
            'max_loss_pct': 3.0,
            'daily_loss_limit': portfolio['balance'] * 0.05,
            'volatility_regime': vol_regime,
            'current_volatility': current_vol,
            'volatility_multiplier': vol_position_multiplier
        }
        
        try:
            ai_analysis = await self.ai_brain.make_trading_decision(market_data, portfolio, risk_limits)
            logger.info(f"\n🤖 AI Analysis:")
            logger.info(f"   Action: {ai_analysis.action}")
            logger.info(f"   Confidence: {ai_analysis.confidence:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            logger.info("   ℹ️ Using fallback RSI-based analysis")
            
            # ✅ FALLBACK: Simple RSI-based logic
            if market_data.rsi_14 < 30:
                ai_analysis = type('obj', (object,), {
                    'action': 'BUY',
                    'confidence': 65.0,
                    'reasoning': 'RSI oversold (AI unavailable)'
                })()
                logger.info(f"\n🤖 Fallback Analysis:")
                logger.info(f"   Action: BUY (RSI: {market_data.rsi_14:.1f} < 30)")
                logger.info(f"   Confidence: 65.0%")
                
            elif market_data.rsi_14 > 70:
                ai_analysis = type('obj', (object,), {
                    'action': 'SELL',
                    'confidence': 65.0,
                    'reasoning': 'RSI overbought (AI unavailable)'
                })()
                logger.info(f"\n🤖 Fallback Analysis:")
                logger.info(f"   Action: SELL (RSI: {market_data.rsi_14:.1f} > 70)")
                logger.info(f"   Confidence: 65.0%")
                
            else:
                ai_analysis = type('obj', (object,), {
                    'action': 'WAIT',
                    'confidence': 0.0,
                    'reasoning': 'Neutral market (AI unavailable)'
                })()
                logger.info(f"\n🤖 Fallback Analysis:")
                logger.info(f"   Action: WAIT (RSI: {market_data.rsi_14:.1f} neutral)")
                logger.info(f"   Confidence: 0.0%")
        
        # Trust layer
        market_state = self._create_market_state(asset, market_data)
        trust_score = self.trust_layer.evaluate_decision(
            action=ai_analysis.action,
            market_regime=regime,
            predicted_confidence=ai_analysis.confidence,
            market_state=market_state
        )
        
        adjustment_factor = max(0.85, trust_score.adjustment_factor) if trust_score.sample_size < 20 else trust_score.adjustment_factor
        calibrated_confidence = ai_analysis.confidence * adjustment_factor
        
        # PHASE 1 ENHANCEMENT: VOLATILITY CONFIDENCE ADJUSTMENT
        vol_confidence_penalty = 1.0 - (current_vol * 0.5)
        calibrated_confidence *= vol_confidence_penalty
        
        logger.info(f"\n🚪 GATE 2 - Trust Layer:")
        logger.info(f"   Calibrated Confidence: {calibrated_confidence:.1f}% | Threshold: {adaptive_params['min_confidence']:.1f}%")
        logger.info(f"   Volatility Penalty: {vol_confidence_penalty:.2f}x (Current vol: {current_vol:.1%})")
        
        if calibrated_confidence < adaptive_params['min_confidence']:
            logger.warning(f"   ❌ BLOCKED: Confidence too low")
            return self._create_blocked_decision('TRUST', calibrated_confidence, adaptive_params, selected_timeframe)
        logger.info(f"   ✅ PASSED")
        
        # Probability scoring
        prob_score = self.probability_scorer.calculate_probability(
            market_state,
            ai_analysis,
            self.rl_supervisor.multi_asset_manager.learners[asset].experience
        )
        
        # Calculate leverage ONCE
        optimal_leverage = self._calculate_optimal_leverage_AGGRESSIVE(
            market_data, regime, selected_timeframe, calibrated_confidence
        )
        
        vol_adjusted_leverage = self.volatility_forecaster.get_leverage_adjustment(optimal_leverage)
        leverage_reduction = ((optimal_leverage - vol_adjusted_leverage) / optimal_leverage) * 100 if optimal_leverage > 0 else 0
        optimal_leverage = vol_adjusted_leverage
        
        # Use leverage in EV calculation
        calculated_ev = self._calculate_realistic_expected_value(
            prob_score, 
            optimal_leverage,
            market_data, 
            regime
        )
        
        logger.info(f"\n🚪 GATE 3 - Probability:")
        logger.info(f"   Win Prob: {prob_score.win_probability*100:.1f}% | Threshold: {adaptive_params['min_win_prob']*100:.1f}%")
        logger.info(f"   Expected Value: {calculated_ev*100:.2f}% | Threshold: {adaptive_params['expected_value_threshold']*100:.2f}%")
        logger.info(f"   Volatility Impact: {leverage_reduction:.1f}% leverage reduction")
        
        if prob_score.win_probability < adaptive_params['min_win_prob']:
            logger.warning(f"   ❌ BLOCKED: Win probability too low")
            return self._create_blocked_decision('PROBABILITY', prob_score.win_probability, adaptive_params, selected_timeframe)
        
        if calculated_ev < adaptive_params['expected_value_threshold']:
            logger.warning(f"   ❌ BLOCKED: Expected value too low ({calculated_ev*100:.2f}% < {adaptive_params['expected_value_threshold']*100:.2f}%)")
            return self._create_blocked_decision('EXPECTED_VALUE', calculated_ev, adaptive_params, selected_timeframe)
        
        logger.info(f"   ✅ PASSED")       
        
        # STEP 5: EXECUTE TRADE (if gates passed)
        if ai_analysis.action in ['BUY', 'SELL']:
            logger.info(f"\n⚡ LEVERAGE CALCULATION:")
            logger.info(f"   Timeframe Base: {selected_timeframe} → {optimal_leverage:.1f}x")
            logger.info(f"   Volatility: {market_data.volatility_24h*100:.2f}% → {optimal_leverage:.1f}x")
            logger.info(f"   Regime: {regime}")
            logger.info(f"   Final Leverage: {optimal_leverage:.1f}x (Vol-adjusted)")
            
            # Calculate position sizing
            base_size = portfolio['balance'] * (prob_score.recommended_size_pct / 100)
            position_size = base_size * adaptive_params['position_size_multiplier']
            position_size = min(position_size, portfolio['balance'] * 0.30)
            
            # Elite agent influence
            elite_influence = 0.0
            if elite_agent and elite_agent['fitness'] > 70.0:
                elite_influence = 0.1
                position_size *= (1 + elite_influence)
                logger.info(f"   🏆 Elite Agent Boost: +{elite_influence*100:.0f}% position size")
            
            effective_size = position_size * optimal_leverage
            
            # Liquidation price
            liquidation_price = self._calculate_liquidation_price(
                market_data, 
                optimal_leverage, 
                ai_analysis.action
            )
            
            # Stop loss and take profit
            if ai_analysis.action == 'BUY':
                stop_loss = market_data.price * (1 - adaptive_params['stop_loss_distance'])
                take_profit = market_data.price * (1 + adaptive_params['take_profit_distance'])
            else:
                stop_loss = market_data.price * (1 + adaptive_params['stop_loss_distance'])
                take_profit = market_data.price * (1 - adaptive_params['take_profit_distance'])
            
            # Validate stop loss
            if ai_analysis.action == 'BUY':
                if stop_loss <= liquidation_price:
                    logger.warning(f"⚠️ Stop loss too tight, adjusting above liquidation")
                    stop_loss = liquidation_price * 1.02
            else:
                if stop_loss >= liquidation_price:
                    logger.warning(f"⚠️ Stop loss too tight, adjusting below liquidation")
                    stop_loss = liquidation_price * 0.98
            
            # Risk/Reward with leverage
            if ai_analysis.action == 'BUY':
                potential_gain = (take_profit - market_data.price) / market_data.price
                potential_loss = (market_data.price - stop_loss) / market_data.price
            else:
                potential_gain = (market_data.price - take_profit) / market_data.price
                potential_loss = (stop_loss - market_data.price) / market_data.price
            
            # Apply leverage multiplier
            leverage_gain = potential_gain * optimal_leverage
            leverage_loss = potential_loss * optimal_leverage
            
            actual_rr = leverage_gain / leverage_loss if leverage_loss > 1e-6 else 0
            
            # Validate R/R
            if actual_rr < 1.5:
                logger.error(f"   ❌ R/R validation failed: {actual_rr:.2f}:1 < 1.5:1")
                return self._create_blocked_decision('RISK_REWARD', actual_rr, adaptive_params, selected_timeframe)
            
            # Calculate safety margin from liquidation
            if ai_analysis.action == 'BUY':
                liquidation_buffer_pct = ((market_data.price - liquidation_price) / market_data.price) * 100
            else:
                liquidation_buffer_pct = ((liquidation_price - market_data.price) / market_data.price) * 100
            
            logger.info(f"\n🎯 LEVERAGED DECISION:")
            logger.info(f"   Action: {ai_analysis.action}")
            logger.info(f"   Leverage: {optimal_leverage:.1f}x (Vol-adjusted)")
            logger.info(f"   Base Position: ${position_size:.2f}")
            logger.info(f"   Effective Size: ${effective_size:.2f}")
            logger.info(f"   Entry: ${market_data.price:.4f}")
            logger.info(f"   Stop Loss: ${stop_loss:.4f} ({adaptive_params['stop_loss_distance']*100:.2f}%)")
            logger.info(f"   Take Profit: ${take_profit:.4f} ({adaptive_params['take_profit_distance']*100:.2f}%)")
            logger.info(f"   Liquidation: ${liquidation_price:.4f} ⚠️ ({liquidation_buffer_pct:.1f}% buffer)")
            logger.info(f"   Risk/Reward (Leveraged): {actual_rr:.2f}:1")
            logger.info(f"   Expected Return: {calculated_ev*100:.2f}%")
            logger.info(f"   Volatility Regime: {vol_regime}")
            if elite_agent:
                logger.info(f"   Elite Influence: {elite_influence*100:.0f}% boost")
            logger.info(f"   ✅ ALL VALIDATIONS PASSED")
            logger.info(f"{'='*70}\n")
            
            # Decision metadata
            decision_metadata = {
                'volatility_regime': vol_regime,
                'current_volatility': current_vol,
                'volatility_trend': vol_trend,
                'forecast_confidence': forecast_confidence,
                'elite_agent_used': elite_agent is not None,
                'elite_agent_id': elite_agent['agent_id'] if elite_agent else None,
                'elite_agent_fitness': elite_agent['fitness'] if elite_agent else 0.0,
                'vol_position_multiplier': vol_position_multiplier,
                'vol_stop_adjustment': vol_stop_adjustment,
                'vol_leverage_adjustment': leverage_reduction,
                'original_leverage': optimal_leverage / (1 - leverage_reduction/100) if leverage_reduction > 0 else optimal_leverage,
                'liquidation_buffer_pct': liquidation_buffer_pct,
                'liquidation_price_accurate': liquidation_price
            }
            
            return {
                'action': ai_analysis.action,
                'confidence': calibrated_confidence,
                'timeframe': selected_timeframe,
                'leverage': optimal_leverage,
                'entry_price': market_data.price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'liquidation_price': liquidation_price,
                'position_size': position_size,
                'effective_size': effective_size,
                'expected_return': calculated_ev,
                'risk_reward': actual_rr,
                'gate_status': 'ALL_PASSED',
                'adaptive_params': adaptive_params,
                'market_regime': regime,
                'max_holding_hours': adaptive_params['max_holding_hours'],
                'decision_metadata': decision_metadata
            }
        else:
            logger.info(f"\n⏸️ META-RL V5 DECISION: WAIT")
            logger.info(f"   Reason: AI recommends {ai_analysis.action}")
            if vol_regime in ["EXTREME", "HIGH"]:
                logger.info(f"   Additional: High volatility regime ({vol_regime})")
            logger.info(f"{'='*70}\n")
            
            return {
                'action': 'WAIT',
                'confidence': 0,
                'timeframe': selected_timeframe,
                'reason': f"AI recommends {ai_analysis.action} + Volatility: {vol_regime}",
                'gate_status': 'AI_WAIT',
                'volatility_regime': vol_regime
            }

    
    def _select_optimal_timeframe(self, market_data: MarketSnapshot, regime: str) -> str:
        """Select best timeframe"""
        volatility = market_data.volatility_24h
        trend_strength = abs((market_data.ema_9 - market_data.ema_50) / market_data.ema_50) if market_data.ema_50 > 0 else 0
        
        if volatility > 0.06:
            return 'short_term'
        
        if trend_strength > 0.05:
            if regime in ['bull_strong', 'bear_strong']:
                return 'long_term'
            else:
                return 'mid_term'
        
        if regime == 'ranging':
            return 'short_term'
        
        return 'mid_term'
    
    def _detect_regime(self, m: MarketSnapshot) -> str:
        """Detect market regime"""
        if m.price < m.price_24h_ago * 0.85:
            return 'crash'
        if m.volatility_24h > 0.08:
            return 'high_volatility'
        if m.ema_20 > m.ema_50:
            return 'bull_strong' if m.volume_24h > m.volume_avg_7d * 1.5 else 'bull_weak'
        elif m.ema_20 < m.ema_50:
            return 'bear_strong' if m.volume_24h > m.volume_avg_7d * 1.5 else 'bear_weak'
        else:
            return 'ranging'
    
    def _create_market_state(self, asset: str, m: MarketSnapshot) -> MarketState:
        """Convert MarketSnapshot to MarketState"""
        return MarketState(
            asset=asset,
            price=m.price,
            rsi=m.rsi_14,
            macd=m.macd,
            macd_signal=m.macd_signal,
            ema_9=m.ema_9,
            ema_20=m.ema_20,
            ema_50=m.ema_50,
            volume_ratio=m.volume_24h / m.volume_avg_7d if m.volume_avg_7d > 0 else 1.0,
            volatility=m.volatility_24h,
            trend_strength=(m.ema_9 - m.ema_50) / m.ema_50 if m.ema_50 > 0 else 0,
            order_flow=m.buy_sell_ratio - 1.0
        )
    
    def _create_blocked_decision(self, gate: str, value: float, adaptive_params: Dict, timeframe: str) -> Dict:
        """Create blocked decision"""
        return {
            'action': 'WAIT',
            'confidence': 0,
            'timeframe': timeframe,
            'gate_status': f'{gate}_BLOCKED',
            'blocked_value': value
        }

    def _calculate_realistic_expected_value(self, prob_score, leverage: float, 
                                           market_data, regime: str) -> float:
        """
        ✅ PATCHED: Robust EV calculation with input validation
        
        FIXES:
        1. Input validation (leverage > 0, prob in [0,1])
        2. NaN/Inf protection at every step
        3. Proper regime-specific bounds
        4. Safe division with fallbacks
        """
        
        # ✅ FIX #1: Input validation
        if leverage <= 0 or np.isnan(leverage) or np.isinf(leverage):
            logger.warning(f"⚠️ Invalid leverage: {leverage}, defaulting to 3.0x")
            leverage = 3.0
        
        leverage = float(np.clip(leverage, 1.0, 10.0))
        
        win_prob = prob_score.win_probability
        if not (0 <= win_prob <= 1) or np.isnan(win_prob):
            logger.warning(f"⚠️ Invalid win_prob: {win_prob}, defaulting to 0.5")
            win_prob = 0.5
        
        # ✅ FIX #2: Regime-specific base moves with validation
        regime_configs = {
            'bull_strong': {'gain': 0.035, 'loss': 0.020},
            'bear_strong': {'gain': 0.035, 'loss': 0.020},
            'bull_weak': {'gain': 0.025, 'loss': 0.018},
            'bear_weak': {'gain': 0.025, 'loss': 0.018},
            'ranging': {'gain': 0.020, 'loss': 0.015},
            'high_volatility': {'gain': 0.030, 'loss': 0.022},
            'crash': {'gain': 0.015, 'loss': 0.030}
        }
        
        config = regime_configs.get(regime, {'gain': 0.025, 'loss': 0.018})
        base_gain = config['gain']
        base_loss = config['loss']
        
        # ✅ FIX #3: Apply leverage with bounds
        potential_gain = base_gain * leverage
        potential_loss = base_loss * leverage
        
        # Clip to prevent extreme values
        potential_gain = float(np.clip(potential_gain, 0.001, 2.0))  # 0.1% to 200%
        potential_loss = float(np.clip(potential_loss, 0.001, 2.0))
        
        # ✅ FIX #4: EV formula with safe calculation
        ev = (win_prob * potential_gain) - ((1 - win_prob) * potential_loss)
        
        # Validate intermediate result
        if np.isnan(ev) or np.isinf(ev):
            logger.error(f"❌ EV calculation produced invalid result: win_prob={win_prob}, gain={potential_gain}, loss={potential_loss}")
            return 0.0
        
        # ✅ FIX #5: Volatility adjustment with validation
        volatility = market_data.volatility_24h
        if np.isnan(volatility) or volatility < 0:
            volatility = 0.02
        
        volatility = float(np.clip(volatility, 0.001, 0.50))
        
        if volatility > 0.08:
            vol_factor = 0.6
        elif volatility > 0.05:
            vol_factor = 0.8
        elif volatility < 0.02:
            vol_factor = 1.2
        else:
            vol_factor = 1.0
        
        ev *= vol_factor
        
        # ✅ FIX #6: Confidence adjustment with safe division
        recommended_size = getattr(prob_score, 'recommended_size_pct', 50.0)
        if np.isnan(recommended_size) or recommended_size <= 0:
            recommended_size = 50.0
        
        recommended_size = float(np.clip(recommended_size, 1.0, 100.0))
        confidence_factor = 0.8 + (recommended_size / 100) * 0.4
        ev *= confidence_factor
        
        # ✅ FIX #7: Final validation and clipping
        if np.isnan(ev) or np.isinf(ev):
            logger.error(f"❌ Final EV is invalid after adjustments, returning 0.0")
            return 0.0
        
        ev = float(np.clip(ev, -0.50, 0.50))  # -50% to +50%
        
        # ✅ FIX #8: Log if EV is suspiciously extreme
        if abs(ev) > 0.30:
            logger.debug(f"⚠️ Extreme EV detected: {ev*100:.2f}% (leverage: {leverage:.1f}x, win_prob: {win_prob:.2f})")
        
        return ev


class IntegratedTradingBotV6Leveraged:
    """V6 LEVERAGED GRAND MASTER - 3x to 10x perpetuals"""    
    """
    FIXED __init__ for IntegratedTradingBotV6Leveraged
    Replace the existing __init__ method with this
    """

    def __init__(self, enable_parallel_evolution: bool = True):
        self.ASSETS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
        
        initial_balance = float(os.getenv('INITIAL_BALANCE', '10.0'))

        # ✅ FIX: Create organized directory structure
        self._create_directory_structure()
        
        # ✅ FIX #1: Use EnhancedCentralPortfolio (not CentralPortfolio)
        self.central_portfolio = EnhancedCentralPortfolio(initial_capital=initial_balance)
        
        # ✅ FIX #2: Initialize price_history BEFORE supervisor
        self.price_history = {asset: deque(maxlen=200) for asset in self.ASSETS}
        
        # Initialize supervisor
        self.supervisor = RLTradingSupervisor(
            initial_balance=initial_balance,
            assets=self.ASSETS
        )
        
        self.meta_rl_v5 = MetaRLSupervisorV5(initial_balance)
        self.brain = HedgeFundBrain(api_key=os.getenv('DEEPSEEK_API_KEY'))
        
        self.exchange = EnhancedHyperliquidExchange(
            wallet_address=os.getenv('HYPERLIQUID_WALLET_ADDRESS'),
            api_wallet_private_key=os.getenv('HYPERLIQUID_API_PRIVATE_KEY'),
            testnet=os.getenv('HYPERLIQUID_TESTNET', 'false').lower() == 'true'
        )
        
        self.decision_engine = None
        self.active_positions = {}
        self.last_trade_time = {}
        self.meta_rl_trained = False
        
        self.evolutionary_system: Optional[LiveEvolutionaryLeveragedTrading] = None
        self.enable_parallel_evolution = enable_parallel_evolution
        self.evolution_running = False
        
        logger.info("🚀 INTEGRATED TRADING BOT V6 - LEVERAGED GRAND MASTER")
        logger.info(f"   Leverage Range: 3x - 10x")
        logger.info(f"   Assets: {', '.join(self.ASSETS)}")
        logger.info(f"   Initial Balance: ${initial_balance:.2f}")
        logger.info(f"   Central Portfolio: ${self.central_portfolio.total_capital:.2f} total capital")

    # ADD THIS NEW METHOD to IntegratedTradingBotV6Leveraged class
    def _create_directory_structure(self):
        """Create organized directory structure for logs and data"""
        import os
        
        directories = [
            'data/elite_agents',
            'data/meta_rl_states',
            'logs/trading',
            'logs/evolution',
            'logs/performance'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("📁 Directory structure created")
    
    async def initialize_meta_rl_v5_leveraged(self):
        """Initialize Meta-RL V5 with LEVERAGED evolutionary training"""
        logger.info("\n🎓 META-RL V5 LEVERAGED EVOLUTIONARY TRAINING...")
        logger.info("   Using REAL Hyperliquid data + 3x-10x leverage")
        
        try:
            meta_rl_v5, insights, evolutionary_system = await bootstrap_meta_rl_with_leveraged_evolution(
                hyperliquid_exchange=self.exchange,
                enable_parallel=self.enable_parallel_evolution
            )
            
            self.meta_rl_v5 = meta_rl_v5
            self.evolutionary_system = evolutionary_system
            self.meta_rl_trained = True
            
            logger.info("✅ Meta-RL V5 LEVERAGED Training Completed!")
            logger.info(f"   Total Trades: {insights['total_trades_generated']}")
            
        except Exception as e:
            logger.error(f"❌ LEVERAGED training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # ✅ FIX: Pass price_history to Decision Engine
        self.decision_engine = LeveragedDecisionEngine(
            self.meta_rl_v5,
            self.brain,
            self.supervisor,
            price_history=self.price_history  # ✅ ADD THIS
        )
        logger.info("✅ Leveraged Decision Engine Initialized")
        return True
    
    async def fetch_market_data(self, asset: str) -> Optional[MarketSnapshot]:
        """
        ✅ FIXED: Proper volatility calculation (raw std dev, not annualized)
        
        CRITICAL FIXES:
        1. Removed sqrt(365) annualization
        2. Better outlier filtering (10% max moves)
        3. Ensure price_history is updated BEFORE calculations
        4. Extended window to 200 samples for stability
        """
        
        # ✅ Ensure price_history exists
        if not hasattr(self, 'price_history'):
            self.price_history = {asset: deque(maxlen=200) for asset in self.ASSETS}
        
        if asset not in self.price_history:
            self.price_history[asset] = deque(maxlen=200)
        
        try:
            data = self.exchange.get_complete_market_data(asset)
            
            if not data['orderbook']:
                return None
            
            ob = data['orderbook']
            funding = data.get('funding')
            oi = data.get('open_interest')
            liq = data.get('liquidations')
            
            price = ob.mid_price
            
            # ✅ UPDATE PRICE HISTORY IMMEDIATELY
            self.price_history[asset].append({
                'timestamp': datetime.now(),
                'price': price
            })
            
            # Historical prices (estimates)
            funding_1h = funding.funding_1h_avg if funding else 0.0
            funding_24h = funding.funding_24h_avg if funding else 0.0
            
            price_1h_ago = price * (1 - funding_1h)
            price_24h_ago = price * (1 - funding_24h)
            price_7d_ago = price * 0.97
            
            # Volume estimates
            volume_24h = abs(oi.oi_change_24h * oi.open_interest_usd / 100) if oi else 1_000_000
            volume_1h = volume_24h / 24
            volume_avg_7d = volume_24h * 0.85
            
            # ✅ FIX: Proper volatility calculation (NO annualization)
            volatility_1h = 0.02  # Default
            volatility_24h = 0.02
            
            try:
                price_history = list(self.price_history[asset])
                
                if len(price_history) >= 20:  # Need minimum 20 samples
                    # Calculate returns
                    returns = []
                    for i in range(1, len(price_history)):
                        prev_price = price_history[i-1]['price']
                        curr_price = price_history[i]['price']
                        
                        if prev_price > 0:
                            ret = (curr_price - prev_price) / prev_price
                            # ✅ Filter outliers (10% max move between samples)
                            if abs(ret) < 0.10:
                                returns.append(ret)
                    
                    if len(returns) >= 10:
                        # ✅ RAW std dev (NO sqrt(365) annualization!)
                        volatility_1h = float(np.std(returns))
                        
                        # ✅ Clip to realistic range: 0.1% to 50%
                        volatility_1h = np.clip(volatility_1h, 0.001, 0.50)
                        
                        # Scale to 24h (slightly higher)
                        volatility_24h = volatility_1h * np.sqrt(24)  # Only scale by time, not annualize
                        volatility_24h = np.clip(volatility_24h, 0.005, 0.60)
                        
                        logger.debug(f"📊 {asset}: 1h vol = {volatility_1h:.3%}, 24h vol = {volatility_24h:.3%} from {len(returns)} returns")
                        
            except Exception as e:
                logger.warning(f"⚠️ Volatility calculation failed for {asset}: {e}")
                volatility_1h = 0.02
                volatility_24h = 0.03
            
            # ATR estimate
            atr = price * volatility_24h
            
            # RSI from funding (rough proxy)
            rsi_14 = 50.0 + (funding.funding_rate * 1000 if funding else 0)
            rsi_14 = np.clip(rsi_14, 20, 80)
            rsi_7 = rsi_14
            
            # MACD from funding
            macd = funding.funding_rate * 10 if funding else 0.0
            macd_signal = funding.funding_8h_avg * 10 if funding else 0.0
            
            # EMAs
            ema_9 = price * 0.998
            ema_20 = price * 0.995
            ema_50 = price * 0.99
            ema_200 = price * 0.95
            
            # Bollinger bands
            bollinger_middle = price
            band_width = price * volatility_24h * 2
            bollinger_upper = price + band_width
            bollinger_lower = price - band_width
            
            # Whale data
            whale_transactions_1h = len(liq.major_long_liqui_levels + liq.major_short_liqui_levels) if liq else 0
            whale_net_flow = (liq.long_liquidations_24h - liq.short_liquidations_24h) if liq else 0
            
            # Fear & Greed
            fear_greed_index = 50 + (ob.imbalance_ratio - 1) * 50
            fear_greed_index = np.clip(fear_greed_index, 0, 100)
            
            # Liquidation levels
            liquidation_levels = liq.major_long_liqui_levels[:3] if liq and liq.major_long_liqui_levels else []
            
            return MarketSnapshot(
                timestamp=datetime.now(),
                price=price,
                price_1h_ago=price_1h_ago,
                price_24h_ago=price_24h_ago,
                price_7d_ago=price_7d_ago,
                volume_1h=volume_1h,
                volume_24h=volume_24h,
                volume_avg_7d=volume_avg_7d,
                volatility_1h=volatility_1h,  # ✅ Raw, not annualized
                volatility_24h=volatility_24h,  # ✅ Raw, not annualized
                atr=atr,
                bid_ask_spread=ob.best_ask - ob.best_bid,
                order_book_depth=ob.bid_liquidity_usd + ob.ask_liquidity_usd,
                rsi_14=rsi_14,
                rsi_7=rsi_7,
                macd=macd,
                macd_signal=macd_signal,
                ema_9=ema_9,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                bollinger_upper=bollinger_upper,
                bollinger_middle=bollinger_middle,
                bollinger_lower=bollinger_lower,
                whale_transactions_1h=whale_transactions_1h,
                whale_net_flow=whale_net_flow,
                exchange_netflow=0.0,
                fear_greed_index=fear_greed_index,
                news_sentiment=0.0,
                social_volume=5000.0,
                funding_rate=funding.funding_rate if funding else 0.0,
                btc_correlation=0.85 if asset != 'BTC' else 1.0,
                eth_correlation=0.78 if asset != 'ETH' else 1.0,
                buy_sell_ratio=ob.imbalance_ratio,
                liquidation_levels=liquidation_levels
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch {asset} data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def execute_leveraged_trade(self, asset: str, decision: Dict):
        """Execute leveraged trade with portfolio management"""
        try:
            leverage = decision['leverage']
            requested_size = decision['position_size']
            
            # ✅ 1. CHECK PORTFOLIO AVAILABILITY
            available_capital = self.central_portfolio.get_available_capital()
            max_trade_size = available_capital * 0.3  # 30% of available
            
            if requested_size > max_trade_size:
                logger.warning(f"⚠️ Reducing trade size: ${requested_size:.2f} → ${max_trade_size:.2f}")
                base_size = max_trade_size
            else:
                base_size = requested_size
            
            # ✅ 2. CREATE UNIQUE AGENT ID FOR THIS TRADE
            agent_id = f"{asset}_{self.central_portfolio.trade_counter}"
            
            # ✅ 3. ALLOCATE CAPITAL FROM CENTRAL PORTFOLIO
            if not self.central_portfolio.allocate_trade_capital(agent_id, base_size):
                logger.error(f"❌ Insufficient capital for {asset}: ${base_size:.2f} > ${available_capital:.2f}")
                return
            
            # ✅ 4. CALCULATE EFFECTIVE SIZE WITH LEVERAGE
            effective_size = base_size * leverage
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🎯 EXECUTING LEVERAGED TRADE (PORTFOLIO MANAGED)")
            logger.info(f"{'='*70}")
            logger.info(f"Asset: {asset}")
            logger.info(f"Action: {decision['action']}")
            logger.info(f"Base Size: ${base_size:.2f} (${requested_size:.2f} requested)")
            logger.info(f"Leverage: {leverage:.1f}x")
            logger.info(f"Effective Size: ${effective_size:.2f}")
            logger.info(f"Portfolio Usage: {base_size/self.central_portfolio.total_capital*100:.1f}%")
            logger.info(f"Available Capital: ${self.central_portfolio.get_available_capital():.2f}")
            logger.info(f"{'='*70}\n")
            
            is_buy = decision['action'] == 'BUY'
            
            # ✅ 5. EXECUTE WITH PROPER SIZE
            result = await self.exchange.place_order(
                symbol=asset,
                is_buy=is_buy,
                size=base_size,  # Now properly sized
                price=decision['entry_price']
            )
            
            if result.success:
                self.active_positions[asset] = {
                    'entry_price': result.entry_price,
                    'size': base_size,
                    'leverage': leverage,
                    'effective_size': effective_size,
                    'side': 'LONG' if is_buy else 'SHORT',
                    'stop_loss': decision['stop_loss'],
                    'take_profit': decision['take_profit'],
                    'liquidation_price': decision['liquidation_price'],
                    'entry_time': datetime.now(),
                    'decision_metadata': decision,
                    'timeframe': decision['timeframe'],
                    'max_holding_hours': decision['max_holding_hours'],
                    # ✅ 6. TRACK ALLOCATED CAPITAL AND AGENT ID
                    'allocated_capital': base_size,
                    'agent_id': agent_id
                }
                
                logger.info(f"✅ Leveraged trade executed (${base_size:.2f} allocated)")
            else:
                # ✅ 7. RELEASE CAPITAL ON FAILURE
                self.central_portfolio.release_trade_capital(agent_id)
                logger.error(f"❌ Trade failed: {result.error}")
        
        except Exception as e:
            # ✅ 8. RELEASE CAPITAL ON ERROR
            if 'agent_id' in locals():
                self.central_portfolio.release_trade_capital(agent_id)
            logger.error(f"Execute error: {e}")
    
    async def manage_leveraged_position(self, asset: str):
        """
        ✅ FIXED: Proper win tracking + clear logging
        REPLACES: Original manage_leveraged_position() in IntegratedTradingBotV6Leveraged
        """
        if asset not in self.active_positions:
            return
        
        position = self.active_positions[asset]
        
        try:
            data = self.exchange.get_complete_market_data(asset)
            if not data['orderbook']:
                return
            
            current_price = data['orderbook'].mid_price
            entry_price = position['entry_price']
            side = position['side']
            leverage = position['leverage']
            
            # Calculate P&L with leverage
            if side == 'LONG':
                price_move_pct = (current_price - entry_price) / entry_price
                leverage_pnl_pct = price_move_pct * leverage * 100
            else:  # SHORT
                price_move_pct = (entry_price - current_price) / entry_price
                leverage_pnl_pct = price_move_pct * leverage * 100
            
            pnl_usd = position['size'] * (leverage_pnl_pct / 100)
            
            should_close = False
            close_reason = ""
            
            # Liquidation check
            if side == 'LONG' and current_price <= position['liquidation_price']:
                should_close = True
                close_reason = "LIQUIDATED"
                pnl_usd = -position['size'] * 0.9
                leverage_pnl_pct = -90.0
                self.decision_engine.liquidation_count += 1
            
            # Stop loss check
            elif (side == 'LONG' and current_price <= position['stop_loss']) or \
                 (side == 'SHORT' and current_price >= position['stop_loss']):
                should_close = True
                close_reason = f"Stop loss ({leverage_pnl_pct:.2f}%)"
            
            # Take profit check
            elif (side == 'LONG' and current_price >= position['take_profit']) or \
                 (side == 'SHORT' and current_price <= position['take_profit']):
                should_close = True
                close_reason = f"Take profit ({leverage_pnl_pct:.2f}%)"
            
            # Time-based exit
            holding_time = datetime.now() - position['entry_time']
            if holding_time > timedelta(hours=position['max_holding_hours']):
                should_close = True
                close_reason = f"Max time ({leverage_pnl_pct:+.2f}%)"
            
            # Log position status
            logger.info(f"   {asset} [{leverage:.1f}x]: ${current_price:.4f} | "
                       f"P&L: {leverage_pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            
            if should_close:
                logger.info(f"\n🔴 Closing {asset}: {close_reason}")
                
                result = await self.exchange.place_order(
                    symbol=asset,
                    is_buy=(side == 'SHORT'),
                    size=position['size'],
                    price=current_price,
                    reduce_only=True
                )
                
                if result.success:
                    # ✅ Create temporary trade object for win check
                    temp_trade = type('obj', (object,), {
                        'pnl': pnl_usd,
                        'position_size': position['size']
                    })()
                    is_win = is_winning_trade(temp_trade)
                    
                    # Update central portfolio
                    agent_id = position.get('agent_id')
                    if agent_id:
                        self.central_portfolio.close_trade(agent_id, pnl_usd)
                        
                        portfolio_status = self.central_portfolio.get_portfolio_status()
                        logger.info(f"   💰 Portfolio Update: ${pnl_usd:+.2f} | "
                                   f"New Total: ${portfolio_status['total_capital']:.2f} | "
                                   f"Available: ${portfolio_status['available_capital']:.2f}")
                    else:
                        logger.error(f"⚠️ {asset} missing agent_id, cannot update portfolio!")
                    
                    # Record trade outcome in MetaRL
                    decision_meta = position.get('decision_metadata', {})
                    timeframe = position.get('timeframe', 'mid_term')
                    
                    # ✅ FIX: Pass complete parameters for RL training
                    parameters_used = {
                        'leverage': leverage,
                        'min_confidence': decision_meta.get('confidence', 50.0),
                        'stop_loss_distance': abs(position['entry_price'] - position['stop_loss']) / position['entry_price'],
                        'take_profit_distance': abs(position['take_profit'] - position['entry_price']) / position['entry_price'],
                        'position_size_base': position['size'] / self.central_portfolio.total_capital,
                        'max_holding_hours': position.get('max_holding_hours', 12.0),
                        'volatility_z_threshold': 2.5,  # From decision
                        'expected_value_threshold': 0.015,
                        'aggression': 0.8 if timeframe == 'short_term' else 0.6,
                        'patience': 0.2 if timeframe == 'short_term' else 0.5
                    }

                    self.meta_rl_v5.record_trade_outcome(
                        traded=True,
                        action=side,
                        market_regime=decision_meta.get('market_regime', 'ranging'),
                        pnl=pnl_usd,
                        pnl_pct=leverage_pnl_pct,
                        success=is_win,
                        confidence_used=decision_meta.get('confidence', 0),
                        win_prob_used=0.5,
                        parameters_used=parameters_used,  # ✅ COMPLETE PARAMS
                        timeframe=timeframe
                    )
                    
                    # Update timeframe performance stats
                    if timeframe in self.decision_engine.timeframe_performance:
                        stats = self.decision_engine.timeframe_performance[timeframe]
                        stats['trades'] += 1
                        stats['total_pnl'] += pnl_usd
                        stats['avg_leverage'] = (stats['avg_leverage'] * (stats['trades']-1) + leverage) / stats['trades']
                        if is_win:
                            stats['wins'] += 1
                    
                    # ✅ FIX: Clear win/loss logging
                    if is_win:
                        logger.info(f"✅ {asset} closed [{leverage:.1f}x] | WIN: ${pnl_usd:+.2f} ({leverage_pnl_pct:+.2f}%)")
                    elif pnl_usd < -0.01:
                        logger.info(f"❌ {asset} closed [{leverage:.1f}x] | LOSS: ${pnl_usd:+.2f} ({leverage_pnl_pct:+.2f}%)")
                    else:
                        logger.info(f"➖ {asset} closed [{leverage:.1f}x] | TIE: ${pnl_usd:+.2f} ({leverage_pnl_pct:+.2f}%)")
                    
                    # Remove from active positions
                    del self.active_positions[asset]
                else:
                    logger.error(f"❌ Failed to close {asset}: {result.error}")
            
            # Update position tracking for open positions
            else:
                position['current_price'] = current_price
                position['current_pnl'] = pnl_usd
                position['current_pnl_pct'] = leverage_pnl_pct
                
                # Liquidation warning
                if side == 'LONG':
                    liquidation_buffer_pct = ((current_price - position['liquidation_price']) / current_price) * 100
                    if liquidation_buffer_pct < 5.0:
                        logger.warning(f"⚠️ {asset} approaching liquidation: {liquidation_buffer_pct:.1f}% buffer")
                else:
                    liquidation_buffer_pct = ((position['liquidation_price'] - current_price) / current_price) * 100
                    if liquidation_buffer_pct < 5.0:
                        logger.warning(f"⚠️ {asset} approaching liquidation: {liquidation_buffer_pct:.1f}% buffer")
        
        except Exception as e:
            logger.error(f"Position management error for {asset}: {e}")
            import traceback
            traceback.print_exc()



    async def log_active_positions_status(self):
        """Log current status of all active positions with capital allocation"""
        portfolio_status = self.central_portfolio.get_portfolio_status()
        
        if not self.active_positions:
            logger.info(f"📊 No active positions | Available: ${portfolio_status['available_capital']:.2f} / ${portfolio_status['total_capital']:.2f}")
            return
        
        logger.info("\n📊 ACTIVE POSITIONS STATUS:")
        logger.info("-" * 80)
        
        total_allocated = 0.0
        total_unrealized_pnl = 0.0
        
        for asset, position in self.active_positions.items():
            try:
                data = self.exchange.get_complete_market_data(asset)
                if not data['orderbook']:
                    continue
                
                current_price = data['orderbook'].mid_price
                entry_price = position['entry_price']
                side = position['side']
                leverage = position['leverage']
                allocated = position.get('allocated_capital', 0.0)
                
                # Calculate current P&L
                if side == 'LONG':
                    pnl_pct = ((current_price - entry_price) / entry_price) * leverage * 100
                else:
                    pnl_pct = ((entry_price - current_price) / entry_price) * leverage * 100
                
                pnl_usd = position['size'] * (pnl_pct / 100)
                
                # Calculate liquidation buffer
                if side == 'LONG':
                    liq_buffer = ((current_price - position['liquidation_price']) / current_price) * 100
                else:
                    liq_buffer = ((position['liquidation_price'] - current_price) / current_price) * 100
                
                holding_time = datetime.now() - position['entry_time']
                hours_held = holding_time.total_seconds() / 3600
                
                logger.info(f"   {asset} [{leverage:.1f}x {side}]: "
                           f"${current_price:.4f} | "
                           f"P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f}) | "
                           f"Capital: ${allocated:.2f} | "
                           f"Liq Buffer: {liq_buffer:.1f}% | "
                           f"Time: {hours_held:.1f}h")
                
                total_allocated += allocated
                total_unrealized_pnl += pnl_usd
            
            except Exception as e:
                logger.error(f"   {asset}: Error getting status - {e}")
        
        logger.info("-" * 80)
        logger.info(f"📊 TOTALS: ${total_allocated:.2f} allocated | "
                    f"${total_unrealized_pnl:+.2f} unrealized P&L | "
                    f"${portfolio_status['available_capital']:.2f} available")
        
        # ✅ Show total portfolio value (capital + unrealized P&L)
        total_portfolio_value = portfolio_status['total_capital'] + total_unrealized_pnl
        logger.info(f"💎 TOTAL PORTFOLIO VALUE: ${total_portfolio_value:.2f} "
                    f"(Capital: ${portfolio_status['total_capital']:.2f} + Unrealized: ${total_unrealized_pnl:+.2f})")
        logger.info("-" * 80)


    async def log_detailed_status_report(self):
        """Comprehensive status report - UPDATED VERSION"""
        logger.info("\n" + "🌟" * 40)
        logger.info("📊 COMPREHENSIVE STATUS REPORT")
        logger.info("🌟" * 40)
        
        # 1. Portfolio integrity check
        integrity_ok = await self.verify_portfolio_integrity()
        if not integrity_ok:
            logger.error("⚠️ PORTFOLIO INTEGRITY ISSUES DETECTED!")
        
        # 2. Active positions status
        await self.log_active_positions_status()
        
        # 3. Performance summary
        await self.log_performance_summary()
        
        # 4. Evolutionary system status (if available)
        if self.evolutionary_system:
            evo_status = self.evolutionary_system.get_live_status()
            logger.info("\n🧬 EVOLUTIONARY SYSTEM:")
            logger.info(f"   Generation: {evo_status['generation']}")
            logger.info(f"   Total Trades: {evo_status['total_trades']}")
            logger.info(f"   Active Trades: {evo_status['active_trades']}")
            
            for timeframe, stats in evo_status['timeframe_stats'].items():
                if stats['total_trades'] > 0:
                    logger.info(f"   {timeframe.upper()}: {stats['total_trades']} trades, "
                               f"{stats['win_rate']:.1f}% WR, ${stats['total_pnl']:+.2f}")
        
        # 5. Decision engine stats
        if self.decision_engine:
            logger.info("\n🧠 DECISION ENGINE:")
            logger.info(f"   Liquidations: {self.decision_engine.liquidation_count}")
            if self.decision_engine.leverage_history:
                avg_leverage = np.mean(list(self.decision_engine.leverage_history))
                logger.info(f"   Avg Leverage Used: {avg_leverage:.1f}x")
        
        logger.info("🌟" * 40)


    # FIX #1: Add missing maintenance method to IntegratedTradingBotV6Leveraged class
    async def _perform_maintenance(self):
        """Periodic system maintenance"""
        logger.info("\n🔧 PERFORMING SYSTEM MAINTENANCE...")
        
        # 1. Verify portfolio integrity
        integrity_ok = await self.verify_portfolio_integrity()
        if not integrity_ok:
            logger.warning("⚠️ Portfolio rebalanced during maintenance")
        
        # 2. Clean up stale price history
        if hasattr(self, 'price_history'):
            cutoff_time = datetime.now() - timedelta(hours=48)
            for asset in self.ASSETS:
                if asset in self.price_history:
                    before = len(self.price_history[asset])
                    self.price_history[asset] = deque(
                        [p for p in self.price_history[asset] if p['timestamp'] > cutoff_time],
                        maxlen=200
                    )
                    after = len(self.price_history[asset])
                    if before > after:
                        logger.info(f"   🗑️ Cleaned {before - after} old price points for {asset}")
        
        # 3. Get memory usage stats from evolutionary system
        if self.evolutionary_system:
            memory_stats = self.evolutionary_system.get_memory_usage_stats()
            logger.info(f"   💾 Memory: {memory_stats['total_price_points']} price points, "
                    f"{memory_stats['memory_estimate_mb']:.2f} MB")
        
        # 4. Save elite agents
        if self.decision_engine and hasattr(self.decision_engine, 'agent_registry'):
            # Save top performers from evolutionary system
            if self.evolutionary_system and len(self.evolutionary_system.agents) > 0:
                top_agents = sorted(
                    self.evolutionary_system.agents,
                    key=lambda a: a.dna.fitness_score,
                    reverse=True
                )[:10]
                
                saved_count = 0
                for agent in top_agents:
                    if agent.dna.fitness_score > 50.0 and agent.dna.total_trades >= 10:
                        regime = 'ranging'  # Default, could be more sophisticated
                        market_conditions = {
                            'timestamp': datetime.now().isoformat(),
                            'fitness': agent.dna.fitness_score,
                            'trades': agent.dna.total_trades
                        }
                        
                        success = self.decision_engine.agent_registry.save_elite_agent(
                            agent, regime, market_conditions
                        )
                        if success:
                            saved_count += 1
                
                if saved_count > 0:
                    logger.info(f"   💾 Saved {saved_count} elite agents to registry")
        
        # 5. Check for orphaned positions
        orphaned = []
        for asset, position in self.active_positions.items():
            agent_id = position.get('agent_id')
            if agent_id and agent_id not in self.central_portfolio.active_trades:
                orphaned.append(asset)
        
        if orphaned:
            logger.warning(f"   ⚠️ Found {len(orphaned)} orphaned positions: {orphaned}")
            # Force close orphaned positions
            for asset in orphaned:
                logger.info(f"   🔴 Force closing orphaned position: {asset}")
                await self.manage_leveraged_position(asset)
        
        logger.info("✅ Maintenance completed")
    
    async def log_performance_summary(self):
        """
        ✅ FIXED: Show CentralPortfolio balance + accurate win rate
        REPLACES: Original log_performance_summary() in IntegratedTradingBotV6Leveraged
        """
        logger.info("\n" + "="*80)
        logger.info("📈 LEVERAGED GRAND MASTER PERFORMANCE")
        logger.info("="*80)
        
        # ✅ FIX: Use CentralPortfolio as single source of truth
        portfolio_status = self.central_portfolio.get_portfolio_status()
        
        logger.info(f"💰 Total Capital: ${portfolio_status['total_capital']:.2f}")
        logger.info(f"💵 Available Capital: ${portfolio_status['available_capital']:.2f}")
        logger.info(f"🔒 Allocated Capital: ${portfolio_status['allocated_capital']:.2f}")
        logger.info(f"📊 Capital Utilization: {portfolio_status['utilization_pct']:.1f}%")
        logger.info(f"🎯 Active Positions: {len(self.active_positions)}")
        logger.info(f"🔢 Total Trades Executed: {self.central_portfolio.trade_counter}")
        
        # Calculate realized P&L
        initial_capital = float(os.getenv('INITIAL_BALANCE', '10.0'))
        realized_pnl = portfolio_status['total_capital'] - initial_capital
        realized_pnl_pct = (realized_pnl / initial_capital) * 100 if initial_capital > 0 else 0
        
        logger.info(f"📊 Realized P&L: ${realized_pnl:+.2f} ({realized_pnl_pct:+.2f}%)")
        
        # ✅ FIX: Show ACTUAL win rate from timeframe performance
        if self.decision_engine:
            total_trades = sum(stats['trades'] for stats in self.decision_engine.timeframe_performance.values())
            total_wins = sum(stats['wins'] for stats in self.decision_engine.timeframe_performance.values())
            
            if total_trades > 0:
                actual_win_rate = (total_wins / total_trades) * 100
                logger.info(f"🎯 Actual Win Rate: {actual_win_rate:.1f}% ({total_wins}/{total_trades} wins)")
            
            if self.decision_engine.leverage_history:
                avg_leverage = np.mean(list(self.decision_engine.leverage_history))
                logger.info(f"⚡ Avg Leverage Used: {avg_leverage:.1f}x")
            
            if self.decision_engine.liquidation_count > 0:
                logger.info(f"⚠️ Liquidations: {self.decision_engine.liquidation_count}")
            
            logger.info("\n⏰ TIMEFRAME PERFORMANCE:")
            for tf, stats in self.decision_engine.timeframe_performance.items():
                if stats['trades'] > 0:
                    win_rate = (stats['wins'] / stats['trades']) * 100
                    logger.info(
                        f"   {tf.upper():<12}: {stats['trades']:2d} trades | "
                        f"{win_rate:5.1f}% WR ({stats['wins']} wins) | "
                        f"${stats['total_pnl']:+.2f} | "
                        f"{stats['avg_leverage']:.1f}x avg leverage"
                    )
        
        logger.info("="*80)

    async def verify_portfolio_integrity(self):
        """FIXED: Proper portfolio verification"""
        try:
            portfolio_status = self.central_portfolio.get_portfolio_status()
            
            # Calculate expected values from active positions
            expected_allocated = sum(
                pos.get('allocated_capital', pos.get('size', 0.0)) 
                for pos in self.active_positions.values()
            )
            
            actual_allocated = portfolio_status['allocated_capital']
            actual_available = portfolio_status['available_capital']
            actual_total = portfolio_status['total_capital']
            
            # Allow for small rounding errors
            tolerance = 0.01
            
            allocated_ok = abs(expected_allocated - actual_allocated) <= tolerance
            total_ok = abs((actual_allocated + actual_available) - actual_total) <= tolerance
            
            if not allocated_ok or not total_ok:
                logger.warning(f"🔧 Portfolio needs rebalancing:")
                logger.warning(f"   Expected allocated: ${expected_allocated:.2f}")
                logger.warning(f"   Actual allocated: ${actual_allocated:.2f}")
                logger.warning(f"   Available: ${actual_available:.2f}")
                logger.warning(f"   Total: ${actual_total:.2f}")
                
                # Auto-repair
                self.central_portfolio.available_capital = actual_total - expected_allocated
                self.central_portfolio.total_capital = actual_total
                
                logger.info("✅ Portfolio auto-rebalanced")
                return False
            
            logger.debug(f"✅ Portfolio integrity verified")
            return True
            
        except Exception as e:
            logger.error(f"Portfolio verification error: {e}")
            return False
    
    async def start_continuous_evolution(self):
        """Start continuous evolutionary learning in background"""
        if not self.evolutionary_system:
            logger.warning("⚠️ No evolutionary system initialized")
            return
        
        self.evolution_running = True
        logger.info("🧬 CONTINUOUS LEVERAGED EVOLUTION STARTED")
        
        while self.evolution_running:
            try:
                await self.evolutionary_system.run_evolution_cycle(cycles_per_generation=20)
                
                training_data = self.evolutionary_system.export_training_data_for_meta_rl()
                new_trades = [t for t in training_data if t not in getattr(self, '_fed_trades', [])]
                
                for trade in new_trades:
                    self.meta_rl_v5.record_trade_outcome(
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
                
                if not hasattr(self, '_fed_trades'):
                    self._fed_trades = []
                self._fed_trades.extend(new_trades)
                
                if new_trades:
                    logger.info(f"🔄 Fed {len(new_trades)} new LEVERAGED trades to Meta-RL")
                
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ Continuous evolution error: {e}")
                await asyncio.sleep(60)
    
    def stop_continuous_evolution(self):
        """Stop background evolution"""
        self.evolution_running = False
        logger.info("🛑 CONTINUOUS EVOLUTION STOPPED")
        

    async def run_trading_cycle(self):
        """
        ✅ ENHANCED: Add portfolio integrity verification
        REPLACES: Original run_trading_cycle() in IntegratedTradingBotV6Leveraged
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔄 LEVERAGED TRADING CYCLE")
        logger.info("=" * 80)
        
        # ✅ FIX: Use CentralPortfolio balance as single source of truth
        portfolio_status = self.central_portfolio.get_portfolio_status()
        
        portfolio = {
            "balance": portfolio_status['total_capital'],
            "available": portfolio_status['available_capital'],
            "positions": len(self.active_positions),
            "total_value": portfolio_status['total_capital']
        }
        
        logger.info(f"💰 Portfolio: ${portfolio['balance']:.2f} total | "
                    f"${portfolio['available']:.2f} available | "
                    f"{portfolio['positions']} active")
        
        # ✅ ADD: Verify portfolio integrity BEFORE trading
        integrity_ok = await self.verify_portfolio_integrity()
        if not integrity_ok:
            logger.error("⚠️ Portfolio integrity issues detected - skipping trading cycle")
            return
        
        # Phase 1: Analyze and execute trades
        for asset in self.ASSETS:
            logger.info(f"\n📊 ANALYZING {asset}...")
            
            market_data = await self.fetch_market_data(asset)
            if not market_data:
                continue
            
            decision = await self.decision_engine.make_leveraged_decision(
                asset, market_data, portfolio
            )
            
            if decision.get("action") in ["BUY", "SELL"] and decision.get("gate_status") == "ALL_PASSED":
                if asset in self.active_positions:
                    logger.info(f"   ⚠️ {asset} already has active position, skipping")
                    continue
                
                await self.execute_leveraged_trade(asset, decision)
            
            # Manage existing positions
            if asset in self.active_positions:
                await self.manage_leveraged_position(asset)
        
        # ✅ ADD: Verify portfolio integrity AFTER trading
        await self.verify_portfolio_integrity()
        
        # Phase 2: Comprehensive logging
        logger.info("\n" + "📈" * 30)
        logger.info("CYCLE COMPLETED - STATUS REPORT")
        logger.info("📈" * 30)
        
        # Show current positions first
        await self.log_active_positions_status()
        
        # Then show overall performance
        await self.log_performance_summary()
        
        # Evolutionary system status
        if self.evolutionary_system:
            status = self.evolutionary_system.get_live_status()
            logger.info("\n🧬 EVOLUTIONARY STATUS:")
            logger.info(f"   Generation: {status['generation']}")
            logger.info(f"   Avg Leverage: {status['avg_active_leverage']:.1f}x")
            logger.info(f"   Active Trades: {status['active_trades']}")
        
        logger.info("=" * 80)

    
    async def run(self):
        """Main loop with proper cleanup"""
        logger.info("🚀 LEVERAGED GRAND MASTER STARTING...")
        
        evolution_task = None
        
        try:
            # Initialization
            success = await self.initialize_meta_rl_v5_leveraged()
            if not success:
                logger.error("❌ Initialization failed")
                return
            
            # Start background evolution
            if self.evolutionary_system and self.enable_parallel_evolution:
                evolution_task = asyncio.create_task(self.start_continuous_evolution())
            
            # Main loop
            cycle_count = 0
            while True:
                try:
                    cycle_count += 1
                    await self.run_trading_cycle()
                    
                    # Periodic maintenance
                    if cycle_count % 10 == 0:
                        await self._perform_maintenance()
                    
                    await asyncio.sleep(300)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Stopped by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(60)
        
        finally:
            # ✅ FIX: Proper cleanup
            logger.info("\n🧹 CLEANING UP...")
            
            # Stop evolution
            if evolution_task:
                self.stop_continuous_evolution()
                evolution_task.cancel()
                try:
                    await evolution_task
                except asyncio.CancelledError:
                    pass
            
            # Close all positions
            logger.info("   Closing all active positions...")
            for asset in list(self.active_positions.keys()):
                try:
                    await self.manage_leveraged_position(asset)
                except Exception as e:
                    logger.error(f"   ❌ Failed to close {asset}: {e}")
            
            # Final report
            await self.log_detailed_status_report()
            
            logger.info("✅ Cleanup completed")



async def main():
    """Main entry point"""
    bot = IntegratedTradingBotV6Leveraged(enable_parallel_evolution=True)

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())