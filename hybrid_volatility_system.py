# hybrid_volatility_system.py
"""
Hybrid Volatility Forecasting System
Combines GARCH (short-term) + TTM (medium-term) for 3-stage classification

3 Volatility Stages:
- LOW: <20% annualized (stable markets, long-term agents)
- MID: 20-50% annualized (normal crypto vol, mid-term agents)
- HIGH: >50% annualized (extreme vol, short-term agents)
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from volatility_forecaster import ARCHGARCHForecaster
from ttm_forecaster import EnhancedTTMForecaster

logger = logging.getLogger(__name__)


@dataclass
class VolatilityPrediction:
    """Complete volatility prediction with confidence scores"""
    
    # Classification
    stage: str  # 'LOW', 'MID', 'HIGH'
    confidence: float  # 0.0 - 1.0
    
    # Numerical predictions
    garch_vol: float  # GARCH prediction (annualized)
    ttm_vol: float    # TTM prediction (annualized)
    hybrid_vol: float # Combined prediction
    
    # Forecasts
    forecast_1min: float
    forecast_5min: float
    forecast_15min: float
    forecast_1hour: float
    
    # Stage probabilities
    stage_probabilities: Dict[str, float]  # {'LOW': 0.2, 'MID': 0.5, 'HIGH': 0.3}
    
    # Metadata
    asset: str
    timestamp: datetime
    data_quality: float  # 0.0 - 1.0
    
    # Regime context
    current_regime: str  # Market regime
    predicted_regime: str  # Predicted regime
    
    # Agent recommendations
    recommended_timeframe: str  # 'short_term', 'mid_term', 'long_term'
    recommended_leverage: float  # 3.0 - 10.0


class HybridVolatilitySystem:
    """
    🎯 3-STAGE VOLATILITY PREDICTION SYSTEM
    
    Workflow:
    1. Collect real-time Hyperliquid data
    2. GARCH forecasts next 1-15 minutes
    3. TTM forecasts next 15-60 minutes
    4. Combine predictions → 3-stage classification
    5. Feed stage to agent selector
    """
    
    def __init__(self, assets: List[str]):
        self.assets = assets
        
        # Initialize forecasters
        self.garch_forecaster = ARCHGARCHForecaster(
            lookback=100,
            warmup_period=20
        )
        
        self.ttm_forecaster = EnhancedTTMForecaster(
            assets=assets,
            context_length=512,
            forecast_horizon=60  # 60 minutes ahead
        )
        
        # Volatility stage thresholds (annualized)
        self.STAGE_THRESHOLDS = {
            'LOW': 0.20,    # <20%
            'MID': 0.50,    # 20-50%
            'HIGH': 0.50    # >50%
        }
        
        # Prediction history for accuracy tracking
        self.prediction_history = deque(maxlen=1000)
        self.accuracy_metrics = {
            'garch': {'correct': 0, 'total': 0},
            'ttm': {'correct': 0, 'total': 0},
            'hybrid': {'correct': 0, 'total': 0}
        }
        
        # Data collection
        self.hyperliquid_data = {}  # Store raw Hyperliquid feeds
        
        logger.info("🎯 HYBRID VOLATILITY SYSTEM INITIALIZED")
        logger.info(f"   Assets: {', '.join(assets)}")
        logger.info(f"   GARCH: Real-time short-term forecasting")
        logger.info(f"   TTM: Zero-shot medium-term forecasting")
        logger.info(f"   Stages: LOW (<20%), MID (20-50%), HIGH (>50%)")
    
    async def update_from_hyperliquid(self, 
                                      asset: str, 
                                      market_data: Dict):
        """
        Update system with real-time Hyperliquid data
        
        Args:
            asset: Asset symbol
            market_data: Complete market snapshot from Hyperliquid
        """
        
        price = market_data['price']
        timestamp = market_data.get('timestamp', datetime.now())
        
        # Store raw data
        if asset not in self.hyperliquid_data:
            self.hyperliquid_data[asset] = deque(maxlen=1000)
        
        self.hyperliquid_data[asset].append({
            'price': price,
            'timestamp': timestamp,
            'volume': market_data.get('volume_24h', 0),
            'funding_rate': market_data.get('funding_rate', 0),
            'bid_ask_spread': market_data.get('bid_ask_spread', 0)
        })
        
        # Update GARCH
        self.garch_forecaster.update(price)
        
        # Update TTM
        self.ttm_forecaster.update_price(asset, price, timestamp)
    
    async def predict_volatility(self, 
                                 asset: str,
                                 current_regime: str = 'ranging') -> Optional[VolatilityPrediction]:
        """
        🎯 MAIN PREDICTION METHOD
        
        Returns complete volatility prediction with 3-stage classification
        """
        
        # Check data availability
        if asset not in self.hyperliquid_data or len(self.hyperliquid_data[asset]) < 100:
            logger.warning(f"⏳ Insufficient data for {asset}")
            return None
        
        # 1. GARCH Forecast (1-15 minutes)
        garch_vol_current = self.garch_forecaster.get_current_volatility()
        garch_vol_5min = self.garch_forecaster.forecast_volatility(horizon=5)
        garch_vol_15min = self.garch_forecaster.forecast_volatility(horizon=15)
        
        # 2. TTM Forecast (15-60 minutes)
        ttm_forecast = self.ttm_forecaster.forecast_volatility(asset, horizon_minutes=60)
        
        if not ttm_forecast:
            logger.warning(f"⚠️ TTM forecast failed for {asset}")
            return None
        
        ttm_vol = ttm_forecast['predicted_vol']
        
        # 3. Hybrid Combination (weighted by time horizon)
        # Short-term (1-5 min): 80% GARCH, 20% TTM
        # Medium-term (15-30 min): 50% GARCH, 50% TTM
        # Long-term (30-60 min): 20% GARCH, 80% TTM
        
        forecast_1min = garch_vol_current
        forecast_5min = 0.8 * garch_vol_5min + 0.2 * ttm_vol
        forecast_15min = 0.5 * garch_vol_15min + 0.5 * ttm_vol
        forecast_1hour = 0.2 * garch_vol_15min + 0.8 * ttm_vol
        
        # 4. Overall hybrid volatility (time-weighted average)
        hybrid_vol = (
            0.3 * forecast_5min +   # 30% weight on 5-min
            0.4 * forecast_15min +  # 40% weight on 15-min
            0.3 * forecast_1hour    # 30% weight on 1-hour
        )
        
        # 5. Classify into 3 stages
        stage, confidence, stage_probs = self._classify_stage(hybrid_vol, ttm_forecast)
        
        # 6. Agent recommendations based on stage
        recommended_timeframe, recommended_leverage = self._get_agent_recommendations(
            stage, hybrid_vol, current_regime
        )
        
        # 7. Predict regime change
        predicted_regime = self._predict_regime_change(
            asset, hybrid_vol, current_regime
        )
        
        # 8. Data quality score
        data_quality = self._calculate_data_quality(asset)
        
        # Create prediction object
        prediction = VolatilityPrediction(
            stage=stage,
            confidence=confidence,
            garch_vol=garch_vol_current,
            ttm_vol=ttm_vol,
            hybrid_vol=hybrid_vol,
            forecast_1min=forecast_1min,
            forecast_5min=forecast_5min,
            forecast_15min=forecast_15min,
            forecast_1hour=forecast_1hour,
            stage_probabilities=stage_probs,
            asset=asset,
            timestamp=datetime.now(),
            data_quality=data_quality,
            current_regime=current_regime,
            predicted_regime=predicted_regime,
            recommended_timeframe=recommended_timeframe,
            recommended_leverage=recommended_leverage
        )
        
        # Store for accuracy tracking
        self.prediction_history.append({
            'prediction': prediction,
            'actual_vol': None  # Will be filled later
        })
        
        return prediction
    
    def _classify_stage(self, 
                       hybrid_vol: float,
                       ttm_forecast: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify volatility into 3 stages with confidence
        
        Returns:
            (stage, confidence, stage_probabilities)
        """
        
        LOW_THRESHOLD = self.STAGE_THRESHOLDS['LOW']
        HIGH_THRESHOLD = self.STAGE_THRESHOLDS['HIGH']
        
        # Base classification
        if hybrid_vol < LOW_THRESHOLD:
            stage = 'LOW'
            # Confidence: how far below threshold
            confidence = 1.0 - (hybrid_vol / LOW_THRESHOLD)
            confidence = min(confidence, 0.95)  # Cap at 95%
            
        elif hybrid_vol < HIGH_THRESHOLD:
            stage = 'MID'
            # MID has lower base confidence (transition zone)
            distance_from_center = abs(hybrid_vol - (LOW_THRESHOLD + HIGH_THRESHOLD) / 2)
            max_distance = (HIGH_THRESHOLD - LOW_THRESHOLD) / 2
            confidence = 1.0 - (distance_from_center / max_distance)
            confidence = np.clip(confidence, 0.5, 0.85)
            
        else:  # HIGH
            stage = 'HIGH'
            # Confidence: how far above threshold
            confidence = min(1.0, (hybrid_vol - HIGH_THRESHOLD) / HIGH_THRESHOLD + 0.5)
            confidence = np.clip(confidence, 0.6, 0.95)
        
        # Adjust confidence based on TTM agreement
        ttm_stage = ttm_forecast['regime']
        if ttm_stage == stage:
            confidence *= 1.1  # 10% boost for agreement
        else:
            confidence *= 0.85  # 15% penalty for disagreement
        
        confidence = np.clip(confidence, 0.3, 0.95)
        
        # Calculate stage probabilities using TTM + GARCH
        ttm_probs = ttm_forecast['regime_probabilities']
        
        # Weighted combination
        stage_probs = {}
        for s in ['LOW', 'MID', 'HIGH']:
            if s == stage:
                stage_probs[s] = confidence
            else:
                stage_probs[s] = ttm_probs.get(s, 0.0) * (1 - confidence)
        
        # Normalize
        total = sum(stage_probs.values())
        stage_probs = {k: v/total for k, v in stage_probs.items()}
        
        return stage, float(confidence), stage_probs
    
    def _get_agent_recommendations(self, 
                                  stage: str,
                                  volatility: float,
                                  regime: str) -> Tuple[str, float]:
        """
        Recommend timeframe and leverage based on volatility stage
        
        Returns:
            (timeframe, leverage)
        """
        
        if stage == 'LOW':
            # Low volatility → Long-term positions, lower leverage
            timeframe = 'long_term'
            base_leverage = 3.5
            
        elif stage == 'MID':
            # Normal volatility → Mid-term positions, medium leverage
            timeframe = 'mid_term'
            base_leverage = 5.0
            
        else:  # HIGH
            # High volatility → Short-term scalping, higher leverage (but tight stops!)
            timeframe = 'short_term'
            base_leverage = 7.0
        
        # Adjust leverage based on regime
        regime_adjustments = {
            'bull_strong': 1.2,
            'bull_weak': 1.0,
            'bear_strong': 0.8,
            'bear_weak': 0.9,
            'ranging': 0.9,
            'high_volatility': 0.6,
            'crash': 0.5
        }
        
        leverage = base_leverage * regime_adjustments.get(regime, 1.0)
        leverage = np.clip(leverage, 3.0, 10.0)
        
        return timeframe, float(leverage)
    
    def _predict_regime_change(self, 
                              asset: str,
                              predicted_vol: float,
                              current_regime: str) -> str:
        """
        Predict if regime will change based on volatility forecast
        """
        
        # Simple heuristic: HIGH vol often precedes regime change
        if predicted_vol > 0.60:  # Extreme volatility
            if current_regime in ['bull_strong', 'bull_weak']:
                return 'high_volatility'  # Possible top
            elif current_regime in ['bear_strong', 'bear_weak']:
                return 'crash'  # Possible capitulation
        
        return current_regime  # No change predicted
    
    def _calculate_data_quality(self, asset: str) -> float:
        """
        Calculate quality score for available data
        
        Higher score = more reliable predictions
        """
        
        if asset not in self.hyperliquid_data:
            return 0.0
        
        data_points = len(self.hyperliquid_data[asset])
        
        # Scoring
        if data_points < 100:
            score = 0.3
        elif data_points < 512:
            score = 0.5 + (data_points - 100) / 824  # Linear 0.5 → 1.0
        else:
            score = 1.0
        
        # Adjust for data freshness
        latest = self.hyperliquid_data[asset][-1]
        age_seconds = (datetime.now() - latest['timestamp']).total_seconds()
        
        if age_seconds > 300:  # >5 minutes old
            score *= 0.7
        elif age_seconds > 60:  # >1 minute old
            score *= 0.9
        
        return float(np.clip(score, 0.0, 1.0))
    
    async def validate_prediction_accuracy(self, 
                                          asset: str,
                                          actual_volatility: float):
        """
        Validate predictions against actual realized volatility
        
        This is CRITICAL for continuous improvement
        """
        
        # Find predictions from 1 hour ago
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        for record in self.prediction_history:
            pred = record['prediction']
            
            # Check if this is the right asset and time
            if (pred.asset == asset and 
                record['actual_vol'] is None and
                (datetime.now() - pred.timestamp).total_seconds() >= 3600):
                
                # Store actual volatility
                record['actual_vol'] = actual_volatility
                
                # Calculate errors
                garch_error = abs(pred.garch_vol - actual_volatility)
                ttm_error = abs(pred.ttm_vol - actual_volatility)
                hybrid_error = abs(pred.hybrid_vol - actual_volatility)
                
                # Update accuracy metrics
                threshold = 0.10  # 10% error tolerance
                
                if garch_error < threshold:
                    self.accuracy_metrics['garch']['correct'] += 1
                self.accuracy_metrics['garch']['total'] += 1
                
                if ttm_error < threshold:
                    self.accuracy_metrics['ttm']['correct'] += 1
                self.accuracy_metrics['ttm']['total'] += 1
                
                if hybrid_error < threshold:
                    self.accuracy_metrics['hybrid']['correct'] += 1
                self.accuracy_metrics['hybrid']['total'] += 1
                
                logger.debug(f"✅ Validated prediction for {asset}:")
                logger.debug(f"   Predicted: {pred.hybrid_vol:.2%}")
                logger.debug(f"   Actual: {actual_volatility:.2%}")
                logger.debug(f"   Error: {hybrid_error:.2%}")
    
    def get_accuracy_report(self) -> Dict:
        """
        Get accuracy report for all forecasters
        """
        
        report = {}
        
        for name, metrics in self.accuracy_metrics.items():
            if metrics['total'] > 0:
                accuracy = metrics['correct'] / metrics['total']
                report[name] = {
                    'accuracy': accuracy,
                    'correct': metrics['correct'],
                    'total': metrics['total']
                }
            else:
                report[name] = {'accuracy': 0.0, 'correct': 0, 'total': 0}
        
        return report
    
    async def predict_all_assets(self, 
                                current_regimes: Dict[str, str]) -> Dict[str, VolatilityPrediction]:
        """
        Predict volatility for all assets simultaneously
        
        Args:
            current_regimes: {'BTC': 'bull_strong', 'ETH': 'ranging', ...}
            
        Returns:
            {'BTC': VolatilityPrediction, 'ETH': VolatilityPrediction, ...}
        """
        
        predictions = {}
        
        for asset in self.assets:
            regime = current_regimes.get(asset, 'ranging')
            
            try:
                prediction = await self.predict_volatility(asset, regime)
                if prediction:
                    predictions[asset] = prediction
            except Exception as e:
                logger.error(f"❌ Prediction failed for {asset}: {e}")
        
        return predictions


# ===========================
# AGENT SELECTOR
# ===========================

class VolatilityBasedAgentSelector:
    """
    🎯 SELECT BEST AGENTS BASED ON VOLATILITY STAGE
    
    Workflow:
    1. Get volatility prediction (LOW/MID/HIGH)
    2. Filter agents by recommended timeframe
    3. Find agents with best performance in this volatility stage
    4. Return top N agents for trading
    """
    
    def __init__(self, 
                 agent_registry,
                 evolutionary_system):
        self.registry = agent_registry
        self.evolutionary_system = evolutionary_system
        
        # Track agent performance by volatility stage
        self.agent_performance_by_stage = {
            'LOW': {},   # agent_id -> metrics
            'MID': {},
            'HIGH': {}
        }
        
        logger.info("🎯 VOLATILITY-BASED AGENT SELECTOR INITIALIZED")
    
    def select_agents_for_stage(self, 
                                prediction: VolatilityPrediction,
                                top_n: int = 5) -> List[Dict]:
        """
        Select best agents for current volatility stage
        
        Args:
            prediction: VolatilityPrediction object
            top_n: Number of agents to return
            
        Returns:
            List of agent dicts sorted by fitness for this stage
        """
        
        stage = prediction.stage
        timeframe = prediction.recommended_timeframe
        
        logger.info(f"🔍 Selecting agents for {stage} volatility ({timeframe})")
        
        # Get all agents for this timeframe
        timeframe_agents = [
            a for a in self.evolutionary_system.agents
            if a.dna.timeframe == timeframe
        ]
        
        # Filter by volatility stage performance
        candidates = []
        
        for agent in timeframe_agents:
            # Get agent's historical performance in this stage
            perf = self.agent_performance_by_stage[stage].get(
                agent.dna.agent_id,
                {'trades': 0, 'wins': 0, 'pnl': 0.0, 'fitness': 0.0}
            )
            
            # Scoring: combine overall fitness + stage-specific performance
            if perf['trades'] >= 5:  # Has experience in this stage
                # 70% stage-specific, 30% overall fitness
                stage_win_rate = perf['wins'] / perf['trades']
                score = 0.7 * stage_win_rate * 100 + 0.3 * agent.dna.fitness_score
            else:
                # No stage experience, use overall fitness
                score = agent.dna.fitness_score * 0.5  # Penalty for no experience
            
            candidates.append({
                'agent': agent,
                'score': score,
                'stage_experience': perf['trades'],
                'stage_win_rate': perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            })
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top N
        selected = candidates[:top_n]
        
        logger.info(f"✅ Selected {len(selected)} agents:")
        for i, cand in enumerate(selected, 1):
            logger.info(f"   #{i} Agent {cand['agent'].dna.agent_id}: "
                       f"Score={cand['score']:.1f}, "
                       f"Stage XP={cand['stage_experience']} trades")
        
        return selected
    
    def update_agent_performance(self, 
                                agent_id: int,
                                volatility_stage: str,
                                trade_result: Dict):
        """
        Update agent's performance record for specific volatility stage
        
        Called after each trade closes
        """
        
        if volatility_stage not in self.agent_performance_by_stage:
            logger.warning(f"Unknown stage: {volatility_stage}")
            return
        
        if agent_id not in self.agent_performance_by_stage[volatility_stage]:
            self.agent_performance_by_stage[volatility_stage][agent_id] = {
                'trades': 0,
                'wins': 0,
                'pnl': 0.0,
                'fitness': 0.0
            }
        
        perf = self.agent_performance_by_stage[volatility_stage][agent_id]
        perf['trades'] += 1
        if trade_result['success']:
            perf['wins'] += 1
        perf['pnl'] += trade_result['pnl']
        
        # Update fitness
        if perf['trades'] > 0:
            win_rate = perf['wins'] / perf['trades']
            perf['fitness'] = win_rate * 100 + (perf['pnl'] / 1000) * 10
    
    def get_stage_statistics(self) -> Dict:
        """Get performance statistics by volatility stage"""
        
        stats = {}
        
        for stage, agents in self.agent_performance_by_stage.items():
            if not agents:
                stats[stage] = {'agents': 0, 'total_trades': 0}
                continue
            
            total_trades = sum(a['trades'] for a in agents.values())
            total_wins = sum(a['wins'] for a in agents.values())
            avg_win_rate = total_wins / total_trades if total_trades > 0 else 0
            
            stats[stage] = {
                'agents': len(agents),
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'total_pnl': sum(a['pnl'] for a in agents.values())
            }
        
        return stats


# ===========================
# TESTING & VALIDATION
# ===========================

async def test_hybrid_system_with_hyperliquid():
    """
    🧪 TEST HYBRID SYSTEM WITH REAL HYPERLIQUID DATA
    
    This collects real data and validates prediction accuracy
    """
    
    from fixed_hyperliquid import EnhancedHyperliquidExchange
    import os
    
    logger.info("🧪 TESTING HYBRID VOLATILITY SYSTEM")
    logger.info("="*80)
    
    # Initialize exchange
    exchange = EnhancedHyperliquidExchange(
        wallet_address=os.getenv('HYPERLIQUID_WALLET_ADDRESS'),
        api_wallet_private_key=os.getenv('HYPERLIQUID_API_PRIVATE_KEY'),
        testnet=True  # Use testnet for testing
    )
    
    # Initialize hybrid system
    assets = ['BTC', 'ETH', 'SOL']
    hybrid_system = HybridVolatilitySystem(assets)
    
    logger.info(f"\n📊 Collecting data for {len(assets)} assets...")
    logger.info("   Phase 1: Collect 600 data points (10 hours at 1-min frequency)")
    logger.info("   Phase 2: Make predictions")
    logger.info("   Phase 3: Validate accuracy\n")
    
    # Phase 1: Data collection
    for i in range(600):
        logger.info(f"📡 Collecting data point {i+1}/600...")
        
        for asset in assets:
            try:
                # Fetch real market data
                data = exchange.get_complete_market_data(asset)
                
                if data['orderbook']:
                    market_snapshot = {
                        'price': data['orderbook'].mid_price,
                        'timestamp': datetime.now(),
                        'volume_24h': data.get('funding', {}).volume_24h if data.get('funding') else 0,
                        'funding_rate': data.get('funding', {}).funding_rate if data.get('funding') else 0,
                        'bid_ask_spread': data['orderbook'].spread_bps / 10000
                    }
                    
                    # Update hybrid system
                    await hybrid_system.update_from_hyperliquid(asset, market_snapshot)
                    
                    if (i+1) % 100 == 0:
                        logger.info(f"   {asset}: ${market_snapshot['price']:.2f}")
            
            except Exception as e:
                logger.error(f"   Error fetching {asset}: {e}")
        
        # Wait 1 minute between data points
        if i < 599:  # Don't wait after last point
            await asyncio.sleep(60)
    
    # Phase 2: Make predictions
    logger.info("\n📊 DATA COLLECTION COMPLETE")
    logger.info("Making predictions...\n")
    
    current_regimes = {asset: 'ranging' for asset in assets}
    predictions = await hybrid_system.predict_all_assets(current_regimes)
    
    for asset, prediction in predictions.items():
        logger.info(f"✅ {asset} PREDICTION:")
        logger.info(f"   Stage: {prediction.stage} ({prediction.confidence:.0%} confidence)")
        logger.info(f"   Hybrid Vol: {prediction.hybrid_vol:.2%}")
        logger.info(f"   GARCH: {prediction.garch_vol:.2%}")
        logger.info(f"   TTM: {prediction.ttm_vol:.2%}")
        logger.info(f"   Forecasts:")
        logger.info(f"     1-min: {prediction.forecast_1min:.2%}")
        logger.info(f"     5-min: {prediction.forecast_5min:.2%}")
        logger.info(f"     15-min: {prediction.forecast_15min:.2%}")
        logger.info(f"     1-hour: {prediction.forecast_1hour:.2%}")
        logger.info(f"   Recommended: {prediction.recommended_timeframe} agents, "
                   f"{prediction.recommended_leverage:.1f}x leverage")
        logger.info(f"   Data Quality: {prediction.data_quality:.0%}\n")
    
    # Phase 3: Wait 1 hour and validate
    logger.info("⏳ Waiting 1 hour to validate predictions...")
    await asyncio.sleep(3600)
    
    logger.info("\n🔬 VALIDATING PREDICTIONS...")
    
    for asset in assets:
        # Calculate actual realized volatility over past hour
        if asset in hybrid_system.hyperliquid_data:
            recent_prices = [d['price'] for d in list(hybrid_system.hyperliquid_data[asset])[-60:]]
            returns = np.diff(np.log(recent_prices))
            actual_vol = np.std(returns) * np.sqrt(252 * 24 * 60)
            
            await hybrid_system.validate_prediction_accuracy(asset, actual_vol)
    
    # Get accuracy report
    accuracy_report = hybrid_system.get_accuracy_report()
    
    logger.info("\n📊 ACCURACY REPORT:")
    logger.info("="*80)
    for model, metrics in accuracy_report.items():
        logger.info(f"{model.upper()}:")
        logger.info(f"   Accuracy: {metrics['accuracy']:.1%}")
        logger.info(f"   Correct: {metrics['correct']}/{metrics['total']}")
    
    logger.info("\n✅ TEST COMPLETE")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(test_hybrid_system_with_hyperliquid())