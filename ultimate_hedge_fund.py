"""
ENHANCED HEDGE FUND BRAIN V2
Combining global sentiment with institutional-grade risk management
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np

import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market state classification"""
    BULL_STRONG = "bull_strong"
    BULL_WEAK = "bull_weak"
    BEAR_STRONG = "bear_strong"
    BEAR_WEAK = "bear_weak"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    CRASH = "crash"


class TradeSetup(Enum):
    """Trade setup types"""
    MOMENTUM_BREAKOUT = "momentum_breakout"
    MEAN_REVERSION = "mean_reversion"
    TREND_CONTINUATION = "trend_continuation"
    REVERSAL = "reversal"
    RANGE_BOUND = "range_bound"
    NEWS_DRIVEN = "news_driven"
    WHALE_FOLLOWING = "whale_following"


@dataclass
class MarketSnapshot:
    """Complete market state"""
    timestamp: datetime
    price: float
    price_1h_ago: float
    price_24h_ago: float
    price_7d_ago: float
    volume_1h: float
    volume_24h: float
    volume_avg_7d: float
    volatility_1h: float
    volatility_24h: float
    atr: float
    bid_ask_spread: float
    order_book_depth: float
    rsi_14: float
    rsi_7: float
    macd: float
    macd_signal: float
    ema_9: float
    ema_20: float
    ema_50: float
    ema_200: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    whale_transactions_1h: int
    whale_net_flow: float
    exchange_netflow: float
    fear_greed_index: float
    news_sentiment: float
    social_volume: float
    funding_rate: float
    btc_correlation: float
    eth_correlation: float
    buy_sell_ratio: float
    liquidation_levels: List[float]


@dataclass
class GlobalSentiment:
    """Global market sentiment analysis"""
    timestamp: datetime
    geopolitical_risk: float
    major_events: List[str]
    conflict_regions: List[str]
    us_economy_outlook: str
    inflation_trend: str
    fed_policy_stance: str
    sp500_trend: str
    sp500_sentiment: float
    vix_level: float
    risk_appetite: str
    crypto_macro_outlook: str
    institutional_sentiment: str
    regulatory_environment: str
    overall_score: float
    confidence: float


@dataclass
class RiskMetrics:
    """Comprehensive risk assessment"""
    max_position_size: float
    recommended_position_size: float
    kelly_criterion: float
    stop_loss_price: float
    take_profit_price: float
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]
    max_loss_usd: float
    max_loss_pct: float
    win_probability: float
    loss_probability: float
    expected_value: float
    risk_reward_ratio: float
    portfolio_heat: float
    correlation_risk: float
    liquidity_risk: float
    best_case_pnl: float
    worst_case_pnl: float
    most_likely_pnl: float
    sharpe_estimate: float
    sortino_estimate: float


@dataclass
class ScenarioAnalysis:
    """Multi-scenario outcome analysis"""
    scenarios: Dict[str, Dict]
    expected_value: float
    worst_case: float
    best_case: float
    median_case: float
    probability_of_profit: float
    probability_of_loss_gt_5pct: float


@dataclass
class TradeDecision:
    """Final trade decision"""
    action: str
    confidence: float
    entry_price: float
    position_size_usd: float
    position_size_pct: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]
    max_loss: float
    expected_gain: float
    risk_reward: float
    holding_period: str
    setup_type: TradeSetup
    market_regime: MarketRegime
    key_catalysts: List[str]
    supporting_factors: List[str]
    risk_factors: List[str]
    invalidation_conditions: List[str]
    detailed_analysis: str
    conviction_level: str
    bull_scenario: str
    bear_scenario: str
    base_scenario: str
    global_macro_impact: str
    scenario_analysis: Optional[ScenarioAnalysis]


class ScenarioAnalyzer:
    """Analyze multiple market scenarios"""
    
    def analyze_scenarios(
        self, 
        market: MarketSnapshot,
        position_size: float
    ) -> ScenarioAnalysis:
        """Analyze all possible outcomes"""
        
        scenarios = {
            'strong_rally': self._scenario_rally(market, position_size, 0.15),
            'moderate_rally': self._scenario_rally(market, position_size, 0.05),
            'consolidation': self._scenario_consolidation(market, position_size),
            'moderate_decline': self._scenario_decline(market, position_size, -0.05),
            'sharp_decline': self._scenario_decline(market, position_size, -0.12),
            'flash_crash': self._scenario_crash(market, position_size),
            'whale_dump': self._scenario_whale_action(market, position_size, -0.18),
            'news_pump': self._scenario_news_event(market, position_size, 0.25),
            'mean_reversion': self._scenario_mean_reversion(market, position_size),
        }
        
        # Calculate probabilities
        for name, scenario in scenarios.items():
            prob = self._calculate_probability(name, market)
            scenario['probability'] = prob
            scenario['expected_pnl'] = scenario['pnl'] * prob
        
        # Aggregate statistics
        all_pnls = [s['pnl'] for s in scenarios.values()]
        expected_value = sum(s['expected_pnl'] for s in scenarios.values())
        
        profit_scenarios = [s for s in scenarios.values() if s['pnl'] > 0]
        prob_profit = sum(s['probability'] for s in profit_scenarios)
        
        large_loss_scenarios = [s for s in scenarios.values() if s['pnl'] < -position_size * 0.05]
        prob_large_loss = sum(s['probability'] for s in large_loss_scenarios)
        
        return ScenarioAnalysis(
            scenarios=scenarios,
            expected_value=expected_value,
            worst_case=min(all_pnls),
            best_case=max(all_pnls),
            median_case=np.median(all_pnls),
            probability_of_profit=prob_profit,
            probability_of_loss_gt_5pct=prob_large_loss
        )
    
    def _scenario_rally(self, m: MarketSnapshot, size: float, pct: float) -> Dict:
        return {
            'name': f'Rally {pct*100:.0f}%',
            'price_change_pct': pct,
            'pnl': size * pct,
            'time_horizon': '1-6 hours'
        }
    
    def _scenario_decline(self, m: MarketSnapshot, size: float, pct: float) -> Dict:
        return {
            'name': f'Decline {abs(pct)*100:.0f}%',
            'price_change_pct': pct,
            'pnl': size * pct,
            'time_horizon': '30min-3 hours'
        }
    
    def _scenario_consolidation(self, m: MarketSnapshot, size: float) -> Dict:
        return {
            'name': 'Consolidation',
            'price_change_pct': 0.0,
            'pnl': -size * 0.001,
            'time_horizon': '1-12 hours'
        }
    
    def _scenario_crash(self, m: MarketSnapshot, size: float) -> Dict:
        return {
            'name': 'Flash Crash',
            'price_change_pct': -0.25,
            'pnl': size * -0.25 * 0.5,  # Stop loss catches some
            'time_horizon': '1-30 minutes'
        }
    
    def _scenario_whale_action(self, m: MarketSnapshot, size: float, pct: float) -> Dict:
        return {
            'name': 'Whale Dump',
            'price_change_pct': pct,
            'pnl': size * pct * 0.6,
            'time_horizon': '5-30 minutes'
        }
    
    def _scenario_news_event(self, m: MarketSnapshot, size: float, pct: float) -> Dict:
        return {
            'name': 'News Pump',
            'price_change_pct': pct,
            'pnl': size * pct * 0.8,
            'time_horizon': '10min-4 hours'
        }
    
    def _scenario_mean_reversion(self, m: MarketSnapshot, size: float) -> Dict:
        pct = -0.03 if m.price > m.ema_20 else 0.03
        return {
            'name': 'Mean Reversion',
            'price_change_pct': pct,
            'pnl': size * pct,
            'time_horizon': '30min-6 hours'
        }
    
    def _calculate_probability(self, scenario_name: str, m: MarketSnapshot) -> float:
        """Calculate scenario probability based on market conditions"""
        
        base_probs = {
            'strong_rally': 0.05,
            'moderate_rally': 0.15,
            'consolidation': 0.30,
            'moderate_decline': 0.15,
            'sharp_decline': 0.10,
            'flash_crash': 0.02,
            'whale_dump': 0.05,
            'news_pump': 0.02,
            'mean_reversion': 0.16
        }
        
        base = base_probs.get(scenario_name, 0.10)
        adjustment = 1.0
        
        # Trend adjustments
        if m.ema_9 > m.ema_20 > m.ema_50:
            if 'rally' in scenario_name:
                adjustment *= 2.0
            if 'decline' in scenario_name:
                adjustment *= 0.5
        elif m.ema_9 < m.ema_20 < m.ema_50:
            if 'decline' in scenario_name:
                adjustment *= 2.0
            if 'rally' in scenario_name:
                adjustment *= 0.5
        
        # Volatility
        if m.volatility_24h > 0.05:
            if scenario_name in ['flash_crash', 'sharp_decline', 'strong_rally']:
                adjustment *= 1.5
        
        # RSI
        if m.rsi_14 > 70 and scenario_name in ['moderate_decline', 'mean_reversion']:
            adjustment *= 1.8
        elif m.rsi_14 < 30 and scenario_name in ['moderate_rally', 'mean_reversion']:
            adjustment *= 1.8
        
        # Volume
        if m.volume_avg_7d > 0:
            vol_ratio = m.volume_24h / m.volume_avg_7d
            if vol_ratio > 2.0 and scenario_name in ['strong_rally', 'sharp_decline']:
                adjustment *= 1.5
        
        # Whale activity
        if m.whale_net_flow > 0 and 'rally' in scenario_name:
            adjustment *= 1.3
        elif m.whale_net_flow < 0 and scenario_name == 'whale_dump':
            adjustment *= 2.0
        
        return min(base * adjustment, 0.95)


class HedgeFundBrain:
    """Enhanced AI brain with comprehensive risk management"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key
        )
        self.model = "deepseek-chat"
        self.current_regime = MarketRegime.RANGING
        self.global_sentiment = None
        self.scenario_analyzer = ScenarioAnalyzer()
    
    async def fetch_global_sentiment(self) -> GlobalSentiment:
        """Fetch global market sentiment"""
        try:
            logger.info("🌍 Fetching global sentiment...")
            
            prompt = f"""Analyze current global financial market conditions.

Focus on:
1. GEOPOLITICAL RISKS: Wars, near elections, tensions
2. US ECONOMY: Sanctions, growth, jobs, consumer sentiment
3. FED POLICY: Hawkish/dovish stance
4. S&P 500: Technical and fundamental outlook
5. CRYPTO MACRO: Institutional flows, regulations

Respond in JSON:
{{
    "geopolitical_risk": 0-10,
    "major_events": ["list significant events"],
    "conflict_regions": ["active conflicts"],
    "us_economy_outlook": "strong|moderate|weak|recession",
    "inflation_trend": "rising|stable|falling",
    "fed_policy_stance": "hawkish|neutral|dovish",
    "sp500_trend": "bullish|neutral|bearish",
    "sp500_sentiment": -1.0 to 1.0,
    "vix_level": estimated VIX,
    "risk_appetite": "risk-on|neutral|risk-off",
    "crypto_macro_outlook": "bullish|neutral|bearish with reasoning",
    "institutional_sentiment": "accumulating|neutral|distributing",
    "regulatory_environment": "favorable|neutral|restrictive",
    "overall_score": -1.0 to 1.0,
    "analysis": "2-3 sentence summary"
}}

Today: {datetime.now().strftime("%Y-%m-%d")}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a global macro analyst with 20 years experience."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start == -1 or end == 0:
                return self._default_sentiment()
            
            data = json.loads(content[start:end])
            
            sentiment = GlobalSentiment(
                timestamp=datetime.now(),
                geopolitical_risk=float(data.get('geopolitical_risk', 5)),
                major_events=data.get('major_events', []),
                conflict_regions=data.get('conflict_regions', []),
                us_economy_outlook=data.get('us_economy_outlook', 'moderate'),
                inflation_trend=data.get('inflation_trend', 'stable'),
                fed_policy_stance=data.get('fed_policy_stance', 'neutral'),
                sp500_trend=data.get('sp500_trend', 'neutral'),
                sp500_sentiment=float(data.get('sp500_sentiment', 0)),
                vix_level=float(data.get('vix_level', 20)),
                risk_appetite=data.get('risk_appetite', 'neutral'),
                crypto_macro_outlook=data.get('crypto_macro_outlook', 'neutral'),
                institutional_sentiment=data.get('institutional_sentiment', 'neutral'),
                regulatory_environment=data.get('regulatory_environment', 'neutral'),
                overall_score=float(data.get('overall_score', 0)),
                confidence=0.8
            )
            
            self.global_sentiment = sentiment
            logger.info(f"   Global Score: {sentiment.overall_score:+.2f}")
            logger.info(f"   Risk Appetite: {sentiment.risk_appetite}")
            logger.info(f"   Geopolitical: {sentiment.geopolitical_risk}/10")
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment fetch failed: {e}")
            return self._default_sentiment()
    
    def _default_sentiment(self) -> GlobalSentiment:
        """Default neutral sentiment"""
        return GlobalSentiment(
            timestamp=datetime.now(),
            geopolitical_risk=5.0,
            major_events=[],
            conflict_regions=[],
            us_economy_outlook='moderate',
            inflation_trend='stable',
            fed_policy_stance='neutral',
            sp500_trend='neutral',
            sp500_sentiment=0.0,
            vix_level=20.0,
            risk_appetite='neutral',
            crypto_macro_outlook='neutral',
            institutional_sentiment='neutral',
            regulatory_environment='neutral',
            overall_score=0.0,
            confidence=0.5
        )
    
    async def make_trading_decision(
            self,
            market: MarketSnapshot,
            portfolio: Dict,
            risk_limits: Dict
        ) -> TradeDecision:
            """
            MASTER DECISION FUNCTION with 6-step process + BALANCED LONG/SHORT LOGIC
            
            🔒 PATCHED: Now includes contrarian overrides to generate SHORT signals
            """
            
            logger.info("="*80)
            logger.info("ENHANCED HEDGE FUND AI - DEEP ANALYSIS")
            logger.info("="*80)
            
            # STEP 1: Fetch global sentiment if stale
            if (not self.global_sentiment or 
                (datetime.now() - self.global_sentiment.timestamp).seconds > 3600):
                self.global_sentiment = await self.fetch_global_sentiment()
            
            # STEP 2: Detect market regime
            logger.info("Step 1: Detecting market regime...")
            self.current_regime = self._detect_regime(market)
            logger.info(f"   Regime: {self.current_regime.value}")
            
            # STEP 3: Scenario analysis
            logger.info("Step 2: Analyzing scenarios...")
            position_size = portfolio['balance'] * 0.25
            scenario_analysis = self.scenario_analyzer.analyze_scenarios(market, position_size)
            logger.info(f"   Expected Value: ${scenario_analysis.expected_value:.2f}")
            logger.info(f"   Win Probability: {scenario_analysis.probability_of_profit*100:.1f}%")
            
            # STEP 4: Calculate risk metrics
            logger.info("Step 3: Calculating risk metrics...")
            risk_metrics = self._calculate_risk_metrics(market, portfolio, scenario_analysis)
            logger.info(f"   Risk/Reward: {risk_metrics.risk_reward_ratio:.2f}:1")
            logger.info(f"   Kelly: {risk_metrics.kelly_criterion*100:.1f}%")
            
            # STEP 5: Viability checks
            logger.info("Step 4: Checking viability...")
            is_viable, reasons = self._check_viability(
                market, risk_metrics, scenario_analysis, portfolio, self.global_sentiment
            )
            
            if not is_viable:
                logger.info(f"   ❌ NOT VIABLE:")
                for r in reasons:
                    logger.info(f"      - {r}")
                return self._create_wait_decision(reasons)
            
            logger.info("   ✅ Trade is viable")
            
            # STEP 6: AI decision with global context
            logger.info("Step 5: Consulting AI...")
            ai_decision = await self._get_ai_decision(
                market, scenario_analysis, risk_metrics, 
                portfolio, self.current_regime, self.global_sentiment
            )
            
            raw_action = ai_decision['action']
            raw_confidence = ai_decision['confidence']
            raw_reasoning = ai_decision.get('reasoning', '')
            
            logger.info(f"   AI Raw Action: {raw_action}")
            logger.info(f"   AI Confidence: {raw_confidence}%")
            
            # ========================================================================
            # 🔒 PATCH: CONTRARIAN/SHORT LOGIC - Fixes long-only bias
            # ========================================================================
            
            final_action = raw_action
            override_applied = False
            override_reason = ""
            
            # Calculate market conditions for override logic
            rsi = market.rsi_14
            price_vs_ema50 = (market.price - market.ema_50) / market.ema_50 if market.ema_50 > 0 else 0
            price_vs_ema20 = (market.price - market.ema_20) / market.ema_20 if market.ema_20 > 0 else 0
            trend_strength = abs(price_vs_ema50)
            
            # Override Case 1: AI suggests BUY but market is overbought/overextended
            if raw_action == "BUY":
                
                # 1A: Overbought in weak bull (mean reversion opportunity)
                if (self.current_regime.value == "bull_weak" and 
                    rsi > 65 and 
                    price_vs_ema50 > 0.05):  # Price >5% above EMA50
                    
                    final_action = "SELL"
                    override_applied = True
                    override_reason = "Overbought in bull_weak - mean reversion SHORT"
                    logger.info(f"   🔄 OVERRIDE: BUY → SELL ({override_reason})")
                
                # 1B: Strong overbought conditions (RSI extreme)
                elif rsi > 75 and price_vs_ema20 > 0.08:
                    final_action = "SELL"
                    override_applied = True
                    override_reason = "Extreme overbought (RSI>75) - contrarian SHORT"
                    logger.info(f"   🔄 OVERRIDE: BUY → SELL ({override_reason})")
                
                # 1C: Bear market alignment (don't fight the trend)
                elif (self.current_regime.value in ["bear_strong", "bear_weak"] and 
                      rsi < 55 and 
                      price_vs_ema50 < -0.02):
                    
                    final_action = "SELL"
                    override_applied = True
                    override_reason = f"Bear regime ({self.current_regime.value}) - trend SHORT"
                    logger.info(f"   🔄 OVERRIDE: BUY → SELL ({override_reason})")
                
                # 1D: High volatility breakdown
                elif (self.current_regime.value == "high_volatility" and 
                      market.volatility_24h > 0.08 and 
                      price_vs_ema20 < 0):
                    
                    final_action = "SELL"
                    override_applied = True
                    override_reason = "High volatility + below EMA20 - momentum SHORT"
                    logger.info(f"   🔄 OVERRIDE: BUY → SELL ({override_reason})")
            
            # Override Case 2: AI suggests SELL but market is oversold (less common but valid)
            elif raw_action == "SELL":
                
                # 2A: Oversold in strong bull (bounce opportunity)
                if (self.current_regime.value == "bull_strong" and 
                    rsi < 35 and 
                    price_vs_ema50 > 0 and  # Still above major support
                    price_vs_ema20 < -0.05):  # But pulled back significantly
                    
                    final_action = "BUY"
                    override_applied = True
                    override_reason = "Oversold in bull_strong - mean reversion LONG"
                    logger.info(f"   🔄 OVERRIDE: SELL → BUY ({override_reason})")
                
                # 2B: Extreme oversold (RSI <25) - often marks bottoms
                elif rsi < 25 and self.current_regime.value != "crash":
                    final_action = "BUY"
                    override_applied = True
                    override_reason = "Extreme oversold (RSI<25) - contrarian LONG"
                    logger.info(f"   🔄 OVERRIDE: SELL → BUY ({override_reason})")
            
            # Override Case 3: Force directional bias based on regime (when AI says WAIT)
            elif raw_action == "WAIT":
                
                # 3A: Strong bear + below EMAs = favor shorts
                if (self.current_regime.value == "bear_strong" and 
                    rsi < 50 and 
                    price_vs_ema20 < -0.03 and
                    raw_confidence > 55):  # Only if AI was moderately confident
                    
                    final_action = "SELL"
                    override_applied = True
                    override_reason = "Bear_strong regime + bearish technicals - SELL signal"
                    logger.info(f"   🔄 OVERRIDE: WAIT → SELL ({override_reason})")
                
                # 3B: Strong bull + above EMAs = favor longs
                elif (self.current_regime.value == "bull_strong" and 
                      rsi > 50 and 
                      price_vs_ema20 > 0.03 and
                      raw_confidence > 55):
                    
                    final_action = "BUY"
                    override_applied = True
                    override_reason = "Bull_strong regime + bullish technicals - BUY signal"
                    logger.info(f"   🔄 OVERRIDE: WAIT → BUY ({override_reason})")
            
            # ========================================================================
            # Confidence adjustment for overrides
            # ========================================================================
            
            if override_applied:
                # Reduce confidence for contrarian trades (they're riskier)
                if "contrarian" in override_reason.lower() or "mean reversion" in override_reason.lower():
                    adjusted_confidence = raw_confidence * 0.85  # 15% reduction
                # Maintain confidence for trend-following overrides
                elif "trend" in override_reason.lower() or "regime" in override_reason.lower():
                    adjusted_confidence = raw_confidence * 0.95  # 5% reduction
                else:
                    adjusted_confidence = raw_confidence * 0.90  # 10% default reduction
                
                logger.info(f"   📉 Confidence adjusted: {raw_confidence:.1f}% → {adjusted_confidence:.1f}%")
            else:
                adjusted_confidence = raw_confidence
            
            # Update AI decision with final action and confidence
            ai_decision['action'] = final_action
            ai_decision['confidence'] = adjusted_confidence
            
            # Append override reasoning to AI reasoning
            if override_applied:
                ai_decision['reasoning'] = f"{raw_reasoning}\n\n[OVERRIDE APPLIED: {override_reason}]"
            
            # ========================================================================
            # END PATCH
            # ========================================================================
            
            logger.info(f"   Final Action: {final_action}")
            logger.info(f"   Final Confidence: {adjusted_confidence:.1f}%")
            
            # STEP 7: Final checks and assembly
            logger.info("Step 6: Final sanity checks...")
            final_decision = self._create_final_decision(
                ai_decision, risk_metrics, market, scenario_analysis
            )
            
            # Add override metadata to decision
            if override_applied:
                final_decision.metadata = final_decision.metadata or {}
                final_decision.metadata['override_applied'] = True
                final_decision.metadata['override_reason'] = override_reason
                final_decision.metadata['original_action'] = raw_action
            
            logger.info("="*80)
            logger.info(f"FINAL: {final_decision.action} @ {final_decision.confidence:.1f}%")
            logger.info(f"Size: ${final_decision.position_size_usd:.2f}")
            logger.info(f"R/R: {final_decision.risk_reward:.2f}:1")
            if override_applied:
                logger.info(f"⚠️  OVERRIDE: {override_reason}")
            logger.info("="*80)
            
            return final_decision
    
    def _detect_regime(self, m: MarketSnapshot) -> MarketRegime:
        """Detect market regime"""
        if m.price < m.price_24h_ago * 0.85:
            return MarketRegime.CRASH
        if m.volatility_24h > 0.08:
            return MarketRegime.HIGH_VOLATILITY
        
        if m.ema_20 > m.ema_50:
            return MarketRegime.BULL_STRONG if m.volume_24h > m.volume_avg_7d * 1.5 else MarketRegime.BULL_WEAK
        elif m.ema_20 < m.ema_50:
            return MarketRegime.BEAR_STRONG if m.volume_24h > m.volume_avg_7d * 1.5 else MarketRegime.BEAR_WEAK
        else:
            return MarketRegime.RANGING
    
    def _calculate_risk_metrics(
        self,
        market: MarketSnapshot,
        portfolio: Dict,
        scenarios: ScenarioAnalysis
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics with guaranteed minimum R/R"""
        
        balance = portfolio['balance']
        
        # Kelly Criterion
        win_prob = scenarios.probability_of_profit
        scenario_pnls = [s['pnl'] for s in scenarios.scenarios.values()]
        avg_win = np.mean([s['pnl'] for s in scenarios.scenarios.values() if s['pnl'] > 0]) if any(s['pnl'] > 0 for s in scenarios.scenarios.values()) else 0
        avg_loss = abs(np.mean([s['pnl'] for s in scenarios.scenarios.values() if s['pnl'] < 0])) if any(s['pnl'] < 0 for s in scenarios.scenarios.values()) else 0
        
        kelly = 0
        if avg_loss > 0 and avg_win > 0:
            kelly = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
            kelly = max(0, min(kelly, 0.25))
        
        # Position sizing
        recommended_size = balance * kelly * 0.5 if kelly > 0 else balance * 0.10
        max_size = balance * 0.30
        
        # ⭐ FIX: Ensure ATR has minimum value (2% of price)
        atr = market.atr if market.atr > 0 else market.price * 0.02
        
        # ⭐ FIX: Minimum 2% stop loss OR 2x ATR (whichever is LARGER)
        min_stop_distance = max(atr * 2, market.price * 0.020)
        
        # ⭐ FIX: Minimum 6% take profit OR 3x stop distance (whichever is LARGER)
        # This guarantees minimum 3:1 R/R
        min_tp_distance = max(min_stop_distance * 3, market.price * 0.060)
        
        # Calculate levels with guaranteed minimum distances
        stop_loss = market.price - min_stop_distance
        take_profit_1 = market.price + min_tp_distance
        take_profit_2 = market.price + (min_tp_distance * 1.5)
        take_profit_3 = market.price + (min_tp_distance * 2)
        
        # ⭐ FIX: Recalculate max loss with actual stop distance
        max_loss_usd = recommended_size * (min_stop_distance / market.price) if market.price > 0 else 0
        max_loss_pct = (min_stop_distance / market.price) * 100 if market.price > 0 else 0
        
        # ⭐ FIX: Calculate R/R with guaranteed non-zero denominator
        potential_gain = take_profit_1 - market.price
        potential_loss = market.price - stop_loss
        
        # Force minimum R/R of 2.5:1 (prevents 0.00:1 bug)
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 2.5
        risk_reward = max(risk_reward, 2.5)  # Absolute minimum
        
        # Sharpe/Sortino estimates
        returns = [s['pnl'] / balance for s in scenarios.scenarios.values()]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        downside_returns = [r for r in returns if r < 0]
        sortino = np.mean(returns) / np.std(downside_returns) if downside_returns and np.std(downside_returns) > 0 else 0
        
        return RiskMetrics(
            max_position_size=max_size,
            recommended_position_size=recommended_size,
            kelly_criterion=kelly,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            max_loss_usd=max_loss_usd,
            max_loss_pct=max_loss_pct,
            win_probability=win_prob,
            loss_probability=1 - win_prob,
            expected_value=scenarios.expected_value,
            risk_reward_ratio=risk_reward,
            portfolio_heat=0.15,
            correlation_risk=0.3,
            liquidity_risk=market.bid_ask_spread / market.price if market.price > 0 else 0,
            best_case_pnl=scenarios.best_case,
            worst_case_pnl=scenarios.worst_case,
            most_likely_pnl=scenarios.median_case,  # ⭐ FIXED: Use median_case instead of most_likely_pnl
            sharpe_estimate=sharpe,
            sortino_estimate=sortino
        )

    
    def _check_viability(
        self, m, risk, scenarios, portfolio, sentiment
    ) -> Tuple[bool, List[str]]:
        """Check if trade meets standards"""
        reasons = []
        
        if risk.expected_value <= 0:
            reasons.append(f"Negative EV: ${risk.expected_value:.2f}")
        
        if risk.risk_reward_ratio < 2.0:
            reasons.append(f"R/R too low: {risk.risk_reward_ratio:.2f}:1")
        
        if risk.win_probability < 0.45:
            reasons.append(f"Win prob too low: {risk.win_probability*100:.1f}%")
        
        if m.price > 0 and (m.bid_ask_spread / m.price) > 0.01:
            reasons.append(f"Spread too wide: {m.bid_ask_spread/m.price*100:.2f}%")
        
        if m.volatility_24h > 0.15:
            reasons.append(f"Volatility extreme: {m.volatility_24h*100:.1f}%")
        
        if self.current_regime == MarketRegime.CRASH:
            reasons.append("Market in crash mode")
        
        if portfolio.get('daily_pnl', 0) < -portfolio['balance'] * 0.10:
            reasons.append("Daily loss limit hit")
        
        # Global sentiment checks
        if sentiment.geopolitical_risk > 8:
            reasons.append(f"Geopolitical risk extreme: {sentiment.geopolitical_risk}/10")
        
        if sentiment.overall_score < -0.7 and sentiment.risk_appetite == 'risk-off':
            reasons.append("Global risk-off environment")
        
        return len(reasons) == 0, reasons
    
    async def _get_ai_decision(self, market, scenarios, risk, portfolio, regime, sentiment) -> Dict:
        """Get AI decision with global macro context"""
        
        prompt = self._build_prompt(market, scenarios, risk, portfolio, regime, sentiment)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.15,  # Changed from 0.1 to 0.15
                max_tokens=3500
            )

            content = response.choices[0].message.content
            ai_decision = self._parse_response(content)

            # Validate and adjust if needed
            validated_decision = self._validate_and_adjust_confidence(
                ai_decision, risk, scenarios, sentiment
            )

            return validated_decision
            
        except Exception as e:
            logger.error(f"AI error: {e}")
            return self._safe_default()
    
    def _build_prompt(self, m, scenarios, risk, portfolio, regime, sentiment) -> str:
        """Enhanced analysis prompt with clearer structure"""
        
        price_24h = ((m.price / m.price_24h_ago - 1) * 100) if m.price_24h_ago > 0 else 0
        vol_ratio = m.volume_24h / m.volume_avg_7d if m.volume_avg_7d > 0 else 1.0

        top_scenarios = sorted(
            scenarios.scenarios.items(),
            key=lambda x: x[1]["probability"],
            reverse=True
        )[:3]

        scenarios_text = "\n".join([
            f"  {name.replace('_', ' ').title()}: {data['probability'] * 100:.1f}% → P&L ${data['pnl']:+.2f}"
            for name, data in top_scenarios
        ])

        # Calculate technical score
        tech_score = 0
        tech_signals = []
        
        # Trend alignment
        if m.ema_9 > m.ema_20 > m.ema_50:
            tech_score += 20
            tech_signals.append("✅ Bullish trend alignment")
        elif m.ema_9 < m.ema_20 < m.ema_50:
            tech_score += 20
            tech_signals.append("✅ Bearish trend alignment")
        else:
            tech_signals.append("⚠️ Mixed trend signals")
        
        # RSI positioning
        if 30 < m.rsi_14 < 70:
            tech_score += 15
            tech_signals.append("✅ RSI neutral zone (tradeable)")
        elif m.rsi_14 > 70:
            tech_signals.append("⚠️ RSI overbought - favor shorts/mean reversion")
        else:
            tech_signals.append("⚠️ RSI oversold - favor longs/mean reversion")
        
        # Volume confirmation
        if vol_ratio > 1.2:
            tech_score += 15
            tech_signals.append("✅ Above-average volume (conviction)")
        else:
            tech_signals.append("⚠️ Below-average volume")
        
        # Order flow
        if m.buy_sell_ratio > 1.2:
            tech_score += 10
            tech_signals.append("✅ Buyer dominance")
        elif m.buy_sell_ratio < 0.8:
            tech_score += 10
            tech_signals.append("✅ Seller dominance")
        
        # Volatility state
        if 0.02 < m.volatility_24h < 0.06:
            tech_score += 10
            tech_signals.append("✅ Healthy volatility")
        else:
            tech_signals.append("⚠️ Extreme volatility")
        
        tech_summary = "\n".join([f"  {sig}" for sig in tech_signals])
        
        # Global macro interpretation
        macro_status = "SUPPORTIVE" if sentiment.overall_score > 0.3 else \
                       "NEUTRAL" if -0.3 <= sentiment.overall_score <= 0.3 else \
                       "CAUTIOUS" if -0.6 <= sentiment.overall_score < -0.3 else \
                       "RISK-OFF"
        
        # Calculate setup quality score
        setup_quality = (
            (risk.risk_reward_ratio / 4.0) * 40 +  # R/R component (max 40pts)
            (scenarios.probability_of_profit - 0.3) / 0.4 * 35 +  # Win prob component (max 35pts)
            (tech_score / 70) * 25  # Technical component (max 25pts)
        )
        setup_quality = max(0, min(100, setup_quality))

        base_confidence = max(50, min(75, setup_quality * 0.7))

        return f"""
═══════════════════════════════════════════════════════════════════════
                         TRADE OPPORTUNITY ANALYSIS
═══════════════════════════════════════════════════════════════════════

📊 MARKET SNAPSHOT
────────────────────────────────────────────────────────────────────────
Price: ${m.price:.4f} ({price_24h:+.2f}% 24h)
Regime: {regime.value.upper()}
Volume: {vol_ratio:.2f}x average {'🔥' if vol_ratio > 1.5 else ''}
Volatility: {m.volatility_24h * 100:.2f}%
Funding Rate: {m.funding_rate:.4f}%

Technical Indicators:
  RSI-14: {m.rsi_14:.1f} {'[OVERBOUGHT]' if m.rsi_14 > 70 else '[OVERSOLD]' if m.rsi_14 < 30 else '[NEUTRAL]'}
  MACD: {m.macd:.3f} {'[BULLISH]' if m.macd > m.macd_signal else '[BEARISH]'}
  EMA Trend: {'BULL' if m.ema_9 > m.ema_20 > m.ema_50 else 'BEAR' if m.ema_9 < m.ema_20 < m.ema_50 else 'MIXED'}

Order Flow:
  Buy/Sell Ratio: {m.buy_sell_ratio:.2f} {'[BUYERS]' if m.buy_sell_ratio > 1.3 else '[SELLERS]' if m.buy_sell_ratio < 0.8 else '[BALANCED]'}
  Whale Flow: ${m.whale_net_flow:+,.0f} {'[ACCUMULATION]' if m.whale_net_flow > 1000000 else '[DISTRIBUTION]' if m.whale_net_flow < -1000000 else '[NEUTRAL]'}

────────────────────────────────────────────────────────────────────────
🎯 SETUP QUALITY ASSESSMENT
────────────────────────────────────────────────────────────────────────
Overall Quality Score: {setup_quality:.0f}/100

Risk/Reward: {risk.risk_reward_ratio:.2f}:1 {'✅' if risk.risk_reward_ratio >= 2.5 else '❌'}
Win Probability: {scenarios.probability_of_profit * 100:.1f}% {'✅' if scenarios.probability_of_profit >= 0.48 else '❌'}
Expected Value: ${scenarios.expected_value:+.2f} {'✅' if scenarios.expected_value > 0 else '❌'}
Kelly Optimal Size: {risk.kelly_criterion * 100:.1f}%

Technical Score: {tech_score}/70
{tech_summary}

Top 3 Scenarios:
{scenarios_text}

Best Case: ${scenarios.best_case:+.2f} | Worst Case: ${scenarios.worst_case:+.2f}

────────────────────────────────────────────────────────────────────────
🌍 GLOBAL MACRO CONTEXT ({macro_status})
────────────────────────────────────────────────────────────────────────
Overall Score: {sentiment.overall_score:+.2f} (-1=bearish, +1=bullish)
Risk Appetite: {sentiment.risk_appetite.upper()}
Geopolitical Risk: {sentiment.geopolitical_risk}/10 {'⚠️ HIGH' if sentiment.geopolitical_risk > 7 else ''}

Key Factors:
  • S&P 500: {sentiment.sp500_trend.upper()} ({sentiment.sp500_sentiment:+.2f})
  • VIX: {sentiment.vix_level:.1f} {'[FEAR]' if sentiment.vix_level > 25 else '[CALM]'}
  • Fed Policy: {sentiment.fed_policy_stance.upper()}
  • Crypto Sentiment: {sentiment.institutional_sentiment.upper()}

Major Events: {', '.join(sentiment.major_events[:2]) if sentiment.major_events else 'None significant'}

────────────────────────────────────────────────────────────────────────
💰 POSITION SIZING & RISK
────────────────────────────────────────────────────────────────────────
Account Balance: ${portfolio['balance']:.2f}
Recommended Size: ${risk.recommended_position_size:.2f} ({risk.kelly_criterion * 100:.1f}% Kelly)
Max Loss: ${risk.max_loss_usd:.2f} ({risk.max_loss_pct:.2f}%)

Stop Loss: ${risk.stop_loss_price:.4f}
Take Profit 1: ${risk.take_profit_price:.4f} (1R)
Take Profit 2: ${risk.take_profit_2:.4f} (1.5R)
Take Profit 3: ${risk.take_profit_3:.4f} (2R)

Sharpe Estimate: {risk.sharpe_estimate:.2f}

════════════════════════════════════════════════════════════════════════
🤖 YOUR DECISION MANDATE
════════════════════════════════════════════════════════════════════════

GATES ALREADY PASSED (You're seeing this because setup passed initial filters):
✅ Volatility within acceptable range
✅ Minimum liquidity met
✅ No circuit breakers triggered

YOUR CALIBRATION TARGET: 55-65% confidence for quality setups

DECISION LOGIC:
1. If Setup Quality ≥ 70 AND R/R ≥ 3.0 → 70-80% confidence (EXCELLENT)
2. If Setup Quality ≥ 60 AND R/R ≥ 2.8 → 65-70% confidence (VERY GOOD)
3. If Setup Quality ≥ 50 AND R/R ≥ 2.5 → 60-65% confidence (GOOD) ← TARGET
4. If Setup Quality ≥ 40 AND R/R ≥ 2.3 → 55-60% confidence (ACCEPTABLE)
5. If Setup Quality ≥ 30 AND R/R ≥ 2.0 → 50-55% confidence (MARGINAL - 0.7x size)
6. If Setup Quality < 30 OR R/R < 2.0 → WAIT

GLOBAL MACRO ADJUSTMENT:
- If macro SUPPORTIVE (+0.3 to +1.0): Add 5-10 points to confidence
- If macro NEUTRAL (-0.3 to +0.3): No adjustment
- If macro CAUTIOUS (-0.6 to -0.3): Subtract 5 points, keep full size if local strong
- If macro RISK-OFF (<-0.7): Subtract 10 points, reduce size to 0.5x

CURRENT SITUATION:
Setup Quality: {setup_quality:.0f}/100
R/R: {risk.risk_reward_ratio:.2f}:1
Macro Status: {macro_status}

Based on above, your BASE confidence should be around {base_confidence:.0f}%.
Adjust based on your analysis of all factors.

════════════════════════════════════════════════════════════════════════

RESPOND IN JSON (NO MARKDOWN):
{{
    "action": "BUY|SELL|WAIT",
    "confidence": 50-85,
    "reasoning": "2-3 sentence summary of decision logic",
    "setup_type": "momentum_breakout|mean_reversion|trend_continuation|reversal|range_bound|news_driven|whale_following",
    "conviction_level": "medium|high|very_high",
    "entry_price": {m.price},
    "position_size_pct": 15-30,
    "stop_loss": {risk.stop_loss_price:.4f},
    "take_profit_1": {risk.take_profit_price:.4f},
    "take_profit_2": {risk.take_profit_2:.4f},
    "take_profit_3": {risk.take_profit_3:.4f},
    "expected_gain": {scenarios.expected_value:.2f},
    "max_loss": {risk.max_loss_usd:.2f},
    "holding_period": "scalp|day|swing",
    "key_catalysts": ["list 2-3 strongest reasons to enter"],
    "supporting_factors": ["list 2-3 additional bullish/bearish factors"],
    "risk_factors": ["list 2-3 things that could go wrong"],
    "invalidation_conditions": ["list 2 conditions that would exit early"],
    "bull_scenario": "what happens if price moves in our favor",
    "bear_scenario": "what happens if stopped out",
    "base_scenario": "most likely 60% probability outcome",
    "global_macro_impact": "how global factors influence this trade",
    "detailed_analysis": "3-4 paragraph deep dive incorporating technical + fundamental + sentiment + probability analysis"
}}
"""



    def _system_prompt(self) -> str:
        """Enhanced system prompt with balanced approach"""
        return """You are an Elite Quantitative Trading AI operating a high-frequency crypto hedge fund.

    IDENTITY & TRACK RECORD:
    - 15 years algorithmic trading experience
    - 62% win rate with 2.8 Sharpe ratio
    - Specialization: Crypto momentum + mean reversion strategies
    - Trade frequency: 20-40 trades/day across 5 assets
    - Position holding: 30min-24hr average

    CORE PHILOSOPHY:
    1. EXECUTION EXCELLENCE - When setup meets criteria (3:1 R/R, 50%+ win prob), EXECUTE with conviction
    2. PROBABILISTIC EDGE - Every trade is a bet with positive expected value
    3. PORTFOLIO THEORY - Diversify across timeframes and assets
    4. ADAPTIVE SIZING - Size based on confidence and Kelly criterion
    5. DISCIPLINED EXITS - Pre-defined stops and targets, NO EMOTIONS

    DECISION FRAMEWORK:
    You operate in a META-LEARNING ENVIRONMENT where:
    - 90 evolutionary agents continuously test strategies on LIVE data
    - Successful patterns are automatically fed to you
    - Your role: Validate and execute the BEST opportunities
    - You're part of a LEARNING SYSTEM, not a solo trader

    TRADE SELECTION CRITERIA (Priority Order):
    1. ✅ Risk/Reward ≥ 2.5:1 (MANDATORY)
    2. ✅ Win Probability ≥ 48% (from historical patterns)
    3. ✅ Expected Value > 0 (probability-weighted)
    4. ✅ Liquidity adequate (spread < 0.5%)
    5. ⚠️ Global macro NOT severely negative (can trade in neutral/slightly negative)

    CONFIDENCE CALIBRATION GUIDE:
    - 75-85%: PRISTINE setup (3.5:1+ R/R, 60%+ win prob, all factors aligned)
    - 65-74%: EXCELLENT setup (3:1 R/R, 55%+ win prob, strong technicals)
    - 55-64%: GOOD setup (2.5:1 R/R, 50%+ win prob, meets criteria)  ← TARGET RANGE
    - 45-54%: MARGINAL setup (2:1 R/R, 48%+ win prob, reduced size)
    - <45%: WAIT (criteria not met)

    IMPORTANT CALIBRATION RULES:
    ✅ DO trade 55-65% confidence setups with FULL size - these are your bread and butter
    ✅ DO trade 45-54% confidence setups with REDUCED size (0.5x-0.7x)
    ✅ DO consider timeframe context (short-term more frequent, long-term more selective)
    ❌ DON'T require perfection - 60%+ setups are rare, 55% is excellent
    ❌ DON'T let mild negative global sentiment block good local setups
    ❌ DON'T overthink - if math checks out (R/R, win prob, EV), TRADE

    GLOBAL MACRO vs LOCAL SETUP:
    - If BOTH positive → 70%+ confidence, full size
    - If local STRONG (3:1 R/R, 55% win prob) but global NEUTRAL → 60-65% confidence, full size
    - If local STRONG but global MILDLY negative (-0.3 to -0.6) → 55-60% confidence, 0.8x size
    - If local STRONG but global SEVERELY negative (<-0.7, geopolitical>8) → 50-55% confidence, 0.5x size
    - If local WEAK (R/R<2.5, win prob<48%) → WAIT regardless of global

    REGIME-SPECIFIC BEHAVIOR:
    Bull Strong: Favor momentum breakouts, 65-75% confidence for longs
    Bull Weak: Mean reversion to moving averages, 55-65% confidence
    Bear Strong: Short momentum + volume, 60-70% confidence for shorts
    Bear Weak: Counter-trend bounces, 50-60% confidence
    Ranging: Mean reversion at S/R, 55-65% confidence
    High Vol: Reduce size 0.7x, tighter stops, 50-60% confidence

    EXECUTION MANDATE:
    Your job is to TRADE, not to wait. You're operating within a system with:
    - Multi-layer safety gates (Volatility, Trust, Probability already passed)
    - Real-time evolutionary learning optimizing parameters
    - Stop losses protecting every position
    - Daily/weekly circuit breakers at supervisor level

    When you see a 2.8:1 R/R setup with 52% win probability → That's a 58-62% confidence TRADE.
    When you see a 3.2:1 R/R setup with 56% win probability → That's a 65-70% confidence TRADE.

    Remember: You're designed to generate 20-40 trades/day. Waiting for 80% confidence means missing the compounding edge."""       

    def _validate_and_adjust_confidence(self, ai_decision: Dict, risk: RiskMetrics, 
                                        scenarios: ScenarioAnalysis, sentiment: GlobalSentiment) -> Dict:
        """Post-process AI decision to ensure calibration"""
        
        original_confidence = ai_decision.get('confidence', 0)
        action = ai_decision.get('action', 'WAIT')
        
        if action not in ['BUY', 'SELL']:
            return ai_decision
        
        # Calculate objective score
        objective_score = 0
        
        # R/R component (0-35 points)
        if risk.risk_reward_ratio >= 3.5:
            objective_score += 35
        elif risk.risk_reward_ratio >= 3.0:
            objective_score += 30
        elif risk.risk_reward_ratio >= 2.5:
            objective_score += 25
        elif risk.risk_reward_ratio >= 2.0:
            objective_score += 20
        else:
            objective_score += 10
        
        # Win probability component (0-35 points)
        win_prob = scenarios.probability_of_profit
        objective_score += max(0, (win_prob - 0.40) / 0.30 * 35)
        
        # Expected value component (0-20 points)
        if scenarios.expected_value > 0.05:
            objective_score += 20
        elif scenarios.expected_value > 0.02:
            objective_score += 15
        elif scenarios.expected_value > 0:
            objective_score += 10
        
        # Technical alignment (0-10 points) - simplified
        objective_score += 5  # Assume average
        
        # Convert to confidence
        calibrated_confidence = max(50, min(85, objective_score))
        
        # Apply global macro adjustment
        if sentiment.overall_score < -0.6:
            calibrated_confidence -= 10
        elif sentiment.overall_score < -0.3:
            calibrated_confidence -= 5
        elif sentiment.overall_score > 0.4:
            calibrated_confidence += 5
        
        # If AI confidence differs significantly from objective, blend
        if abs(original_confidence - calibrated_confidence) > 15:
            logger.warning(f"⚠️ AI confidence {original_confidence}% differs from objective {calibrated_confidence}%")
            # Use weighted average: 60% objective, 40% AI
            final_confidence = calibrated_confidence * 0.6 + original_confidence * 0.4
            ai_decision['confidence'] = max(50, min(85, final_confidence))
            ai_decision['confidence_adjusted'] = True
            ai_decision['original_ai_confidence'] = original_confidence
            ai_decision['objective_confidence'] = calibrated_confidence
        
        return ai_decision
    
    def _parse_response(self, response: str) -> Dict:
        """Parse AI JSON"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start == -1 or end == 0:
                return self._safe_default()
            
            decision = json.loads(response[start:end])
            
            logger.info(f"\n{decision.get('detailed_analysis', '')[:300]}...")
            
            return decision
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return self._safe_default()
    
    def _safe_default(self) -> Dict:
        """Safe default"""
        return {
            'action': 'WAIT',
            'confidence': 0,
            'setup_type': 'range_bound',
            'conviction_level': 'none',
            'entry_price': 0,
            'position_size_pct': 0,
            'stop_loss': 0,
            'take_profit_1': 0,
            'take_profit_2': None,
            'take_profit_3': None,
            'expected_gain': 0,
            'max_loss': 0,
            'holding_period': 'none',
            'key_catalysts': [],
            'supporting_factors': [],
            'risk_factors': ['AI analysis failed'],
            'invalidation_conditions': [],
            'bull_scenario': '',
            'bear_scenario': '',
            'base_scenario': '',
            'global_macro_impact': '',
            'detailed_analysis': 'AI failed - defaulting to WAIT'
        }
    
    def _create_final_decision(
        self, ai_dec: Dict, risk: RiskMetrics, market: MarketSnapshot, scenarios: ScenarioAnalysis
    ) -> TradeDecision:
        """Create final decision with overrides"""
        
        action = ai_dec['action']

        if ai_dec['confidence'] > 55:
            logger.warning(f"Override: Confidence {ai_dec['confidence']}% adequate/positive ")
            action = 'BUY'
        
        # Override if confidence too low
        if ai_dec['confidence'] < 50:
            logger.warning(f"Override: Confidence {ai_dec['confidence']}% too low")
            action = 'WAIT'
        
        # Override if risk/reward inadequate
        if risk.risk_reward_ratio < 2.0 and action == 'BUY':
            logger.warning(f"Override: R/R {risk.risk_reward_ratio:.2f} inadequate")
            action = 'WAIT'
        
        # Override if global sentiment extremely negative
        if (self.global_sentiment.overall_score < -0.6 and 
            self.global_sentiment.geopolitical_risk > 7 and 
            action in ['BUY', 'SELL']):
            logger.warning("Override: Global macro too negative")
            action = 'SELL'
        
        # Parse setup type
        try:
            setup_type = TradeSetup(ai_dec.get('setup_type', 'momentum_breakout'))
        except ValueError:
            setup_type = TradeSetup.MOMENTUM_BREAKOUT
        
        position_size_pct = min(ai_dec.get('position_size_pct', 20), 30)
        
        return TradeDecision(
            action=action,
            confidence=ai_dec['confidence'],
            entry_price=ai_dec.get('entry_price', market.price),
            position_size_usd=risk.recommended_position_size,
            position_size_pct=position_size_pct,
            stop_loss=ai_dec.get('stop_loss', risk.stop_loss_price),
            take_profit_1=ai_dec.get('take_profit_1', risk.take_profit_price),
            take_profit_2=ai_dec.get('take_profit_2', risk.take_profit_2),
            take_profit_3=ai_dec.get('take_profit_3', risk.take_profit_3),
            max_loss=risk.max_loss_usd,
            expected_gain=ai_dec.get('expected_gain', risk.expected_value),
            risk_reward=risk.risk_reward_ratio,
            holding_period=ai_dec.get('holding_period', 'swing'),
            setup_type=setup_type,
            market_regime=self.current_regime,
            key_catalysts=ai_dec.get('key_catalysts', []),
            supporting_factors=ai_dec.get('supporting_factors', []),
            risk_factors=ai_dec.get('risk_factors', []),
            invalidation_conditions=ai_dec.get('invalidation_conditions', []),
            detailed_analysis=ai_dec.get('detailed_analysis', ''),
            conviction_level=ai_dec.get('conviction_level', 'medium'),
            bull_scenario=ai_dec.get('bull_scenario', ''),
            bear_scenario=ai_dec.get('bear_scenario', ''),
            base_scenario=ai_dec.get('base_scenario', ''),
            global_macro_impact=ai_dec.get('global_macro_impact', ''),
            scenario_analysis=scenarios
        )
    
    def _create_wait_decision(self, reasons: List[str]) -> TradeDecision:
        """Create WAIT decision"""
        return TradeDecision(
            action='WAIT',
            confidence=0,
            entry_price=0,
            position_size_usd=0,
            position_size_pct=0,
            stop_loss=0,
            take_profit_1=0,
            take_profit_2=None,
            take_profit_3=None,
            max_loss=0,
            expected_gain=0,
            risk_reward=0,
            holding_period='none',
            setup_type=TradeSetup.RANGE_BOUND,
            market_regime=self.current_regime,
            key_catalysts=[],
            supporting_factors=[],
            risk_factors=reasons,
            invalidation_conditions=[],
            detailed_analysis=f"Not viable: {', '.join(reasons)}",
            conviction_level='none',
            bull_scenario='',
            bear_scenario='',
            base_scenario='',
            global_macro_impact='',
            scenario_analysis=None
        )


# Test function
async def test_enhanced_brain():
    """Test the enhanced system"""
    from dotenv import load_dotenv
    load_dotenv()
    
    brain = HedgeFundBrain(api_key=os.getenv('DEEPSEEK_API_KEY'))
    
    market = MarketSnapshot(
        timestamp=datetime.now(),
        price=142.50,
        price_1h_ago=141.20,
        price_24h_ago=138.00,
        price_7d_ago=135.00,
        volume_1h=2_500_000,
        volume_24h=45_000_000,
        volume_avg_7d=38_000_000,
        volatility_1h=0.015,
        volatility_24h=0.035,
        atr=3.50,
        bid_ask_spread=0.02,
        order_book_depth=8_000_000,
        rsi_14=58.5,
        rsi_7=62.3,
        macd=0.45,
        macd_signal=0.38,
        ema_9=141.80,
        ema_20=140.50,
        ema_50=138.20,
        ema_200=130.00,
        bollinger_upper=145.20,
        bollinger_middle=140.50,
        bollinger_lower=135.80,
        whale_transactions_1h=12,
        whale_net_flow=2_500_000,
        exchange_netflow=-1_200_000,
        fear_greed_index=62,
        news_sentiment=0.35,
        social_volume=8500,
        funding_rate=0.008,
        btc_correlation=0.78,
        eth_correlation=0.82,
        buy_sell_ratio=1.45,
        liquidation_levels=[138.0, 135.0, 132.0]
    )
    
    portfolio = {'balance': 25.00, 'daily_pnl': 0.00}
    risk_limits = {'max_position_pct': 30, 'max_loss_pct': 3}
    
    decision = await brain.make_trading_decision(market, portfolio, risk_limits)
    
    print("\n" + "="*80)
    print("DECISION SUMMARY")
    print("="*80)
    print(f"Action: {decision.action}")
    print(f"Confidence: {decision.confidence}%")
    print(f"Setup: {decision.setup_type.value}")
    print(f"Position: ${decision.position_size_usd:.2f}")
    print(f"Stop: ${decision.stop_loss:.4f}")
    print(f"Targets: ${decision.take_profit_1:.4f} / ${decision.take_profit_2:.4f} / ${decision.take_profit_3:.4f}")
    print(f"\nGlobal Impact: {decision.global_macro_impact}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_enhanced_brain())