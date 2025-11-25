"""
ENHANCED Hyperliquid Integration V2
Complete Market Data + BNB/XRP Support + Advanced Features
"""

import os
import json
import time
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import requests
from collections import deque

import eth_account
from eth_account import Account
from eth_utils import keccak
import msgpack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderbookSnapshot:
    """Complete orderbook data"""
    timestamp: datetime
    symbol: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float
    bid_depth_5: float
    ask_depth_5: float
    bid_depth_20: float
    ask_depth_20: float
    bid_liquidity_usd: float
    ask_liquidity_usd: float
    imbalance_ratio: float
    orderbook_pressure: str  # "BUY"|"SELL"|"NEUTRAL"


@dataclass
class FundingData:
    """Funding rate data"""
    symbol: str
    funding_rate: float
    funding_rate_8h: float
    predicted_funding: float
    time_to_funding: int
    funding_1h_avg: float
    funding_8h_avg: float
    funding_24h_avg: float
    funding_trend: str  # "RISING"|"FALLING"|"STABLE"


@dataclass
class LiquidationData:
    """Liquidation levels and risks"""
    symbol: str
    long_liquidations_24h: float
    short_liquidations_24h: float
    liquidation_ratio: float
    major_long_liqui_levels: List[float]
    major_short_liqui_levels: List[float]
    distance_to_nearest_long_liqui: float
    distance_to_nearest_short_liqui: float
    liquidation_risk_score: float  # 0-10


@dataclass
class OpenInterestData:
    """Open interest data"""
    symbol: str
    open_interest_usd: float
    oi_change_1h: float
    oi_change_24h: float
    long_short_ratio: float
    oi_trend: str  # "RISING"|"FALLING"|"STABLE"


@dataclass
class VolumeProfile:
    """Volume analysis"""
    symbol: str
    volume_1h: float
    volume_24h: float
    volume_7d_avg: float
    volume_ratio_1h: float  # vs 7d avg
    volume_ratio_24h: float
    buy_volume_pct: float  # Percentage of buy volume
    large_trades_1h: int  # Trades > $100k
    whale_activity: str  # "HIGH"|"MEDIUM"|"LOW"


@dataclass
class MarketSentiment:
    """Aggregated market sentiment"""
    symbol: str
    timestamp: datetime
    funding_sentiment: float  # -1 to 1
    oi_sentiment: float
    volume_sentiment: float
    orderbook_sentiment: float
    liquidation_sentiment: float
    overall_sentiment: float  # -1 (bearish) to 1 (bullish)
    sentiment_label: str  # "VERY_BEARISH"|"BEARISH"|"NEUTRAL"|"BULLISH"|"VERY_BULLISH"
    confidence: float  # 0-1


@dataclass
class TradeExecution:
    success: bool
    exchange: str
    tx_hash: Optional[str]
    entry_price: float
    filled_amount: float
    fees: float
    slippage: float
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class AssetMetadata:
    """Asset configuration"""
    symbol: str
    index: int
    sz_decimals: int
    max_leverage: int
    min_size: float
    tick_size: float
    is_active: bool


class EnhancedHyperliquidExchange:
    """
    Enhanced Hyperliquid Integration with Complete Market Data
    Supports: BTC, ETH, SOL, BNB, XRP and all available assets
    """
    
    def __init__(self, wallet_address: str, api_wallet_private_key: str, testnet: bool = False):
        try:
            self.wallet_address = wallet_address.lower()
            
            if not api_wallet_private_key.startswith('0x'):
                api_wallet_private_key = '0x' + api_wallet_private_key
            
            self.api_wallet = Account.from_key(api_wallet_private_key)
            self.api_wallet_address = self.api_wallet.address.lower()
            
            if testnet:
                self.base_url = "https://api.hyperliquid-testnet.xyz"
                self.chain_id = 421614
            else:
                self.base_url = "https://api.hyperliquid.xyz"
                self.chain_id = 42161
            
            self.info_url = f"{self.base_url}/info"
            self.exchange_url = f"{self.base_url}/exchange"
            
            # Asset registry
            self.asset_cache: Dict[str, AssetMetadata] = {}
            self.supported_assets = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'AVAX', 'MATIC', 'ARB']
            
            # Historical data for analysis
            self.funding_history = {}
            self.oi_history = {}
            self.volume_history = {}
            self.price_history = {}
            
            # Rate limiting
            self.last_request_time = {}
            self.min_request_interval = 0.1  # 100ms between requests
            
            self._refresh_asset_cache()
            
            logger.info(f"✅ Enhanced Hyperliquid initialized:")
            logger.info(f"  Main Wallet: {self.wallet_address}")
            logger.info(f"  API Wallet: {self.api_wallet_address}")
            logger.info(f"  Network: {'TESTNET' if testnet else 'MAINNET'}")
            logger.info(f"  Assets: {len(self.asset_cache)} available")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hyperliquid: {e}")
            raise
    
    def _rate_limit(self, endpoint: str):
        """Simple rate limiting"""
        now = time.time()
        if endpoint in self.last_request_time:
            elapsed = now - self.last_request_time[endpoint]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time[endpoint] = time.time()
    
    def _refresh_asset_cache(self):
        """Get asset metadata from API"""
        try:
            payload = {"type": "meta"}
            response = requests.post(self.info_url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'universe' in data:
                for idx, asset in enumerate(data['universe']):
                    symbol = asset['name']
                    
                    self.asset_cache[symbol] = AssetMetadata(
                        symbol=symbol,
                        index=idx,
                        sz_decimals=asset.get('szDecimals', 8),
                        max_leverage=asset.get('maxLeverage', 50),
                        min_size=10 ** (-asset.get('szDecimals', 8)),
                        tick_size=0.01,  # Default
                        is_active=True
                    )
            
            logger.info(f"✅ Loaded {len(self.asset_cache)} assets")
            
            # Verify supported assets
            for asset in self.supported_assets:
                if asset in self.asset_cache:
                    logger.info(f"  ✓ {asset} available")
                else:
                    logger.warning(f"  ✗ {asset} not found on exchange")
            
        except Exception as e:
            logger.error(f"Failed to load asset metadata: {e}")
    
    def is_asset_supported(self, symbol: str) -> bool:
        """Check if asset is supported"""
        return symbol in self.asset_cache
    
    def get_complete_orderbook(self, symbol: str) -> Optional[OrderbookSnapshot]:
        """Get complete orderbook with depth analysis"""
        try:
            if not self.is_asset_supported(symbol):
                logger.error(f"{symbol} not supported")
                return None
            
            self._rate_limit('orderbook')
            
            payload = {
                "type": "l2Book",
                "coin": symbol
            }
            
            response = requests.post(self.info_url, json=payload, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if 'levels' not in data or len(data['levels']) < 2:
                return None
            
            bids = data['levels'][1]
            asks = data['levels'][0]
            
            if not bids or not asks:
                return None
            
            # Parse orderbook
            best_bid = float(bids[0]['px'])
            best_ask = float(asks[0]['px'])
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate spread
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000
            
            # Calculate depth
            bid_depth_5 = sum(float(level['sz']) for level in bids[:5])
            ask_depth_5 = sum(float(level['sz']) for level in asks[:5])
            bid_depth_20 = sum(float(level['sz']) for level in bids[:20])
            ask_depth_20 = sum(float(level['sz']) for level in asks[:20])
            
            # Calculate USD liquidity
            bid_liquidity_usd = sum(float(level['px']) * float(level['sz']) 
                                   for level in bids[:20])
            ask_liquidity_usd = sum(float(level['px']) * float(level['sz']) 
                                   for level in asks[:20])
            
            # Orderbook imbalance
            imbalance_ratio = bid_depth_20 / ask_depth_20 if ask_depth_20 > 0 else 0
            
            # Orderbook pressure
            if imbalance_ratio > 1.5:
                pressure = "BUY"
            elif imbalance_ratio < 0.67:
                pressure = "SELL"
            else:
                pressure = "NEUTRAL"
            
            return OrderbookSnapshot(
                timestamp=datetime.now(),
                symbol=symbol,
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid_price,
                spread_bps=spread_bps,
                bid_depth_5=bid_depth_5,
                ask_depth_5=ask_depth_5,
                bid_depth_20=bid_depth_20,
                ask_depth_20=ask_depth_20,
                bid_liquidity_usd=bid_liquidity_usd,
                ask_liquidity_usd=ask_liquidity_usd,
                imbalance_ratio=imbalance_ratio,
                orderbook_pressure=pressure
            )
            
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            return None
    
    def get_funding_data(self, symbol: str) -> Optional[FundingData]:
        """Get complete funding rate data"""
        try:
            if not self.is_asset_supported(symbol):
                return None
            
            self._rate_limit('funding')
            
            payload = {"type": "metaAndAssetCtxs"}
            response = requests.post(self.info_url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            universe = data[0]['universe']
            asset_ctxs = data[1]
            
            idx = self.asset_cache[symbol].index
            ctx = asset_ctxs[idx]
            
            # Current funding rate
            funding_rate = float(ctx.get('funding', 0))
            
            # Store in history
            if symbol not in self.funding_history:
                self.funding_history[symbol] = deque(maxlen=200)
            
            self.funding_history[symbol].append({
                'timestamp': datetime.now(),
                'rate': funding_rate
            })
            
            # Calculate averages
            rates = [f['rate'] for f in self.funding_history[symbol]]
            funding_1h_avg = sum(rates[-8:]) / len(rates[-8:]) if len(rates) >= 8 else funding_rate
            funding_8h_avg = sum(rates[-64:]) / len(rates[-64:]) if len(rates) >= 64 else funding_rate
            funding_24h_avg = sum(rates) / len(rates) if rates else funding_rate
            
            # Funding trend
            if len(rates) > 10:
                recent_avg = sum(rates[-10:]) / 10
                older_avg = sum(rates[-20:-10]) / 10 if len(rates) >= 20 else recent_avg
                
                if recent_avg > older_avg * 1.2:
                    trend = "RISING"
                elif recent_avg < older_avg * 0.8:
                    trend = "FALLING"
                else:
                    trend = "STABLE"
            else:
                trend = "STABLE"
            
            # Annualize to 8h period
            funding_rate_8h = funding_rate * 3
            
            # Predicted funding
            predicted_funding = funding_rate
            if len(rates) > 10:
                recent_trend = (rates[-1] - rates[-10]) / 10
                predicted_funding = funding_rate + recent_trend
            
            # Time to next funding
            current_hour = datetime.now().hour
            next_funding_hour = ((current_hour // 8) + 1) * 8 % 24
            hours_to_funding = (next_funding_hour - current_hour) % 24
            time_to_funding = hours_to_funding * 3600
            
            return FundingData(
                symbol=symbol,
                funding_rate=funding_rate,
                funding_rate_8h=funding_rate_8h,
                predicted_funding=predicted_funding,
                time_to_funding=time_to_funding,
                funding_1h_avg=funding_1h_avg,
                funding_8h_avg=funding_8h_avg,
                funding_24h_avg=funding_24h_avg,
                funding_trend=trend
            )
            
        except Exception as e:
            logger.error(f"Failed to get funding data for {symbol}: {e}")
            return None
    
    def get_open_interest_data(self, symbol: str) -> Optional[OpenInterestData]:
        """Get open interest data"""
        try:
            if not self.is_asset_supported(symbol):
                return None
            
            self._rate_limit('oi')
            
            payload = {"type": "metaAndAssetCtxs"}
            response = requests.post(self.info_url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            asset_ctxs = data[1]
            
            idx = self.asset_cache[symbol].index
            ctx = asset_ctxs[idx]
            
            oi_usd = float(ctx.get('openInterest', 0))
            funding_rate = float(ctx.get('funding', 0))
            
            # Store in history
            if symbol not in self.oi_history:
                self.oi_history[symbol] = deque(maxlen=200)
            
            self.oi_history[symbol].append({
                'timestamp': datetime.now(),
                'oi': oi_usd
            })
            
            # Calculate changes
            oi_change_1h = 0.0
            oi_change_24h = 0.0
            
            if len(self.oi_history[symbol]) > 1:
                oi_1h_ago = next((oi['oi'] for oi in reversed(self.oi_history[symbol])
                                 if (datetime.now() - oi['timestamp']).seconds >= 3600),
                                self.oi_history[symbol][-1]['oi'])
                
                if oi_1h_ago > 0:
                    oi_change_1h = ((oi_usd - oi_1h_ago) / oi_1h_ago) * 100
                
                oi_24h_ago = self.oi_history[symbol][0]['oi']
                if oi_24h_ago > 0:
                    oi_change_24h = ((oi_usd - oi_24h_ago) / oi_24h_ago) * 100
            
            # OI trend
            if abs(oi_change_1h) > 5:
                trend = "RISING" if oi_change_1h > 0 else "FALLING"
            else:
                trend = "STABLE"
            
            # Estimate long/short ratio from funding
            long_short_ratio = 1.0 + (funding_rate * 10)
            
            return OpenInterestData(
                symbol=symbol,
                open_interest_usd=oi_usd,
                oi_change_1h=oi_change_1h,
                oi_change_24h=oi_change_24h,
                long_short_ratio=long_short_ratio,
                oi_trend=trend
            )
            
        except Exception as e:
            logger.error(f"Failed to get OI data for {symbol}: {e}")
            return None
    
    def get_volume_profile(self, symbol: str) -> Optional[VolumeProfile]:
        """Get volume analysis"""
        try:
            if not self.is_asset_supported(symbol):
                return None
            
            # This would require historical candle data
            # Simplified version using current data
            
            return VolumeProfile(
                symbol=symbol,
                volume_1h=0,
                volume_24h=0,
                volume_7d_avg=0,
                volume_ratio_1h=1.0,
                volume_ratio_24h=1.0,
                buy_volume_pct=50.0,
                large_trades_1h=0,
                whale_activity="MEDIUM"
            )
            
        except Exception as e:
            logger.error(f"Failed to get volume profile for {symbol}: {e}")
            return None
    
    def estimate_liquidation_levels(self, symbol: str, orderbook: OrderbookSnapshot) -> Optional[LiquidationData]:
        """Estimate major liquidation levels from orderbook clusters"""
        try:
            self._rate_limit('liq')
            
            payload = {
                "type": "l2Book",
                "coin": symbol
            }
            
            response = requests.post(self.info_url, json=payload, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if 'levels' not in data:
                return None
            
            bids = data['levels'][1]
            asks = data['levels'][0]
            
            # Find clusters
            def find_clusters(levels, current_price, is_bid=True):
                clusters = []
                for i, level in enumerate(levels[:50]):
                    price = float(level['px'])
                    size = float(level['sz'])
                    
                    if i > 0:
                        prev_size = float(levels[i-1]['sz'])
                        if size > prev_size * 2:  # 2x size jump
                            distance_pct = abs((price - current_price) / current_price) * 100
                            clusters.append((price, distance_pct, size))
                
                clusters.sort(key=lambda x: x[2], reverse=True)
                return [c[0] for c in clusters[:5]]
            
            current_price = orderbook.mid_price
            
            long_liqui_levels = find_clusters(bids, current_price, is_bid=True)
            short_liqui_levels = find_clusters(asks, current_price, is_bid=False)
            
            # Calculate distances
            distance_to_long = min([abs((p - current_price) / current_price * 100) 
                                   for p in long_liqui_levels]) if long_liqui_levels else 10.0
            distance_to_short = min([abs((p - current_price) / current_price * 100) 
                                    for p in short_liqui_levels]) if short_liqui_levels else 10.0
            
            # Estimate 24h liquidations from OI changes
            oi_data = self.get_open_interest_data(symbol)
            if oi_data:
                oi_change_abs = abs(oi_data.oi_change_24h / 100 * oi_data.open_interest_usd)
                long_liq_24h = oi_change_abs * 0.6
                short_liq_24h = oi_change_abs * 0.4
                liq_ratio = long_liq_24h / short_liq_24h if short_liq_24h > 0 else 1.0
            else:
                long_liq_24h = 0
                short_liq_24h = 0
                liq_ratio = 1.0
            
            # Liquidation risk score (0-10)
            risk_score = 0
            if distance_to_long < 2 or distance_to_short < 2:
                risk_score += 4
            if distance_to_long < 5 or distance_to_short < 5:
                risk_score += 3
            if liq_ratio > 2 or liq_ratio < 0.5:
                risk_score += 3
            
            return LiquidationData(
                symbol=symbol,
                long_liquidations_24h=long_liq_24h,
                short_liquidations_24h=short_liq_24h,
                liquidation_ratio=liq_ratio,
                major_long_liqui_levels=long_liqui_levels,
                major_short_liqui_levels=short_liqui_levels,
                distance_to_nearest_long_liqui=distance_to_long,
                distance_to_nearest_short_liqui=distance_to_short,
                liquidation_risk_score=min(risk_score, 10)
            )
            
        except Exception as e:
            logger.error(f"Failed to estimate liquidations for {symbol}: {e}")
            return None
    
    def calculate_market_sentiment(
        self,
        symbol: str,
        orderbook: Optional[OrderbookSnapshot],
        funding: Optional[FundingData],
        oi_data: Optional[OpenInterestData],
        liquidations: Optional[LiquidationData]
    ) -> MarketSentiment:
        """Calculate aggregated market sentiment"""
        
        sentiments = []
        
        # Funding sentiment
        if funding:
            if funding.funding_rate > 0.01:
                funding_sentiment = -0.5  # Bearish (too much long leverage)
            elif funding.funding_rate < -0.01:
                funding_sentiment = 0.5  # Bullish (shorts paying longs)
            else:
                funding_sentiment = 0.0
            sentiments.append(funding_sentiment)
        else:
            funding_sentiment = 0.0
        
        # OI sentiment
        if oi_data:
            if oi_data.oi_trend == "RISING" and oi_data.oi_change_1h > 5:
                oi_sentiment = 0.3  # Interest increasing
            elif oi_data.oi_trend == "FALLING" and oi_data.oi_change_1h < -5:
                oi_sentiment = -0.3  # Interest declining
            else:
                oi_sentiment = 0.0
            sentiments.append(oi_sentiment)
        else:
            oi_sentiment = 0.0
        
        # Volume sentiment (placeholder - would need real volume data)
        volume_sentiment = 0.0
        sentiments.append(volume_sentiment)
        
        # Orderbook sentiment
        if orderbook:
            if orderbook.orderbook_pressure == "BUY":
                orderbook_sentiment = 0.4
            elif orderbook.orderbook_pressure == "SELL":
                orderbook_sentiment = -0.4
            else:
                orderbook_sentiment = 0.0
            sentiments.append(orderbook_sentiment)
        else:
            orderbook_sentiment = 0.0
        
        # Liquidation sentiment
        if liquidations:
            if liquidations.liquidation_ratio > 1.5:
                liquidation_sentiment = -0.3  # More longs liquidated (bearish)
            elif liquidations.liquidation_ratio < 0.67:
                liquidation_sentiment = 0.3  # More shorts liquidated (bullish)
            else:
                liquidation_sentiment = 0.0
            sentiments.append(liquidation_sentiment)
        else:
            liquidation_sentiment = 0.0
        
        # Overall sentiment
        overall = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        # Sentiment label
        if overall > 0.5:
            label = "VERY_BULLISH"
        elif overall > 0.2:
            label = "BULLISH"
        elif overall > -0.2:
            label = "NEUTRAL"
        elif overall > -0.5:
            label = "BEARISH"
        else:
            label = "VERY_BEARISH"
        
        # Confidence based on data availability
        confidence = len(sentiments) / 5.0
        
        return MarketSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            funding_sentiment=funding_sentiment,
            oi_sentiment=oi_sentiment,
            volume_sentiment=volume_sentiment,
            orderbook_sentiment=orderbook_sentiment,
            liquidation_sentiment=liquidation_sentiment,
            overall_sentiment=overall,
            sentiment_label=label,
            confidence=confidence
        )
    
    def get_complete_market_data(self, symbol: str) -> Dict:
        """Get ALL market data in one call"""
        logger.info(f"🔍 Fetching complete market data for {symbol}...")
        
        orderbook = self.get_complete_orderbook(symbol)
        funding = self.get_funding_data(symbol)
        oi_data = self.get_open_interest_data(symbol)
        liquidations = self.estimate_liquidation_levels(symbol, orderbook) if orderbook else None
        volume = self.get_volume_profile(symbol)
        
        # Calculate sentiment
        sentiment = self.calculate_market_sentiment(
            symbol, orderbook, funding, oi_data, liquidations
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'orderbook': orderbook,
            'funding': funding,
            'open_interest': oi_data,
            'liquidations': liquidations,
            'volume': volume,
            'sentiment': sentiment
        }
    
    def get_multi_asset_snapshot(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Get snapshot for multiple assets"""
        if symbols is None:
            symbols = self.supported_assets
        
        logger.info(f"📊 Fetching data for {len(symbols)} assets...")
        
        results = {}
        for symbol in symbols:
            if self.is_asset_supported(symbol):
                try:
                    results[symbol] = self.get_complete_market_data(symbol)
                    time.sleep(0.2)  # Rate limiting
                except Exception as e:
                    logger.error(f"Failed to get data for {symbol}: {e}")
        
        return results
    
    def _sign_l1_action(self, action: Dict, nonce: int, vault_address: Optional[str] = None) -> Dict:
        """Sign action using MessagePack + Keccak"""
        try:
            serialized = msgpack.packb(action, use_bin_type=True)
            encoded = serialized + nonce.to_bytes(8, 'big')
            
            if vault_address:
                encoded += b'\x01' + bytes.fromhex(vault_address[2:])
            else:
                encoded += b'\x00'
            
            msg_hash = keccak(encoded)
            signed = self.api_wallet.sign_message_hash(msg_hash)
            
            return {
                'action': action,
                'nonce': nonce,
                'signature': {
                    'r': hex(signed.r),
                    's': hex(signed.s),
                    'v': signed.v
                },
                'vaultAddress': vault_address
            }
            
        except Exception as e:
            logger.error(f"Failed to sign action: {e}")
            raise
    
    async def place_order(
        self,
        symbol: str,
        is_buy: bool,
        size: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        post_only: bool = False
    ) -> TradeExecution:
        """Place order with multiple execution options"""
        try:
            if not self.is_asset_supported(symbol):
                raise ValueError(f"{symbol} not supported")
            
            # Get price if not provided
            if price is None:
                orderbook = self.get_complete_orderbook(symbol)
                if not orderbook:
                    raise ValueError(f"Could not get orderbook for {symbol}")
                
                mid_price = orderbook.mid_price
                # Add slippage buffer
                if is_buy:
                    price = mid_price * 1.002  # 0.2% slippage
                else:
                    price = mid_price * 0.998
            
            # Calculate size in contracts
            sz = size / price
            asset_idx = self.asset_cache[symbol].index
            
            # Build order
            order_type = {"limit": {"tif": "Alk" if post_only else "Ioc"}}
            
            order = {
                "type": "order",
                "orders": [{
                    "a": asset_idx,
                    "b": is_buy,
                    "p": str(price),
                    "s": str(sz),
                    "r": reduce_only,
                    "t": order_type
                }],
                "grouping": "na"
            }
            
            nonce = int(time.time() * 1000)
            signed_action = self._sign_l1_action(order, nonce)
            
            response = requests.post(
                self.exchange_url,
                json=signed_action,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('status') == 'ok':
                statuses = result.get('response', {}).get('data', {}).get('statuses', [])
                filled = statuses[0] if statuses else {}
                
                filled_px = filled.get('filled', {}).get('avgPx', price)
                actual_slippage = abs(float(filled_px) - price) / price if filled_px else 0.01
                
                return TradeExecution(
                    success=True,
                    exchange='Hyperliquid',
                    tx_hash=str(filled.get('filled', 'pending')),
                    entry_price=float(filled_px) if filled_px else price,
                    filled_amount=sz,
                    fees=size * 0.00025,  # 2.5 bps maker/taker
                    slippage=actual_slippage,
                    timestamp=datetime.now()
                )
            else:
                error = result.get('response', 'Unknown error')
                raise Exception(f"Order failed: {error}")
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return TradeExecution(
                success=False,
                exchange='Hyperliquid',
                tx_hash=None,
                entry_price=0,
                filled_amount=0,
                fees=0,
                slippage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def place_market_order(
        self,
        symbol: str,
        is_buy: bool,
        size_usd: float,
        max_slippage_pct: float = 0.5
    ) -> TradeExecution:
        """Place market order with slippage protection"""
        try:
            orderbook = self.get_complete_orderbook(symbol)
            if not orderbook:
                raise ValueError(f"Could not get orderbook for {symbol}")
            
            # Calculate limit price with slippage tolerance
            if is_buy:
                limit_price = orderbook.mid_price * (1 + max_slippage_pct / 100)
            else:
                limit_price = orderbook.mid_price * (1 - max_slippage_pct / 100)
            
            return await self.place_order(
                symbol=symbol,
                is_buy=is_buy,
                size=size_usd,
                price=limit_price,
                reduce_only=False,
                post_only=False
            )
            
        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return TradeExecution(
                success=False,
                exchange='Hyperliquid',
                tx_hash=None,
                entry_price=0,
                filled_amount=0,
                fees=0,
                slippage=0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def get_account_state(self) -> Optional[Dict]:
        """Get account balance and positions"""
        try:
            self._rate_limit('account')
            
            payload = {
                "type": "clearinghouseState",
                "user": self.wallet_address
            }
            
            response = requests.post(self.info_url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            balance = float(data.get('marginSummary', {}).get('accountValue', 0))
            positions = data.get('assetPositions', [])
            
            return {
                'balance': balance,
                'positions': positions,
                'margin_used': float(data.get('marginSummary', {}).get('totalMarginUsed', 0)),
                'available_balance': balance - float(data.get('marginSummary', {}).get('totalMarginUsed', 0))
            }
            
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return None
    
    def close_position(self, symbol: str) -> bool:
        """Close position for symbol"""
        try:
            account = self.get_account_state()
            if not account:
                return False
            
            # Find position
            position = next((p for p in account['positions'] 
                           if p['position']['coin'] == symbol), None)
            
            if not position:
                logger.info(f"No position found for {symbol}")
                return True
            
            size = abs(float(position['position']['szi']))
            is_long = float(position['position']['szi']) > 0
            
            # Close with market order
            result = self.place_order(
                symbol=symbol,
                is_buy=not is_long,  # Opposite direction to close
                size=size,
                reduce_only=True
            )
            
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return False


# Enhanced test function
async def test_enhanced_v2():
    """Comprehensive test of enhanced features"""
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    hl = EnhancedHyperliquidExchange(
        wallet_address=os.getenv('HYPERLIQUID_WALLET_ADDRESS'),
        api_wallet_private_key=os.getenv('HYPERLIQUID_API_PRIVATE_KEY'),
        testnet=True
    )
    
    # Test multiple assets including BNB and XRP
    test_symbols = ['SOL', 'BNB', 'XRP', 'ETH']
    
    print(f"\n{'='*80}")
    print(f"ENHANCED HYPERLIQUID V2 - MULTI-ASSET TEST")
    print(f"{'='*80}\n")
    
    for symbol in test_symbols:
        if not hl.is_asset_supported(symbol):
            print(f"❌ {symbol} not supported on this network")
            continue
        
        print(f"\n{'='*70}")
        print(f"📊 {symbol} COMPLETE MARKET DATA")
        print(f"{'='*70}")
        
        data = hl.get_complete_market_data(symbol)
        
        # Orderbook
        if data['orderbook']:
            ob = data['orderbook']
            print(f"\n🔹 ORDERBOOK:")
            print(f"   Mid Price: ${ob.mid_price:.4f}")
            print(f"   Spread: {ob.spread_bps:.2f} bps")
            print(f"   Bid Liquidity: ${ob.bid_liquidity_usd:,.0f}")
            print(f"   Ask Liquidity: ${ob.ask_liquidity_usd:,.0f}")
            print(f"   Imbalance: {ob.imbalance_ratio:.2f}")
            print(f"   Pressure: {ob.orderbook_pressure}")
        
        # Funding
        if data['funding']:
            fd = data['funding']
            print(f"\n🔹 FUNDING:")
            print(f"   Current: {fd.funding_rate:.6f}% ({fd.funding_rate_8h:.4f}% per 8h)")
            print(f"   Trend: {fd.funding_trend}")
            print(f"   24H Avg: {fd.funding_24h_avg:.6f}%")
            print(f"   Next in: {fd.time_to_funding/3600:.1f}h")
        
        # Open Interest
        if data['open_interest']:
            oi = data['open_interest']
            print(f"\n🔹 OPEN INTEREST:")
            print(f"   Total: ${oi.open_interest_usd:,.0f}")
            print(f"   1H Change: {oi.oi_change_1h:+.2f}%")
            print(f"   24H Change: {oi.oi_change_24h:+.2f}%")
            print(f"   Trend: {oi.oi_trend}")
            print(f"   Long/Short: {oi.long_short_ratio:.2f}")
        
        # Liquidations
        if data['liquidations']:
            liq = data['liquidations']
            print(f"\n🔹 LIQUIDATIONS:")
            print(f"   Risk Score: {liq.liquidation_risk_score}/10")
            print(f"   Distance to Long Liqui: {liq.distance_to_nearest_long_liqui:.2f}%")
            print(f"   Distance to Short Liqui: {liq.distance_to_nearest_short_liqui:.2f}%")
            if liq.major_long_liqui_levels:
                print(f"   Long Levels: {[f'${p:.2f}' for p in liq.major_long_liqui_levels[:3]]}")
            if liq.major_short_liqui_levels:
                print(f"   Short Levels: {[f'${p:.2f}' for p in liq.major_short_liqui_levels[:3]]}")
        
        # Sentiment
        if data['sentiment']:
            sent = data['sentiment']
            print(f"\n🔹 MARKET SENTIMENT:")
            print(f"   Overall: {sent.sentiment_label} ({sent.overall_sentiment:+.2f})")
            print(f"   Confidence: {sent.confidence*100:.0f}%")
            print(f"   Funding: {sent.funding_sentiment:+.2f}")
            print(f"   OI: {sent.oi_sentiment:+.2f}")
            print(f"   Orderbook: {sent.orderbook_sentiment:+.2f}")
            print(f"   Liquidations: {sent.liquidation_sentiment:+.2f}")
        
        time.sleep(0.5)  # Rate limiting between assets
    
    # Test account state
    print(f"\n{'='*70}")
    print(f"💰 ACCOUNT STATE")
    print(f"{'='*70}")
    
    account = hl.get_account_state()
    if account:
        print(f"Balance: ${account['balance']:.2f}")
        print(f"Margin Used: ${account['margin_used']:.2f}")
        print(f"Available: ${account['available_balance']:.2f}")
        print(f"Positions: {len(account['positions'])}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_enhanced_v2())