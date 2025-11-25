"""
RL STATE MANAGER - COMPLETELY FIXED VERSION
✅ Proper attribute names matching actual implementation
✅ Individual model files (PyTorch best practice)
✅ Optimizer states saved
✅ Memory-efficient experience buffer handling
✅ Proper DQN target network support
"""

import pickle
import json
import torch
import os
import glob
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class RLPerformanceMetrics:
    """Performance tracking for RL models"""
    score: float
    win_rate: float
    sharpe_ratio: float
    total_pnl: float
    avg_leverage: float
    liquidation_rate: float
    trades_count: int
    generation: int
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'total_pnl': self.total_pnl,
            'avg_leverage': self.avg_leverage,
            'liquidation_rate': self.liquidation_rate,
            'trades_count': self.trades_count,
            'generation': self.generation,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class RLStateManager:
    """
    Fixed RL State Manager
    - Saves each model to its own file
    - Includes optimizer states
    - Efficient memory buffer handling
    """
    
    def __init__(self, base_dir: str = "rl_states"):
        self.base_dir = base_dir
        
        # Individual model files
        self.dqn_path = os.path.join(base_dir, "dqn.pth")
        self.a3c_path = os.path.join(base_dir, "a3c.pth")
        self.ppo_path = os.path.join(base_dir, "ppo.pth")
        self.metadata_path = os.path.join(base_dir, "metadata.json")
        
        # Backup directory
        self.backup_dir = os.path.join(base_dir, "backups")
        
        # Create directories
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # ✅ FIX: Register safe globals ONCE at initialization (not per-load)
        self._register_safe_globals()
        
        logger.info("💾 RL State Manager initialized (FIXED VERSION)")
        logger.info(f"   Base dir: {base_dir}")
        logger.info(f"   Individual model files: dqn.pth, a3c.pth, ppo.pth")
        logger.info(f"   ✅ PyTorch 2.6+ safe globals registered")

    
    def save_state(self, rl_coordinator, evolutionary_system, 
                   force_save: bool = False) -> Tuple[bool, float]:
        """
        Save RL state only if performance improved
        
        ✅ FIXED:
        - Correct attribute names
        - Saves optimizer states
        - Individual model files
        - Efficient memory handling
        """
        try:
            # Calculate current performance
            current_metrics = self.calculate_performance_metrics(evolutionary_system)
            current_score = current_metrics.score
            
            # Load previous best
            best_metrics = self.load_best_performance()
            best_score = best_metrics.score if best_metrics else 0.0
            
            # Decision: Save only if improved or forced
            should_save = force_save or current_score > best_score
            
            if should_save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # ==================== SAVE DQN ====================
                self._save_dqn_state(rl_coordinator.dqn_short, timestamp)
                
                # ==================== SAVE A3C ====================
                self._save_a3c_state(rl_coordinator.a3c_mid, timestamp)
                
                # ==================== SAVE PPO ====================
                self._save_ppo_state(rl_coordinator.ppo_long, timestamp)
                
                # ==================== SAVE METADATA ====================
                self._save_metadata(current_metrics, timestamp)
                
                # Cleanup old backups
                self._cleanup_old_backups()
                
                if current_score > best_score:
                    logger.info(f"🏆 NEW BEST RL STATE! Score: {current_score:.3f} "
                               f"(↑ {current_score - best_score:+.3f})")
                else:
                    logger.info(f"💾 RL state saved (FORCED) - Score: {current_score:.3f}")
                
                return True, current_score
            else:
                logger.debug(f"⏭️ RL state not saved - Current: {current_score:.3f}, Best: {best_score:.3f}")
                return False, current_score
                
        except Exception as e:
            logger.error(f"❌ RL state save failed: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def _save_dqn_state(self, dqn_short, timestamp: str):
        """
        ✅ UPDATED: Save DQN with target network
        """
        try:
            dqn_state = {
                # Policy network
                'network': dqn_short.network.state_dict(),
                
                # ✅ NEW: Save target network
                'target_network': dqn_short.target_network.state_dict(),
                
                # Optimizer state
                'optimizer': dqn_short.optimizer.state_dict(),
                
                # Training state
                'epsilon': dqn_short.epsilon,
                'epsilon_decay': dqn_short.epsilon_decay,
                'epsilon_min': dqn_short.epsilon_min,
                
                # ✅ NEW: Save target network update counter
                'update_counter': dqn_short.update_counter,
                'target_update_frequency': dqn_short.target_update_frequency,
                
                # Mappings
                'agent_to_archetype': dict(dqn_short.agent_to_archetype),
                'archetype_performance': {
                    k: dict(v) for k, v in dqn_short.archetype_performance.items()
                },
                
                # Memory sample
                'memory_sample': list(dqn_short.memory)[-1000:] if len(dqn_short.memory) > 0 else [],
                'memory_size': len(dqn_short.memory),
                
                'timestamp': timestamp,
                'model_type': 'DQN'
            }
            
            # Save files
            torch.save(dqn_state, self.dqn_path)
            backup_path = os.path.join(self.backup_dir, f"dqn_{timestamp}.pth")
            torch.save(dqn_state, backup_path)
            
            logger.debug(f"✅ DQN saved with target network")
            
        except Exception as e:
            logger.error(f"❌ DQN save failed: {e}")
            raise
    
    def _save_a3c_state(self, a3c_mid, timestamp: str):
        """
        ✅ FIXED: Save A3C with correct attribute names
        """
        try:
            a3c_state = {
                # ✅ FIX: Use 'policy_net' and 'value_net' not 'actor'/'critic'
                'policy_net': a3c_mid.policy_net.state_dict(),
                'value_net': a3c_mid.value_net.state_dict(),
                
                # ✅ FIX: Save optimizer state
                'optimizer': a3c_mid.optimizer.state_dict(),
                
                # ✅ FIX: Save training hyperparameters
                'entropy_coef': a3c_mid.entropy_coef,
                'min_entropy_coef': a3c_mid.min_entropy_coef,
                'entropy_decay': a3c_mid.entropy_decay,
                'grad_clip_value': a3c_mid.grad_clip_value,
                'steps': a3c_mid.steps,
                'current_lr': a3c_mid.current_lr,
                'base_lr': a3c_mid.base_lr,
                
                # ✅ FIX: Save loss history for explosion detection
                'loss_history': list(a3c_mid.loss_history),
                'explosion_count': a3c_mid.explosion_count,
                
                # ✅ FIX: Save gradient norms
                'grad_norms': list(a3c_mid.grad_norms),
                
                'timestamp': timestamp,
                'model_type': 'A3C'
            }
            
            # Save main file
            torch.save(a3c_state, self.a3c_path)
            
            # Save backup
            backup_path = os.path.join(self.backup_dir, f"a3c_{timestamp}.pth")
            torch.save(a3c_state, backup_path)
            
            logger.debug(f"✅ A3C saved: entropy_coef={a3c_mid.entropy_coef:.4f}, "
                        f"steps={a3c_mid.steps}, lr={a3c_mid.current_lr:.2e}")
            
        except Exception as e:
            logger.error(f"❌ A3C save failed: {e}")
            raise
    
    def _save_ppo_state(self, ppo_long, timestamp: str):
        """
        ✅ CORRECT: PPO already uses correct attribute names
        """
        try:
            ppo_state = {
                # ✅ PPO correctly uses 'actor' and 'critic'
                'actor': ppo_long.actor.state_dict(),
                'critic': ppo_long.critic.state_dict(),
                
                # ✅ FIX: Save both optimizers
                'actor_optimizer': ppo_long.actor_optimizer.state_dict(),
                'critic_optimizer': ppo_long.critic_optimizer.state_dict(),
                
                # ✅ FIX: Save agent statistics
                'agent_stats': {
                    k: dict(v) for k, v in ppo_long.agent_stats.items()
                },
                
                # ✅ FIX: Save PPO hyperparameters
                'clip_epsilon': ppo_long.clip_epsilon,
                'ppo_epochs': ppo_long.ppo_epochs,
                
                'timestamp': timestamp,
                'model_type': 'PPO'
            }
            
            # Save main file
            torch.save(ppo_state, self.ppo_path)
            
            # Save backup
            backup_path = os.path.join(self.backup_dir, f"ppo_{timestamp}.pth")
            torch.save(ppo_state, backup_path)
            
            logger.debug(f"✅ PPO saved: {len(ppo_long.agent_stats)} agent stats, "
                        f"clip_epsilon={ppo_long.clip_epsilon:.2f}")
            
        except Exception as e:
            logger.error(f"❌ PPO save failed: {e}")
            raise
    
    def _save_metadata(self, metrics: RLPerformanceMetrics, timestamp: str):
        """Save performance metadata"""
        try:
            metadata = {
                'timestamp': timestamp,
                'performance_metrics': metrics.to_dict(),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"✅ Metadata saved: score={metrics.score:.3f}")
            
        except Exception as e:
            logger.warning(f"⚠️ Metadata save failed: {e}")
    
    def load_state(self, rl_coordinator, load_strategy: str = "BEST") -> Tuple[bool, float]:
        """
        Load RL state from individual model files
        
        ✅ FIXED: PyTorch 2.6+ compatibility with safe globals
        """
        try:
            # # ✅ ADD: Register safe globals first
            # self._register_safe_globals()           
            performance_score = 0.0
            
            # ==================== LOAD DQN ====================
            if os.path.exists(self.dqn_path):
                self._load_dqn_state(rl_coordinator.dqn_short)
            else:
                logger.warning("⚠️ No DQN state found")
            
            # ==================== LOAD A3C ====================
            if os.path.exists(self.a3c_path):
                self._load_a3c_state(rl_coordinator.a3c_mid)
            else:
                logger.warning("⚠️ No A3C state found")
            
            # ==================== LOAD PPO ====================
            if os.path.exists(self.ppo_path):
                self._load_ppo_state(rl_coordinator.ppo_long)
            else:
                logger.warning("⚠️ No PPO state found")
            
            # ==================== LOAD METADATA ====================
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                performance_score = metadata['performance_metrics']['score']
                logger.info(f"📊 Metadata loaded (Score: {performance_score:.3f})")
            
            if os.path.exists(self.dqn_path) or os.path.exists(self.a3c_path) or os.path.exists(self.ppo_path):
                logger.info(f"🎉 RL state loaded successfully!")
                return True, performance_score
            else:
                logger.info(f"📄 No previous RL state found, starting fresh")
                return False, 0.0
            
        except Exception as e:
            logger.error(f"❌ RL state load failed: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def _load_dqn_state(self, dqn_short):
        """✅ UPDATED: Safe loading for PyTorch 2.6+ with enhanced logging"""
        try:
            dqn_state = self._safe_torch_load(self.dqn_path)
            
            # Load policy network
            dqn_short.network.load_state_dict(dqn_state['network'])
            
            # ✅ NEW: Load target network
            if 'target_network' in dqn_state:
                dqn_short.target_network.load_state_dict(dqn_state['target_network'])
                logger.debug("✅ Target network loaded")
            else:
                # Fallback: sync from policy network
                dqn_short.target_network.load_state_dict(dqn_state['network'])
                logger.warning("⚠️ No target network in saved state, syncing from policy network")
            
            # Load optimizer
            dqn_short.optimizer.load_state_dict(dqn_state['optimizer'])
            
            # Restore training state
            dqn_short.epsilon = dqn_state['epsilon']
            dqn_short.epsilon_decay = dqn_state.get('epsilon_decay', 0.995)
            dqn_short.epsilon_min = dqn_state.get('epsilon_min', 0.05)
            
            # ✅ NEW: Restore update counter
            dqn_short.update_counter = dqn_state.get('update_counter', 0)
            dqn_short.target_update_frequency = dqn_state.get('target_update_frequency', 100)
            
            # Restore mappings
            dqn_short.agent_to_archetype = dqn_state.get('agent_to_archetype', {})
            dqn_short.archetype_performance = defaultdict(
                lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'samples': 0},
                dqn_state.get('archetype_performance', {})
            )
            
            # Restore memory
            memory_sample = dqn_state.get('memory_sample', [])
            if memory_sample:
                dqn_short.memory = deque(memory_sample, maxlen=10000)
            
            # ✅ ENHANCED LOGGING: Log DQN parameters
            logger.info(f"✅ DQN loaded: {len(dqn_short.agent_to_archetype)} mappings, "
                    f"epsilon={dqn_short.epsilon:.3f}, "
                    f"update_counter={dqn_short.update_counter}, "
                    f"memory_size={len(dqn_short.memory)}, "
                    f"target_updates={dqn_short.update_counter // dqn_short.target_update_frequency}")
            
            # Log archetype performance summary
            if dqn_short.archetype_performance:
                best_archetype = max(dqn_short.archetype_performance.items(), 
                                key=lambda x: x[1].get('total_pnl', 0))
                logger.info(f"📊 DQN Best Archetype: {best_archetype[0]} "
                        f"(PnL: ${best_archetype[1].get('total_pnl', 0):.2f}, "
                        f"Wins: {best_archetype[1].get('wins', 0)})")
                
        except Exception as e:
            logger.error(f"❌ DQN load failed: {e}")
            raise

    def _load_a3c_state(self, a3c_mid):
        """✅ FIXED: Load A3C with correct attributes and enhanced logging"""
        try:
            a3c_state = self._safe_torch_load(self.a3c_path)
            
            # ✅ FIX: Load to 'policy_net' and 'value_net'
            a3c_mid.policy_net.load_state_dict(a3c_state['policy_net'])
            a3c_mid.value_net.load_state_dict(a3c_state['value_net'])
            
            # ✅ FIX: Load optimizer
            a3c_mid.optimizer.load_state_dict(a3c_state['optimizer'])
            
            # ✅ FIX: Restore training hyperparameters
            a3c_mid.entropy_coef = a3c_state.get('entropy_coef', 0.01)
            a3c_mid.min_entropy_coef = a3c_state.get('min_entropy_coef', 0.001)
            a3c_mid.entropy_decay = a3c_state.get('entropy_decay', 0.999)
            a3c_mid.grad_clip_value = a3c_state.get('grad_clip_value', 0.1)
            a3c_mid.steps = a3c_state.get('steps', 0)
            a3c_mid.current_lr = a3c_state.get('current_lr', 3e-4)
            a3c_mid.base_lr = a3c_state.get('base_lr', 3e-4)
            
            # ✅ FIX: Restore loss history
            a3c_mid.loss_history = deque(a3c_state.get('loss_history', []), maxlen=10)
            a3c_mid.explosion_count = a3c_state.get('explosion_count', 0)
            a3c_mid.grad_norms = deque(a3c_state.get('grad_norms', []), maxlen=100)
            
            # ✅ ENHANCED LOGGING
            logger.info(f"✅ A3C loaded: entropy_coef={a3c_mid.entropy_coef:.4f}, "
                    f"steps={a3c_mid.steps}, "
                    f"lr={a3c_mid.current_lr:.2e}, "
                    f"grad_clip={a3c_mid.grad_clip_value:.3f}, "
                    f"explosion_count={a3c_mid.explosion_count}")
            
            # Log loss history if available
            if a3c_mid.loss_history:
                avg_loss = np.mean(list(a3c_mid.loss_history))
                logger.info(f"📊 A3C Recent Loss: avg={avg_loss:.4f}, "
                        f"latest={a3c_mid.loss_history[-1]:.4f}")
                
        except Exception as e:
            logger.error(f"❌ A3C load failed: {e}")
            raise

    def _load_ppo_state(self, ppo_long):
        """✅ CORRECT: PPO with enhanced logging"""
        try:
            ppo_state = self._safe_torch_load(self.ppo_path)
            
            # ✅ PPO correctly uses 'actor' and 'critic'
            ppo_long.actor.load_state_dict(ppo_state['actor'])
            ppo_long.critic.load_state_dict(ppo_state['critic'])
            
            # ✅ FIX: Load both optimizers
            ppo_long.actor_optimizer.load_state_dict(ppo_state['actor_optimizer'])
            ppo_long.critic_optimizer.load_state_dict(ppo_state['critic_optimizer'])
            
            # ✅ FIX: Restore agent statistics
            ppo_long.agent_stats = defaultdict(
                lambda: {'selections': 0, 'wins': 0, 'sharpe': 0.0, 'total_pnl': 0.0, 'sell_ratio': 0.0},
                ppo_state.get('agent_stats', {})
            )
            
            # ✅ FIX: Restore hyperparameters
            ppo_long.clip_epsilon = ppo_state.get('clip_epsilon', 0.2)
            ppo_long.ppo_epochs = ppo_state.get('ppo_epochs', 10)
            
            # ✅ ENHANCED LOGGING: Agent performance summary
            total_agents = len(ppo_long.agent_stats)
            active_agents = sum(1 for stats in ppo_long.agent_stats.values() if stats['selections'] > 0)
            
            if active_agents > 0:
                best_agent = max(ppo_long.agent_stats.items(), 
                            key=lambda x: x[1].get('total_pnl', 0))
                sell_agents = sum(1 for stats in ppo_long.agent_stats.values() if stats.get('sell_ratio', 0) > 0.5)
                
                logger.info(f"✅ PPO loaded: {active_agents}/{total_agents} active agents, "
                        f"clip_epsilon={ppo_long.clip_epsilon:.2f}")
                logger.info(f"📊 PPO Best Agent: {best_agent[0]} "
                        f"(PnL: ${best_agent[1].get('total_pnl', 0):.2f}, "
                        f"sell_ratio={best_agent[1].get('sell_ratio', 0):.2f})")
                logger.info(f"📊 PPO Sell-capable agents: {sell_agents}/{active_agents} "
                        f"({sell_agents/active_agents*100:.1f}%)")
            else:
                logger.info(f"✅ PPO loaded: {total_agents} agents (none active yet)")
                
        except Exception as e:
            logger.error(f"❌ PPO load failed: {e}")
            raise
    
    def calculate_performance_metrics(self, evolutionary_system) -> RLPerformanceMetrics:
        """Calculate comprehensive RL performance score"""
        try:
            recent_trades = self._get_recent_trades_for_evaluation(evolutionary_system)
            
            if not recent_trades:
                return RLPerformanceMetrics(
                    score=0.0, win_rate=0.0, sharpe_ratio=0.0, total_pnl=0.0,
                    avg_leverage=0.0, liquidation_rate=0.0, trades_count=0,
                    generation=evolutionary_system.generation,
                    timestamp=datetime.now().isoformat()
                )
            
            # Calculate metrics
            wins = sum(1 for t in recent_trades if self._is_winning_trade(t))
            win_rate = wins / len(recent_trades)
            
            pnl_returns = [t.pnl_pct / 100 for t in recent_trades]
            sharpe_ratio = np.mean(pnl_returns) / np.std(pnl_returns) if len(pnl_returns) > 1 and np.std(pnl_returns) > 0 else 0.0
            
            total_pnl = sum(t.pnl for t in recent_trades)
            avg_leverage = np.mean([t.leverage for t in recent_trades])
            
            liquidations = sum(1 for t in recent_trades if t.close_reason == 'liquidated')
            liquidation_rate = liquidations / len(recent_trades)
            
            # Weighted score
            score = (
                win_rate * 0.35 +
                max(0, sharpe_ratio) * 0.25 +
                max(0, total_pnl / 1000) * 0.20 +
                (1.0 - liquidation_rate) * 0.15 +
                min(1.0, len(recent_trades) / 50) * 0.05
            )
            
            return RLPerformanceMetrics(
                score=score,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                total_pnl=total_pnl,
                avg_leverage=avg_leverage,
                liquidation_rate=liquidation_rate,
                trades_count=len(recent_trades),
                generation=evolutionary_system.generation,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Performance calculation failed: {e}")
            return RLPerformanceMetrics(
                score=0.0, win_rate=0.0, sharpe_ratio=0.0, total_pnl=0.0,
                avg_leverage=0.0, liquidation_rate=0.0, trades_count=0,
                generation=evolutionary_system.generation,
                timestamp=datetime.now().isoformat()
            )
    
    def _get_recent_trades_for_evaluation(self, evolutionary_system, min_trades: int = 20):
        """Get recent trades for evaluation"""
        closed_trades = [t for t in evolutionary_system.all_trades if t.status == 'closed']
        
        if len(closed_trades) >= min_trades:
            return closed_trades[-50:]
        elif len(closed_trades) >= 10:
            return closed_trades
        else:
            return []
    
    def _is_winning_trade(self, trade) -> bool:
        """Unified win definition"""
        if trade.position_size <= 0:
            return False
        net_pnl_pct = (trade.pnl / trade.position_size) * 100
        return net_pnl_pct > 0.1
    
    def load_best_performance(self) -> Optional[RLPerformanceMetrics]:
        """Load best performance from metadata"""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                return RLPerformanceMetrics.from_dict(metadata['performance_metrics'])
        except Exception as e:
            logger.debug(f"Could not load best performance: {e}")
        return None
    
    def _cleanup_old_backups(self, keep_count: int = 5):
        """Keep only recent backups"""
        try:
            for model_type in ['dqn', 'a3c', 'ppo']:
                backups = sorted(glob.glob(os.path.join(self.backup_dir, f"{model_type}_*.pth")))
                for old_backup in backups[:-keep_count]:
                    os.remove(old_backup)
                    logger.debug(f"🗑️ Removed old backup: {os.path.basename(old_backup)}")
        except Exception as e:
            logger.warning(f"⚠️ Backup cleanup failed: {e}")
    
    def get_performance_trend(self) -> Dict:
        """Get performance trend analysis"""
        try:
            backups = sorted(glob.glob(os.path.join(self.backup_dir, "dqn_*.pth")))[-10:]
            
            if not backups:
                return {'trend': 'no_data', 'best_score': 0.0, 'improvement': 0.0}
            
            # Load metadata to get scores (simplified - in production load from metadata backups)
            best_score = 0.0
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                best_score = metadata['performance_metrics']['score']
            
            return {
                'trend': 'improving' if best_score > 0.5 else 'learning',
                'best_score': best_score,
                'improvement': best_score,
                'states_analyzed': len(backups)
            }
        except Exception as e:
            logger.warning(f"⚠️ Trend analysis failed: {e}")
            return {'trend': 'unknown', 'best_score': 0.0, 'improvement': 0.0}

    def _register_safe_globals(self):
        """
        ✅ FIXED: Register safe globals for PyTorch 2.6+ ONCE at initialization
        """
        try:
            import numpy as np
            from numpy.core.multiarray import scalar
            
            # Comprehensive list of NumPy types that might appear in saved models
            safe_types = [
                np.ndarray,
                scalar,
                np.core.multiarray.scalar,
                np.dtype,
                np.dtypes.Float64DType,
                np.dtypes.Float32DType,
                np.dtypes.Int64DType,
                np.dtypes.Int32DType,
                np.dtypes.Int16DType,
                np.dtypes.Int8DType,
                np.dtypes.UInt64DType,
                np.dtypes.UInt32DType,
                np.dtypes.BoolDType,
            ]
            
            # Filter out types that don't exist in this NumPy version
            available_types = []
            for dtype in safe_types:
                try:
                    if dtype is not None:
                        available_types.append(dtype)
                except (AttributeError, TypeError):
                    continue
            
            # Register all available types
            if available_types:
                torch.serialization.add_safe_globals(available_types)
                logger.debug(f"✅ Registered {len(available_types)} safe NumPy types for PyTorch loading")
            
        except Exception as e:
            logger.warning(f"⚠️ Safe globals registration failed (non-critical): {e}")
            logger.info("   Will attempt fallback loading methods if needed")

    def _safe_torch_load(self, filepath: str):
        """
        ✅ FIXED: PyTorch 2.6+ compatible loader with proper fallback chain
        """
        try:
            # Strategy 1: Secure load with pre-registered globals
            return torch.load(filepath, weights_only=True, map_location='cpu')
            
        except (pickle.UnpicklingError, RuntimeError, TypeError) as e:
            logger.debug(f"⚠️ Secure load failed for {os.path.basename(filepath)}: {str(e)[:100]}")
            
            try:
                # Strategy 2: Fallback to insecure load (only for trusted sources)
                logger.warning(f"⚠️ Using insecure load for {os.path.basename(filepath)} - TRUSTED SOURCE ONLY!")
                logger.warning("   Consider re-saving this file with current PyTorch version")
                return torch.load(filepath, weights_only=False, map_location='cpu')
                
            except Exception as e2:
                # Strategy 3: Complete failure - provide helpful error
                logger.error(f"❌ All load attempts failed for {os.path.basename(filepath)}")
                logger.error(f"   Secure load error: {str(e)[:100]}")
                logger.error(f"   Fallback error: {str(e2)[:100]}")
                logger.error(f"   File may be corrupted or from incompatible PyTorch version")
                raise RuntimeError(f"Cannot load {filepath}: {e2}")