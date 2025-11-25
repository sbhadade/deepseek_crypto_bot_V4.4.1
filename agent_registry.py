"""
AGENT REGISTRY - Persistent storage for elite agents
✅ ENHANCED: Saves complete agent state including neural network weights
✅ FIXED: Perfect cloning capability for next generation
✅ Complete implementation - ready to use
"""

import pickle
import json
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from collections import defaultdict, deque
import logging
import os
import glob

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Persistent storage for elite agents across generations"""
    
    def __init__(self, registry_file: str = "data/elite_agents/registry.pkl", max_elite_per_regime: int = 20):
        self.registry_file = registry_file
        self.max_elite_per_regime = max_elite_per_regime
        self.regime_specialists = defaultdict(list)
        self.agent_performance_db = {}
        
        # Create directory structure
        os.makedirs(os.path.dirname(registry_file), exist_ok=True)
        
        # Load existing registry
        self.load_registry()
        
        logger.info("🏆 Agent Registry initialized")
    
    def save_elite_agent(self, agent, regime: str, market_conditions: Dict = None) -> bool:
        """
        ✅ FIXED: Save complete agent state including learned parameters
        
        CRITICAL FIXES:
        1. Serialize full DNA parameters (not just metadata)
        2. Save agent's learned model weights if they exist
        3. Store complete performance history
        4. Enable perfect cloning in next generation
        """
        try:
            agent_id = agent.dna.agent_id
            
            # ✅ Calculate comprehensive fitness
            if agent.dna.total_trades > 0:
                win_rate = agent.dna.winning_trades / agent.dna.total_trades
                avg_pnl = agent.dna.total_pnl / agent.dna.total_trades
                fitness = agent.dna.fitness_score if hasattr(agent.dna, 'fitness_score') else 0.0
            else:
                return False
            
            # ✅ Only save truly elite agents
            # ✅ REPLACE WITH progressive thresholds:
            if agent.dna.total_trades < 3:
                min_fitness = 15.0  # Low bar for new agents
            elif agent.dna.total_trades < 10:
                min_fitness = 25.0  # Medium bar for learning agents
            else:
                min_fitness = 40.0  # High bar for veterans
            
            if fitness < min_fitness:
                return False
            
            # ✅ Serialize complete DNA (all parameters)
            dna_dict = {
                'agent_id': agent.dna.agent_id,
                'generation': agent.dna.generation,
                'timeframe': agent.dna.timeframe,
                'min_confidence': agent.dna.min_confidence,
                'min_win_prob': agent.dna.min_win_prob,
                'volatility_z_threshold': agent.dna.volatility_z_threshold,
                'position_size_base': agent.dna.position_size_base,
                'risk_reward_threshold': agent.dna.risk_reward_threshold,
                'expected_value_threshold': agent.dna.expected_value_threshold,
                'stop_loss_distance': agent.dna.stop_loss_distance,
                'take_profit_distance': agent.dna.take_profit_distance,
                'trailing_stop_activation': agent.dna.trailing_stop_activation,
                'min_holding_minutes': agent.dna.min_holding_minutes,
                'max_holding_hours': agent.dna.max_holding_hours,
                'aggression': agent.dna.aggression,
                'patience': agent.dna.patience,
                'contrarian_bias': agent.dna.contrarian_bias,
                'loss_aversion': agent.dna.loss_aversion,
                'asset_preference': agent.dna.asset_preference,
                'regime_preference': agent.dna.regime_preference,
                'parent_ids': agent.dna.parent_ids if hasattr(agent.dna, 'parent_ids') else [],
                'regime_performance': agent.dna.regime_performance if hasattr(agent.dna, 'regime_performance') else {},
                'total_trades': agent.dna.total_trades,
                'winning_trades': agent.dna.winning_trades,
                'total_pnl': agent.dna.total_pnl,
                'fitness_score': fitness
            }
            
            # ✅ NEW: Save learned model weights if agent has neural network
            model_weights = None
            if hasattr(agent, 'policy_net'):
                try:
                    model_weights = {
                        'policy_net': agent.policy_net.state_dict(),
                        'value_net': agent.value_net.state_dict() if hasattr(agent, 'value_net') else None
                    }
                    logger.debug(f"   💾 Saved neural network weights for agent {agent_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not save model weights: {e}")
            
            # ✅ Create elite agent record with full state
            elite_record = {
                'agent_id': agent_id,
                'dna': dna_dict,
                'model_weights': model_weights,  # ✅ NEW: Neural network weights
                'fitness': fitness,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_trades': agent.dna.total_trades,
                'total_pnl': agent.dna.total_pnl,
                'regime': regime,
                'market_conditions': market_conditions or {},
                'timestamp': datetime.now().isoformat(),
                'timeframe': agent.dna.timeframe,
                'balance': agent.balance,
                'preferred_leverage': agent.preferred_leverage if hasattr(agent, 'preferred_leverage') else 5.0,
                'trade_history_sample': [
                    {
                        'asset': t.asset,
                        'action': t.action,
                        'pnl': t.pnl,
                        'leverage': t.leverage if hasattr(t, 'leverage') else 1.0,
                        'holding_hours': t.realized_holding_hours if hasattr(t, 'realized_holding_hours') else 0.0
                    }
                    for t in agent.trade_history[-10:] if hasattr(agent, 'trade_history')
                ]
            }
            
            # ✅ Update regime specialists
            if regime not in self.regime_specialists:
                self.regime_specialists[regime] = []
            
            # ✅ Remove old version of same agent if exists
            self.regime_specialists[regime] = [
                a for a in self.regime_specialists[regime] 
                if a['agent_id'] != agent_id
            ]
            
            self.regime_specialists[regime].append(elite_record)
            
            # ✅ Keep only top performers per regime (sorted by fitness)
            self.regime_specialists[regime].sort(key=lambda x: x['fitness'], reverse=True)
            self.regime_specialists[regime] = self.regime_specialists[regime][:self.max_elite_per_regime]
            
            # ✅ Update global performance database
            self.agent_performance_db[agent_id] = elite_record
            
            # ✅ Save to disk
            self._save_to_disk()
            
            logger.info(f"💾 Elite agent {agent_id} saved: Fitness={fitness:.1f}, WR={win_rate*100:.1f}%, Regime={regime}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save elite agent {agent.dna.agent_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_to_disk(self):
        """
        ✅ NEW: Persist elite agents to disk with rotation
        """
        try:
            save_data = {
                'regime_specialists': dict(self.regime_specialists),
                'agent_performance_db': self.agent_performance_db,
                'timestamp': datetime.now().isoformat()
            }
            
            # Create timestamped backup
            save_dir = os.path.dirname(self.registry_file)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{save_dir}/registry_backup_{timestamp}.pkl"
            
            # Save current state
            with open(self.registry_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Save backup
            with open(backup_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Keep only last 5 backups
            backups = sorted(glob.glob(f"{save_dir}/registry_backup_*.pkl"))
            for old_backup in backups[:-5]:
                os.remove(old_backup)
            
            logger.debug(f"💾 Registry saved to {self.registry_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save to disk: {e}")
    
    def load_registry(self) -> bool:
        """
        ✅ ENHANCED: Load elite agents from disk with fallback
        """
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'rb') as f:
                    save_data = pickle.load(f)
                
                self.regime_specialists = defaultdict(list, save_data.get('regime_specialists', {}))
                self.agent_performance_db = save_data.get('agent_performance_db', {})
                
                total_agents = sum(len(agents) for agents in self.regime_specialists.values())
                logger.info(f"📂 Loaded {total_agents} elite agents from registry")
                
                # Log per-regime stats
                for regime, agents in self.regime_specialists.items():
                    if agents:
                        best = agents[0]
                        logger.info(f"   {regime}: {len(agents)} agents (best: ID {best['agent_id']}, fitness {best['fitness']:.1f})")
                
                return True
            else:
                logger.info("📂 No existing registry found, starting fresh")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load registry: {e}")
            
            # Try to load from backup
            try:
                save_dir = os.path.dirname(self.registry_file)
                backups = sorted(glob.glob(f"{save_dir}/registry_backup_*.pkl"))
                
                if backups:
                    latest_backup = backups[-1]
                    logger.info(f"📂 Loading from backup: {latest_backup}")
                    
                    with open(latest_backup, 'rb') as f:
                        save_data = pickle.load(f)
                    
                    self.regime_specialists = defaultdict(list, save_data.get('regime_specialists', {}))
                    self.agent_performance_db = save_data.get('agent_performance_db', {})
                    
                    logger.info("✅ Successfully loaded from backup")
                    return True
            except Exception as backup_error:
                logger.error(f"❌ Backup load also failed: {backup_error}")
            
            # Initialize empty registry
            self.regime_specialists = defaultdict(list)
            self.agent_performance_db = {}
            return False
    
    def get_regime_specialists(self, regime: str, min_fitness: float = 0.0, limit: int = 5) -> List[Dict]:
        """Get top specialists for a specific regime"""
        specialists = self.regime_specialists.get(regime, [])
        filtered = [agent for agent in specialists if agent['fitness'] >= min_fitness]
        return filtered[:limit]
    
    def get_best_agent_for_conditions(self, regime: str, timeframe: str = None, min_trades: int = 10) -> Optional[Dict]:
        """Get best-performing agent for current market conditions"""
        specialists = self.regime_specialists.get(regime, [])
        
        if not specialists:
            return None
        
        # Filter by criteria
        candidates = [
            agent for agent in specialists 
            if agent['total_trades'] >= min_trades and 
            (timeframe is None or agent.get('timeframe') == timeframe)
        ]
        
        if not candidates:
            return None
            
        # Return highest fitness agent
        return max(candidates, key=lambda x: x['fitness'])
    
    def get_ensemble_agents(self, regime: str, count: int = 3) -> List[Dict]:
        """Get multiple top agents for ensemble decision making"""
        specialists = self.regime_specialists.get(regime, [])
        return specialists[:count]
    
    def save_registry(self) -> bool:
        """Legacy method - calls _save_to_disk()"""
        self._save_to_disk()
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            'total_agents': len(self.agent_performance_db),
            'regimes_covered': len(self.regime_specialists),
            'agents_per_regime': {},
            'top_agents': {}
        }
        
        for regime, agents in self.regime_specialists.items():
            stats['agents_per_regime'][regime] = len(agents)
            if agents:
                stats['top_agents'][regime] = {
                    'best_fitness': agents[0]['fitness'],
                    'best_win_rate': agents[0]['win_rate'],
                    'best_agent_id': agents[0]['agent_id']
                }
        
        return stats
    
    def clear_registry(self) -> bool:
        """Clear the entire registry"""
        try:
            self.regime_specialists = defaultdict(list)
            self.agent_performance_db = {}
            
            if os.path.exists(self.registry_file):
                os.remove(self.registry_file)
            
            # Clear backups
            save_dir = os.path.dirname(self.registry_file)
            backups = glob.glob(f"{save_dir}/registry_backup_*.pkl")
            for backup in backups:
                os.remove(backup)
                
            logger.info("🧹 Agent registry cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear registry: {e}")
            return False
    
    def get_agent_performance_history(self, agent_id: int) -> Optional[Dict]:
        """Get performance history for a specific agent"""
        return self.agent_performance_db.get(agent_id)
    
    def get_all_regimes(self) -> List[str]:
        """Get list of all regimes with stored specialists"""
        return list(self.regime_specialists.keys())
    
    def is_agent_elite(self, agent_id: int, regime: str = None) -> bool:
        """Check if an agent is in the elite registry"""
        if regime:
            specialists = self.regime_specialists.get(regime, [])
            return any(agent['agent_id'] == agent_id for agent in specialists)
        else:
            return agent_id in self.agent_performance_db
    
    def clone_elite_agent(self, agent_id: int):
        """
        ✅ NEW: Clone an elite agent with full state restoration
        
        Returns:
            LeveragedEvolutionaryAgent with restored parameters and weights
        """
        try:
            if agent_id not in self.agent_performance_db:
                logger.warning(f"⚠️ Agent {agent_id} not found in registry")
                return None
            
            elite_record = self.agent_performance_db[agent_id]
            
            # Reconstruct DNA
            from evolutionary_paper_trading_2 import AgentDNA
            from evolutionary_paper_trading_leverage import LeveragedEvolutionaryAgent
            
            dna_dict = elite_record['dna'].copy()
            dna = AgentDNA(**dna_dict)
            
            # Create agent
            agent = LeveragedEvolutionaryAgent(dna, elite_record['balance'])
            
            # ✅ Restore neural network weights if they exist
            if elite_record.get('model_weights') and hasattr(agent, 'policy_net'):
                try:
                    agent.policy_net.load_state_dict(elite_record['model_weights']['policy_net'])
                    if elite_record['model_weights'].get('value_net') and hasattr(agent, 'value_net'):
                        agent.value_net.load_state_dict(elite_record['model_weights']['value_net'])
                    logger.info(f"✅ Restored neural network weights for agent {agent_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not restore model weights: {e}")
            
            logger.info(f"🧬 Cloned elite agent {agent_id}: Fitness={elite_record['fitness']:.1f}")
            return agent
            
        except Exception as e:
            logger.error(f"❌ Failed to clone agent {agent_id}: {e}")
            import traceback
            traceback.print_exc()
            return None