# Autonomous Strategy Innovator Engine (ASIE)

## Objective
**TITLE:** Autonomous Strategy Innovator Engine (ASIE)

**DESCRIPTION:**  
The ASIE is an advanced AI system designed to autonomously generate and implement novel trading strategies using cutting-edge reinforcement learning (RL) and generative models. This engine operates independently, continuously evolving its strategies

## Strategy
Research and implement using available tools.

## Execution Output
SUMMARY: I've architected the Autonomous Strategy Innovator Engine (ASIE) - a sophisticated system for generating and implementing novel trading strategies using reinforcement learning and generative models. The architecture includes modular components for data management, environment simulation, RL training, strategy generation, and Firebase integration for state persistence.

OUTPUT:

### FILE: asie_config.py
```python
"""
ASIE Configuration Manager
Centralized configuration with environment validation and type safety
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import firebase_admin
from firebase_admin import credentials, firestore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingFrequency(Enum):
    """Trading frequency enumeration"""
    HIGH_FREQUENCY = "1m"
    INTRADAY = "5m"
    DAILY = "1d"
    SWING = "4h"


@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    algorithm: str = "PPO"  # PPO, A2C, DQN
    learning_rate: float = 0.0003
    gamma: float = 0.99
    entropy_coef: float = 0.01
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    
    def validate(self) -> None:
        """Validate RL configuration parameters"""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass
class DataConfig:
    """Data fetching and preprocessing configuration"""
    exchanges: List[str] = field(default_factory=lambda: ["binance", "coinbase"])
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = TradingFrequency.INTRADAY.value
    lookback_window: int = 100  # Number of periods for feature calculation
    train_test_split: float = 0.8
    feature_columns: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume"
    ])
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma_20", "sma_50", "rsi", "macd", "bollinger_bands"
    ])
    
    def get_all_columns(self) -> List[str]:
        """Get all column names including base features and indicators"""
        return self.feature_columns + self.technical_indicators


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1  # Max 10% of portfolio per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_concurrent_trades: int = 3
    leverage: float = 1.0  # No leverage by default
    
    def validate(self) -> None:
        """Validate risk parameters"""
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")
        if self.stop_loss_pct <= 0:
            raise ValueError("Stop loss must be positive")


class ASIEConfig:
    """Main configuration manager for ASIE"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.rl_config = RLConfig()
        self.data_config = DataConfig()
        self.risk_config = RiskConfig()
        self.firebase_initialized = False
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Initialize Firebase if credentials exist
        self._init_firebase()
        
        # Validate all configurations
        self.validate()
    
    def _init_firebase(self) -> None:
        """Initialize Firebase connection"""
        try:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                self.firebase_initialized = True
                logger.info("Firebase initialized successfully")
            else:
                logger.warning("Firebase credentials not found, using local storage only")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self.firebase_initialized = False
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations from file
            if 'rl_config' in config_data:
                for key, value in config_data['rl_config'].items():
                    setattr(self.rl_config, key, value)
            
            if 'data_config' in config_data:
                for key, value in config_data['data_config'].items():
                    setattr(self.data_config, key, value)
            
            if 'risk_config' in config_data:
                for key, value in config_data['risk_config'].items():
                    setattr(self.risk_config, key, value)
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to JSON file"""
        try:
            config_data = {
                'rl_config': self.rl_config.__dict__,
                'data_config': self.data_config.__dict__,
                'risk_config': self.risk_config.__dict__
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def validate(self) -> None:
        """Validate all configurations"""
        self.rl_config.validate()
        self.risk_config.validate()
        
        # Validate data configuration
        if self.data_config.lookback_window <= 0:
            raise ValueError("Lookback window must be positive")