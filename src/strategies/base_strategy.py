"""
Base strategy class.
Defines interface for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
from dataclasses import dataclass
import logging

from src.config import config


logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    action: str  # 'buy', 'sell', 'close', 'hold'
    confidence: float
    regime: str
    size: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'regime': self.regime,
            'size': self.size,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'reason': self.reason,
        }


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.min_confidence = config.min_confidence
        
        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        ml_signal: int,
        ml_confidence: float,
        regime: str,
        features: Dict,
        current_price: float
    ) -> Optional[Signal]:
        """
        Generate trading signal.
        
        Args:
            symbol: Trading symbol
            ml_signal: ML model signal (1, 0, -1)
            ml_confidence: ML model confidence
            regime: Current market regime
            features: Dictionary of features
            current_price: Current market price
            
        Returns:
            Signal object or None
        """
        pass
    
    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if valid
        """
        if signal.confidence < self.min_confidence:
            logger.debug(f"Signal rejected: confidence {signal.confidence:.3f} "
                        f"< minimum {self.min_confidence}")
            return False
        
        if signal.action not in ['buy', 'sell', 'close', 'hold']:
            logger.warning(f"Invalid signal action: {signal.action}")
            return False
        
        return True
