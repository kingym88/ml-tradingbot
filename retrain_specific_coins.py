#!/usr/bin/env python3
"""
Retrain specific coins with fresh data collection.
Usage: python3 retrain_specific_coins.py ETH ZEC
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.utils.logging_setup import setup_logging
from src.data.binance_collector import BinanceCollector
from src.data.data_manager import DataManager
from src.ml.trainer import ModelTrainer


def collect_and_train_coins(coin_list: list):
    """
    Collect data and train models for specific coins.
    
    Args:
        coin_list: List of coin symbols to process
    """
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info(f"Retraining {len(coin_list)} coins: {', '.join(coin_list)}")
    logger.info("=" * 60)
    
    # Initialize components
    collector = BinanceCollector()
    data_manager = DataManager()
    trainer = ModelTrainer()
    
    lookback_periods = config.get('LOOKBACK_PERIODS', 200)
    
    for i, coin in enumerate(coin_list, 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"[{i}/{len(coin_list)}] Processing {coin}")
        logger.info("=" * 60)
        
        # Step 1: Collect fresh data
        logger.info(f"Step 1/2: Collecting data for {coin}...")
        try:
            # Check existing data
            existing_data = data_manager.load_data(coin)
            
            if len(existing_data) < lookback_periods:
                logger.warning(f"Insufficient data for {coin}: {len(existing_data)} rows")
                logger.info(f"Fetching fresh {lookback_periods} periods...")
                
                # Fetch fresh data
                df = collector.fetch_ohlcv(coin, limit=lookback_periods)
                
                if df.empty:
                    logger.error(f"Failed to fetch data for {coin}")
                    continue
                
                # Save data
                data_manager.save_data(coin, df)
                logger.info(f"Saved {len(df)} rows for {coin}")
            else:
                logger.info(f"Existing data sufficient: {len(existing_data)} rows")
                logger.info("Updating with latest candles...")
                updated_data = collector.update_data(coin, existing_data)
                data_manager.save_data(coin, updated_data)
                logger.info(f"Updated to {len(updated_data)} rows")
        
        except Exception as e:
            logger.error(f"Error collecting data for {coin}: {e}")
            continue
        
        # Step 2: Train models
        logger.info(f"Step 2/2: Training models for {coin}...")
        try:
            results = trainer.train_single_coin(coin)
            
            if results:
                logger.info(f"✅ Successfully trained {len(results)} models for {coin}")
                for regime, metrics in results.items():
                    if 'error' not in metrics:
                        acc = metrics.get('accuracy', 0)
                        logger.info(f"  - {regime}: accuracy={acc:.3f}")
                    else:
                        logger.error(f"  - {regime}: {metrics['error']}")
            else:
                logger.error(f"❌ Failed to train models for {coin}")
        
        except Exception as e:
            logger.error(f"Error training {coin}: {e}")
            continue
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Retraining Complete!")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 retrain_specific_coins.py COIN1 COIN2 ...")
        print("Example: python3 retrain_specific_coins.py ETH ZEC")
        return 1
    
    coins = [coin.upper() for coin in sys.argv[1:]]
    collect_and_train_coins(coins)
    return 0


if __name__ == '__main__':
    sys.exit(main())
