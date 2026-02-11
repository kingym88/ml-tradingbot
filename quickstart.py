#!/usr/bin/env python3
"""
Quick start script to verify installation and test basic functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.utils.logging_setup import setup_logging
from src.data.binance_collector import BinanceCollector
from src.features.feature_pipeline import FeaturePipeline
from src.regime.regime_classifier import RegimeClassifier


def main():
    """Run quick start verification."""
    print("=" * 60)
    print("Hyperliquid ML Trading Bot - Quick Start")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting quick start verification...")
    
    # Test 1: Configuration
    print("\n[1/5] Testing configuration...")
    print(f"  - Coins: {len(config.coin_list)}")
    print(f"  - Max positions: {config.max_positions}")
    print(f"  - Min confidence: {config.min_confidence}")
    print(f"  - Live trading: {config.enable_live_trading}")
    print(f"  - Testnet: {config.use_testnet}")
    print("  [OK] Configuration loaded successfully")
    
    # Test 2: Data Collection
    print("\n[2/5] Testing data collection...")
    collector = BinanceCollector()
    test_coin = config.coin_list[0]
    print(f"  - Fetching sample data for {test_coin}...")
    df = collector.fetch_ohlcv(test_coin, limit=100)
    if not df.empty:
        print(f"  - Fetched {len(df)} candles")
        print(f"  - Latest price: {df['close'].iloc[-1]:.2f}")
        print("  [OK] Data collection working")
    else:
        print("  [WARNING] Could not fetch data (check internet connection)")
    
    # Test 3: Feature Engineering
    print("\n[3/5] Testing feature engineering...")
    if not df.empty:
        feature_pipeline = FeaturePipeline()
        df_features = feature_pipeline.compute_features(df)
        print(f"  - Computed {len(df_features.columns)} features")
        print(f"  - Sample features: {list(df_features.columns[:5])}")
        print("  [OK] Feature engineering working")
    else:
        print("  [SKIPPED] No data available")
    
    # Test 4: Regime Detection
    print("\n[4/5] Testing regime detection...")
    if not df.empty:
        regime_classifier = RegimeClassifier()
        df_regimes = regime_classifier.classify_regimes(df_features)
        current_regime, confidence = regime_classifier.get_current_regime(df_regimes)
        print(f"  - Current regime: {current_regime}")
        print(f"  - Confidence: {confidence:.3f}")
        stats = regime_classifier.get_regime_statistics(df_regimes)
        print(f"  - Regime distribution: {stats['regime_percentages']}")
        print("  [OK] Regime detection working")
    else:
        print("  [SKIPPED] No data available")
    
    # Test 5: Model Directory
    print("\n[5/5] Checking model directory...")
    model_dir = config.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - Model directory: {model_dir}")
    print(f"  - Directory exists: {model_dir.exists()}")
    print("  [OK] Model directory ready")
    
    # Summary
    print("\n" + "=" * 60)
    print("Quick Start Verification Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Configure your API keys in config/.env")
    print("2. Review and customize config/settings.yaml")
    print("3. Train initial models: python3 main.py --train-only")
    print("4. Run in testnet mode: python3 main.py")
    print("\nFor more information, see README.md")
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
