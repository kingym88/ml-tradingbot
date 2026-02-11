
import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = "data/trades.db"

def run_report():
    if not os.path.exists(DB_PATH):
        print("❌ No database found. Run the bot or backfill first.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # query recent trades with AI data
    query = """
    SELECT 
        id, 
        timestamp,
        symbol, 
        side, 
        entry_price, 
        quantity, 
        leverage, 
        regime, 
        confidence, 
        strategy_name,
        stop_loss_price as SL, 
        take_profit_price as TP,
        realized_pnl,
        is_backfilled
    FROM trades 
    ORDER BY timestamp DESC
    LIMIT 20
    """
    
    df = pd.read_sql_query(query, conn)
    
    print("\n📊 RECENT TRADES PERFORMANCE TRACKER")
    print("=" * 100)
    
    if df.empty:
        print("No trades found.")
        return

    # Check if realized_pnl exists in dataframe columns (in case of old DB)
    has_pnl = 'realized_pnl' in df.columns

    # Format output
    for _, row in df.iterrows():
        # Status icon
        icon = "📝" if row['is_backfilled'] else "🤖"
        
        # Format Strategy info
        if row['is_backfilled']:
            ai_info = "Unknown (Backfilled)"
        else:
            conf_str = f"{row['confidence']*100:.0f}%" if row['confidence'] else "N/A"
            ai_info = f"{row['regime']} | {conf_str} | {row['strategy_name']}"
        
        # Format Side
        side = row['side'].upper()
        side_color = "🟢" if side == "BUY" else "🔴"
        
        # PnL String
        pnl_str = ""
        if has_pnl and row['realized_pnl']:
            pnl_val = row['realized_pnl']
            if pnl_val > 0:
                pnl_str = f" | 💰 PnL: +${pnl_val:.2f}"
            elif pnl_val < 0:
                pnl_str = f" | 💸 PnL: -${abs(pnl_val):.2f}"
            else:
                pnl_str = " | PnL: $0.00"
        
        print(f"{icon} {row['timestamp']} | {side_color} {row['symbol']:<5} {side:<4} | Price: ${row['entry_price']:.4f} | Lev: {row['leverage']}x{pnl_str}")
        print(f"   Context:  {ai_info}")
        if row['SL'] and row['TP']:
            print(f"   Risk:     SL: ${row['SL']:.4f} | TP: ${row['TP']:.4f}")
        print("-" * 100)

    print(f"\nTotal Trades Tracked: {pd.read_sql_query('SELECT COUNT(*) FROM trades', conn).iloc[0,0]}")
    conn.close()

if __name__ == "__main__":
    run_report()
