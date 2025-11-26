import pandas as pd
import strategy_engine
import os

# --- UPDATE THIS PATH ---
DATA_FOLDER = "/home/shashank-vaibhav/Desktop/yms-assignment1/customdata_new/trading.end_time__15-30-00" 
# ------------------------

def main():
    print("1. Loading Data...")
    uni = strategy_engine.build_universe(DATA_FOLDER, ncores=4)
    if uni.empty:
        print("No data found!")
        return

    print("2. Running Strategy...")
    # Parameters: Entry Z=1.5, TakeProfit=0.5, StopLoss=2.0
    params = strategy_engine.RelZParams(entry=1.5, tp_off=0.5, stop_off=2.0)
    
    results = []
    for underlying, df in uni.groupby('underlying'):
        res = strategy_engine.simulate_mean_reversion((underlying, df), params)
        # Calculate Market % (Approximation)
        total_vol = df['ttq_FUT1_delta'].sum() + df['ttq_FUT2_delta'].sum()
        res['total_volume'] = total_vol
        res['market_perc'] = res['total_lots_traded'] / total_vol if total_vol > 0 else 0
        
        # Missing fields required by problem statement
        res['slippage_fut1'] = 0 # (Calculated inside net_pnl, usually split out)
        res['slippage_fut2'] = 0 
        
        results.append(res)
        
    df_res = pd.DataFrame(results)
    
    # Sort by Net PnL
    df_res = df_res.sort_values('net_pnl', ascending=False)
    
    # Save
    df_res.to_csv("Results.csv", index=False)
    print("Done! Saved Results.csv")
    print(df_res[['stock_name', 'net_pnl', 'total_lots_traded']])

if __name__ == "__main__":
    main()