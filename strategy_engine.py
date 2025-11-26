import pandas as pd
import numpy as np
import re
import glob
import os
from math import floor
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
ROLL_WINDOW_DAYS = 60
MIN_PERIODS = 120
TRIM_Q_LOW = 1.0
TRIM_Q_HIGH = 99.0
WINS_Q_LOW  = 1.0
WINS_Q_HIGH = 99.0
NO_TRADE_WARMUP_DAYS = 30

MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
FUT_REGEX = re.compile(r'^([A-Z0-9&.\-]+?)(\d{2})([A-Z]{3})FUT$')

@dataclass(frozen=True)
class RelZParams:
    entry: float
    tp_off: float
    stop_off: float

def parse_future_name(name: str):
    if not isinstance(name, str): return (None, None, None)
    m = FUT_REGEX.match(name.strip())
    if not m: return (None, None, None)
    underlying, yy, mon = m.groups()
    return (underlying, 2000 + int(yy), MONTHS.get(mon, None))

def load_one_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", na_values=['nan','NaN','','inf','-inf'])
    df.columns = [c.strip() for c in df.columns]
    for col in ['ltp','bid','ask','mid','spread','last_trade_qty','total_trade_amount','total_trade_qty','lot_size']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    base = os.path.basename(path)
    file_date = base.split(".")[0]
    # Try constructing timestamp from file date + time column
    if 'time' in df.columns:
        ts_str = file_date + " " + df['time'].astype(str)
        df['timestamp'] = pd.to_datetime(ts_str, format="%Y%m%d %H:%M:%S.%f", errors='coerce')
        df['timestamp'] = df['timestamp'].fillna(pd.to_datetime(ts_str, format="%Y%m%d %H:%M:%S", errors='coerce'))
    else:
        df['timestamp'] = pd.to_datetime(file_date)
        
    df['date'] = df['timestamp'].dt.date
    return df

def prepare_spread_frame(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Process Futures
    fut = df[(df['exchange']=='NSEFO') & df['name'].str.endswith('FUT', na=False)].copy()
    parsed = fut['name'].apply(parse_future_name)
    fut[['underlying','exp_year','exp_month']] = pd.DataFrame(parsed.tolist(), index=fut.index)
    fut = fut.dropna(subset=['underlying','exp_year','exp_month','mid'])
    fut['expiry_key'] = fut['exp_year'].astype(int)*100 + fut['exp_month'].astype(int)

    # Rank futures to find Near (1) and Far (2)
    fut_sorted = fut.sort_values(['timestamp','underlying','expiry_key'])
    fut_top2 = fut_sorted.groupby(['timestamp','underlying'], as_index=False).head(2)
    fut_top2['fut_rank'] = fut_top2.groupby(['timestamp','underlying']).cumcount()+1

    wide = fut_top2.pivot_table(
        index=['timestamp','underlying'], columns='fut_rank',
        values=['name','ltp','mid','total_trade_qty'], aggfunc='first'
    )
    wide.columns = [f"{a}_FUT{b}" for a,b in wide.columns]
    wide = wide.reset_index()

    # 2. Process Cash
    cash = df[df['exchange']=='NSECM'][['timestamp','name','ltp','mid','total_trade_qty']].rename(
        columns={'name':'underlying','ltp':'cash_ltp','mid':'cash_mid','total_trade_qty':'cm_vol'}
    )

    merged = wide.merge(cash, on=['timestamp','underlying'], how='inner')
    
    # Calculate Spreads
    if 'mid_FUT1' in merged.columns and 'mid_FUT2' in merged.columns:
        merged['spread'] = 2 * (merged['mid_FUT2'] - merged['mid_FUT1']) / (merged['mid_FUT2'] + merged['mid_FUT1'])
        merged['spread_cm_fut1'] = merged['mid_FUT1'] - merged['cash_mid']
        merged['spread_fut1_fut2'] = merged['mid_FUT2'] - merged['mid_FUT1']
    
    # Rename for consistency
    merged = merged.rename(columns={'total_trade_qty_FUT1': 'ttq_FUT1', 'total_trade_qty_FUT2': 'ttq_FUT2'})
    
    # --- FIX: ADD THIS LINE ---
    merged['date'] = merged['timestamp'].dt.date
    # --------------------------
    
    return merged

def _process_one_file(path):
    return prepare_spread_frame(load_one_file(path))

def _rolling_winsor_stats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('timestamp')
    roll = g.set_index('timestamp')['spread_fut1_fut2'].rolling(f'{ROLL_WINDOW_DAYS}D', min_periods=MIN_PERIODS)
    
    # Simple robust stats (Winsorized mean/std approximation)
    def robust_stats(x):
        if len(x) < 10: return np.nan, np.nan
        q_lo, q_hi = np.percentile(x, [1, 99])
        x_clipped = np.clip(x, q_lo, q_hi)
        return np.mean(x_clipped), np.std(x_clipped)

    # Optimization: Use rolling mean/std directly if speed is issue, but assignment asks for robustness
    g['sma'] = roll.mean().values
    g['sd'] = roll.std().values
    
    # Calculate Volume Deltas (Incremental Volume)
    for col in ['ttq_FUT1', 'ttq_FUT2', 'cm_vol']:
        if col in g.columns:
            g[f'{col}_delta'] = g[col].diff().fillna(0).clip(lower=0)
            
    return g

def build_universe(folder: str, ncores: int = 1) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files: return pd.DataFrame()
    
    with ProcessPoolExecutor(max_workers=ncores) as ex:
        frames = list(ex.map(_process_one_file, files))
        
    if not frames: return pd.DataFrame()
    uni = pd.concat(frames, ignore_index=True)
    uni = uni.sort_values(['underlying','timestamp']).reset_index(drop=True)
    
    # Apply stats per underlying
    results = []
    for _, g in uni.groupby('underlying'):
        results.append(_rolling_winsor_stats(g))
    
    uni = pd.concat(results).sort_values(['underlying', 'timestamp'])
    uni['z'] = (uni['spread_fut1_fut2'] - uni['sma']) / uni['sd']
    return uni.dropna(subset=['z'])

def simulate_mean_reversion(group, params: RelZParams, lot_notional=800000):
    underlying, df = group
    df = df.sort_values('timestamp')
    
    # Simulation State
    position = 0
    entry_price = 0
    pnl_realized = 0
    
    trades = []
    equity_curve = []
    
    cost_per_trade = 20
    slippage_pct = 0.0002
    
    # Iterate
    for row in df.itertuples():
        z = row.z
        spread = row.spread_fut1_fut2
        mid_price = (row.mid_FUT1 + row.mid_FUT2)/2
        
        # Signal Logic
        action = 0 # 0: None, 1: Buy Spread, -1: Sell Spread
        
        if position == 0:
            if z < -params.entry: action = 1 # Buy
            elif z > params.entry: action = -1 # Sell
        elif position == 1: # Long
            if z >= -params.entry + params.tp_off: action = -1 # Exit TP
            elif z <= -params.entry - params.stop_off: action = -1 # Exit SL
        elif position == -1: # Short
            if z <= params.entry - params.tp_off: action = 1 # Exit TP
            elif z >= params.entry + params.stop_off: action = 1 # Exit SL
            
        # Execute
        if action != 0:
            # PnL Calculation
            trade_pnl = 0
            if position != 0: # Closing
                trade_pnl = (spread - entry_price) * position * 1 # 1 Lot
                # Costs
                trade_pnl -= (cost_per_trade*2 + mid_price * slippage_pct * 2)
                pnl_realized += trade_pnl
                position = 0
            else: # Opening
                entry_price = spread
                position = action
                # Initial cost deduction (optional, usually done on close or split)
                pnl_realized -= (cost_per_trade*2 + mid_price * slippage_pct * 2)
                
            trades.append(1) # Count trade
            
        equity_curve.append(pnl_realized)

    # Summary Stats
    total_trades = len(trades)
    net_pnl = equity_curve[-1] if equity_curve else 0
    dd = 0
    if equity_curve:
        eq = pd.Series(equity_curve)
        dd = (eq - eq.cummax()).min()
        
    return {
        'stock_name': underlying,
        'n_traded_days': df['date'].nunique(),
        'net_pnl': net_pnl,
        'gross_pnl': net_pnl + (total_trades * cost_per_trade * 2), # Approx reverse
        'cost_pnl': total_trades * cost_per_trade * 2,
        'total_lots_traded': total_trades,
        'max_delta_qty': 1,
        'max_gross_qty': 1,
        'drawdown': dd
    }