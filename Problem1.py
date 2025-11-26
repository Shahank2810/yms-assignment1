import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import strategy_engine
import os

# --- UPDATE THIS PATH ---
DATA_FOLDER = "/home/shashank-vaibhav/Desktop/yms-assignment1/customdata_new/trading.end_time__15-30-00" 
# ------------------------

def main():
    print("1. Loading Data...")
    uni = strategy_engine.build_universe(DATA_FOLDER, ncores=4)
    if uni.empty:
        print("No data found! Check DATA_FOLDER path.")
        return

    print("2. Calculating Metrics...")
    # Calculate Volume Ratios (using the fixed cm_vol)
    # Using 15min deltas (approx) or raw volume
    uni['vol_ratio_cm_fut1'] = uni['cm_vol_delta'] / (uni['ttq_FUT1_delta'] + 1)
    uni['vol_ratio_fut1_fut2'] = uni['ttq_FUT1_delta'] / (uni['ttq_FUT2_delta'] + 1)
    
    # Days to Expiry (DTE) Approximation
    # We use 'fut_rank' implication. FUT1 is near. 
    # Just plotting against time or 'exp_month' for now as proxy.
    
    print("3. Generating PDF...")
    with PdfPages("Problem1_Analysis.pdf") as pdf:
        # A. Spreads
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.lineplot(data=uni, x='timestamp', y='spread_cm_fut1', hue='underlying', ax=axes[0])
        axes[0].set_title("CM-FUT1 Spread over Time")
        sns.lineplot(data=uni, x='timestamp', y='spread_fut1_fut2', hue='underlying', ax=axes[1])
        axes[1].set_title("FUT1-FUT2 Spread over Time")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # B. Volume Ratios
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.lineplot(data=uni, x='timestamp', y='vol_ratio_cm_fut1', hue='underlying', ax=axes[0])
        axes[0].set_title("Vol Ratio CM/FUT1")
        sns.lineplot(data=uni, x='timestamp', y='vol_ratio_fut1_fut2', hue='underlying', ax=axes[1])
        axes[1].set_title("Vol Ratio FUT1/FUT2")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # C. Distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.kdeplot(data=uni, x='spread_cm_fut1', hue='underlying', fill=True, ax=axes[0])
        axes[0].set_title("Dist. CM-FUT1 Spread")
        sns.kdeplot(data=uni, x='spread_fut1_fut2', hue='underlying', fill=True, ax=axes[1])
        axes[1].set_title("Dist. FUT1-FUT2 Spread")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print("Done! Saved Problem1_Analysis.pdf")

if __name__ == "__main__":
    main()