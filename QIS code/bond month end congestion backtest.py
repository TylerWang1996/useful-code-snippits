import pandas as pd
import numpy as np
from datetime import datetime
#from blp import blp  # Custom package by Matthew Gilbert :contentReference[oaicite:0]{index=0}
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Data retrieval and cleaning

def fetch_data(tickers, start_date, end_date):
    """
    Pull historical data for the given tickers.
    Fields: PX_Bid, PX_Ask, PX_Last.
    Returns a multi-index DataFrame (index: date, outer level of columns: ticker,
    inner level: field).
    """
    bquery = blp.BlpQuery().start()
    fields = ['PX_Bid', 'PX_Ask', 'PX_Last']
    data = bquery.bdh(tickers, fields, start_date=start_date, end_date=end_date, options={})
    data_pivot = data.pivot(index='date', columns='security')
    data_pivot.index = pd.to_datetime(data_pivot.index)
    # Swap levels so that the outer level is ticker and inner level is field.
    data_pivot = data_pivot.swaplevel(axis=1)
    # Sort by ticker then by field.
    data_pivot.sort_index(axis=1, level=0, inplace=True)
    data_pivot.sort_index(axis=1, level=1, inplace=True)
    return data_pivot

def clean_data(data):
    """Forward-fill missing data."""
    return data.ffill()

# --------------------------------------------------
# 2. Identify trade days per month

def get_month_end_trade_days(df):
    """
    Given a DataFrame with a DateTime index (daily trading days),
    return a list of tuples (T_minus_4, T_minus_3, T_minus_2, T_minus_1, T)
    for each month where T is the last trading day.
    """
    # Use 'ME' (Month End) instead of 'M'
    month_ends = df.index.to_series().resample('ME').last()
    trade_periods = []
    dates = df.index.sort_values()
    for T in month_ends:
        pos = dates.get_loc(T)
        if pos < 4:
            continue
        T_minus_4 = dates[pos - 4]
        T_minus_3 = dates[pos - 3]
        T_minus_2 = dates[pos - 2]
        T_minus_1 = dates[pos - 1]
        trade_periods.append((T_minus_4, T_minus_3, T_minus_2, T_minus_1, T))
    return trade_periods


# --------------------------------------------------
# 3. Compute 3-month lookback volatility

def compute_volatility(mid_series, vol_start, vol_end):
    """
    Compute annualized volatility from mid price series between vol_start and vol_end.
    Use daily log returns and annualize by sqrt(252).
    """
    period_data = mid_series.loc[vol_start:vol_end]
    if len(period_data) < 2:
        return np.nan
    returns = np.log(period_data / period_data.shift(1)).dropna()
    daily_vol = returns.std()
    ann_vol = daily_vol * np.sqrt(252)
    return ann_vol

# --------------------------------------------------
# 4. Simulate daily returns for one asset

def simulate_daily_returns_for_asset(asset_df, target_vol=0.15):
    """
    For an asset (DataFrame with columns: PX_Bid, PX_Ask, PX_Last; index: trading days),
    simulate daily strategy returns based on the congestion trade.
    Returns a Series of daily returns.
    """
    df = asset_df.copy().sort_index()
    # Compute mid price for volatility estimation.
    df['Mid'] = (df['PX_Bid'] + df['PX_Ask']) / 2

    # Get trade periods (tuples of T-4, T-3, T-2, T-1, T) using the PX_Last column.
    trade_periods = get_month_end_trade_days(df)
    
    # Initialize dictionary for daily return contributions.
    daily_returns = {date: 0.0 for date in df.index}

    for period in trade_periods:
        T_m4, T_m3, T_m2, T_m1, T_exit = period
        dates = df.index.sort_values()
        pos_exit = dates.get_loc(T_exit)
        if pos_exit < 5:
            continue
        T_vol_end = dates[pos_exit - 5]
        vol_start = T_exit - pd.DateOffset(months=3)
        vol = compute_volatility(df['Mid'], vol_start, T_vol_end)
        if pd.isna(vol) or vol == 0:
            continue
        # Volatility-targeted notional size.
        N = target_vol / vol
        
        # Define legs: (entry_date, holding_days)
        leg_info = [
            (T_m4, 4),
            (T_m3, 3),
            (T_m2, 2),
            (T_m1, 1)
        ]

        for entry_date, holding_days in leg_info:
            try:
                entry_price = df.loc[entry_date, 'PX_Ask']
                exit_price = df.loc[T_exit, 'PX_Bid']
            except KeyError:
                continue
            R_leg = (exit_price - entry_price) / entry_price
            daily_r = (1 + R_leg) ** (1 / holding_days) - 1
            leg_weight = 0.25 * N

            dates_list = df.index.sort_values()
            pos_entry = dates_list.get_loc(entry_date)
            pos_exit = dates_list.get_loc(T_exit)
            if pos_exit - pos_entry != holding_days:
                continue
            for pos in range(pos_entry + 1, pos_exit + 1):
                day = dates_list[pos]
                daily_returns[day] += leg_weight * daily_r

    daily_return_series = pd.Series(daily_returns).sort_index()
    return daily_return_series

# --------------------------------------------------
# 5. Backtest for multiple assets (and combine portfolio)

def backtest_daily_strategy(data, tickers, target_vol=0.15):
    """
    For each ticker, simulate the daily returns.
    Then combine asset returns into an equal-weighted portfolio.
    Returns:
      - A dict mapping ticker to its daily return Series.
      - A combined portfolio DataFrame with daily returns and a wealth index.
    """
    asset_daily_returns = {}
    for ticker in tickers:
        try:
            df_ticker = data[ticker].dropna()
        except KeyError:
            continue
        df_ticker = df_ticker.sort_index()
        df_ticker = clean_data(df_ticker)
        dr = simulate_daily_returns_for_asset(df_ticker, target_vol=target_vol)
        asset_daily_returns[ticker] = dr

    combined_df = None
    for ticker, series in asset_daily_returns.items():
        s = series.rename(ticker)
        if combined_df is None:
            combined_df = s.to_frame()
        else:
            combined_df = combined_df.join(s, how='outer')
    if combined_df is None:
        combined_df = pd.DataFrame()
    else:
        combined_df = combined_df.sort_index().fillna(0)
        combined_df['PortfolioReturn'] = combined_df.mean(axis=1)
        combined_df['Wealth'] = (1 + combined_df['PortfolioReturn']).cumprod()
    return asset_daily_returns, combined_df

# --------------------------------------------------
# 6. Simulated Data Generator (for testing)

def simulate_bloomberg_data(tickers, sim_start_date, sim_end_date):
    """
    Simulates Bloomberg historical data for given tickers.
    The simulated data covers a larger history (from sim_start_date to sim_end_date)
    so that the backtest period (a subset) has sufficient lookback.
    Generates daily trading data (business days) with columns: PX_Bid, PX_Ask, PX_Last.
    The returned DataFrame has a MultiIndex on columns where the outer level is the ticker
    and inner level is the field.
    """
    dates = pd.bdate_range(start=sim_start_date, end=sim_end_date)
    simulated_dfs = []
    for ticker in tickers:
        np.random.seed(abs(hash(ticker)) % 2**32)
        daily_returns = np.random.normal(0, 0.001, len(dates))
        prices = 100 * np.cumprod(1 + daily_returns)
        spread = 0.1
        bid_prices = prices - spread/2
        ask_prices = prices + spread/2
        df = pd.DataFrame({
            'PX_Bid': bid_prices,
            'PX_Ask': ask_prices,
            'PX_Last': prices,
            'security': ticker
        }, index=dates)
        simulated_dfs.append(df.reset_index().rename(columns={'index': 'date'}))
    
    combined = pd.concat(simulated_dfs, ignore_index=True)
    pivoted = combined.pivot(index='date', columns='security')
    pivoted.index = pd.to_datetime(pivoted.index)
    # Swap levels so that outer level is ticker, inner level is field.
    pivoted = pivoted.swaplevel(axis=1)
    pivoted.sort_index(axis=1, level=0, inplace=True)
    pivoted.sort_index(axis=1, level=1, inplace=True)
    return pivoted

# --------------------------------------------------
# 7. Main routine for real data

def main():
    # User parameters for real data:
    tickers = ['BOND1 US Equity', 'BOND2 US Equity']  # Replace with your actual tickers.
    
    # Date range to fetch full data history (sufficient for volatility lookback).
    data_start = '2022-01-01'
    data_end   = '2023-12-31'
    
    # Backtest period (a subset of the fetched data).
    backtest_start = '2023-01-01'
    backtest_end   = '2023-07-31'
    
    target_vol = 0.15  # Target annual volatility (15%)
    
    # Fetch and clean full data history from Bloomberg.
    data = fetch_data(tickers, data_start, data_end)
    data = clean_data(data)
    
    # Restrict data to the backtest period.
    data = data.loc[backtest_start:backtest_end]
    
    # Backtest daily strategy for each asset and combine portfolio daily returns.
    asset_returns_dict, portfolio_df = backtest_daily_strategy(data, tickers, target_vol=target_vol)
    
    # Print sample individual asset daily returns.
    for ticker, series in asset_returns_dict.items():
        print(f"Sample daily returns for {ticker}:")
        print(series.head(10))
        print("\n")
    
    # Print combined portfolio daily returns and wealth series.
    if not portfolio_df.empty:
        print("Combined portfolio daily returns and wealth series (sample):")
        print(portfolio_df[['PortfolioReturn', 'Wealth']].head(15))
        portfolio_df['Wealth'].plot(title="Portfolio Wealth Index (Daily)")
        plt.show()
    else:
        print("No trade periods were generated; please check the data period or trade logic.")

# --------------------------------------------------
# 8. Test Case using simulated data

def test_simulated_data():
    # Define simulation period (full data history) and backtest period.
    tickers = ['BOND1 US Equity', 'BOND2 US Equity']
    sim_start_date = '2022-01-01'
    sim_end_date = '2023-12-31'
    backtest_start = '2023-01-01'
    backtest_end = '2023-07-31'
    target_vol = 0.15
    
    simulated_data = simulate_bloomberg_data(tickers, sim_start_date, sim_end_date)
    # For backtest, filter the simulated data to the backtest period.
    simulated_data = simulated_data.loc[backtest_start:backtest_end]
    
    print("Simulated Bloomberg data (head):")
    print(simulated_data.head())
    
    asset_returns_dict, portfolio_df = backtest_daily_strategy(simulated_data, tickers, target_vol=target_vol)
    
    for ticker, series in asset_returns_dict.items():
        print(f"\nSample daily returns for {ticker}:")
        print(series.head(30))
    
    if not portfolio_df.empty:
        print("\nCombined portfolio daily returns and wealth series (head):")
        print(portfolio_df[['PortfolioReturn', 'Wealth']].head(30))
        portfolio_df['Wealth'].plot(title="Simulated Portfolio Wealth Index (Daily)")
        plt.show()
    else:
        print("No trade periods were generated; please check the simulation period or trade logic.")

# --------------------------------------------------
# Run the test case if executed as main

if __name__ == '__main__':
    # To test with simulated data, uncomment the line below:
    test_simulated_data()
    # For real data, comment out the above and uncomment the line below:
    # main()
