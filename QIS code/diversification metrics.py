import pandas as pd
import numpy as np
import statsmodels.api as sm

# =============================================================================
# USER INPUTS
# =============================================================================
# Data source: Set CSV_DATA_FILE to a file path (string) if you have a CSV of real data.
# The CSV is expected to have dates as index (or a column that can be parsed as dates).
CSV_DATA_FILE = None  # e.g., "path/to/your/data.csv" or None to simulate data

# Simulation settings (used only when CSV_DATA_FILE is None):
START_DATE = "2015-01-01"
END_DATE = "2020-12-01"
FREQ = "MS"  # Monthly start dates

# Ticker settings: List your benchmark and strategy tickers.
BENCHMARKS = ['SPY', 'AGG']            # Benchmarks
STRATEGIES = ['Strategy1', 'Strategy2', 'Strategy3']  # Strategies

# Crisis periods defined as a dictionary (name: (start_date, end_date)).
CRISIS_PERIODS = {
    "Crisis_2018": ("2018-01-01", "2018-12-01"),
    "Crisis_2020": ("2020-02-01", "2020-04-01")
}

# Output settings:
OUTPUT_FILENAME = "strategy_analysis.xlsx"
ROLLING_WINDOW = 12  # 1-year rolling window (in months)

# Random seed for reproducibility:
np.random.seed(42)

# =============================================================================
# DATA LOADING / SIMULATION FUNCTIONS
# =============================================================================
def simulate_data(start_date, end_date, freq, benchmarks, strategies):
    """
    Simulate monthly total return index data for benchmarks and strategies.
    Returns a DataFrame of simulated index values.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(dates)
    tickers = benchmarks + strategies
    simulated_returns = pd.DataFrame(index=dates, columns=tickers)
    
    # Simulate returns: benchmarks (lower mean & volatility)
    for ticker in benchmarks:
        simulated_returns[ticker] = np.random.normal(loc=0.005, scale=0.02, size=n)
    # Simulate returns: strategies (higher mean & volatility)
    for ticker in strategies:
        simulated_returns[ticker] = np.random.normal(loc=0.01, scale=0.04, size=n)
    
    # Convert returns to a total return index (starting at 100)
    df = (1 + simulated_returns).cumprod() * 100
    return df, simulated_returns

def load_data():
    """
    Load real data from CSV if CSV_DATA_FILE is specified;
    otherwise, simulate data.
    """
    if CSV_DATA_FILE:
        df = pd.read_csv(CSV_DATA_FILE, index_col=0, parse_dates=True)
        simulated_returns = None
    else:
        df, simulated_returns = simulate_data(START_DATE, END_DATE, FREQ, BENCHMARKS, STRATEGIES)
    return df, simulated_returns

# =============================================================================
# METRIC CALCULATION FUNCTIONS
# =============================================================================
def calculate_cagr(series):
    """
    Calculate the Compound Annual Growth Rate (CAGR) from an index series.
    """
    if len(series) < 2:
        return np.nan
    total_return = series.iloc[-1] / series.iloc[0]
    years = (series.index[-1] - series.index[0]).days / 365.25
    return total_return ** (1/years) - 1

def compute_metrics_for_benchmark(bm, df, returns, strategies, crisis_periods):
    """
    Compute correlations and CAGR metrics for a given benchmark.
    The metrics are computed with respect to the current benchmark (bm)
    and include all instruments: the current benchmark, the other benchmarks,
    and the strategies.
    Returns a dictionary of metrics.
    """
    metrics = {}
    # Construct all_tickers: current benchmark, other benchmarks, then strategies
    other_benchmarks = [x for x in BENCHMARKS if x != bm]
    all_tickers = [bm] + other_benchmarks + strategies
    
    # --- Correlations ---
    # Full correlation (over the entire period)
    full_corr = returns[all_tickers].corr().loc[bm, all_tickers]
    metrics["Full Corr"] = full_corr

    # Downside correlation: when benchmark returns are negative
    downside_condition = returns[bm] < 0
    if downside_condition.sum() > 1:
        downside_corr = returns[downside_condition][all_tickers].corr().loc[bm, all_tickers]
    else:
        downside_corr = pd.Series(data=np.nan, index=all_tickers)
    metrics["Downside Corr"] = downside_corr

    # Drawdown correlation: using the drawdown of index values
    drawdowns = df.div(df.cummax()) - 1
    drawdown_corr = drawdowns[all_tickers].corr().loc[bm, all_tickers]
    metrics["Drawdown Corr"] = drawdown_corr

    # Crisis period correlations and CAGRs for each defined crisis period
    crisis_corrs = {}
    crisis_cagrs = {}
    for name, (start, end) in crisis_periods.items():
        condition = (returns.index >= start) & (returns.index <= end)
        if returns[condition].shape[0] > 1:
            corr = returns[condition][all_tickers].corr().loc[bm, all_tickers]
        else:
            corr = pd.Series(data=np.nan, index=all_tickers)
        crisis_corrs[name] = corr
        
        # Compute CAGR during the crisis period for each ticker using aligned index
        cagr_dict = {}
        for ticker in all_tickers:
            aligned_series = df[ticker].loc[returns.index]
            filtered_series = aligned_series[condition]
            cagr_dict[ticker] = calculate_cagr(filtered_series) if len(filtered_series) > 1 else np.nan
        crisis_cagrs[name] = pd.Series(cagr_dict)
    metrics["Crisis Corrs"] = crisis_corrs
    metrics["Crisis CAGRs"] = crisis_cagrs

    # Bottom 10% analysis: when the benchmark is in its worst 10% of historical returns
    bottom10_threshold = returns[bm].quantile(0.10)
    bottom10_condition = returns[bm] < bottom10_threshold
    if bottom10_condition.sum() > 1:
        bottom10_corr = returns[bottom10_condition][all_tickers].corr().loc[bm, all_tickers]
    else:
        bottom10_corr = pd.Series(data=np.nan, index=all_tickers)
    metrics["Bottom10 Corr"] = bottom10_corr
    
    bottom10_cagr = {}
    for ticker in all_tickers:
        aligned_series = df[ticker].loc[returns.index]
        filtered_series = aligned_series[bottom10_condition]
        bottom10_cagr[ticker] = calculate_cagr(filtered_series) if len(filtered_series) > 1 else np.nan
    metrics["Bottom10 CAGRs"] = pd.Series(bottom10_cagr)
    
    # Overall CAGR for each ticker (entire period)
    overall_cagr = df[all_tickers].apply(calculate_cagr)
    metrics["Overall CAGRs"] = overall_cagr

    return metrics

def compute_rolling_metrics(bm, returns, strategies, window):
    """
    Compute 1-year (rolling window) correlations and betas for strategies vs. the benchmark.
    Returns two DataFrames: rolling correlations and rolling betas.
    """
    rolling_corrs = {}
    rolling_betas = {}
    for ticker in strategies:
        # Rolling correlation between benchmark and strategy
        rolling_corr = returns[bm].rolling(window=window).corr(returns[ticker])
        rolling_corrs[ticker] = rolling_corr

        # Rolling beta: covariance / variance
        rolling_cov = returns[bm].rolling(window=window).cov(returns[ticker])
        rolling_var = returns[bm].rolling(window=window).var()
        rolling_beta = rolling_cov / rolling_var
        rolling_betas[ticker] = rolling_beta

    rolling_corr_df = pd.DataFrame(rolling_corrs)
    rolling_beta_df = pd.DataFrame(rolling_betas)
    return rolling_corr_df, rolling_beta_df

def create_summary_table(metrics, bm, strategies, crisis_periods, df, returns):
    """
    Create a summary table combining correlation and CAGR metrics.
    The table now includes columns for all instruments (benchmarks and strategies).
    """
    other_benchmarks = [x for x in BENCHMARKS if x != bm]
    all_tickers = [bm] + other_benchmarks + strategies
    
    summary_dict = {"Metric": []}
    for ticker in all_tickers:
        summary_dict[ticker] = []
    
    # Full correlation row
    summary_dict["Metric"].append("Full Corr")
    for ticker in all_tickers:
        summary_dict[ticker].append(metrics["Full Corr"].get(ticker, np.nan))
        
    # Downside correlation row
    summary_dict["Metric"].append("Downside Corr")
    for ticker in all_tickers:
        summary_dict[ticker].append(metrics["Downside Corr"].get(ticker, np.nan))
        
    # Drawdown correlation row
    summary_dict["Metric"].append("Drawdown Corr")
    for ticker in all_tickers:
        summary_dict[ticker].append(metrics["Drawdown Corr"].get(ticker, np.nan))
        
    # Bottom 10% correlation row
    summary_dict["Metric"].append("Bottom 10% Corr")
    for ticker in all_tickers:
        summary_dict[ticker].append(metrics["Bottom10 Corr"].get(ticker, np.nan))
        
    # Overall CAGR row
    summary_dict["Metric"].append("Overall CAGR")
    for ticker in all_tickers:
        summary_dict[ticker].append(metrics["Overall CAGRs"].get(ticker, np.nan))
        
    # Downside CAGR row (using periods when benchmark returns are negative)
    downside_cagr = {}
    for ticker in all_tickers:
        aligned_series = df[ticker].loc[returns.index]
        filtered_series = aligned_series[returns[bm] < 0]
        downside_cagr[ticker] = calculate_cagr(filtered_series) if len(filtered_series) > 1 else np.nan
    summary_dict["Metric"].append("Downside CAGR")
    for ticker in all_tickers:
        summary_dict[ticker].append(downside_cagr.get(ticker, np.nan))
        
    # Bottom 10% CAGR row
    summary_dict["Metric"].append("Bottom 10% CAGR")
    for ticker in all_tickers:
        summary_dict[ticker].append(metrics["Bottom10 CAGRs"].get(ticker, np.nan))
        
    # Crisis period CAGR rows (one row per crisis period)
    for crisis, _ in crisis_periods.items():
        summary_dict["Metric"].append(f"{crisis} CAGR")
        crisis_cagr = metrics["Crisis CAGRs"][crisis]
        for ticker in all_tickers:
            summary_dict[ticker].append(crisis_cagr.get(ticker, np.nan))
            
    summary_table = pd.DataFrame(summary_dict)
    return summary_table

def write_metrics_to_excel(writer, sheet_name, summary_table, rolling_corr_df, rolling_beta_df):
    """
    Write the summary table and rolling metrics to an Excel sheet.
    """
    # Write summary table at the top of the sheet
    summary_table.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    start_row = summary_table.shape[0] + 3

    # Write rolling correlations (strategies only)
    writer.sheets[sheet_name].write(start_row - 1, 0, "Rolling 1Y Correlations (Benchmark vs. Strategies)")
    rolling_corr_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=0)
    start_row += rolling_corr_df.shape[0] + 3

    # Write rolling betas (strategies only)
    writer.sheets[sheet_name].write(start_row - 1, 0, "Rolling 1Y Betas (Strategies relative to Benchmark)")
    rolling_beta_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=0)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    # Load data (either from CSV or simulate)
    df, _ = load_data()
    # Compute monthly returns from index values (returns has one fewer row than df)
    returns = df.pct_change().dropna()
    
    # Create an Excel writer to store each benchmark's results in a separate tab
    with pd.ExcelWriter(OUTPUT_FILENAME, engine='xlsxwriter') as writer:
        for bm in BENCHMARKS:
            sheet_name = bm
            
            # Compute metrics for the current benchmark (using bm as the reference)
            metrics = compute_metrics_for_benchmark(bm, df, returns, STRATEGIES, CRISIS_PERIODS)
            # Compute rolling correlations and betas (only for strategies)
            rolling_corr_df, rolling_beta_df = compute_rolling_metrics(bm, returns, STRATEGIES, ROLLING_WINDOW)
            # Create a summary table with the key metrics for all instruments
            summary_table = create_summary_table(metrics, bm, STRATEGIES, CRISIS_PERIODS, df, returns)
            
            # Write results to the corresponding Excel sheet
            summary_table.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
            start_row = summary_table.shape[0] + 3
            writer.sheets[sheet_name].write(start_row - 1, 0, "Rolling 1Y Correlations (Benchmark vs. Strategies)")
            rolling_corr_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=0)
            start_row += rolling_corr_df.shape[0] + 3
            writer.sheets[sheet_name].write(start_row - 1, 0, "Rolling 1Y Betas (Strategies relative to Benchmark)")
            rolling_beta_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=0)

    print("Excel file created:", OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
