import pandas as pd
import numpy as np
import blp
from datetime import datetime
import xlsxwriter

# ----- Helper Functions for Metrics Calculation -----

def calculate_cagr(series):
    """
    Calculate the annualized compound growth rate (CAGR) given a cumulative series.
    """
    if len(series) < 2:
        return np.nan
    start = series.iloc[0]
    end = series.iloc[-1]
    num_months = len(series) - 1  # number of periods
    if num_months <= 0 or start == 0:
        return np.nan
    return (end / start) ** (12 / num_months) - 1

def calculate_annualized_vol(monthly_returns):
    """
    Calculate the annualized volatility from monthly returns.
    """
    return monthly_returns.std() * np.sqrt(12)

def calculate_info_ratio(cagr, vol):
    """
    Information Ratio as CAGR / Volatility.
    """
    return cagr / vol if vol != 0 else np.nan

def calculate_avg_return(monthly_returns):
    return monthly_returns.mean()

def percent_positive(monthly_returns):
    if len(monthly_returns) == 0:
        return np.nan
    return (monthly_returns > 0).sum() / len(monthly_returns)

def avg_positive_return(monthly_returns):
    pos = monthly_returns[monthly_returns > 0]
    return pos.mean() if len(pos) > 0 else np.nan

def avg_negative_return(monthly_returns):
    neg = monthly_returns[monthly_returns < 0]
    return neg.mean() if len(neg) > 0 else np.nan

def max_drawdown(series):
    """
    Calculate maximum drawdown of a cumulative return series.
    """
    roll_max = series.cummax()
    drawdown = series / roll_max - 1
    return drawdown.min()

def compute_metrics(cum_series):
    """
    Given a cumulative return series (rebased to start at 1),
    compute monthly returns (dropping the undefined first return) and return a dictionary of metrics.
    """
    monthly_returns = cum_series.pct_change().dropna()
    cagr = calculate_cagr(cum_series)
    vol = calculate_annualized_vol(monthly_returns)
    info_ratio = calculate_info_ratio(cagr, vol)
    mdd = max_drawdown(cum_series)
    avg_return = calculate_avg_return(monthly_returns)
    pos_pct = percent_positive(monthly_returns)
    avg_pos = avg_positive_return(monthly_returns)
    avg_neg = avg_negative_return(monthly_returns)
    
    return {
        'CAGR': cagr,
        'Volatility': vol,
        'Information Ratio': info_ratio,
        'Max Drawdown': mdd,
        'Average Monthly Return': avg_return,
        '% Positive Months': pos_pct,
        'Avg Return (Positive Months)': avg_pos,
        'Avg Return (Negative Months)': avg_neg
    }

def create_year_month_pivot(monthly_returns):
    """
    Create a pivot table of monthly returns with years as rows and months as columns.
    """
    df = monthly_returns.to_frame(name='Return')
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    pivot = df.pivot(index='Year', columns='Month', values='Return')
    return pivot

# ----- New Function: Check Date/Month Matrix Accuracy -----

def check_date_month_matrix(pivot, monthly_returns):
    """
    Check that each cell in the pivot table (year/month matrix) exactly matches
    the monthly return from the original monthly_returns series for that year and month.
    Prints out any discrepancies.
    """
    discrepancies = []
    for year in pivot.index:
        for month in pivot.columns:
            pivot_value = pivot.loc[year, month]
            matching_dates = [d for d in monthly_returns.index if d.year == year and d.month == month]
            if not matching_dates:
                if pd.notnull(pivot_value):
                    discrepancies.append(f"Cell ({year}, {month}) has value {pivot_value} but no matching date in monthly_returns.")
            else:
                if len(matching_dates) > 1:
                    discrepancies.append(f"Multiple dates found for cell ({year}, {month}) in monthly_returns.")
                else:
                    date = matching_dates[0]
                    actual_value = monthly_returns.loc[date]
                    if not np.isclose(pivot_value, actual_value, atol=1e-8, equal_nan=True):
                        discrepancies.append(f"Mismatch in cell ({year}, {month}): pivot value {pivot_value} != actual value {actual_value} from {date}.")
    if discrepancies:
        print("Discrepancies found in the date/month matrix:")
        for d in discrepancies:
            print("  -", d)
    else:
        print("All cells in the date/month matrix match the monthly returns time series.")

# ----- Helper Function: Generate Unique Sheet Names -----

def generate_sheet_name(base, existing_names):
    """
    Generate a unique sheet name (max length 31) given a base name and a set of names already used.
    """
    name = base[:31]
    if name not in existing_names:
        existing_names.add(name)
        return name
    else:
        counter = 1
        while True:
            suffix = f"_{counter}"
            allowed_len = 31 - len(suffix)
            new_name = base[:allowed_len] + suffix
            if new_name not in existing_names:
                existing_names.add(new_name)
                return new_name
            counter += 1

# ----- Main Analysis Function -----

def analyze_strategies(cum_df, oos_dict, portfolio_weights, excel_filename='output.xlsx'):
    """
    cum_df: DataFrame where index is datetime (monthly) and columns are strategies' cumulative returns.
            Each series may have a different base; they will be uniformly rebased to start at 1.
    oos_dict: dictionary with out-of-sample start dates for each strategy (e.g., {"Strategy_A": "2018-01-01", ...})
    portfolio_weights: dictionary of weights (e.g., {"Strategy_A": 0.4, "Strategy_B": 0.6})
    excel_filename: filename for the output Excel workbook.
    """
    # Rebase each strategy so that the first available value is 1
    cum_df_rebased = cum_df.copy()
    for col in cum_df_rebased.columns:
        first_valid = cum_df_rebased[col].dropna().iloc[0]
        cum_df_rebased[col] = cum_df_rebased[col] / first_valid

    # Ensure the index is a DatetimeIndex
    if not isinstance(cum_df_rebased.index, pd.DatetimeIndex):
        cum_df_rebased.index = pd.to_datetime(cum_df_rebased.index)

    # Prepare dictionaries to hold metrics and pivot tables for each strategy
    metrics_dict = {}
    pivot_dict = {}
    # For portfolio and correlation, include the first month by filling NaN with 0.
    monthly_returns_all = pd.DataFrame(index=cum_df_rebased.index)

    # Define the periods to analyze for strategies
    periods = {
        '1Y': 12,
        '5Y': 60,
        'Full': None,
        'OOS': 'oos'
    }

    for strat in cum_df_rebased.columns:
        strat_series = cum_df_rebased[strat].dropna()
        strat_metrics = {}
        # Use dropna for metrics and pivot table
        strat_monthly_returns_metrics = strat_series.pct_change().dropna()
        # Use fillna(0) for portfolio calculations so that the first month is included.
        strat_monthly_returns_for_portfolio = strat_series.pct_change().fillna(0)
        monthly_returns_all[strat] = strat_monthly_returns_for_portfolio

        for label, period in periods.items():
            if period == 'oos':
                oos_start = pd.to_datetime(oos_dict.get(strat))
                if oos_start not in strat_series.index:
                    valid_dates = strat_series.index[strat_series.index >= oos_start]
                    if valid_dates.empty:
                        sub_series = pd.Series(dtype=float)
                    else:
                        sub_series = strat_series.loc[valid_dates[0]:]
                else:
                    sub_series = strat_series.loc[oos_start:]
            elif period is None:
                sub_series = strat_series
            else:
                if len(strat_series) >= period:
                    sub_series = strat_series.iloc[-period:]
                else:
                    sub_series = strat_series

            if len(sub_series) > 1:
                strat_metrics[label] = compute_metrics(sub_series)
            else:
                strat_metrics[label] = {key: np.nan for key in [
                    'CAGR', 'Volatility', 'Information Ratio', 'Max Drawdown',
                    'Average Monthly Return', '% Positive Months',
                    'Avg Return (Positive Months)', 'Avg Return (Negative Months)'
                ]}
        
        metrics_df = pd.DataFrame({period: pd.Series(metrics) for period, metrics in strat_metrics.items()})
        metrics_dict[strat] = metrics_df

        pivot_df = create_year_month_pivot(strat_monthly_returns_metrics)
        pivot_dict[strat] = pivot_df

        print(f"Checking date/month matrix for {strat}:")
        check_date_month_matrix(pivot_df, strat_monthly_returns_metrics)
        print("-" * 50)

    # ----- Portfolio Metrics and Pivot for Portfolio Sheet -----
    monthly_returns_all = monthly_returns_all.sort_index()
    corr_matrix = monthly_returns_all.corr()

    # Compute portfolio returns using the provided weights (only use strategies present)
    common_strats = [s for s in portfolio_weights if s in monthly_returns_all.columns]
    weights_series = pd.Series(portfolio_weights)[common_strats]
    portfolio_returns = (monthly_returns_all[common_strats] * weights_series).sum(axis=1)
    # Rebuild portfolio cumulative series from monthly returns so that the first month is included.
    portfolio_cum_full = (1 + portfolio_returns).cumprod()
    
    # Define portfolio periods: 1Y, 5Y, and Full sample (omit OOS for portfolio)
    portfolio_periods = {'1Y': 12, '5Y': 60, 'Full': None}
    portfolio_metrics_dict = {}
    for label, period in portfolio_periods.items():
        if period is None:
            sub_series = portfolio_cum_full
        else:
            if len(portfolio_cum_full) >= period:
                sub_series = portfolio_cum_full.iloc[-period:]
            else:
                sub_series = portfolio_cum_full
        if len(sub_series) > 1:
            portfolio_metrics_dict[label] = compute_metrics(sub_series)
        else:
            portfolio_metrics_dict[label] = {key: np.nan for key in [
                'CAGR', 'Volatility', 'Information Ratio', 'Max Drawdown',
                'Average Monthly Return', '% Positive Months',
                'Avg Return (Positive Months)', 'Avg Return (Negative Months)'
            ]}
    portfolio_metrics_df = pd.DataFrame({period: pd.Series(metrics) for period, metrics in portfolio_metrics_dict.items()})
    
    portfolio_pivot = create_year_month_pivot(portfolio_returns)

    # Function to rebase cumulative series to 100
    def rebase_to_100(series):
        return series * 100

    # ----- Write to Excel with Conditional Formatting -----
    writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
    workbook  = writer.book

    # Create Portfolio sheet
    portfolio_sheet = 'Portfolio'
    # Write portfolio metrics table at top
    portfolio_metrics_df.to_excel(writer, sheet_name=portfolio_sheet, startrow=1, startcol=0)
    
    # --- NEW SECTION: Write Full Sample Metrics for All Strategies (Transposed) ---
    full_metrics_list = {}
    for strat, metrics_df in metrics_dict.items():
        if 'Full' in metrics_df.columns:
            full_metrics_list[strat] = metrics_df['Full']
    # Now metrics are rows and strategies are columns.
    full_metrics_df = pd.DataFrame(full_metrics_list)
    full_metrics_start_row = portfolio_metrics_df.shape[0] + 4
    full_metrics_df.to_excel(writer, sheet_name=portfolio_sheet, startrow=full_metrics_start_row, startcol=0)
    
    # Write correlation matrix below full sample metrics table
    corr_start_row = full_metrics_start_row + full_metrics_df.shape[0] + 4
    corr_matrix.to_excel(writer, sheet_name=portfolio_sheet, startrow=corr_start_row, startcol=0)
    ws_portfolio = writer.sheets[portfolio_sheet]
    nrows, ncols = corr_matrix.shape
    corr_range = xlsxwriter.utility.xl_range(corr_start_row+1, 0, corr_start_row+nrows, ncols-1)
    ws_portfolio.conditional_format(corr_range, {
        'type': '3_color_scale',
        'min_color': "#F8696B",
        'mid_color': "#FFEB84",
        'max_color': "#63BE7B"
    })
    
    # Write portfolio year/month pivot below correlation matrix
    pivot_start_row = corr_start_row + nrows + 4
    portfolio_pivot.to_excel(writer, sheet_name=portfolio_sheet, startrow=pivot_start_row, startcol=0)
    ws_portfolio.conditional_format(
        xlsxwriter.utility.xl_range(pivot_start_row+1, 1, pivot_start_row+portfolio_pivot.shape[0], portfolio_pivot.shape[1]),
        {
            'type': '3_color_scale',
            'min_color': "#F8696B",
            'mid_color': "#FFEB84",
            'max_color': "#63BE7B"
        }
    )
    # Write portfolio cumulative return series (rebased to 100) at bottom
    port_cum_100 = rebase_to_100(portfolio_cum_full)
    port_cum_df = port_cum_100.to_frame(name='Total Return (Base=100)')
    cum_start_row = pivot_start_row + portfolio_pivot.shape[0] + 4
    port_cum_df.to_excel(writer, sheet_name=portfolio_sheet, startrow=cum_start_row, startcol=0)

    # ----- Create Unique Sheet Names for Each Strategy and Write Their Data -----
    used_sheet_names = set()
    for strat in cum_df_rebased.columns:
        sheet_name = generate_sheet_name(strat, used_sheet_names)
        # Write the strategy's metrics table at top
        metrics_dict[strat].to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0)
        # Write the strategy's year/month pivot table below metrics table
        pivot_df = pivot_dict[strat]
        start_row = metrics_dict[strat].shape[0] + 4
        pivot_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=0)
        ws = writer.sheets[sheet_name]
        nrows_pivot, ncols_pivot = pivot_df.shape
        pivot_range = xlsxwriter.utility.xl_range(start_row+1, 1, start_row+nrows_pivot, ncols_pivot)
        ws.conditional_format(pivot_range, {
            'type': '3_color_scale',
            'min_color': "#F8696B",
            'mid_color': "#FFEB84",
            'max_color': "#63BE7B"
        })
        # Write the cumulative return series (rebased to 100) at bottom
        strat_cum_100 = rebase_to_100(cum_df_rebased[strat])
        strat_cum_df = strat_cum_100.to_frame(name='Total Return (Base=100)')
        cum_start_row = start_row + nrows_pivot + 4
        strat_cum_df.to_excel(writer, sheet_name=sheet_name, startrow=cum_start_row, startcol=0)

    writer.close()
    print(f"Excel workbook saved as {excel_filename}")



def get_bbg_px_last(tickers_dict, start_date, end_date, freq='ME'):
    """
    Retrieves Bloomberg PX_LAST data for a set of tickers specified in tickers_dict.
    
    Parameters:
        tickers_dict (dict): Dictionary where keys are Bloomberg tickers (e.g., 'IBM US Equity') 
                             and values are display names (e.g., 'IBM').
        start_date (str or datetime): Start date for the data retrieval (e.g., '2010-01-01').
        end_date (str or datetime): End date for the data retrieval (e.g., '2020-12-31').
        freq (str): Frequency for the output index; default is 'ME' (month-end).
    
    Returns:
        pd.DataFrame: DataFrame with a DatetimeIndex (with the specified frequency) and columns
                      renamed to the display names provided in tickers_dict.
    """
    # Extract Bloomberg tickers from the dictionary keys.
    bbg_tickers = list(tickers_dict.keys())
    
    # Retrieve Bloomberg historical data for PX_LAST using monthly periodicity.
    data = blp.bdh(bbg_tickers, 'PX_LAST', start_date, end_date, Per='M')
    
    # If the returned DataFrame has a MultiIndex on columns, extract the 'PX_LAST' level.
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs('PX_LAST', axis=1, level=1)
    
    # Create a full date range based on the provided frequency to ensure consistency.
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = data.reindex(dates)
    data.sort_index(inplace=True)
    
    # Rename the columns using the provided mapping.
    data.rename(columns=tickers_dict, inplace=True)
    
    return data

# ----- Example Usage -----

if __name__ == "__main__":
    # Define a dictionary where keys are Bloomberg tickers and values are display names.
    tickers_dict = {
        'Strategy_A_Ticker': 'Strategy_A',
        'Strategy_B_Ticker': 'Strategy_B',
        'Strategy_C_Ticker': 'Strategy_C'
    }
    
    # Set the start and end dates.
    start_date = '2010-01-01'
    end_date   = '2020-12-31'
    
    # Retrieve the PX_LAST data.
    df = get_bbg_px_last(tickers_dict, start_date, end_date, freq='ME')
    
    # Define out-of-sample start dates and portfolio weights (as before).
    oos_dict = {
        'Strategy_A': '2015-01-01',
        'Strategy_B': '2016-01-01',
        'Strategy_C': '2014-06-01'
    }
    
    portfolio_weights = {
        'Strategy_A': 0.3,
        'Strategy_B': 0.5,
        'Strategy_C': 0.2
    }
    
    # Call the analysis function with the retrieved Bloomberg data.
    analyze_strategies(df, oos_dict, portfolio_weights, excel_filename='strategy_analysis_test.xlsx')
    print("BBG data test case executed and Excel workbook 'strategy_analysis_test.xlsx' has been created.")
