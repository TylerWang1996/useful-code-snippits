import pandas as pd
import numpy as np
import blp  # if still needed elsewhere; can be removed if not used
from datetime import datetime
import xlsxwriter

# ----- Helper Functions for Metrics Calculation -----

def calculate_cagr(series):
    """
    Calculate the annualized compound growth rate (CAGR) given a cumulative series.
    Assumes the series is rebased (first valid value = 1) and is sampled monthly.
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

# ----- Alternate Main Analysis Function (Using Portfolio Ticker from Excel) -----
# This version uses a single common out-of-sample date and the portfolio ticker series is provided separately.

def analyze_strategies_alternate(strategies_df, portfolio_df, oos_date, excel_filename='strategy_analysis_from_excel.xlsx'):
    """
    strategies_df: DataFrame with individual strategies' cumulative returns.
                   Columns are ticker symbols (to be converted to friendly names using ticker_mapping)
                   and the index contains month-end dates.
    portfolio_df: DataFrame with a single column representing the portfolio ticker's cumulative returns.
                  The column header is a ticker symbol (to be converted to a friendly name).
    oos_date: A single out-of-sample date (string or datetime) to be applied to all series.
    excel_filename: Filename for the output Excel workbook.
    """
    # --- Process Ticker to Name Conversion ---
    # Define your mapping dictionary here.
    ticker_mapping = {
        'PortfolioTicker': 'Portfolio',
        'Strategy_A_Ticker': 'Strategy_A',
        'Strategy_B_Ticker': 'Strategy_B',
        'Strategy_C_Ticker': 'Strategy_C'
        # Add additional mappings as needed.
    }
    # Rename columns in strategies_df and portfolio_df according to ticker_mapping.
    strategies_df.rename(columns=ticker_mapping, inplace=True)
    portfolio_df.rename(columns=ticker_mapping, inplace=True)
    
    # --- Process Individual Strategies ---
    # Rebase each strategy so that its first valid value is 1.
    cum_df_rebased = strategies_df.copy()
    for col in cum_df_rebased.columns:
        first_valid = cum_df_rebased[col].dropna().iloc[0]
        cum_df_rebased[col] = cum_df_rebased[col] / first_valid

    # The index is assumed to already be a DatetimeIndex (month-end dates).
    metrics_dict = {}
    pivot_dict = {}
    monthly_returns_all = pd.DataFrame(index=cum_df_rebased.index)  # for correlation purposes
    
    common_oos_date = pd.to_datetime(oos_date)
    
    # Define periods for individual strategies.
    periods = {
        '1Y': 12,
        '5Y': 60,
        'Full': None,
        'OOS': 'oos'
    }
    
    for strat in cum_df_rebased.columns:
        strat_series = cum_df_rebased[strat].dropna()
        strat_metrics = {}
        # Compute monthly returns for metrics and pivot table.
        strat_monthly_returns_metrics = strat_series.pct_change().dropna()
        # For portfolio calculations, fill the first month with 0.
        strat_monthly_returns_for_portfolio = strat_series.pct_change().fillna(0)
        monthly_returns_all[strat] = strat_monthly_returns_for_portfolio

        for label, period in periods.items():
            if period == 'oos':
                # Use the common out-of-sample date.
                oos_start = common_oos_date
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

    # --- Process Portfolio Ticker ---
    portfolio_series = portfolio_df.squeeze()  # Assume a single column
    first_valid = portfolio_series.dropna().iloc[0]
    portfolio_series = portfolio_series / first_valid  # Rebase so first valid = 1

    # Compute monthly returns for the portfolio (fill the first month with 0).
    portfolio_monthly_returns = portfolio_series.pct_change().fillna(0)

    # Compute portfolio metrics (using periods: 1Y, 5Y, Full; omit OOS for portfolio).
    portfolio_periods = {'1Y': 12, '5Y': 60, 'Full': None}
    portfolio_metrics_dict = {}
    for label, period in portfolio_periods.items():
        if period is None:
            sub_series = portfolio_series
        else:
            if len(portfolio_series) >= period:
                sub_series = portfolio_series.iloc[-period:]
            else:
                sub_series = portfolio_series
        if len(sub_series) > 1:
            portfolio_metrics_dict[label] = compute_metrics(sub_series)
        else:
            portfolio_metrics_dict[label] = {key: np.nan for key in [
                'CAGR', 'Volatility', 'Information Ratio', 'Max Drawdown',
                'Average Monthly Return', '% Positive Months',
                'Avg Return (Positive Months)', 'Avg Return (Negative Months)'
            ]}
    portfolio_metrics_df = pd.DataFrame({period: pd.Series(metrics) for period, metrics in portfolio_metrics_dict.items()})
    
    portfolio_pivot = create_year_month_pivot(portfolio_monthly_returns)

    # Function to rebase cumulative series to 100.
    def rebase_to_100(series):
        return series * 100

    # --- Write to Excel with Conditional Formatting ---
    writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
    workbook  = writer.book

    # Create Portfolio sheet.
    portfolio_sheet = 'Portfolio'
    portfolio_metrics_df.to_excel(writer, sheet_name=portfolio_sheet, startrow=1, startcol=0)
    
    # --- NEW SECTION: Write Full Sample Metrics for All Strategies (Transposed) ---
    full_metrics_list = {}
    for strat, metrics_df in metrics_dict.items():
        if 'Full' in metrics_df.columns:
            full_metrics_list[strat] = metrics_df['Full']
    full_metrics_df = pd.DataFrame(full_metrics_list)
    full_metrics_start_row = portfolio_metrics_df.shape[0] + 4
    full_metrics_df.to_excel(writer, sheet_name=portfolio_sheet, startrow=full_metrics_start_row, startcol=0)
    
    # Write correlation matrix below full sample metrics.
    monthly_returns_all = monthly_returns_all.sort_index()
    corr_matrix = monthly_returns_all.corr()
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
    
    # Write portfolio year/month pivot below correlation matrix.
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
    # Write portfolio cumulative return series (rebased to 100) at bottom.
    port_cum_100 = rebase_to_100(portfolio_series)
    port_cum_df = port_cum_100.to_frame(name='Total Return (Base=100)')
    cum_start_row = pivot_start_row + portfolio_pivot.shape[0] + 4
    port_cum_df.to_excel(writer, sheet_name=portfolio_sheet, startrow=cum_start_row, startcol=0)

    # --- Create Unique Sheets for Each Strategy ---
    used_sheet_names = set()
    for strat in cum_df_rebased.columns:
        sheet_name = generate_sheet_name(strat, used_sheet_names)
        metrics_dict[strat].to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0)
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
        strat_cum_100 = rebase_to_100(cum_df_rebased[strat])
        strat_cum_df = strat_cum_100.to_frame(name='Total Return (Base=100)')
        cum_start_row = start_row + nrows_pivot + 4
        strat_cum_df.to_excel(writer, sheet_name=sheet_name, startrow=cum_start_row, startcol=0)

    writer.close()
    print(f"Excel workbook saved as {excel_filename}")

# ----- Main Section: Read Data from Excel and Process -----

if __name__ == "__main__":
    # Read the provided Excel file (e.g., 'input_data.xlsx').
    # The Excel file should have a 'Date' column, a portfolio ticker column, and one or more strategy ticker columns.
    input_file = 'input_data.xlsx'  # Update with your actual file name.
    df = pd.read_excel(input_file)
    
    # Convert the 'Date' column to datetime and set it as the index.
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Define your mapping dictionary to convert ticker headers to friendly names.
    # For example, if the Excel columns are 'IBM US Equity', 'AAPL US Equity', etc.
    ticker_mapping = {
        'PortfolioTicker': 'Strategy',
        'Strategy_A_Ticker': 'Strategy_A',
        'Strategy_B_Ticker': 'Strategy_B',
        'Strategy_C_Ticker': 'Strategy_C'
        # Add additional mappings as needed.
    }
    
    # Apply the mapping to all columns.
    df.rename(columns=ticker_mapping, inplace=True)
    
    # Separate the portfolio ticker series and the individual strategy tickers.
    portfolio_df = df[['Strategy']]
    strategies_df = df.drop(columns=['Strategy'])
    
    # Define a common out-of-sample date (applied to all series).
    oos_date = '2015-01-01'
    
    # Call the alternate analysis function with the provided data.
    analyze_strategies_alternate(strategies_df, portfolio_df, oos_date, excel_filename='strategy_analysis_from_excel.xlsx')
    print("Excel file processed and output workbook 'strategy_analysis_from_excel.xlsx' has been created.")
