import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def analyze_month_end_premium(df, x_days_list=[4, 3, 2, 1]):
    # Convert the 'date' column to datetime objects to ensure proper time-based operations.
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort the DataFrame by 'ticker' and 'date' so that each ticker's data is in chronological order.
    df = df.sort_values(by=['ticker', 'date']).copy()
    
    # Calculate daily returns for each ticker.
    # The lambda function computes simple returns: (price_today / price_yesterday) - 1.
    # Using 'transform' ensures that the returned Series aligns with the original DataFrame's index.
    df['return'] = df.groupby('ticker')['price'].transform(lambda x: x / x.shift(1) - 1)
    
    # Remove the first observation for each ticker (NaN returns, since there is no prior price).
    df = df.dropna(subset=['return'])
    
    # Create a 'month' column that identifies the month for each trading day.
    # This groups all trading days within the same calendar month together.
    df['month'] = df['date'].dt.to_period('M')
    
    # Initialize an empty list to store the month-level analysis results.
    results = []
    
    # Loop over each ticker separately.
    for ticker, group in df.groupby('ticker'):
        # For each ticker, loop over each month.
        for month, month_group in group.groupby('month'):
            # Reset the index for the month_group to use integer-based slicing with iloc.
            month_group = month_group.reset_index(drop=True)
            # Count the number of trading days in the current month.
            n_days = len(month_group)
            
            # Loop through each specified window (last X trading days) in x_days_list.
            for x in x_days_list:
                # Skip the month if it doesn't have enough trading days for the given window.
                if n_days < x:
                    continue
                
                # Define the month-end period as the last X trading days of the month.
                end_period = month_group.iloc[-x:]['return']
                # Define the rest-of-month period as all trading days excluding the last X days.
                rest_period = month_group.iloc[:-x]['return']
                
                # Append the computed metrics for the current month, ticker, and x window to the results list.
                results.append({
                    'ticker': ticker,                              # The ticker symbol.
                    'month': month.to_timestamp(),                 # The month as a timestamp.
                    'x': x,                                        # Number of month-end trading days.
                    'mean_return_end': end_period.mean(),          # Average return for month-end days.
                    'mean_return_rest': rest_period.mean(),        # Average return for the rest of the month.
                    'n_end': len(end_period),                      # Number of observations in month-end period.
                    'n_rest': len(rest_period),                    # Number of observations in the rest-of-month period.
                    'returns_end': end_period.values,              # Array of returns for month-end period (for aggregation).
                    'returns_rest': rest_period.values             # Array of returns for the rest-of-month period.
                })
    
    # Convert the list of dictionaries to a DataFrame.
    monthly_results = pd.DataFrame(results)
    
    # Initialize a list to store the final aggregated results.
    final_results = []
    # Group the monthly results by ticker and the x window size.
    for (ticker, x), subdf in monthly_results.groupby(['ticker', 'x']):
        # Concatenate all month-end returns across different months into one array.
        all_end_returns = np.concatenate(subdf['returns_end'].values)
        # Concatenate all rest-of-month returns across different months into one array.
        all_rest_returns = np.concatenate(subdf['returns_rest'].values)
        
        # Compute the overall average returns for month-end and rest-of-month periods.
        overall_mean_end = all_end_returns.mean()
        overall_mean_rest = all_rest_returns.mean()
        # Calculate the difference between the month-end and rest-of-month average returns.
        diff = overall_mean_end - overall_mean_rest
        
        # Perform a Welch's t-test (unequal variances) to test the statistical significance of the difference.
        t_stat, p_val = ttest_ind(all_end_returns, all_rest_returns, equal_var=False)
        
        # Append the aggregated metrics and statistical test results for the current group.
        final_results.append({
            'ticker': ticker,                               # The ticker symbol.
            'n_days': x,                                    # Number of month-end trading days used in the analysis.
            'overall_mean_return_end': overall_mean_end,    # Overall average return for month-end days.
            'overall_mean_return_rest': overall_mean_rest,  # Overall average return for rest-of-month days.
            'mean_difference': diff,                        # Difference between month-end and rest-of-month averages.
            't_statistic': t_stat,                          # t-statistic from the t-test.
            'p_value': p_val                                # p-value from the t-test.
        })
    
    # Convert the aggregated results to a DataFrame for final output.
    final_df = pd.DataFrame(final_results)
    return final_df


def simulate_bond_futures_data(ticker, start_date, end_date, init_price=100):
    """
    Simulate bond futures prices using a simple geometric progression.
    Daily returns are drawn from a normal distribution with mean 0.02% and volatility 0.5%.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(dates)
    # simulate daily returns: mean 0.0002, standard deviation 0.005
    daily_returns = np.random.normal(loc=0.0002, scale=0.005, size=n)
    prices = [init_price]
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    prices = prices[1:]  # drop the initial price
    return pd.DataFrame({'date': dates, 'ticker': ticker, 'price': prices})

if __name__ == '__main__':
    # Simulate data for two tickers over a 6-month period
    data_A = simulate_bond_futures_data("US_BOND_A", "2021-01-01", "2021-06-30")
    data_B = simulate_bond_futures_data("US_BOND_B", "2021-01-01", "2021-06-30")
    df_simulated = pd.concat([data_A, data_B], ignore_index=True)
    
    # Run the month-end premium analysis using the simulated data
    result_df = analyze_month_end_premium(df_simulated, x_days_list=[4, 3, 2, 1])
    print(result_df)
