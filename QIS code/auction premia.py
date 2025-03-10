import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------
# Flexible Event Window Function
# ---------------------------
def get_event_window_returns(auction_date, returns_series, window):
    """
    Extracts returns for an event window spanning from -window to +window trading days 
    relative to the auction_date from the given returns_series.
    """
    if auction_date not in returns_series.index:
        return None
    trading_dates = returns_series.index
    loc = trading_dates.get_loc(auction_date)
    
    # Ensure there are enough trading days before and after the auction.
    if loc < window or loc > len(trading_dates) - window - 1:
        return None
    
    # Extract the event window and reindex it to relative days.
    window_returns = returns_series.iloc[loc - window: loc + window + 1].copy()
    window_returns.index = range(-window, window + 1)
    return window_returns

# ---------------------------
# Auction Event Analysis Function
# ---------------------------
def analyze_auction_events(auctions_df, futures_df, security_term, futures_col, window=5):
    """
    Analyzes auction events for a specific security term by:
      - Computing daily percentage returns for the given futures column.
      - Extracting an event window (from -window to +window) for each auction.
      - Calculating the average return in the pre-auction period (days -window to -1)
        and post-auction period (days +1 to +window).
      - Running a one-sample t-test on the difference (post minus pre) returns.
    
    Returns:
      - avg_event_window: Average return for each relative day across events.
      - pre_returns: List of average pre-auction returns for each event.
      - post_returns: List of average post-auction returns for each event.
      - t_stat, p_value: T-test statistics on the difference (post - pre).
      - event_df: DataFrame of all event windows (each row is one event).
    """
    # Ensure the futures data is sorted by date.
    futures_df = futures_df.sort_index()
    
    # Calculate daily returns for the target futures series.
    returns_col = futures_col + '_return'
    futures_df[returns_col] = futures_df[futures_col].pct_change()
    
    # Filter auctions by security term.
    auctions_sec = auctions_df[auctions_df['security_term'] == security_term]
    
    event_windows = []
    pre_returns = []
    post_returns = []
    
    # Loop through each auction event.
    for auction_date in auctions_sec['auction_date']:
        window_returns = get_event_window_returns(auction_date, futures_df[returns_col], window)
        if window_returns is not None:
            event_windows.append(window_returns)
            # Compute average pre-auction returns (from -window to -1).
            pre_avg = window_returns.loc[range(-window, 0)].mean()
            # Compute average post-auction returns (from +1 to +window).
            post_avg = window_returns.loc[range(1, window + 1)].mean()
            pre_returns.append(pre_avg)
            post_returns.append(post_avg)
    
    # Create a DataFrame with each event as a row (columns are relative days).
    event_df = pd.DataFrame(event_windows)
    # Average returns at each relative day across all events.
    avg_event_window = event_df.mean()
    
    # Convert lists to arrays for t-test computation.
    pre_returns = np.array(pre_returns)
    post_returns = np.array(post_returns)
    diff = post_returns - pre_returns
    t_stat, p_value = stats.ttest_1samp(diff, 0)
    
    return avg_event_window, pre_returns, post_returns, t_stat, p_value, event_df

# ---------------------------
# Example Usage
# ---------------------------
# Assume:
# - futures_df has dates as the index and columns: 'US 5 Year Futures' and 'US 10 Year Futures'
# - auctions_df has columns: 'security_term' (e.g. '5-Year', '10-Year') and 'auction_date' (datetime)
# Ensure auction_date is a datetime type.
auctions_df['auction_date'] = pd.to_datetime(auctions_df['auction_date'])

# Set a flexible window length (e.g., 5 trading days before and after the auction).
window = 5

# Analyze 5-Year Auctions for US 5 Year Futures.
avg_event_window_5Y, pre_returns_5Y, post_returns_5Y, t_stat_5Y, p_value_5Y, event_df_5Y = analyze_auction_events(
    auctions_df, futures_df, '5-Year', 'US 5 Year Futures', window)

print(f"5-Year Futures: t-statistic = {t_stat_5Y:.3f}, p-value = {p_value_5Y:.3f}")

# Analyze 10-Year Auctions for US 10 Year Futures.
avg_event_window_10Y, pre_returns_10Y, post_returns_10Y, t_stat_10Y, p_value_10Y, event_df_10Y = analyze_auction_events(
    auctions_df, futures_df, '10-Year', 'US 10 Year Futures', window)

print(f"10-Year Futures: t-statistic = {t_stat_10Y:.3f}, p-value = {p_value_10Y:.3f}")

# ---------------------------
# Visualization
# ---------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot average event window returns for 5-Year Futures.
axs[0].plot(avg_event_window_5Y.index, avg_event_window_5Y.values, marker='o', linestyle='-')
axs[0].set_title('Average US 5Y Futures Returns around 5-Year Auctions')
axs[0].set_xlabel('Relative Day')
axs[0].set_ylabel('Average Return')
axs[0].axhline(0, color='gray', linestyle='--')
axs[0].grid(True)

# Plot average event window returns for 10-Year Futures.
axs[1].plot(avg_event_window_10Y.index, avg_event_window_10Y.values, marker='o', linestyle='-')
axs[1].set_title('Average US 10Y Futures Returns around 10-Year Auctions')
axs[1].set_xlabel('Relative Day')
axs[1].set_ylabel('Average Return')
axs[1].axhline(0, color='gray', linestyle='--')
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Optional: Bar chart comparing average Pre- and Post-Auction Returns for both series.
labels = ['5-Year', '10-Year']
avg_pre = [np.mean(pre_returns_5Y), np.mean(pre_returns_10Y)]
avg_post = [np.mean(post_returns_5Y), np.mean(post_returns_10Y)]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width/2, avg_pre, width, label=f'Pre-Auction (Days -{window} to -1)')
ax.bar(x + width/2, avg_post, width, label=f'Post-Auction (Days +1 to +{window})')

ax.set_ylabel('Average Return')
ax.set_title('Average Pre vs. Post Auction Returns')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
