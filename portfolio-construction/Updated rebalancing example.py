import pandas as pd
import numpy as np


class PortfolioBacktester:
    def __init__(self, prices, weights, total_returns):
        """
        Initialize the PortfolioBacktester with prices, weights, and total_returns dataframes.
        """
        self.prices = prices
        self.weights = weights
        self.total_returns = total_returns

    def _extend_prices(self):
        """
        Extend the prices dataframe to include all calendar days and forward-fill missing data.
        """
        all_days = pd.date_range(self.prices.index.min(), self.prices.index.max())
        extended_prices = self.prices.reindex(all_days).ffill()
        return extended_prices

    def _calculate_target_share_counts(self, extended_prices):
        """
        Calculate the target share counts for each stock on rebalancing dates.
        """
        rebalancing_prices = extended_prices.loc[self.weights.index]
        target_share_counts = self.weights.div(rebalancing_prices, axis=0) * 10000

        return target_share_counts


    def _extend_share_counts(self, target_share_counts):
        """
        Extend the share counts dataframe to include all calendar days and forward-fill missing data.
        """
        extended_share_counts = target_share_counts.reindex(
            self.total_returns.index
        ).ffill()
        
        return extended_share_counts


    def _compute_daily_portfolio_value(self, extended_share_counts):
        """
        Compute the daily portfolio value using extended share counts and total returns.
        """
        extended_prices = self._extend_prices()
        aligned_share_counts = extended_share_counts.loc[self.prices.index]
        aligned_prices = extended_prices.loc[self.prices.index]
        portfolio_values = aligned_share_counts * aligned_prices
        portfolio_weights = portfolio_values.div(portfolio_values.sum(axis=1), axis=0)
        daily_portfolio_value = (portfolio_weights * self.total_returns).sum(axis=1)
        daily_portfolio_value = daily_portfolio_value.dropna()
        
        return daily_portfolio_value


    def _calculate_daily_portfolio_returns(self, daily_portfolio_value):
        """
        Calculate the daily portfolio returns.
        """
        daily_portfolio_returns = daily_portfolio_value.pct_change()
        daily_portfolio_returns = daily_portfolio_returns.replace(
            [np.inf, -np.inf], np.nan).dropna(axis=0)
        
        return daily_portfolio_returns

    def _calculate_cumulative_portfolio_returns(self, daily_portfolio_returns):
        """
        Calculate the cumulative portfolio returns.
        """
        cumulative_portfolio_returns = (daily_portfolio_returns + 1).cumprod()
        
        return cumulative_portfolio_returns

    def _calculate_portfolio_value_index(self, cumulative_portfolio_returns):
        """
        Calculate the portfolio value index starting with an initial investment of 100 dollars.
        """
        portfolio_value_index = cumulative_portfolio_returns * 100
        return portfolio_value_index

    def run_backtest(self):
        """
        Run the backtesting process and return the portfolio value index.
        """
        extended_prices = self._extend_prices()
        target_share_counts = self._calculate_target_share_counts(extended_prices)
        extended_share_counts = self._extend_share_counts(target_share_counts)
        daily_portfolio_value = self._compute_daily_portfolio_value(
            extended_share_counts
        )
        daily_portfolio_returns = self._calculate_daily_portfolio_returns(
            daily_portfolio_value
        )
        cumulative_portfolio_returns = self._calculate_cumulative_portfolio_returns(
            daily_portfolio_returns
        )
        portfolio_value_index = self._calculate_portfolio_value_index(
            cumulative_portfolio_returns
        )

        return portfolio_value_index


if __name__ == "__main__":
    
    # Generate random example data
    date_range = pd.date_range('2020-01-01', '2020-12-31', freq='B')
    stocks = ['AAPL', 'GOOG', 'TSLA']
    
    # Generate random prices
    prices_data = np.random.uniform(100, 500, size=(len(date_range), len(stocks)))
    prices = pd.DataFrame(prices_data, index=date_range, columns=stocks)
    
    # Generate random weights
    rebalance_dates = pd.date_range('2020-01-01', '2020-12-31', freq='3M')
    weights_data = np.random.uniform(0, 1, size=(len(rebalance_dates), len(stocks)))
    weights = pd.DataFrame(weights_data, index=rebalance_dates, columns=stocks)
    weights = weights.div(weights.sum(axis=1), axis=0)
    
    # Generate random daily % total returns
    total_returns = prices.copy()
    
    # Create the backtester and run the backtest
    backtester = PortfolioBacktester(prices, weights, total_returns)
    portfolio_value_index = backtester.run_backtest()
    
    print(portfolio_value_index)
