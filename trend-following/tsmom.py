import pandas as pd
import numpy as np
from blp import blp
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
d_tickers = {'Equities': ['ES1 Index', 'NQ1 Index', 'Z 1 Index', 'VG1 Index', 'XU1 Index', 'NK1 Index'],
             'Commodities': ['CL1 Comdty', 'CO1 Comdty', 'NG1 Comdty', 'XB1 Comdty',
                             'HO1 Comdty', 'HG1 Comdty', 'LA1 Comdty', 'LX1 Comdty',
                             'C 1 Comdty', 'S 1 Comdty', 'BO1 Comdty', 'LC1 Comdty',
                             'W 1 Comdty', 'GC1 Comdty', 'PL1 Comdty'],
             'Rates': ['FV1 Comdty', 'TY1 Comdty', 'RX1 Comdty', 'JB1 Comdty'],
             'FX': ['ANT1 Curncy', 'IUS1 Curncy', 'CD1 Curncy', 'LAS1 Curncy', 'AD1 Curncy']}
def ex_ante_vol(data_series, com=60, freq='D'):
    ret_series = data_series.pct_change(periods=1)
    vol = ret_series.ewm(com=com).std()
    ann_vol = vol * np.sqrt(253)
    if freq == 'D':
        ann_vol_prd = ann_vol
    elif freq == 'M':
        ann_vol_prd = ann_vol.resample('BM').last().ffill()
    return ann_vol_prd
def cum_returns(data_series):
    ret_series = data_series.pct_change(periods=1)
    cum_rets = (1 + ret_series).cumprod()
    cum_rets.iloc[0] = 1
    return cum_rets
def tsmom_signal(asset_name, cum_returns, vol, lookback, vol_target=0.1):
    df = pd.concat([cum_returns, vol, cum_returns.pct_change(lookback)],
                      axis = 1,
                      keys = (['cum_returns', 'vol', 'lookback']))
    cum_col = df['cum_returns']
    vol_col = df['vol']
    lback_col = df['lookback']
    col_name = asset_name + " " + str(lookback) + " D"
    pnl = {pd.Timestamp(lback_col.index[lookback]): 0}
    size_dict = {pd.Timestamp(lback_col.index[lookback]): 1}
    for k, v in enumerate(lback_col):
        if k <= lookback:
            continue
        leverage = (vol_target/vol_col[k-1])
        if lback_col.iloc[k-1] > 0:
            pnl[lback_col.index[k]] = ((cum_col.iloc[k] / cum_col.iloc[k - 1]) - 1) * leverage
            size_dict[lback_col.index[k]] = leverage
        elif lback_col.iloc[k-1] < 0:
            size_dict[lback_col.index[k]] = leverage * -1
            pnl[lback_col.index[k]] = ((cum_col.iloc[k] / cum_col.iloc[k - 1]) - 1) * leverage * -1
    new_size = pd.DataFrame.from_dict(size_dict, orient='index', columns=[col_name])
    new_pnl = pd.DataFrame.from_dict(pnl, orient='index', columns=[col_name])
    return new_size, new_pnl
def tsmom_gearing(signal_returns):
    corr_m = signal_returns.ewm(com=60).corr()
    corr_m = corr_m.reset_index()
    corr_m.dropna(inplace=True)
    corr_grouped = corr_m.groupby(['level_0']).sum()
    corr_grouped['Gearing'] = corr_grouped.sum(axis=1)
    corr_grouped['Gearing'] = 1 / np.sqrt(corr_grouped['Gearing'])
    gearing = corr_grouped[['Gearing']]
    return gearing
def comb_pnl(pnl, asset_weights, gearing):
    sum_weights = asset_weights.sum(axis=1)
    df_comb_pnl = pnl.mul(gearing, axis=0)
    df_comb_pnl['PnL'] = df_comb_pnl.sum(axis=1)
    df_comb_pnl['Weight'] = sum_weights
    df_comb_pnl.dropna(inplace=True)
    df_return = df_comb_pnl[['PnL', 'Weight']].copy()
    return df_return
def total_return(asset_returns, exposure_size, period):
    asset_returns = asset_returns.pivot(index='date', columns='security', values='PX_Last')
    if period =='weekly':
        asset_returns = asset_returns.resample('W-WED').last()
    asset_returns.to_csv('asset_returns_inspect.csv')
    asset_returns = asset_returns.pct_change()
    asset_returns = asset_returns.shift(-1).fillna(0)
    df_return = exposure_size.copy()
    df_return['PnL'] = exposure_size.mul(asset_returns).round(2).dropna().sum(axis=1)
    df_return['Pct_PnL'] = df_return['PnL'] / 100000000
    return df_return
def tsmom_port(lookback_days, d_tickers, period=None):
    source = pd.read_csv('data_pull_updated.csv')
    data = source.pivot(index='date', columns='security', values='PX_Last').copy()
    data.dropna(subset=['ES1 Index'], inplace=True)
    data = data.fillna(method='ffill')
    data.reset_index(inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by=['date'])
    data = pd.melt(data, id_vars=['date'])
    data.rename(columns={'variable': 'security', 'value': 'PX_Last'}, inplace=True)
    fx_rate = source.pivot(index='date', columns='security', values='FX_Rate').copy()
    fx_rate = 1 / fx_rate
    fx_rate.dropna(subset=['ES1 Index'], inplace=True)
    fx_rate = fx_rate.fillna(method='ffill')
    fx_rate.reset_index(inplace=True)
    fx_rate['date'] = pd.to_datetime(fx_rate['date'])
    fx_rate.sort_values(by=['date'])
    fx_rate = pd.melt(fx_rate, id_vars=['date'])
    fx_rate.rename(columns={'variable': 'security', 'value': 'FX_Rate'}, inplace=True)
    tickers = []
    for asset in ['Equities', 'Commodities', 'Rates']:
        asset_ticker = d_tickers[asset]
        tickers.extend(asset_ticker)
    cont_size = source[['security', 'FUT_CONT_SIZE']].drop_duplicates().copy()
    cont_value = data.merge(cont_size, left_on='security', right_on='security')
    cont_value = cont_value.merge(fx_rate, left_on=['date', 'security'], right_on=['date', 'security'])
    cont_value['cont_value'] = cont_value['PX_Last'] * cont_value['FUT_CONT_SIZE'] * cont_value['FX_Rate']
    cont_value = cont_value[['date', 'security', 'cont_value']].copy()
    cont_value = cont_value.pivot(index='date', columns='security', values='cont_value')
    cont_value = cont_value[tickers].copy()
    d_asset_weight = {}
    d_asset_pnl = {}
    for ticker in tickers:
        dfs_pnl = []
        dfs_size = []
        df_data_ticker = data[data.security == ticker][['date', 'PX_Last']].copy()
        df_data_ticker.set_index('date', inplace=True)
        data_ticker = df_data_ticker['PX_Last']
        ticker_cum_ret = cum_returns(data_ticker)
        ticker_ex_ante_vol = ex_ante_vol(data_ticker)
        for i in lookback_days:
            ticker_size, ticker_pnl = tsmom_signal(ticker, ticker_cum_ret, ticker_ex_ante_vol, i)
            dfs_pnl.append(ticker_pnl)
            dfs_size.append(ticker_size)
        df_pnl = pd.concat(dfs_pnl, axis=1)
        df_pnl.dropna(inplace=True)
        gearing = tsmom_gearing(df_pnl)
        df_size = pd.concat(dfs_size, axis=1)
        df_gearing = df_size.join(gearing)
        df_gearing.dropna(inplace=True)
        df_weighted = df_gearing.iloc[:,:-1].mul(df_gearing['Gearing'], axis=0)
        df_comb_pnl = comb_pnl(df_pnl, df_weighted, df_gearing['Gearing'])
        d_asset_weight[ticker] = df_comb_pnl['Weight']
        d_asset_pnl[ticker] = df_comb_pnl['PnL']
    df_pnl = pd.DataFrame(d_asset_pnl)
    df_size = pd.DataFrame(d_asset_weight)
    d_classes = {}
    d_ind_weight = {}
    d_class_weight = {}
    d_class_pnl = {}
    for key in ['Equities', 'Commodities', 'Rates']:
        l_tickers = d_tickers[key]
        pnl_class = df_pnl[l_tickers].copy()
        pnl_class.dropna(inplace=True)
        gearing = tsmom_gearing(pnl_class)
        size_class = df_size[l_tickers].copy()
        gearing_class = size_class.join(gearing)
        gearing_class.dropna(inplace=True)
        weighted_class = gearing_class.iloc[:, :-1].mul(gearing_class['Gearing'], axis=0)
        class_pnl = comb_pnl(pnl_class, weighted_class, gearing_class['Gearing'])
        df_class = weighted_class.copy()
        df_class['PnL'] = class_pnl['PnL']
        df_class['CumPnL'] = df_class['PnL'] + 1
        df_class['CumPnL'] = df_class['CumPnL'].cumprod()
        d_class_weight[key] = class_pnl['Weight']
        d_ind_weight[key] = weighted_class.copy()
        d_class_pnl[key] = class_pnl['PnL']
        d_classes[key] = df_class
    df_class_pnl = pd.DataFrame(d_class_pnl)
    gearing = tsmom_gearing(df_class_pnl)
    df_class_size = pd.DataFrame(d_class_weight)
    df_gearing = df_class_size.join(gearing)
    df_gearing.dropna(inplace=True)
    df_weighted = df_gearing.iloc[:, :-1].mul(df_gearing['Gearing'], axis=0)
    df_comb_pnl = comb_pnl(df_class_pnl, df_weighted, df_gearing['Gearing'])
    dfs_weights = []
    for key in ['Equities', 'Commodities', 'Rates']:
        df_weight = d_ind_weight[key].copy()
        df_weight = df_weight.multiply(df_gearing['Gearing'], axis='index')
        df_weight.dropna(inplace=True)
        dfs_weights.append(df_weight)
    df_port = pd.concat(dfs_weights, axis=1)
    df_port = df_port.mul(100000000).div(cont_value).round(0)
    df_port.dropna(inplace=True)
    df_port.to_csv('contracts.csv')
    df_port = df_port.mul(cont_value).dropna()
    if period == 'weekly':
        df_port = df_port.resample('W-WED').last()
    df_output = total_return(data, df_port, period)
    df_output['CumPnL'] = df_output['Pct_PnL'] + 1
    df_output['CumPnL'] = df_output['CumPnL'].cumprod()
    df_test = df_output[['Pct_PnL']].tail(1000).copy()
    df_test = df_test.std() * np.sqrt(253)
    print(df_test)
    d_classes['Equities'].to_csv('equities_results.csv')
    d_classes['Commodities'].to_csv('commodities_results.csv')
    d_classes['Rates'].to_csv('rates_results.csv')
    #d_classes['FX'].to_csv('FX_results.csv')
    df_output.to_csv('cta_results.csv')
    return d_classes

def tsmom_breakpoints(lookback_days, d_tickers):
    source = pd.read_csv('data_pull_updated.csv')
    data = source.pivot(index='date', columns='security', values='PX_Last').copy()
    data.dropna(subset=['ES1 Index'], inplace=True)
    data = data.fillna(method='ffill')
    data.reset_index(inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by=['date'])
    data = pd.melt(data, id_vars=['date'])
    data.rename(columns={'variable': 'security', 'value': 'PX_Last'}, inplace=True)
    tickers = []
    for asset in ['Equities', 'Commodities', 'Rates']:
        asset_ticker = d_tickers[asset]
        tickers.extend(asset_ticker)
    dfs = []
    for ticker in tickers:
        ticker_data = data[data['security'] == ticker]
        ticker_data_filtered_now = ticker_data.iloc[-1].to_frame().transpose()
        ticker_data_filtered_one = ticker_data.iloc[-22].to_frame().transpose()
        ticker_data_filtered_two = ticker_data.iloc[-85].to_frame().transpose()
        ticker_data_filtered_three = ticker_data.iloc[-253].to_frame().transpose()
        ticker_data_filtered = pd.concat([ticker_data_filtered_now,
                                          ticker_data_filtered_one,
                                          ticker_data_filtered_two,
                                          ticker_data_filtered_three])
        ticker_data_filtered.reset_index(inplace=True)
        dfs.append(ticker_data_filtered)
    df_concat = pd.concat(dfs).sort_values(by=['date'])
    df_concat = df_concat.pivot(index='date', columns='security', values='PX_Last')
    df_concat = df_concat[tickers]
    df_div = df_concat.copy()
    df_div.iloc[:] = df_div.iloc[:].div(df_div.iloc[-1])
    df_div = df_div - 1
    df_div.drop(df_div.tail(1).index, inplace=True)
    df_concat.to_csv('breakpoints.csv')
    df_pct = df_concat.div(df_concat.iloc[-1]) - 1
    df_pct.drop(df_pct.tail(1).index,inplace=True)
    df_pct.to_csv('breakpoints_pct.csv')
    
    return df_concat

if __name__ == "__main__":
    test = tsmom_port([21, 84, 252], d_tickers)
    test2 = tsmom_breakpoints([21, 84, 252], d_tickers)