# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:27:46 2022

@author: czhang
"""

import pandas as pd
import numpy as np
from blp import blp


def import_data(start_date, end_date, tickers, fields):
    
    bquery = blp.BlpQuery().start()
    df_pull = bquery.bdh(tickers, fields,
                         start_date=start_date, end_date=end_date,)
    
    return df_pull


def bdp_data(tickers, fields):
    
    bquery = blp.BlpQuery().start()
    df_pull = bquery.bdp(tickers, fields)
    
    return df_pull


def data_update(l_tickers, fx_tickers, start_date, end_date):
    
    data = import_data(start_date, end_date, l_tickers, ['PX_Last'])
    fx_data = import_data(start_date, end_date, fx_tickers, ['PX_Last'])
    fx_data.rename(columns={'security': 'CRNCY', 'PX_Last': 'FX_Rate'}, inplace=True)
    data.to_csv('data_pull_update.csv')
    fx_data.to_csv('fx_data_pull_update.csv')
    
    data_new = pd.read_csv('data_pull_update.csv', index_col=0)
    fx_data_new = pd.read_csv('fx_data_pull_update.csv', index_col=0)
    
    data_new['date'] = pd.to_datetime(data_new['date'])
    fx_data_new['date'] = pd.to_datetime(fx_data_new['date'])
    cont_size = bdp_data(l_tickers, ['FUT_CONT_SIZE', 'CRNCY'])
    cont_size['CRNCY'] = cont_size['CRNCY'] + ' Curncy'
    data_new = data_new.merge(cont_size, left_on='security', right_on='security')
    data_new = data_new.merge(fx_data_new, left_on=['date', 'CRNCY'], right_on=['date', 'CRNCY'], how='left')
    data_new = data_new.fillna(1)
    
    data_master = pd.read_csv('data_pull_mastered.csv')
    data_master['date'] = pd.to_datetime(data_master['date'])
    data_updated = pd.concat([data_master, data_new])
    
    data_updated = data_updated[['date','security', 'PX_Last','FUT_CONT_SIZE', 'CRNCY', 'FX_Rate']] 
    # data_updated.reset_index(inplace=True, drop=True)
    data_updated = data_updated.sort_values(by=['date', 'security']).reset_index(drop=True)

    data_updated.to_csv('data_pull_updated.csv')
    
    return data_new
    
    
start_date_master ='20100101'
end_date_master = '20220531'

start_date_update ='20220601'
end_date_update = '20230209'
     
tickers = pd.read_csv('kiran tickers.csv')
l_tickers = list(tickers['Tickers'].values)
fx_tickers = ['GBP Curncy', 'EUR Curncy', 'JPY Curncy']
data_update(l_tickers, fx_tickers, start_date_update, end_date_update)
