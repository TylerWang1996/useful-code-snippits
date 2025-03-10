# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 22:22:40 2025

@author: Tyler
"""

import pandas as pd

def get_sorted_unique_auction_dates(df, terms=['5-Year', '10-Year']):
    """
    Get sorted and unique auction dates for specified security terms and the 'Note' security type.

    Parameters:
        df (pd.DataFrame): DataFrame containing auction data.
        terms (list): List of security terms to filter by (default: ['5-Year', '10-Year']).

    Returns:
        pd.DataFrame: Sorted and deduplicated DataFrame with 'Security Term' and 'Auction Date'.
    """
    # Filter the DataFrame for specified terms and 'Note' security type
    filtered_df = df[(df['Security Term'].isin(terms)) & (df['Security Type'] == 'Note')]

    # Drop duplicates based on 'Security Term' and 'Auction Date'
    unique_df = filtered_df[['Security Type', 'Security Term', 'Auction Date']].drop_duplicates()

    # Sort the results by 'Security Term' and 'Auction Date'
    sorted_df = unique_df.sort_values(by=['Security Term', 'Auction Date']).reset_index(drop=True)

    return sorted_df

# Example usage:
auction_data = pd.read_csv('Auctions_Query_20150101_20250317.csv')
sorted_auction_dates = get_sorted_unique_auction_dates(auction_data)
