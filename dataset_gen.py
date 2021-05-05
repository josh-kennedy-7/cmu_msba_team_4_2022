import pygrib
import eccodes
import xarray
import glob
import pandas as pd
import os
import numpy as np

d_months = {
     'Jan':1
    ,'Feb':2
    ,'Mar':3
    ,'Apr':4
    ,'May':5
    ,'Jun':6
    ,'Jul':7
    ,'Aug':8
    ,'Sep':9
    ,'Oct':10
    ,'Nov':11
    ,'Dec':12
    ,'January':1
    ,'February':2
    ,'March':3
    ,'April':4
    ,'May':5
    ,'June':6
    ,'July':7
    ,'August':8
    ,'September':9
    ,'October':10
    ,'November':11
    ,'December':12
}

ds = pygrib.open('adaptor.mars.internal-1619785219.3716667-16287-5-b23cc0ac-0d68-448b-b948-ffe7a3a76e82.grib')
grb = ds.select()
df_s = pd.DataFrame()
for i in range(len(grb)):
    staging_list = {
         'parameterName': grb[i]['parameterName']
        ,'year': grb[i]['year']
        ,'month' : grb[i]['month']
        ,'average': grb[i]['average']
        ,'minimum': grb[i]['minimum']
        ,'maximum': grb[i]['maximum']
        ,'standardDeviation': grb[i]['standardDeviation']
    }
    df_s = df_s.append(staging_list, ignore_index=True)

df_s.parameterName.unique()
df_p = df_s.pivot(index=['year','month'], columns='parameterName', values=['average','minimum','maximum','standardDeviation'])
df_p.reset_index(inplace=True)
df_p.columns = [' '.join(col).strip() for col in df_p.columns.values]
df_p.columns = df_p.columns.str.replace(' ','_').str.lower()
env_df = df_p.copy()


# https://fred.stlouisfed.org/series/PCU3117103117102
df = pd.read_csv('ppi_seafood_fresh_frozen_processing.csv')
df = df.rename(columns={'PCU3117103117102': 'seafood_fresh_frozen_processing_ppi'})
df['month'] = pd.DatetimeIndex(df['DATE']).month
df['year'] = pd.DatetimeIndex(df['DATE']).year
ppi_seafood_fresh_frozen_processing = df[['year', 'month', 'seafood_fresh_frozen_processing_ppi']].copy()


# https://investor.thaiunion.com/raw_material.html
df = pd.read_csv('monthly_tuna_prices.csv')
df.month = df.month.map(d_months)
df = df.rename(columns={'monthly_price': 'skipjack_tuna_price'})
monthly_skipjack_tuna_prices = df[['year', 'month', 'skipjack_tuna_price']].copy()


# http://www.fao.org/worldfoodsituation/foodpricesindex/en/
df = pd.read_csv('Food_price_indices_data_apr945.csv',skiprows=2)
df.drop(0, inplace=True)
df['month'] = pd.DatetimeIndex(df['Date']).month.astype('int64')
df['year'] = pd.DatetimeIndex(df['Date']).year.astype('int64')
df = df.rename(columns={
                        'Food Price Index': 'fao_food_price_index'
                        ,'Meat': 'fao_meat_price_index'
                        ,'Dairy': 'fao_dairy_price_index'
                        })
fao_food_price_indices = df[['year', 'month', 'fao_food_price_index','fao_meat_price_index','fao_dairy_price_index']].copy()


# https://fred.stlouisfed.org/series/PSALMUSDM
df = pd.read_csv('global_price_of_fish.csv')
df['month'] = pd.DatetimeIndex(df['DATE']).month
df['year'] = pd.DatetimeIndex(df['DATE']).year
df = df.rename(columns={'PSALMUSDM': 'global_fish_price_index'})
global_fish_prices = df[['year', 'month', 'global_fish_price_index']].copy()


# https://fred.stlouisfed.org/series/PCU445200445200102
df = pd.read_csv('ppi_seafood_fish_seafood_markets.csv')
df = df.rename(columns={'PCU445200445200102': 'seafood_fish_markets_ppi'})
df['month'] = pd.DatetimeIndex(df['DATE']).month
df['year'] = pd.DatetimeIndex(df['DATE']).year
ppi_seafood_fish_markets = df[['year', 'month', 'seafood_fish_markets_ppi']].copy()


# https://fred.stlouisfed.org/series/APU0000707111
df = pd.read_csv('tuna_light_chunk_per_pound_us_cities.csv')
df = df.rename(columns={'APU0000707111': 'tuna_light_chunk_price_per_pound_us_cities'})
df['month'] = pd.DatetimeIndex(df['DATE']).month
df['year'] = pd.DatetimeIndex(df['DATE']).year
tuna_light_chunk_price_us = df[['year', 'month', 'tuna_light_chunk_price_per_pound_us_cities']].copy()


# https://fred.stlouisfed.org/series/IR01000
df = pd.read_csv('import_price_index_fish_shellfish.csv')
df = df.rename(columns={'IR01000': 'import_price_index_fish_shellfish'})
df['month'] = pd.DatetimeIndex(df['DATE']).month
df['year'] = pd.DatetimeIndex(df['DATE']).year
import_price_index_fish_shellfish = df[['year', 'month', 'import_price_index_fish_shellfish']].copy()


# https://fred.stlouisfed.org/series/IQ01000
df = pd.read_csv('export_price_index_fish_shellfish.csv')
df = df.rename(columns={'IQ01000': 'export_price_index_fish_shellfish'})
df['month'] = pd.DatetimeIndex(df['DATE']).month
df['year'] = pd.DatetimeIndex(df['DATE']).year
export_price_index_fish_shellfish = df[['year', 'month', 'export_price_index_fish_shellfish']].copy()


# https://fred.stlouisfed.org/series/PSHRIUSDM
df = pd.read_csv('global_price_of_shrimp.csv')
df = df.rename(columns={'PSHRIUSDM': 'global_price_of_shrimp'})
df['month'] = pd.DatetimeIndex(df['DATE']).month
df['year'] = pd.DatetimeIndex(df['DATE']).year
global_price_of_shrimp = df[['year', 'month', 'global_price_of_shrimp']].copy()


# https://my.ibisworld.com/cn/en/industry/1361/key-statistics
df = pd.read_csv('global_price_of_shrimp.csv')
df = df.rename(columns={'PSHRIUSDM': 'global_price_of_shrimp'})
df['month'] = pd.DatetimeIndex(df['DATE']).month
df['year'] = pd.DatetimeIndex(df['DATE']).year
global_price_of_shrimp = df[['year', 'month', 'global_price_of_shrimp']].copy()


# https://my.ibisworld.com/cn/en/industry/1361/key-statistics
df = pd.read_csv('bts_air_cargo_summary.csv')
df['month'] = df.Month.map(d_months)
df = df.rename(columns={
                         'Year': 'year'
                        ,'Domestic': 'domestic_air_cargo_volume'
                        ,'Atlantic': 'atlantic_air_cargo_volume'
                        ,'Latin America': 'latam_air_cargo_volume'
                        ,'Pacific': 'pacific_air_cargo_volume'
                        ,'International': 'international_air_cargo_volume'
                        ,'Total': 'total_air_cargo_volume'
                        })
bts_air_cargo_summary = df[['year', 'month', 'domestic_air_cargo_volume','atlantic_air_cargo_volume','latam_air_cargo_volume','pacific_air_cargo_volume','international_air_cargo_volume','total_air_cargo_volume']].copy()


# https://nsidc.org/data/g02135
path = r'ice/'
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
df = pd.concat(li, axis=0, ignore_index=True)
df['month'] = df[' mo']
df_p = df.pivot(index=['year','month'], columns=' region', values=[' extent','   area'])
df_p.reset_index(inplace=True)
df_p.columns = [' '.join(col).strip() for col in df_p.columns.values]
df_p.columns = df_p.columns.str.replace(' ','_').str.lower()
ice_df = df_p.copy()

# water temperature data
# https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v5/access/timeseries/
coord = '00N.30N'
coordinates = [
 '00N.30N'
,'00N.90N'
,'20N.90N'
,'20S.20N'
,'30N.60N'
,'30S.00N'
,'60N.90N'
,'60S.30S'
,'60S.60N'
,'90S.00N'
,'90S.20S'
,'90S.60S'
,'90S.90N'
]

ocean_temps = pd.DataFrame
for coord in coordinates:
    url=f'https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v5/access/timeseries/aravg.mon.ocean.{coord}.v5.0.0.202103.asc'
    c = pd.read_csv(url, sep='\t', lineterminator='\n', error_bad_lines=False, header=None)
    df = c[0].str.split(expand=True)
    df.columns = ['year','month','temp_anomaly','total_error','high_error_variance','low_error_variance','bias_error_variance','diag_var_1','diag_var_2','diag_var_3']
    keep_same = ['year','month']
    df = df.rename(columns={col: "ocean__" + col + "__"+coord for col in df.columns if col not in keep_same})
    if ocean_temps.empty:
        ocean_temps = df.copy()
    else:
        ocean_temps = pd.merge(ocean_temps, df,  how='left', left_on=['year','month'], right_on = ['year','month'])

land_temps = pd.DataFrame
for coord in coordinates:
    url=f'https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v5/access/timeseries/aravg.mon.land.{coord}.v5.0.0.202103.asc'
    c = pd.read_csv(url, sep='\t', lineterminator='\n', error_bad_lines=False, header=None)
    df = c[0].str.split(expand=True)
    df.columns = ['year','month','temp_anomaly','total_error','high_error_variance','low_error_variance','bias_error_variance','diag_var_1','diag_var_2','diag_var_3']
    keep_same = ['year','month']
    df = df.rename(columns={col: "land__" + col + "__"+coord for col in df.columns if col not in keep_same})
    if land_temps.empty:
        land_temps = df.copy()
    else:
        land_temps = pd.merge(land_temps, df,  how='left', left_on=['year','month'], right_on = ['year','month'])

datasets = [
     land_temps
    ,env_df
    ,ice_df
    ,ppi_seafood_fresh_frozen_processing
    ,monthly_skipjack_tuna_prices
    ,fao_food_price_indices
    ,global_fish_prices
    ,ppi_seafood_fish_markets
    ,tuna_light_chunk_price_us
    ,import_price_index_fish_shellfish
    ,export_price_index_fish_shellfish
    ,global_price_of_shrimp
    ,global_price_of_shrimp
    ,bts_air_cargo_summary
]

# starting with this one because it has largest date range
consolidated_tuna_data = ocean_temps.copy()


for d in datasets:
    d['year'] = d['year'].astype('str').replace('\.0', '', regex=True)
    d['month'] = d['month'].astype('str').replace('\.0', '', regex=True)
    consolidated_tuna_data = pd.merge(consolidated_tuna_data, d,  how='left', left_on=['year','month'], right_on = ['year','month'])

for i in consolidated_tuna_data.columns:
    print(i)

consolidated_tuna_data.to_csv('consolidated_tuna_data.csv')
