from sys import path
path.insert(0,'../python')
from map_utils import *
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster,HeatMap, Fullscreen
from folium import plugins
import re
from async_request import make_request
from requests import Session

def check_zipcode_format(df,cp_column = 'CP'):
    ''' Checks the zipcodes column in the dataframe
        and ands a column to indicate if the zipcode has a good format
    '''
    zipcode_pattern = "^[0-9]{5}$"
    #creating a new column
    df['good_zip_code'] = False 
    df.loc[(df[cp_column].str.match(zipcode_pattern, na=False)), 'good_zip_code'] = True


def merge_coordinates(df,cp_column = 'CP',
                     zip_codes_path = '../data/zip_codes/codes_for_db1.csv'):

    zip_codes = pd.read_csv(zip_codes_path, index_col=0, dtype=str)
    zip_codes['CP'] = zip_codes['CP'].map(lambda x:  '0' + str(x) if (len(str(x)) == 4) else str(x) )
    zip_codes[['lat','lon']] = zip_codes[['lat','lon']].astype(float)
    zip_codes = zip_codes.drop_duplicates(subset = ['CP'], keep='first')[['CP','lat','lon','address']]
    merged_zipcodes = pd.merge(df, zip_codes, left_on=[cp_column], right_on=['CP'], how = 'left')
    merged_zipcodes['has_coords'] = True
    merged_zipcodes.loc[np.isnan(merged_zipcodes['lat']), 'has_coords'] = False
    return merged_zipcodes


def group_zip_codes(df):
    group_zip_codes = df.groupby(['CP','lat','lon', 'address']).has_coords.agg('count')
    return pd.DataFrame(group_zip_codes).reset_index().rename(columns={'has_coords': 'count'})


def get_associates_cluster_layer(df,
                                layer_name = 'Associates Cluster',show = False,
                                columns = ['CP', 'address']):
    popups = []
    table = df.loc[df.has_coords]

    for i in range(len(table)):
        nombre = table.iloc[i]['Número de personal']
        numero = table.iloc[i]['Nº pers.']
        CP = table.iloc[i]['CP']
        text =  f'''
        <ul>
        <li><b>CP:</b> {CP}</li>
        <li><b>Nombre de personal:</b> {nombre}</li>
        <li><b>Número de personal:</b> {numero}</li>
        </ul>

        '''
        text = folium.Html(text, script=True)
        popups.append(folium.Popup(text, max_width=2650))
        
    pt_lyr = folium.FeatureGroup(name = layer_name,show=show)
    pt_lyr.add_child(MarkerCluster(locations = table[['lat', 'lon']].astype(float).values, popups= popups))
    return pt_lyr


def get_associates_heatmap_layer(df,
                                layer_name = 'Associates Heatmap', show= False):
    
    table = df.loc[df.has_coords]
    pt_lyr = folium.FeatureGroup(name = layer_name, show=show)
    pt_lyr.add_child(HeatMap(table[['lat', 'lon']].astype(float).values, max_val = 1))
    
    return pt_lyr


def get_associates_zip_codes_layer(df,
                                layer_name = 'Associates ZipCodes', show= False):
    
    pt_lyr = folium.FeatureGroup(name = layer_name,show=show)

    table = group_zip_codes(df)
    for i in range(len(table)):
        point = table.iloc[i][['lat', 'lon']].values
        CP = table.iloc[i]['CP']
        num = table.iloc[i]['count']
        text = '<p>CP: {}</p><p>Asociados: {}</p>'.format(CP, num)   
        text =  f'''
        <ul>
        <li><b>CP:</b> {CP}</li>
        <li><b>Número de Asociados:</b> {num}</li>
        </ul>

        '''
        text = folium.Html(text, script=True)
        popup = folium.Popup(text, max_width=2650)
        pt_lyr.add_child(folium.CircleMarker(point,radius = 4,
                                popup=popup,
                                fill=True, # Set fill to True
                                fill_color='white',
                                color = 'red',
                                fill_opacity=1))
    return pt_lyr

def filter_coordinates(df,bounds):
    def filter_coords(coords, bounds):
        return list(map(lambda x: ispointinside(x, bounds) if not np.isnan(x[0]) else False, coords))
    df['in_bounds'] = filter_coords(df[['lat', 'lon']].values, bounds)


