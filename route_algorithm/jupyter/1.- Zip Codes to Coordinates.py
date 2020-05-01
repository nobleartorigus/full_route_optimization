#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sys import path
path.insert(0,'../../python')
from map_utils import *
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster,HeatMap, Fullscreen
from folium import plugins
import re
from async_request import make_request
from requests import Session


# In[2]:


associates = pd.read_excel('../../data/aguascalientes/Direct headcount with ZIP codes 051219.xlsx',dtype=str)


# In[3]:


associates.head()


# In[ ]:





# ## Checamos si los codigos postales son correctos

# In[4]:


def check_zipcode_format(df,cp_column = 'CP'):
    ''' Checks the zipcodes column in the dataframe
        and ands a column to indicate if the zipcode has a good format
    '''
    zipcode_pattern = "^[0-9]{5}$"
    #creating a new column
    df['good_zip_code'] = False 
    df.loc[(df['Cód.postal'].str.match(zipcode_pattern)), 'good_zip_code'] = True
    

check_zipcode_format(associates, cp_column = 'Cód.postal')


# In[5]:


associates.loc[associates.good_zip_code == False]


# ## Obtenemos las coordenadas de los codigos postales

# In[6]:


def merge_coordinates(df,cp_column = 'CP',
                     zip_codes_path = '../../data/zip_codes/codes_for_db1.csv'):

    zip_codes = pd.read_csv(zip_codes_path, index_col=0, dtype=str)
    zip_codes['CP'] = zip_codes['CP'].map(lambda x:  '0' + str(x) if (len(str(x)) == 4) else str(x) )
    zip_codes[['lat','lon']] = zip_codes[['lat','lon']].astype(float)
    zip_codes = zip_codes.drop_duplicates(subset = ['CP'], keep='first')
    merged_zipcodes = pd.merge(associates, zip_codes, left_on=[cp_column], right_on=['CP'], how = 'left')
    merged_zipcodes['has_coords'] = True
    merged_zipcodes.loc[np.isnan(merged_zipcodes['lat']), 'has_coords'] = False
    return merged_zipcodes


# In[7]:


associates = merge_coordinates(associates,cp_column = 'Cód.postal')


# ## Agrupamos los codigos postales similates

# In[8]:


def group_zip_codes(df):
    group_zip_codes = df.groupby(['CP','lat','lon', 'address']).has_coords.agg('count')
    return pd.DataFrame(group_zip_codes).reset_index().rename(columns={'has_coords': 'count'})


# In[9]:


grouped_zip_codes = group_zip_codes(associates)


# In[10]:


def get_associates_cluster_layer(df,
                                layer_name = 'Associates Cluster',show = False):
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


# In[11]:


def get_associates_heatmap_layer(df,
                                layer_name = 'Associates Heatmap', show= False):
    
    table = df.loc[df.has_coords]
    pt_lyr = folium.FeatureGroup(name = layer_name, show=show)
    pt_lyr.add_child(HeatMap(table[['lat', 'lon']].astype(float).values, max_val = 1))
    
    return pt_lyr


# In[12]:


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


# In[13]:


tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Mapeo de Asociados Aguascalientes', show = True).add_to(m)

text = '<h4>Planta Bosch Aguascalientes</h4>'
text = folium.Html(text, script=True)
popup = folium.Popup(text, max_width=2650)
folium.Marker([21.967103, -102.284380], icon = folium.Icon(color='red'), popup=popup).add_to(m)

get_associates_zip_codes_layer(associates, show = True).add_to(m)
get_associates_cluster_layer(associates).add_to(m)
get_associates_heatmap_layer(associates).add_to(m)

Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())
m
#m.save('../maps/Mapeo Asociados Aguascalientes.html')


# ### Coordinates filtering

# In[14]:


bounds = [  [21.611472417139705,-102.87322998046875],
            [21.611472417139705,-101.65924072265625],
            [22.550610920226646,-101.65924072265625],
            [ 22.550610920226646,-102.87322998046875],
            [21.611472417139705,-102.87322998046875]]


# In[15]:


def filter_coordinates(df):
    def filter_coords(coords, bounds):
        return list(map(lambda x: ispointinside(x, bounds) if not np.isnan(x[0]) else False, coords))
    df['in_bounds'] = filter_coords(df[['lat', 'lon']].values, bounds)


# In[16]:


filter_coordinates(associates)


# In[17]:


tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Mapeo de Asociados Aguascalientes', show = True).add_to(m)

text = '<h4>Planta Bosch Aguascalientes</h4>'
text = folium.Html(text, script=True)
popup = folium.Popup(text, max_width=2650)
folium.Marker([21.967103, -102.284380], icon = folium.Icon(color='red'), popup=popup).add_to(m)


table = associates.loc[associates.in_bounds]

get_associates_zip_codes_layer(table, show = True).add_to(m)
get_associates_cluster_layer(table).add_to(m)
get_associates_heatmap_layer(table).add_to(m)

Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())
m
#m.save('../maps/Mapeo Asociados Aguascalientes.html')


# In[ ]:





# ### MONGO

# In[18]:


import pymongo
from pymongo import MongoClient


# In[23]:


mongoClient = MongoClient('localhost',27017)
#db = mongoClient.prueba
db = mongoClient.route_optimization_api
collection = db.associates


# In[34]:


#associates_dict = [{"personal_number": 62000048, "name": "MALDONADO ZAMORA JOSE", "zipcode": 20324}, {"personal_number": 62000060, "name": "BENITEZ MEDINA BRAYAN ALEJANDRO", "zipcode": 20925}]


# In[38]:


#associates_dict[0]
#collection.insert(associates_dict)


# In[20]:


grouped_zip_codes


# In[65]:


for i in prueba_zip.iloc[i]:
    print(i)


# In[21]:


full_zipcodes_dict = grouped_zip_codes.to_dict(orient = 'index')


# In[22]:


full_zipcodes_dict


# In[24]:


""
for e in full_zipcodes_dict:
    try: 
        #print(zip_dict[e])
        collection.insert_one(full_zipcodes_dict[e])
    except: 
        print('ObjectId is not iterable')
""

#collection.insert_many(full_zipcodes_dict)

