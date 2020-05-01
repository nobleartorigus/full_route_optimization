import sys
sys.path.append('route_algorithm/python')

import json
from map_utils import *
import folium
from folium.plugins import MarkerCluster,HeatMap, Fullscreen
from folium import plugins
from requests import Session
from async_request import make_request
from requests import Session
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from zip_codes_to_coordinates import *
from sklearn.cluster import DBSCAN
from numpy import radians, asarray, unique ,where, ones, zeros, std, append
from pandas import Series
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from map_utils import getDistance2
from sklearn.cluster import KMeans

from matplotlib import cm
from matplotlib.colors import Normalize

# -------------------- Functions --------------------------------------------------

def get_all_stops(routes):
    stops = []
    for route in routes:
        for stop in route['stops'][:-1]:
            stops.append(stop)
    stops = stops
    return stops

def get_stops_coords(stops):
    return list(map(lambda stop:stop['coords'], stops))

def is_repeated_stop(stop_coords, all_coords, distance):
    if len(all_coords) == 0: return False
    tree = BallTree(radians(all_coords), leaf_size=2, metric = 'haversine')  
    result = tree.query_radius(radians([stop_coords]), r=calculate_radius(distance), count_only=True)[0]
    if result == 0: return False
    else: return True
    
#stops that are not near from another one
def get_good_stops(stops):
    good_coords = []
    good_stops = []
    for i,stop in enumerate(stops):
        if not is_repeated_stop(stop['coords'], good_coords, 200):
            good_coords.append(stop['coords'])
            good_stops.append(stop)
    return good_stops

def calculate_radius(distance):
    kms_per_radian = 6371.0088
    return (distance*.001) / kms_per_radian

def get_centermost_point(cluster):
    '''
    - Recibe un arreglo con las coordenadas del cluster
    - Retorna el centroide de ese cluster
    '''
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    #centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return asarray(centroid)

def radians_to_meters(radians):
    kms_per_radian = 6371.0088
    return (radians*kms_per_radian) / 0.001

def disperse_coords(df,max_radius = 750):
    random_coords = []
    for i in range(len(df)):
        zip_code = df.iloc[i]
        coords = zip_code[['lat','lon']].values
        count = zip_code['count']
        for j in range(count):
            bearing = np.random.randint(0,360)
            radius = np.random.random()*max_radius
            random_coord = destinationPoint(coords, radius, bearing)
            random_coords.append(random_coord)
    return random_coords

def get_stop_coincidence(associates_coords, stops_coords, distance):

    
    tree = BallTree(radians(associates_coords), leaf_size=2, metric = 'haversine')  
    coincidence, distances = tree.query_radius(radians(stops_coords), r=calculate_radius(distance), count_only=False,
                                   return_distance = True)
    distances = radians_to_meters(distances)
    
    stops_count = np.zeros(len(stops_coords))
    stops_coincidence = [[] for i in range(len(stops_coords))]

    #devide all the coincident coordinates to every stop
    for i in range(len(associates_coords)):
        coincidence_search = [np.where(con == i)[0] for con in coincidence]
        coincidence_search = list(map(lambda x: [x[0], x[1]], enumerate(coincidence_search)))
        coincidence_search = list(filter(lambda x: len(x[1]) == 1, coincidence_search))
        coincidence_search = [[i, con[0]] for i, con in coincidence_search]
        if not coincidence_search:
            continue
        if len(coincidence_search) == 1:
            stop_index, ind =  coincidence_search[0]
            stops_count[stop_index]+=1
            stops_coincidence[stop_index].append(i)
        else:
            stop_indexes = np.asarray([stop_index for stop_index, dist_index in coincidence_search])
            dist = np.asarray([distances[stop_index][dist_index] for stop_index, dist_index in coincidence_search])
            #stop_indexes = stop_indexes[dist.argsort()]
            stop_index = stops_count[stop_indexes].argmin()
            stop_index = stop_indexes[stop_index]
            stops_count[stop_index]+=1
            stops_coincidence[stop_index].append(i)
            
    return stops_coincidence

def select_stops(stop_coincidence, min_samples):
    coincidence_sum = np.asarray([len(con) for con in stop_coincidence])
    return list(np.where(coincidence_sum >= min_samples)[0])

def calculate_DBSCAN(coords, distance, min_samples):
    '''
    - Ejecuta el algoritmo de DBSCAN para obtener los clusters más importantes
    - Retorna los elementos de cada cluster, los centroides y los elementos fuera de los clusters
    '''
    coords = asarray(coords)
    kms_per_radian = 6371.0088
    epsilon = (distance*.001) / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(radians(coords))
    cluster_labels = db.labels_
    labels = unique(cluster_labels)
    clusters = Series([coords[where(cluster_labels == label)[0]] for label in labels])
    noise_index = where(labels == -1)
    mask = zeros(len(clusters),dtype=bool) 
    mask[noise_index] = True

    noise = clusters[mask]
    if len(noise) != 0:
        noise = noise[0]
    clusters = clusters[~mask]

    centers = clusters.map(get_centermost_point)
    
    return clusters, centers, noise

def calculate_clusters_DBSCAN(coords, distance, min_samples):
    '''
    - Recibe un arreglo con coordenadas, una distancia máxima y un mínimo de samples
    - Ejecuta el algoritmo de DBSCAN de manera recursiva para que todos los clusters tengan una varianza uniforme
    - Retorna los elementos de cada cluster, los centroides y los elementos fuera de los clusters
    '''
    distance_array = asarray([100, 500, 1000, 1500, 2500, 5000])
    clusters, centers, noise = calculate_DBSCAN(coords, distance, min_samples)

    new_clusters = []
    new_centers = []
    for i, group in enumerate(zip(clusters, centers)):
        cluster, center = group
        distances = getDistance2(center, cluster)
        deviation = std(distances)
        
        index = where(deviation > distance*.4)[0]
        if len(index) > 0:
            index_dist = where(distance_array == distance)[0]
            if len(index_dist) != 0:
                index_dist = index_dist[0]
                if index_dist != 0:
                    distance1 = distance_array[index_dist -1]
                    clusters1, centers1, noise1 = calculate_clusters_DBSCAN(cluster, distance1, min_samples)
                    for group1 in zip(clusters1, centers1):
                        cluster1, center1 = group1
                        new_clusters.append(cluster1)
                        new_centers.append(center1)      
                    #centers = centers.append(centers1, ignore_index=True)
                    #clusters = clusters.append(clusters1, ignore_index=True)
                    if len(noise) and len(noise1):
                        noise = append(noise,noise1, axis =0)
                    elif len(noise):
                        noise = noise
                    else:
                        noise = noise1
                    continue
        new_clusters.append(cluster)
        new_centers.append(center)
                
    return new_clusters, new_centers, noise

def get_clusters(coords, min_samples, max_distance, iterations= 10):
    distances = np.cumsum([max_distance // iterations for i in range(iterations)])
    
    new_clusters, new_centers, noise = calculate_clusters_DBSCAN(coords, distances[0], min_samples)
    clusters = new_clusters
    if len(noise):
        for distance in distances[1:]:
            new_clusters, new_centers, noise = calculate_clusters_DBSCAN(noise, distance, min_samples)
            if new_clusters:
                clusters = clusters + new_clusters
    return clusters, noise


def get_coverage(coincidence, associates_coords):
    covered_associates_index = np.concatenate(coincidence)
    covered_associates_index.sort()
    mask = np.ones(len(associates_coords),dtype=bool) #np.ones_like(a,dtype=bool)
    mask[covered_associates_index] = False
    not_covered_associates_index = np.arange(len(associates_coords))[mask]
    
    covered_associates = np.asarray(associates_coords)[covered_associates_index]
    not_covered_associates = np.asarray(associates_coords)[not_covered_associates_index]
    return covered_associates, not_covered_associates


def get_associates_clusters(associates_coords, stop_coords, distance, min_samples):
    stop_coincidence = get_stop_coincidence(associates_coords, stop_coords, distance)
    selected_stops_index = select_stops(stop_coincidence, min_samples)
    selected_stops = np.asarray(stop_coords)[selected_stops_index]
    if len(selected_stops_index):
        covered_associates, not_covered_associates = get_coverage(np.asarray(stop_coincidence)[selected_stops_index], associates_coords)
        coords = not_covered_associates
    else:
        coords = associates_coords
    clusters, noise =  get_clusters(coords, min_samples, distance, iterations= 10)
    cluster_centers = np.asarray([get_centermost_point(cluster) for cluster in clusters])
    
    all_clusters = list(selected_stops) + cluster_centers.tolist()
    
    cluster_coincidence = get_stop_coincidence(associates_coords, all_clusters, distance)
    selected_cluster_index = select_stops(cluster_coincidence, min_samples)
    
    selected_clusters = np.asarray(all_clusters)[selected_cluster_index]
    selected_clusters_coincidence = np.asarray(cluster_coincidence)[selected_cluster_index]
    return selected_clusters, selected_clusters_coincidence


associates = pd.read_excel('route_algorithm/data/aguascalientes/example.xlsx',dtype=str)
associates_columns = associates.columns.tolist()

#check the format of the zipcodes
check_zipcode_format(associates, cp_column = 'Cód.postal')

#merge the zipcodes with the coordinates database
associates = merge_coordinates(associates,cp_column = 'Cód.postal')

#filtering coordinates
bounds = [  [21.611472417139705,-102.87322998046875],
            [21.611472417139705,-101.65924072265625],
            [22.550610920226646,-101.65924072265625],
            [ 22.550610920226646,-102.87322998046875],
            [21.611472417139705,-102.87322998046875]]

filter_coordinates(associates, bounds)

with open('route_algorithm/data/aguascalientes/routes_polylines_ags.json') as json_file:
    routes = json.load(json_file)

# ## Obtaining all the stops from the routes

stops = get_all_stops(routes)
good_stops = get_good_stops(stops)

filter_coordinates(associates, bounds)
filter_associates = associates.loc[associates.in_bounds]
fliter_coords = filter_associates[['lat','lon']].values

grouped_zipcodes = group_zip_codes(associates)
filter_coordinates(grouped_zipcodes, bounds)
filter_associates = grouped_zipcodes.loc[grouped_zipcodes.in_bounds]
fliter_coords = filter_associates[['lat','lon']].values


#Get route stops

distance = 1000
min_samples = 5

dispersed_zip_codes = disperse_coords(filter_associates,max_radius = 750)
stop_coincidence = get_stop_coincidence(dispersed_zip_codes, get_stops_coords(good_stops), distance)
selected_stops_index = select_stops(stop_coincidence, min_samples)
selected_stops = np.asarray(good_stops)[selected_stops_index]

covered_associates_index = np.concatenate(np.asarray(stop_coincidence)[selected_stops_index])
covered_associates_index.sort()

mask = np.ones(len(dispersed_zip_codes),dtype=bool) #np.ones_like(a,dtype=bool)
mask[covered_associates_index] = False
not_covered_associates_index = np.arange(len(dispersed_zip_codes))[mask]

covered_associates = np.asarray(dispersed_zip_codes)[covered_associates_index]
not_covered_associates = np.asarray(dispersed_zip_codes)[not_covered_associates_index]

tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Mapeo Asociados Toluca', show = True).add_to(m)
 
for i, point in enumerate(get_stops_coords(selected_stops)):
    folium.Marker(point, icon = folium.Icon(color = 'red')).add_to(m)  


for i, point in enumerate(np.asarray(dispersed_zip_codes)[not_covered_associates_index]):
    folium.CircleMarker(point,radius = 4,
                    
                    fill=True, # Set fill to True|
                    fill_color='white',
                    color = 'gray',
                    fill_opacity=1).add_to(m)
    
    
for i, point in enumerate(np.asarray(dispersed_zip_codes)[covered_associates_index]):
    folium.CircleMarker(point,radius = 4,
                    
                    fill=True, # Set fill to True|
                    fill_color='white',
                    color = 'green',
                    fill_opacity=1).add_to(m)



Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())

# Asociates clustering
#m.save('src/views/routes_algorithm/maps/clustering_aguascalientes.html')


max_distance = 1000
min_samples = 5

coords = not_covered_associates
clusters, noise =  get_clusters(coords, min_samples, max_distance, iterations= 10)
cluster_centers = np.asarray([get_centermost_point(cluster) for cluster in clusters])

sum(len(cluster) for cluster in clusters), len(noise), len(clusters)

all_clusters = get_stops_coords(selected_stops) + cluster_centers.tolist()

len(all_clusters)


colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff', '#9A6324', '#800000', '#aaffc3', '#808000', '#000075', '#a9a9a9', '#ffffff', '#000000']

tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Mapeo Asociados Toluca', show = True).add_to(m)    
    
# for i, point in enumerate(filter_associates[['lat', 'lon']].values):
#     folium.CircleMarker(point,radius = 4,
#                         fill=True, # Set fill to True|
#                         fill_color='white',
#                         color = 'red',
#                         fill_opacity=1).add_to(m)
#     folium.Circle(point,radius = 500,
#                         fill=True, # Set fill to True|
#                         fill_color='white',
#                         color = 'red',
#                         fill_opacity=0.1).add_to(m)
    
max_radius = 750
for i, cluster in enumerate(clusters):
    for coord in cluster:
        folium.CircleMarker(coord,radius = 4,
                        fill=True, # Set fill to True|
                        fill_color='white',
                        color = colors[(i-1)%len(colors)] if i else 'gray',
                        fill_opacity=1).add_to(m)
        
max_radius = 750

for coord in cluster_centers:
    folium.Marker(coord,).add_to(m)


Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())

m.save('src/views/routes_algorithm/maps/clustering_aguascalientes.html')





clusters,clusters_coincidence = get_associates_clusters(dispersed_zip_codes, get_stops_coords(good_stops), distance, min_samples)


