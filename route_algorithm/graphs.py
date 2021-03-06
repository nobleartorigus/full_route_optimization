import sys
sys.path.append('route_algorithm/python')

from zip_codes_to_coordinates import *
from stop_clustering import *
import json

aguascalientes_plant = [21.967103, -102.284380]

#open the file with the zipcodes
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


tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Mapeo de Asociados Aguascalientes', show = True).add_to(m)

text = '<h4>Planta Bosch Aguascalientes</h4>'
text = folium.Html(text, script=True)
popup = folium.Popup(text, max_width=2650)
folium.Marker(aguascalientes_plant, icon = folium.Icon(color='red'), popup=popup).add_to(m)


table = associates.loc[associates.in_bounds]

get_associates_zip_codes_layer(table, show = True).add_to(m)
get_associates_cluster_layer(table).add_to(m)
get_associates_heatmap_layer(table).add_to(m)

Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())

#m.save('../maps/Mapeo Asociados Aguascalientes.html')


# ### Filter the zip codes coordinates

filter_coordinates(associates, bounds)
filter_associates = associates.loc[associates.in_bounds]
fliter_coords = filter_associates[['lat','lon']].values

grouped_zipcodes = group_zip_codes(associates)
filter_coordinates(grouped_zipcodes, bounds)
filter_associates = grouped_zipcodes.loc[grouped_zipcodes.in_bounds]
fliter_coords = filter_associates[['lat','lon']].values


# ### Obtaining the current routes of the plants 
# 
# 2 .- Stop clustering

# ### Defining parameters

distance = 1000
min_samples = 4

with open('route_algorithm/data/aguascalientes/routes_polylines_ags.json') as json_file:
    routes = json.load(json_file)
    
stops = get_all_stops(routes)
good_stops = get_good_stops(stops)

dispersed_zip_codes = disperse_coords(filter_associates,max_radius = 750)

clusters,clusters_coincidence = get_associates_clusters(dispersed_zip_codes, get_stops_coords(good_stops), distance, min_samples)

covered_associates, not_covered_associates = get_coverage(np.asarray(clusters_coincidence), dispersed_zip_codes)


tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Mapeo Asociados Toluca', show = True).add_to(m)
 
for i, point in enumerate(clusters):
    folium.Marker(point, icon = folium.Icon(color = 'red')).add_to(m)  


for i, point in enumerate(not_covered_associates):
    folium.CircleMarker(point,radius = 4,
                    
                    fill=True, # Set fill to True|
                    fill_color='white',
                    color = 'gray',
                    fill_opacity=1).add_to(m)
    
    
for i, point in enumerate(covered_associates):
    folium.CircleMarker(point,radius = 4,
                    
                    fill=True, # Set fill to True|
                    fill_color='white',
                    color = 'green',
                    fill_opacity=1).add_to(m)



Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())
m
#m.save('Maps/Rutas Sugeridas Urbvan Monterrey.html')


# In[14]:


from map_utils import *
from async_request import make_request
from sklearn.preprocessing import RobustScaler


from datetime import datetime, timedelta
def next_weekday_epoch(d, weekday):
    '''
    - Recibe una fecha y el siguiente día de la semana deseado
    - Retorna el día de la semana deseado más próximo
    '''
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0: # Target day already happened this week
        days_ahead += 7
    x = d + timedelta(days_ahead)
    x = x.replace(hour=7, minute=0, second=0, microsecond=0)
    return int(x.timestamp())


def get_clousure_area(a,b, delta, distance):
    bearing = initialBearing(a,b)
    if bearing is None: return [a,a,a,a]
    p1 = destinationPoint(a, -delta, bearing)
    p2 = destinationPoint(b, delta, bearing)
    bearing1 = wrap360(bearing - 90)
    c1 = destinationPoint(p1, distance, bearing1)
    c2 = destinationPoint(p1, -distance, bearing1)
    c3 = destinationPoint(p2, -distance, bearing1)
    c4 = destinationPoint(p2, distance, bearing1)

    return [ p1,c1,c4,p2,c3, c2, p1]

def estimate_time(distance):
    v = 0.17
    return v*distance

def fetch_matrix(session,url, j):
    '''
    - Recibe un objeto de Session() de la librería requests, una url con la petición al API de Google Distance Matrix, y un iterador
    - Es usada para mandar múltiples requests con una misma sesión de manera asíncrona
    - Retorna un diccionario con la respuesta de Google Maps
    '''
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    with session.get(url, headers= headers) as response:
        try:
            data = loads(response.content.decode('utf8'))
            return data
        except Exception:
            return None
        
def  get_matrix_info(origins_index, dest_index, coords, n_destinations):
    LIMIT = 100 #limit of elements in the matrix
    n_origins = int(np.floor(sqrt(LIMIT/n_destinations))) #max number of origins
    n_requests = int(np.ceil((len(coords))/(n_origins))) #number of requests needed
    d = datetime.now()
    next_monday = next_weekday_epoch(d, 0) 
    request_origins = [[] for i in range(n_requests)]
    request_dest = [[] for i in range(n_requests)]
    request_urls = []
    current_request = 0
    current_origins = 0
    for origin, dest in zip(origins_index,dest_index):
        request_origins[current_request].append(origin)
        request_dest[current_request] += dest
        current_origins+=1
        if current_origins >= n_origins:
            current_origins = 0
            current_request+=1
    for origin, dest in zip(request_origins,request_dest):   
        origin_coords = coords[origin].tolist()
        dest_coords = coords[dest].tolist()
        url = distance_matrix(origin_coords,dest_coords, get_url=True, traffic_model='pessimistic', departure_time=next_monday)
        request_urls.append(url)
    print('Requesting...')
    array = make_request(fetch_matrix, request_urls)
    matrices = []
    for element  in array:
        rows_origin = element['rows']
        matrix = []
        for row_origin in rows_origin:
            rows_dest = row_origin['elements']
            temp_row = []
            for row_dest in rows_dest:
                status = row_dest['status']
                if status == 'OK':
                    if 'duration_in_traffic' in row_dest.keys():
                        duration = row_dest['duration_in_traffic']['value']
                    else:
                        duration = row_dest['duration']['value']
                    distance = row_dest['distance']['value']   
                else:
                    distance = sys.maxsize
                    duration = sys.maxsize
                info = {'distance':distance, 'duration':duration}
                temp_row.append(info)
            matrix.append(temp_row)
        matrices.append(matrix)

    current_ind = 0
    durations = []
    distances = []
    for i, matrix in enumerate(matrices): 
        current_times = []
        down = 0
        up = 0
        for j, m in enumerate(matrix):
            up = up + len(dest_index[current_ind])
            info = m[down:up]
            current_duration = list(map(lambda x: x['duration'], info))
            current_distance = list(map(lambda x: x['distance'], info))
            durations.append(current_duration)
            distances.append(current_distance)
            current_ind+=1
            down = up
    return distances, durations

def scale_weights(weights,end_distances, scaler,  w_weights = 1, w_end_distance = 1):
    
    scaled_weights = -scaler.fit_transform(weights.reshape(-1,1)).reshape(1,-1)[0]
    scaled_weights -= np.min(scaled_weights)
    
    scaled_end_distances = scaler.fit_transform(end_distances.reshape(-1,1)).reshape(1,-1)[0]
    scaled_end_distances -= np.min(scaled_end_distances)
    
    scaled_weights*=w_weights
    scaled_end_distances*=w_end_distance
    
    return scaled_weights, scaled_end_distances

def calculate_scaled_edges(edge_distances, scaler, w_edge_distance = 1):
    
    scaled_edge_distances = scaler.fit_transform(edge_distances.reshape(-1,1)).reshape(1,-1)[0]
    scaled_edge_distances -= np.min(scaled_edge_distances)
    
    scaled_edge_distances*=w_edge_distance
    
    return scaled_edge_distances

class Graph:
    
    def __init__(self, coords, dest, weights, n = None, include_dest = False, edge_func = 'full', request_matrix = False,
                 w_weights = 1, w_end_distance = 1, w_edge_distance = 1):
        
        if edge_func == 'full':
            edge_func = self.calculate_node_edges1
        elif edge_func == 'wide':
            edge_func = self.calculate_node_edges2
        elif edge_func == 'narrow':
            edge_func = self.calculate_node_edges3
        
        weights = np.concatenate([[0], weights])
        coords = np.concatenate([[dest], coords])
        end_distances = getDistance2(coords[0], coords)
        
        scaler = RobustScaler(quantile_range=(25, 85))
        scaled_weights, scaled_end_distances = scale_weights(weights,
                                                            end_distances, 
                                                            scaler,  
                                                            w_weights = w_weights, 
                                                            w_end_distance = w_end_distance)
        self.node = [Node(point,weight,
                          end_distance, 
                          scaled_weight, 
                          scaled_end_distance) 
                     for point,
                     weight,
                     end_distance,
                     scaled_weight,
                     scaled_end_distance
                     in zip(coords, weights, 
                            end_distances, 
                            scaled_weights, 
                            scaled_end_distances)]
        
        self.n = len(self.node)
        self.edges = np.ones([self.n ,self.n ])*np.inf
        self.edges_distances_est = np.ones([self.n ,self.n ])*np.inf
        self.edges_durations_est = np.ones([self.n ,self.n ])*np.inf
        origins_index = []
        dest_index = []
        all_indexes = []
        
        for i in range(self.n):
            indexes, distances, times = edge_func(self.node[i], n = n, include_dest = include_dest)
            all_indexes.append(indexes)
            origins_index.append(i)
            dest_index.append(list(indexes))
            for zipped_ind in zip(indexes, distances, times):
                j, distance, time = zipped_ind
                self.edges[i,j] = distance
                self.edges_distances_est[i,j] = distance
                self.edges_durations_est[i,j] = time
        if request_matrix and n is not None:
            try:
                distances, durations = get_matrix_info(origins_index, dest_index, coords, n)
                self.edges_distances = np.ones([self.n ,self.n ])*np.inf
                self.edges_durations = np.ones([self.n ,self.n ])*np.inf
                for i in range(len(all_indexes)):
                    for j in range(len(all_indexes[i])):
                        self.edges[i,all_indexes[i][j]] = distances[i][j]
                        self.edges_distances[i,all_indexes[i][j]] = distances[i][j]
                        self.edges_durations[i,all_indexes[i][j]] = durations[i][j]
                print('Request made!')
            except Exception as e:
                print('Problem making request:', e)

        self.edges_distances_scaled = self.edges_distances.copy()
        edges_index = np.where(self.edges_distances_scaled != np.inf)
        edge_distances = self.edges_distances_scaled[edges_index]
        scaled_edge_distances = calculate_scaled_edges(edge_distances, scaler, w_edge_distance = w_edge_distance)
        self.edges_distances_scaled[edges_index] = scaled_edge_distances

        self.edges_durations_scaled = self.edges_durations.copy()
        edges_index = np.where(self.edges_durations_scaled != np.inf)
        edge_distances = self.edges_durations_scaled[edges_index]
        scaled_edge_distances = calculate_scaled_edges(edge_distances, scaler, w_edge_distance = w_edge_distance)
        self.edges_durations_scaled[edges_index] = scaled_edge_distances


        self.edges1 = self.edges_distances_scaled.copy()

        for i in range(len(self.edges1)):
            for j in range(len(self.edges1)):
                self.edges1[i,j] = self.edges_distances_scaled[i,j] + self.edges_durations_scaled[i,j]
                
    
    def calculate_node_edges1(self, node, n = None, include_dest = False):
        distances = getDistance2([node.x, node.y], [[nd.x, nd.y] for nd in self.node]) # in meters
        times = estimate_time(distances)
        if n is None:
            indexes = list(range(len(self.node)))
            return indexes, distances, times
        else:
            indexes = distances.argsort()[1:n+1]
            if include_dest:
                if 0 not in indexes:
                    indexes = np.append(indexes[:-1], 0)
            return indexes, distances[indexes], times[indexes]
        
    def calculate_node_edges2(self, node, n = None, include_dest = False):
        a = [node.x, node.y]
        b = [self.node[0].x, self.node[0].y]
        if equal_coords(a,b): return [0], [0], [0]
        node_coords = np.asarray([[nd.x, nd.y] for nd in self.node]) 
        x_delta = 10
        y_delta = getDistance(a,b)
        clousure_area = get_clousure_area(a,b,x_delta, y_delta)
        indexes = np.asarray(list(filter(lambda x: ispointinside(node_coords[x], clousure_area), range(len(node_coords)))))
        distances = getDistance2([node.x, node.y], node_coords[indexes]) # in meters
        times = estimate_time(distances)
        if n is None:   
            return indexes, distances, times
        else:
            dist_indexes = distances.argsort()[1:n+1]
            indexes = indexes[dist_indexes]
            if include_dest:
                if 0 not in indexes:
                    indexes = np.append(indexes[:-1], 0)
                    distances = np.append(distances[dist_indexes][:-1], getDistance(a, b))
                    times = estimate_time(distances)
                    return indexes, distances, times
            return indexes, distances[dist_indexes], times[dist_indexes]

            return indexes, distances, times    
        
    def calculate_node_edges3(self, node, n = None, include_dest = False):
        a = [node.x, node.y]
        b = [self.node[0].x, self.node[0].y]
        if equal_coords(a,b): return [0], [0], [0]
        node_coords = np.asarray([[nd.x, nd.y] for nd in self.node]) 
        x_delta = 100
        y_delta = getDistance(a,b)/6
        clousure_area = get_clousure_area(a,b,x_delta, y_delta)
        indexes = np.asarray(list(filter(lambda x: ispointinside(node_coords[x], clousure_area), range(len(node_coords)))))
        distances = getDistance2([node.x, node.y], node_coords[indexes]) # in meters
        times = estimate_time(distances)
        if n is None:   
            return indexes, distances, times
        else:
            dist_indexes = distances.argsort()[1:n+1]
            indexes = indexes[dist_indexes]
            if include_dest:
                if 0 not in indexes:
                    indexes = np.append(indexes[:-1], 0)
                    distances = np.append(distances[dist_indexes][:-1], getDistance(a, b))
                    times = estimate_time(distances)
                    return indexes, distances, times
            return indexes, distances[dist_indexes], times[dist_indexes]


    def cut_edges(self):
        edges = np.ones(self.edges.shape)*np.inf
        for node in range(len(edges)):
            connections = np.where(self.edges_durations[node] != np.inf)
            times = self.edges_durations[node][connections]
            mean = np.mean(times)
            if mean == 0: continue
            std_proportion = np.std(times)*100 / mean
            if std_proportion < 0.2:
                selected = times
                edges[node][connections] = selected
            else:
                inverted_times = max(times) - times
                mean = np.mean(inverted_times)
                std = np.std(inverted_times)
                indexes = np.where(inverted_times >= (mean))
                selected = times[indexes]
                edges[node][connections[0][indexes]] = selected
        self.edges = edges  
        
class Node:
    
    def __init__(self, point,weight,end_distance, scaled_weight, 
                scaled_end_distance):
        self.x = point[0]
        self.y = point[1]
        self.weight = weight
        self.end_distance = end_distance
        self.scaled_weight = scaled_weight
        self.scaled_end_distance = scaled_end_distance
        
    def __str__(self):
        return str([self.x, self.y])

def calculateDistance(n1,n2):
    return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

destination = aguascalientes_plant
stop_coords = clusters
stop_weights = [len(con) for con in clusters_coincidence]

## Creating the graph
graph = Graph(stop_coords, 
             destination, 
             stop_weights, 
             n = 8, 
             include_dest = False, 
             edge_func = 'wide', 
             request_matrix=True,
             w_weights = 1, 
             w_end_distance = 1, 
             w_edge_distance = 1)


def create_edge_table(graph,i,j):
    text = f'''
    <table border="1" style="width:100%">
      <tr>
        <th></th>
        <th> Distance </th>
        <th> Duration </th>
        <th> Sum </th>
      </tr>
      <tr>
        <th> Est. </th>
        <td>{format_distance(graph.edges_distances_est[i,j])}</td>
        <td>{format_duration(graph.edges_durations_est[i,j])}</td>
        <td> - </td>
      </tr>
      <tr>
        <th> Real </th>
        <td>{format_distance(graph.edges_distances[i,j])}</td>
        <td>{format_duration(graph.edges_durations[i,j])}</td>
        <td> - </td>
      </tr>
    </table>
    '''
    text = folium.Html(text, script=True)
    return folium.Popup(text, max_width=2650)


import folium
from folium.plugins import MarkerCluster,HeatMap, Fullscreen
from folium import plugins

tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Mapeo Asociados Toluca', show = True).add_to(m)

folium.Marker(destination, icon = folium.Icon(color='red')).add_to(m) 





for i in range(len(graph.edges)):
    for j in range(len(graph.edges)):
        if graph.edges[i,j] != np.inf:
            node = graph.node[i]
            nd = graph.node[j]
            text = f'<p>{format_distance(graph.edges[i,j])}</p>'
            text = folium.Html(text, script=True)
            popup = folium.Popup(text, max_width=2650)
            popup = create_edge_table(graph,i,j)
            folium.PolyLine([[node.x, node.y], [nd.x, nd.y]], popup=popup, color = 'red').add_to(m)
            
for i in range(len(graph.edges)):
    node = graph.node[i]
    folium.CircleMarker([node.x, node.y],radius = 4, popup=i,
                        fill=True, # Set fill to True|
                        fill_color='white',
                        color = 'blue',
                        fill_opacity=1).add_to(m)  
    

            

Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())

m.save('Maps/Rutas Sugeridas Urbvan Monterrey.html')


graph.cut_edges()

import folium
from folium.plugins import MarkerCluster,HeatMap, Fullscreen
from folium import plugins

tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Mapeo Asociados Toluca', show = True).add_to(m)

folium.Marker(destination, icon = folium.Icon(color='red')).add_to(m) 


for i in range(len(graph.edges)):
    for j in range(len(graph.edges)):
        if graph.edges[i,j] != np.inf:
            node = graph.node[i]
            nd = graph.node[j]
            text = f'<p>{format_distance(graph.edges[i,j])}</p>'
            text = folium.Html(text, script=True)
            popup = folium.Popup(text, max_width=2650)
            popup = create_edge_table(graph,i,j)
            folium.PolyLine([[node.x, node.y], [nd.x, nd.y]], popup=popup, color = 'red').add_to(m)
            
for i in range(len(graph.edges)):
    node = graph.node[i]
    folium.CircleMarker([node.x, node.y],radius = 4, popup=i,
                        fill=True, # Set fill to True|
                        fill_color='white',
                        color = 'blue',
                        fill_opacity=1).add_to(m)  
    

            

Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())
m.save('Maps/Rutas Sugeridas Urbvan Monterrey 464645.html')


