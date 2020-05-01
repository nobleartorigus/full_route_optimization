#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sys import path
path.insert(0,'../../python')
from zip_codes_to_coordinates import *
from stop_clustering import *
from graph_making import *
import json


# ## Obtaining the associates zipcodes
# (1. - Zipcodes to Coordinates)

# In[2]:


aguascalientes_plant = [21.967103, -102.284380]


# In[3]:


#open the file with the zipcodes
associates = pd.read_excel('../../data/aguascalientes/example.xlsx',dtype=str)
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


# In[4]:


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
m
#m.save('../maps/Mapeo Asociados Aguascalientes.html')


# ### Filter the zip codes coordinates

# In[5]:


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

# In[6]:


distance = 1000
min_samples = 4


# In[7]:


with open('../../data/aguascalientes/routes_polylines_ags.json') as json_file:
    routes = json.load(json_file)
    
stops = get_all_stops(routes)
good_stops = get_good_stops(stops)


# In[8]:


dispersed_zip_codes = disperse_coords(filter_associates,max_radius = 750)

clusters,clusters_coincidence = get_associates_clusters(dispersed_zip_codes, get_stops_coords(good_stops), distance, min_samples)


# In[9]:


covered_associates, not_covered_associates = get_coverage(np.asarray(clusters_coincidence), dispersed_zip_codes)


# In[10]:


len(covered_associates),len(not_covered_associates)


# In[11]:


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


# ## Graph making

# In[12]:


destination = aguascalientes_plant
stop_coords = clusters
stop_weights = [len(con) for con in clusters_coincidence]


# In[13]:


## Creating the graph
graph = Graph(stop_coords, 
             destination, 
             stop_weights, 
             n = 8, 
             include_dest = True, 
             edge_func = 'wide', 
             request_matrix=True,
             w_weights = 1, 
             w_end_distance = 1, 
             w_edge_distance = 1)


# In[14]:



graph.cut_edges()


# In[15]:


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
m
#m.save('Maps/Rutas Sugeridas Urbvan Monterrey.html')


# In[16]:


class Colony:
    
    def __init__(self,antNo):
            
        self.queen = Ant()
        self.ant = [Ant() for i in range(antNo)]
    def select_queen(self):
        allAntsFitness = np.asarray([ant.fitness for ant in self.ant])
        minIndex = allAntsFitness.argmin()
        self.queen = self.ant[minIndex]
        

class Ant:
    
    def __init__(self):
        self.fitness = 0
        self.tour = [None]
        
        
def rouletteWheel(P):
    cumSumP = np.cumsum(P)
    r = np.random.random()
    nextNode = np.where(r <= cumSumP)[0]
    if len(nextNode) == 0:
        return None
    return nextNode[0]


# In[17]:


def evaporation(tau, rho, tau_min, tau_max):
    tau = (1-rho)*tau
    tau[np.where(tau < tau_min)] = tau_min
    tau[np.where(tau > tau_max)] = tau_max
    return tau

def updatePhromone(tau, colony):
    queen = colony.queen
    nodeNo = len(queen.tour)
    for j in range(nodeNo - 1):
        currentNode = queen.tour[j]
        nextNode = queen.tour[j+1]
        tau[currentNode, nextNode] = tau[currentNode, nextNode] + 1 / (queen.fitness)
            
    return tau


# In[18]:


def select_best_ant(best_ant, queen):
    if best_ant.fitness < queen.fitness:
        return best_ant
    else:
        return queen
    
def decide_next_node(graph, ant, tau, eta, alpha, beta, avoid_nodes = None, nodes_recurrency = None):
    current_node = ant.tour[-1] #the node where the ant is
    N = get_neighborhood(graph, ant, avoid_nodes) #feasible neighborhood of ant
    if N is None:
        return None #the ant can not go to any other node
    if nodes_recurrency is not None:
        P_allNodes = (tau[current_node,:][N]**alpha) * (eta[current_node,:][N]**beta) *nodes_recurrency[N]
    else:
        P_allNodes = tau[current_node,:][N]**alpha * eta[current_node,:][N]**beta
    P = P_allNodes / np.sum(P_allNodes)   #calculate de probability of selecting each node
    selected_node = rouletteWheel(P)      #stochastically decide the next node
    if selected_node is None:
        return None
    return N[selected_node]


# In[19]:


def get_neighborhood(graph, ant, avoid_nodes = None):
    '''
        Feasible neighborhood of the ant.
        This function can be modified according to the problem.
        Return index of the feasible nodes.
    '''
    current_node = ant.tour[-1]
    N = np.where(graph.edges[current_node,:] != np.inf)[0] #Nodes that the ant can reach
    N = np.asarray(list(set(N) - set(ant.tour))) #Nodes that the ant has not visited yet
    if avoid_nodes is not None:
        N = np.asarray(list(set(N) - set(avoid_nodes)))
    if len(N) == 0:
        return None
    return N

def fitnessFunction(tour, graph, end_node,maxNodes, nodes_recurrency = None):
    fitness = 0
    for i in range(len(tour) - 1):
        currentNode = tour[i]
        nextNode = tour[i+1]
        fitness += graph.edges_durations[currentNode, nextNode]
    if tour[-1] != end_node:
        fitness*=1000
    #fitness *= (abs(maxNodes - len(tour)) + 1)
    fitness *= (abs(maxNodes - len(tour)) + 1) / sum([graph.node[nd].weight for nd in tour])
#     if node_recurrency is not None:
#         fitness *= sum(node_recurrency[tour])
    return fitness


# In[20]:


def ACO(graph, initial_node, end_node, 
        maxNodes = 5,
        maxIter = 200,
        antNo = 20,
        rho = 0.4, #evaporation rate
        alpha = 1, #Phromone exponential parameters
        beta = 0,  #Desiarability exponential parameter
        avoid_nodes = None,
        nodes_recurrency = None,
       ): 

    graph_edges = graph.edges_durations[np.where((graph.edges_durations != np.inf)&(graph.edges_durations != 0))]

    k = 10
    tau_min =  k / (np.max(graph_edges))
    tau_max = k/ np.min(graph_edges)
    tau0 = tau_max

    with np.errstate(divide='ignore'):
        eta = 1 / (graph.edges) #desiarability of each edge 

    tau = tau0*np.ones([graph.n ,graph.n ]) #phromone matrix

    best_ant = Ant()
    best_ant.fitness = np.inf 

    for i in range(maxIter):
        colony = Colony(antNo) #creat ants

        for ant in colony.ant: 
            ant.tour[0] = initial_node
            current_node = ant.tour[0]
            while (current_node != end_node):
                next_node = decide_next_node(graph, ant, tau, eta, alpha, beta, avoid_nodes, nodes_recurrency) #stochastically decide the next node
                if next_node is None:
                    break
                ant.tour.append(next_node)
                current_node = next_node
                if len(ant.tour) > maxNodes*3: #the ant could not find a good solution
                    break
            #end while
        #end for
        for ant in colony.ant:
            ant.fitness = fitnessFunction(ant.tour, graph, end_node,maxNodes, nodes_recurrency)
            
        colony.select_queen()
        best_ant = select_best_ant(best_ant, colony.queen)
        #Evaporation 
        tau = evaporation(tau, rho, tau_min, tau_max)
        tau = updatePhromone(tau, colony)
    return best_ant


# In[21]:


connections = []
for i in range(len(graph.edges)):
    connections.append(len(np.where(graph.edges[:,i] != np.inf)[0]))
connections = np.asarray(connections)


# In[22]:


from polylabel import polylabel


# ## Select the route starts

# In[23]:


center = polylabel([stop_coords])
distances = getDistance2(center, stop_coords)
radius = max(distances)
#bearings = np.random.randint(0,360, size=40)
bearings = [x for x in range(0,365,5)]
starts = [destinationPoint(center, radius, bearing) for bearing in bearings]
indexes = [getDistance_fast(start, stop_coords).argmin() for start in starts]
indexes = np.unique(indexes)
max_index = indexes[distances[indexes].argmax()]


# In[24]:


start_index = []
for start in starts:    
    index = getDistance_fast(start, stop_coords).argmin()
    start_index.append(index)
    point = stop_coords[index]


# In[25]:


#initial_nodes = np.where(connections  < 1)[0]
initial_nodes = np.unique(start_index) + 1


# In[26]:


import folium
from folium.plugins import MarkerCluster,HeatMap, Fullscreen
from folium import plugins

tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Mapeo Asociados Toluca', show = True).add_to(m)

# folium.Marker(center, icon = folium.Icon(color='green')).add_to(m) 
# folium.Circle(center, radius, popup=None, dash_array = '10, 10', color = 'red',fill=True,fill_opacity=0.1).add_to(m)
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
    
for start in starts:
    index = getDistance_fast(start, stop_coords).argmin()
    point = stop_coords[index]
    folium.CircleMarker(point ,radius = 4, popup=index + 1,
                        fill=True, # Set fill to True|
                        fill_color='white',
                        color = 'green',
                        fill_opacity=1).add_to(m) 
            

Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())
m
#m.save('Maps/Rutas Sugeridas Urbvan Monterrey.html')


# In[38]:


initial_nodes


# In[78]:


initial_nodes = [82,64,73,9,74,54,62,76,68]


# In[83]:


nodes_recurrency = np.ones(len(graph.edges)) * (len(initial_nodes))
bestAnts = []
for k in initial_nodes:
    initial_node = k
    end_node = 0
    best_ant = ACO(graph, initial_node, end_node, 
            maxNodes = 5,
            maxIter = 150,
            antNo = 15,
            rho = 0.4, #evaporation rate
            alpha = 1, #Phromone exponential parameters
            beta = 0,  #Desiarability exponential parameter
            nodes_recurrency = nodes_recurrency
           )
    bestAnts.append(best_ant)
    nodes_recurrency[best_ant.tour]-=1


# In[84]:


import folium
from folium.plugins import MarkerCluster,HeatMap, Fullscreen
from folium import plugins

tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Suggested Routes', show = True).add_to(m)

folium.Marker(destination, icon = folium.Icon(color='red')).add_to(m) 

n = 5

for i in range(len(stop_coords)):
    x = stop_coords[i]
    folium.CircleMarker(x,radius = 4, popup=i + 1,
                        fill=True, # Set fill to True|
                        fill_color='white',
                        color = 'gray',
                        fill_opacity=1).add_to(m) 

for ant in bestAnts:
    bestTour = ant.tour
    for i in range(1, len(bestTour)):
        x1 = bestTour[i]
        x = bestTour[i-1]
        popup = create_edge_table(graph,x,x1)
        node = graph.node[x]
        nd = graph.node[x1]
        folium.PolyLine([[node.x, node.y], [nd.x, nd.y]], popup=popup, color = 'red').add_to(m)

for ant in bestAnts:
    bestTour = ant.tour
    for i in range(len(bestTour)):
        x = bestTour[i]
        node = graph.node[x]
        popup = create_node_popup(node, x)
        folium.CircleMarker([node.x, node.y],radius = 4, popup=popup,
                            fill=True, # Set fill to True|
                            fill_color='white',
                            color = 'blue',
                            fill_opacity=1).add_to(m) 


Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())
m
#m.save('Maps/Rutas Sugeridas Urbvan Monterrey.html')


# In[86]:


selected_nodes = []
for ant in bestAnts:
    selected_nodes.append(ant.tour)
    
selected_nodes = np.unique(np.concatenate(selected_nodes))
sum ([graph.node[i].weight for i in selected_nodes])


# In[105]:


252/9


# In[87]:


for ant in bestAnts:
    users = sum([graph.node[nd].weight for nd in ant.tour])
    print(users)


# In[88]:


sum ([graph.node[i].weight for i in range(len(graph.edges))])


# In[89]:


nodes_recurrency


# In[94]:


mask = np.ones(len(graph.edges)-1 ,dtype=bool) #np.ones_like(a,dtype=bool)
mask[selected_nodes] = False
not_selected_nodes = np.arange(len(graph.edges)-1)[mask]


# In[95]:


len(stop_weights)


# In[96]:


stop_coords1 = np.asarray(stop_coords)[not_selected_nodes]
stop_weights1 = np.asarray(stop_weights)[not_selected_nodes]


# In[97]:


## Creating the graph
graph1 = Graph(stop_coords1, 
             destination, 
             stop_weights1, 
             n = 8, 
             include_dest = False, 
             edge_func = 'wide', 
             request_matrix=True,
             w_weights = 1, 
             w_end_distance = 1, 
             w_edge_distance = 1)

graph1.cut_edges()


# In[98]:


center = polylabel([stop_coords1])
distances = getDistance2(center, stop_coords1)
radius = max(distances)
#bearings = np.random.randint(0,360, size=40)
bearings = [x for x in range(0,365,5)]
starts = [destinationPoint(center, radius, bearing) for bearing in bearings]
indexes = [getDistance_fast(start, stop_coords1).argmin() for start in starts]
indexes = np.unique(indexes)
max_index = indexes[distances[indexes].argmax()]


# In[99]:


start_index = []
for start in starts:    
    index = getDistance_fast(start, stop_coords1).argmin()
    start_index.append(index)
    point = stop_coords[index]


# In[ ]:





# In[100]:


#initial_nodes = np.where(connections  < 1)[0]
initial_nodes1 = np.unique(start_index) + 1


# In[101]:


initial_nodes1


# In[102]:


nodes_recurrency = np.ones(len(graph1.edges)) * (len(initial_nodes1))
bestAnts = []
for k in initial_nodes1:
    initial_node = k
    end_node = 0
    best_ant = ACO(graph1, initial_node, end_node, 
            maxNodes = 2,
            maxIter = 100,
            antNo = 10,
            rho = 0.4, #evaporation rate
            alpha = 1, #Phromone exponential parameters
            beta = 0,  #Desiarability exponential parameter
            nodes_recurrency = nodes_recurrency
           )
    bestAnts.append(best_ant)
    nodes_recurrency[best_ant.tour]-=1


# In[103]:


import folium
from folium.plugins import MarkerCluster,HeatMap, Fullscreen
from folium import plugins

tiles_url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
attrb ='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
m = folium.Map(tiles=None)

folium.TileLayer(tiles_url,attr=attrb,name = 'Suggested Routes', show = True).add_to(m)

folium.Marker(destination, icon = folium.Icon(color='red')).add_to(m) 

n = 5

for i in range(len(stop_coords)):
    x = stop_coords[i]
    folium.CircleMarker(x,radius = 4, popup=i + 1,
                        fill=True, # Set fill to True|
                        fill_color='white',
                        color = 'gray',
                        fill_opacity=1).add_to(m) 

for ant in bestAnts:
    bestTour = ant.tour
    for i in range(1, len(bestTour)):
        x = bestTour[i]
        x1 = bestTour[i-1]
        node = graph.node[x]
        nd = graph.node[x1]
        folium.PolyLine([[node.x, node.y], [nd.x, nd.y]], popup=None, color = 'red').add_to(m)

for ant in bestAnts:
    bestTour = ant.tour
    for i in range(len(bestTour)):
        x = bestTour[i]
        node = graph.node[x]
        folium.CircleMarker([node.x, node.y],radius = 4, popup=x,
                            fill=True, # Set fill to True|
                            fill_color='white',
                            color = 'blue',
                            fill_opacity=1).add_to(m) 


Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen', force_separate_button=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds(m.get_bounds())
m
#m.save('Maps/Rutas Sugeridas Urbvan Monterrey.html')


# In[104]:


text = f'''
Associates (total): {len(associates)}
Assiciates with errors: {len(associates.loc[~associates.good_zip_code])}
Associates inside bounds: {len(associates.loc[associates.in_bounds])}
Current stops: {len(stops)}
Stops proposed: {len(clusters)}
Covered associates: {len(covered_associates)} ({round(len(covered_associates) / len(associates.loc[associates.in_bounds]) *100,1)}%)
Not Covered associates: {len(not_covered_associates)} ({round(len(not_covered_associates) / len(associates.loc[associates.in_bounds]) *100,1)}%)

'''
print(text)


# In[ ]:




