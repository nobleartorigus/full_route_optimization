from pandas import read_json, read_csv
from requests import get, Session
from json import load, loads
from numpy import asarray, concatenate, cos, pi, sqrt, arctan2, sin, radians, zeros, arccos, degrees
from numpy.linalg import norm
import sys
from yaml import load as yaml_load, FullLoader

try:
    yaml_file = yaml_load(open('route_algorithm/auth/key.yaml'), Loader=FullLoader)
    api_key = yaml_file['api_key']
except:
    try:
        yaml_file = yaml_load(open('../../auth/key.yaml'), Loader=FullLoader)
        api_key = yaml_file['api_key']
    except Exception as e:
        print(e,
"----------------------------------------------------------------------------\n\
ERROR: NO API KEY FOUND. PLEASE ADD THE FILE key.yaml TO THE PROJECT\n\
-----------------------------------------------------------------------------"
)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

def to_geojson(data):
    geojson = dict()
    geojson["type"] = "FeatureCollection"
    features = list()
    feature = dict()
    features.append(feature)
    feature["type"] = "Feature"
    feature["properties"] = dict()
    geometry = dict()
    geometry["type"] = "LineString"
    coord = list()
    for x in data:
        coord.append([x[1], x[0]])
    geometry["coordinates"] = coord
    feature["geometry"] = geometry
    geojson["features"] = features
    return geojson


def to_geojson_polygon(data):
    geojson = dict()
    geojson["type"] = "FeatureCollection"
    features = list()
    feature = dict()
    features.append(feature)
    feature["type"] = "Feature"
    feature["properties"] = dict()
    geometry = dict()
    geometry["type"] = "Polygon"
    coord = list()
    for x in data:
        coord.append([x[1], x[0]])
    geometry["coordinates"] = [coord]
    feature["geometry"] = geometry
    geojson["features"] = features
    return geojson


def to_geojson_multiple(data):
    geojson = dict()
    geojson["type"] = "FeatureCollection"
    features = list()
    for ruta in data:
        feature = dict()
        feature["type"] = "Feature"
        feature["properties"] = dict()
        geometry = dict()
        geometry["type"] = "LineString"
        coord = list()
        for x in ruta:
            coord.append([x[1], x[0]])
        geometry["coordinates"] = coord
        feature["geometry"] = geometry
        features.append(feature)
    geojson["features"] = features
    return geojson

# https://stackoverflow.com/questions/15380712/how-to-decode-polylines-from-google-maps-direction-api-in-php


def decode_polyline(polyline_str):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}

    # Coordinates have variable length when encoded, so just keep
    # track of whether we've hit the end of the string. In each
    # while loop iteration, a single coordinate is decoded.
    while index < len(polyline_str):
        # Gather lat/lon changes, store them in a dictionary to apply them later
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0

            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20:
                    break

            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)

        lat += changes['latitude']
        lng += changes['longitude']

        coordinates.append((lat / 100000.0, lng / 100000.0))

    return coordinates


def get_Coordinates(session, location, added=' ', get_url=False, print_request=False, get_request=False):
    coordenadas = list()
    direction = location.replace(' ', '+')  # Sustituimos espacion por '+'
    added = added.replace(' ', '+')
    url = 'https://maps.google.com/maps/api/geocode/json?address=' + \
        direction + added + '&key=' + api_key
    if get_url:
        return url
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    # print(url)
    try:
        with session.get(url, headers= headers) as response:
            data = loads(response.content.decode('utf8'))
    except Exception as e:
        return {'status': 'INTERNET_FAIL', 'error_message': e}
    if get_request == True:
        return data
    if print_request is True:
        print(data)
    results = data['results']
    if(len(results) is 0):  # Google Maps te niega el uso de su API arbitrariamente,
        coordenadas = None
    else:
        coordenadas.append(results[0]['geometry']['location']['lat'])
        coordenadas.append(results[0]['geometry']['location']['lng'])
    return coordenadas

def get_directions(origin, destination, mode='driving', alternatives='false', waypoints=None, waypoints_via=False, departure_time=None, arrival_time=None, traffic_model=None, print_request=False, get_request=False, get_url = False):
    waypoints_str = ''
    if waypoints is not None:
        for point in waypoints:
            if(waypoints_via):
                waypoints_str += 'via:'
            waypoints_str += str(point[0])+','+str(point[1])+'|'

    url = 'https://maps.googleapis.com/maps/api/directions/json?' + \
        'origin='+str(origin[0])+','+str(origin[1])+'&' + \
        'destination='+str(destination[0])+','+str(destination[1])+'&' + \
        'mode='+mode+'&' + \
        'alternatives='+alternatives+'&' + \
        'waypoints='+waypoints_str+'&' + \
        'key=' + api_key
    if departure_time is not None:
        url = url + '&departure_time=' + str(departure_time)
        if traffic_model is not None:
            url = url + '&traffic_model=' + traffic_model
    if arrival_time is not None:
        url = url + '&arrival_time=' + str(arrival_time)
        if traffic_model is not None:
            url = url + '&traffic_model=' + traffic_model

    if get_url:
        return url
    # print(url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    # print(url)
    try:
        data = get(url, headers=headers)
    except Exception as e:
        return {'status': 'INTERNET_FAIL', 'error_message': e}
    data = loads(data.content.decode('utf8'))
    if get_request is True:
        return data
    if print_request is True:
        print(data)
    n = len(data['routes'])
    rutas = [None]*n
    durations = [None]*n
    distancias = [None]*n
    for i in range(n):
        rutas[i] = data["routes"][i]["overview_polyline"]["points"]
        distance = []
        duration = []
        for leg in data['routes'][i]['legs']:
            distance.append(leg['distance']['value'])
            duration.append(leg['duration']['value'])
        durations[i] = duration
        distancias[i] = distance
    return rutas, durations, distancias


def get_directions_url(origin, destination, mode='driving', alternatives='false', waypoints=None, departure_time=None, traffic_model=None):
    waypoints_str = ''
    if waypoints is not None:
        for point in waypoints:
            waypoints_str += str(point[0])+','+str(point[1])+'|'

    url = 'https://maps.googleapis.com/maps/api/directions/json?' + \
        'origin='+str(origin[0])+','+str(origin[1])+'&' + \
        'destination='+str(destination[0])+','+str(destination[1])+'&' + \
        'mode='+mode+'&' + \
        'alternatives='+alternatives+'&' + \
        'waypoints=optimize:true|'+waypoints_str+'&' + \
        'key=' + api_key
    if departure_time is not None:
        url = url + '&departure_time=' + departure_time
        if traffic_model is not None:
            url = url + '&traffic_model=' + traffic_model
    return url


def getDistance_fast(inicio, points):
    points1 = asarray(points)
    return ((points1[:, 0] - inicio[0])**2 + (points1[:, 1] - inicio[1])**2)

# def get_distance_matrix(encoded, point):
#     url = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=metric&origins=' + str(point[0]) + ','+ str(point[1]) + '&destinations=enc:' + str(encoded)+ ':&key=' + api_key
#     headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
#     data = get(url, headers=headers)
#     data = loads(data.content)
#     return data


def get_distance_matrix(ruta1, ruta2, point1, point2, mode='driving', print_url=False, print_request=False):
    coords = concatenate([ruta1, ruta2])
    encoded_coords = encode_coords(coords)
    encoded_points = encode_coords([point1, point2])
    url = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=metric&mode='+mode + \
        '&origins=enc:' + str(encoded_points) + ':&destinations=enc:' + \
        str(encoded_coords) + ':&key=' + api_key
    if print_url is True:
        print(url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    data = get(url, headers=headers)
    data = loads(data.content)
    if print_request is True:
        print(data)
    return data


def distance_matrix(origins, destinations, mode='driving', get_url=False, departure_time=None, arrival_time=None, traffic_model=None, print_request=False, get_request=False):
    encoded_origins = encode_coords(origins)
    encoded_destinations = encode_coords(destinations)
    url = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=metric&' + \
        'mode='+mode+'&' +\
        'origins=enc:' + str(encoded_origins) + ':&' +\
        'destinations=enc:' + str(encoded_destinations) + ':&' +\
        'key=' + api_key
    if departure_time is not None:
        url = url + '&departure_time=' + str(departure_time)
        if traffic_model is not None:
            url = url + '&traffic_model=' + traffic_model
    if arrival_time is not None:
        url = url + '&arrival_time=' + str(arrival_time)
        if traffic_model is not None:
            url = url + '&traffic_model=' + traffic_model
    if get_url:
        return url
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    data = get(url, headers=headers)
    data = loads(data.content)
    if print_request is True:
        print(data)
    return data


def places_nearbySearch(session, coord, keyword, radius, language = 'es'):
    url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={},{}&radius={}&keyword={}&key={}&language={}'.format(coord[0], coord[1], radius, keyword, api_key, language)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    with session.get(url, headers= headers) as response:
        data = loads(response.content)
    if data['status'] == 'OK':
        results = []
        for result in data['results']:
            coords = result['geometry']['location']
            dic_ = {'name':result['name'], 'location': [coords['lat'], coords['lng']],
                    'place_id': result['place_id'],
                    'address':result['vicinity'],
                    'type': keyword
                   }
            results.append(dic_)
        return results
    else:
        return None


def search_places(coordinates, typ, radius):
    directions_total = []
    names_total = []
    coords_total = []
    for coord in coordinates:
        coords, names, directions = places_nearbySearch(coord, typ, radius)
        x = []
        for j in range(len(directions)):
            if directions[j] not in directions_total:
                x.append(j)
        directions_total = directions_total + [directions[int(i)] for i in x]
        names_total = names_total + [names[int(i)] for i in x]
        coords_total = coords_total + [coords[int(i)] for i in x]
    coords_total = asarray(coords_total)
    return coords_total, names_total, directions_total

# http://tainguyen.me/blog/python-encode-and-decode-polylines-from-google-direction-api/


def encode_coords(coords):
    '''Encodes a polyline using Google's polyline algorithm

    See http://code.google.com/apis/maps/documentation/polylinealgorithm.html
    for more information.

    :param coords: Coordinates to transform (list of tuples in order: latitude,
    longitude).
    :type coords: list
    :returns: Google-encoded polyline string.
    :rtype: string
    '''

    result = []

    prev_lat = 0
    prev_lng = 0

    for x, y in coords:
        lat, lng = int(y * 1e5), int(x * 1e5)

        d_lat = _encode_value(lat - prev_lat)
        d_lng = _encode_value(lng - prev_lng)

        prev_lat, prev_lng = lat, lng

        result.append(d_lng)
        result.append(d_lat)

    return ''.join(c for r in result for c in r)


def _split_into_chunks(value):
    while value >= 32:  # 2^5, while there are at least 5 bits

        # first & with 2^5-1, zeros out all the bits other than the first five
        # then OR with 0x20 if another bit chunk follows
        yield (value & 31) | 0x20
        value >>= 5
    yield value


def _encode_value(value):
    # Step 2 & 4
    value = ~(value << 1) if value < 0 else (value << 1)

    # Step 5 - 8
    chunks = _split_into_chunks(value)

    # Step 9-10
    return (chr(chunk + 63) for chunk in chunks)

# def getDistance(relativeNullPoint, p):
#     p = asarray(p)
#     deltaLatitude = p[0] - relativeNullPoint[0]
#     deltaLongitude = p[1] - relativeNullPoint[1]
#     latitudeCircumference = 40075160 * cos(asRadians(relativeNullPoint[0]))
#     resultX = deltaLongitude * latitudeCircumference / 360
#     resultY = deltaLatitude * 40008000 / 360
#     return sqrt(resultX**2 + resultY**2)


def getDistance(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    R = 6371E3
    angle1 = radians(lat1)
    angle2 = radians(lat2)
    delta_lat = radians(lat2-lat1)
    delta_lon = radians(lon2-lon1)
    a = sin(delta_lat/2) * sin(delta_lat/2) + cos(angle1) * \
        cos(angle2) * sin(delta_lon/2) * sin(delta_lon/2)
    c = 2 * arctan2(sqrt(a), sqrt(1-a))
    d = R * c
    return d

def getDistance2(point1, point2):
    point2 = asarray(point2)
    lat1, lon1 = point1
    lat2, lon2 = point2[:, 0], point2[:, 1]
    R = 6371E3
    angle1 = radians(lat1)
    angle2 = radians(lat2)
    delta_lat = radians(lat2-lat1)
    delta_lon = radians(lon2-lon1)
    a = sin(delta_lat/2) * sin(delta_lat/2) + cos(angle1) * \
        cos(angle2) * sin(delta_lon/2) * sin(delta_lon/2)
    c = 2 * arctan2(sqrt(a), sqrt(1-a))
    d = R * c
    return d


def directions(origin_text, destination_text, added='', print_url=False):
    origin = get_Coordinates(origin_text, added, print_url=print_url)
    destination = get_Coordinates(destination_text, added, print_url=print_url)
    polyline, tiempos, distancias = get_directions(
        origin, destination, alternatives='true', print_url=print_url)
    rutas = [None]*len(polyline)
    geojson_rutas = [None]*len(polyline)
    for i in range(len(polyline)):
        rutas[i] = decode_polyline(polyline[i])
        geojson_rutas[i] = to_geojson(rutas[i])
    return rutas, geojson_rutas, tiempos, distancias


def get_dist(x, y):
    return sqrt(sum((x-y)**2))


def extend_route(ruta, eps):
    y = []
    for i in range(1, len(ruta)):
        a, b = ruta[i-1], ruta[i]
        d = getDistance(a, b)
        y.append(ruta[i-1])
        if d > eps:
            sections = int(round(d/eps)) + 1
            for i in range(1, sections):
                m = i
                n = sections-i
                y.append(section(a, b, m, n))
    y.append(ruta[-1])
    return y


def section(A, B, m, n):
    x1, y1 = A
    x2, y2 = B
    # Applying section formula
    x = (float)((n * x1)+(m * x2))/(m + n)
    y = (float)((n * y1)+(m * y2))/(m + n)

    # Printing result
    return asarray([x, y])


def angle(pt1, pt2):
    
    x1, y1 = pt1
    x2, y2 = pt2
    #Si el punto es el mismo no tomamos en cuenta esa ruta
    if x1 == y1:
        return 90
    inner_product = x1*x2 + y1*y2
    len1 = norm([x1,y1])
    len2 = norm([x2, y2])
    x = round(inner_product/(len1*len2), 5)
    return arccos(x)*180/pi

    # x1, y1 = pt1
    # x2, y2 = pt2
    # dot = x1*x2 + y1*y2      # dot product
    # det = x1*y2 - y1*x2      # determinant
    # a = degrees(arctan2(det, dot))
    # a %= 360
    # return a  # atan2(y, x) or atan2(sin, cos)


def delta_angle_compare(a1, a2, comp_angle):
    a1 = asarray(a1)
    a2 = asarray(a2)
    return 180 - abs(abs(a1 - a2) - 180) > comp_angle


_eps = 0.00001
_huge = sys.float_info.max
_tiny = sys.float_info.min


def rayintersectseg(p, edge):
    ''' takes a point p=Pt() and an edge of two endpoints a,b=Pt() of a line segment returns boolean
    '''
    a, b = edge
    if a[1] > b[1]:
        a, b = b, a
    if p[1] == a[1] or p[1] == b[1]:
        p[1] = p[1] + _eps

    intersect = False

    if (p[1] > b[1] or p[1] < a[1]) or (
            p[0] > max(a[0], b[0])):
        return False

    if p[0] < min(a[0], b[0]):
        intersect = True
    else:
        if abs(a[0] - b[0]) > _tiny:
            m_red = (b[1] - a[1]) / float(b[0] - a[0])
        else:
            m_red = _huge
        if abs(a[0] - p[0]) > _tiny:
            m_blue = (p[1] - a[1]) / float(p[0] - a[0])
        else:
            m_blue = _huge
        intersect = m_blue >= m_red
    return intersect


def _odd(x): return x % 2 == 1


def ispointinside(p, poly):
    edges = [[poly[x], poly[x+1]] for x in range(len(poly) - 1)]
    return _odd(sum(rayintersectseg(p, edge) for edge in edges))


def format_duration(secs):
    minutes = (secs / 60)
    if minutes > 60:
        hours = minutes / 60
        minutes = (hours % 1) * 60
        hours = int(hours - (hours % 1))
        if hours > 1:
            string = str(hours) + ' hrs ' + str(int(round(minutes))) + ' min'
        else:
            string = str(hours) + ' hr ' + str(int(round(minutes))) + ' min'
    else:
        string = str(int(round(minutes))) + ' min'

    return string


def format_distance(m):
    if m < 1000:
        string = str(m) + ' m'
    else:
        km = (m / 1000)
        string = str(round(km, 1)) + ' km'
    return string

def get_photo_url(session, input_, size = 500):
    fields= "photos,geometry,place_id"
    inputtype = "textquery"
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={}&inputtype={}&fields={}&key={}".format(input_.replace(' ', "+"),inputtype,fields,api_key)
    with session.get(url, headers= headers) as response:
        data = loads(response.content.decode('utf8'))
    if(len(data['candidates']) > 0):
        if 'photos' in  data['candidates'][0].keys():
            place_id = data['candidates'][0]['place_id']
            fields= "photos"
            url = 'https://maps.googleapis.com/maps/api/place/details/json?placeid={}&fields={}&key={}'.format(place_id,fields,api_key)
            with session.get(url, headers= headers) as response:
                data = loads(response.content.decode('utf8'))
            reference = data['result']['photos'][0]['photo_reference']
            url = "https://maps.googleapis.com/maps/api/place/photo?maxwidth={}&photoreference={}&key=API_KEY".format(size,reference)
        else:
            lat = data['candidates'][0]['geometry']['location']['lat']
            lon = data['candidates'][0]['geometry']['location']['lng']
            url = 'https://maps.googleapis.com/maps/api/streetview?size={}x{}&location={},{}&source=outdoor&key=API_KEY'.format(size,size,lat,lon)
    else:
        response = get_Coordinates(session, input_, get_request=True)
        if response['status'] == 'OK':
            lat = response['results'][0]['geometry']['location']['lat']
            lon = response['results'][0]['geometry']['location']['lng']
            url = 'https://maps.googleapis.com/maps/api/streetview?size={}x{}&location={},{}&source=outdoor&key=API_KEY'.format(size,size,lat,lon)
        else:
            return None
        
    return url

def clean_coordinates(coordenadas, inferior_izquierda, superior_derecha, lat_col = 'lat', lon_col = 'lon', negated = False):

    mask = ((inferior_izquierda[0] < coordenadas[lat_col])\
                        &(coordenadas[lat_col] < superior_derecha[0]))\
                        & ((superior_derecha[1]> coordenadas[lon_col])\
                        & (inferior_izquierda[1] < coordenadas[lon_col]))

    if negated:
        mask = ~mask
    coord = coordenadas.loc[mask]
    return coord


import numpy as np
#https://www.movable-type.co.uk/scripts/latlong-vectors.html
def toNvector(point):
    phi = np.radians(point[0])
    lam = np.radians(point[1])
    
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    x = cos_phi*cos_lam
    y = cos_phi*sin_lam
    z = sin_phi

    return np.asarray([x, y, z])

def equal_coords(a,b):
    return (a[0] == b[0]) and (a[1] == b[1])

def wrap360(degrees):
    if (0<=degrees and degrees<360): return degrees
    return (degrees%360+360) % 360

def initialBearing(a,b):
    if equal_coords(a,b): return None
    p1 = toNvector(a)
    p2 = toNvector(b)
    N = [0,0,1]

    c1 = np.cross(p1,p2)
    c2 = np.cross(p1,N)
    sin_phi = np.dot(np.linalg.norm(np.cross(c1,c2)), np.sign(np.dot(np.cross(c1,c2), p1)))
    cos_phi = np.dot(c1,c2)
    angle = np.rad2deg(np.arctan2(sin_phi, cos_phi))
    return wrap360(angle)

def greatCircle_bearing(point, bearing):
    phi = np.radians(point[0])
    lam = np.radians(point[1])
    angle = np.radians(bearing)

    x =  np.sin(lam) * np.cos(angle) - np.sin(phi) * np.cos(lam) * np.sin(angle)
    y = -np.cos(lam) * np.cos(angle) - np.sin(phi) * np.sin(lam) * np.sin(angle)
    z =  np.cos(phi) * np.sin(angle)
    return [x, y, z]

def greatCircle_point(a, b):
    p1 = toNvector(a)
    p2 = toNvector(b)
    return unit_vector(np.cross(p1,p2))

def unit_vector(v):
    return v / (v**2).sum()**0.5

def toLatLon(v):
        x = v[0]
        y = v[1]
        z = v[2]
        phi = np.arctan2(z, np.sqrt(x*x + y*y))
        lam = np.arctan2(y, x)
        return [np.rad2deg(phi), np.rad2deg(lam)]
    
def destinationPoint(point, distance, bearing, radius=6371e3):
    n1 = toNvector(point)
    angular_distance = distance / radius
    angle = np.radians(bearing)

    N = [0,0,1]

    de =unit_vector(np.cross(N,n1))
    dn = np.cross(n1,de)
    

    de_sin_angle = np.dot(de, np.sin(angle));
    dn_cos_angle = np.dot(dn, np.cos(angle));
    d = de_sin_angle + dn_cos_angle


    x = np.dot(n1, np.cos(angular_distance))
    y = np.dot(d, np.sin(angular_distance))
    
    n2 = x + y

    return toLatLon(n2)

def get_square(a, b, height_ratio = 4):
    bearing = initialBearing(a,b)
    if bearing is None: return [a,a,a,a]
    d = getDistance(a,b)
    width_ratio =  map_values(d, in_min = 5000, in_max = 50000, out_min = 8, out_max = 16)
    print(width_ratio)
    bearing1 = wrap360(bearing - 90)
    width_distance = d/width_ratio
    p1 = destinationPoint(a, -width_distance, bearing)
    p2 = destinationPoint(b, width_distance, bearing)
    
    height_ratio =  map_values(d, in_min = 5000, in_max = 50000, out_min = 3, out_max = 10)
    height_distance = d/height_ratio
    print(height_ratio)
    c1 = destinationPoint(p1, height_distance, bearing1)
    c2 = destinationPoint(p1, -height_distance, bearing1)
    c3 = destinationPoint(p2, -height_distance, bearing1)
    c4 = destinationPoint(p2, height_distance, bearing1)
    return [c1,c2,c3,c4,c1]

def map_values(x, in_min = 5000, in_max = 50000, out_min = 8, out_max = 16): return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def get_square(a, b, height_ratio = 4):
    bearing = initialBearing(a,b)
    if bearing is None: return [a,a,a,a]
    d = getDistance(a,b)
    width_ratio =  map_values(d, in_min = 5000, in_max = 50000, out_min = 8, out_max = 16)
    bearing1 = wrap360(bearing - 90)
    width_distance = d/width_ratio
    p1 = destinationPoint(a, 10, bearing)
    p2 = destinationPoint(b, 10, bearing)
    
    height_ratio =  map_values(d, in_min = 5000, in_max = 50000, out_min = 3, out_max = 15)
    height_distance = d/height_ratio
    c1 = destinationPoint(p1, height_distance, bearing1)
    c2 = destinationPoint(p1, -height_distance, bearing1)
    c3 = destinationPoint(p2, -height_distance, bearing1)
    c4 = destinationPoint(p2, height_distance, bearing1)
    return [c1,c2,c3,c4,c1]