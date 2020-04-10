from requests import get, Session
from json import load, loads
from yaml import load as yaml_load, FullLoader
try:
    yaml_file = yaml_load(open('../auth/here.yaml'), Loader=FullLoader)
    HERE_API_KEY = yaml_file['HERE_API_KEY']
except:
    print(
"----------------------------------------------------------------------------\n\
ERROR: NO API KEY FOUND. PLEAS ADD THE FILE here.yaml TO THE PROJECT\n\
-----------------------------------------------------------------------------"
)


#https://developer.here.com/documentation/routing/dev_guide/topics/resource-calculate-matrix.html
def calculatematrix(starts, destinations, session = None,
                    searchrange = 100000, departure= None,
                    traffic = 'enabled', #enabled , disabled
                    mode = 'fastest' #fastest, shortest, balanced
                    ):
    M = len(starts)
    N = len(destinations)
    if N > 100:
        raise Exception("Request should not contain more than 100 destinations")
    if N > 1:
        if M > 15:
            raise ValueError("Request should not contain more than 100 starts")
    else:
        if M > 100:
            raise Exception("Request should not contain more than 15 starts")
        
    
    session = session or Session()
    
    url = 'https://matrix.route.ls.hereapi.com/routing/7.2/calculatematrix.json'

    params = {
       'apiKey': HERE_API_KEY,
        'mode':f'{mode};car;traffic:{traffic}',
        'searchrange':searchrange,
        'summaryAttributes':'tt,di'
    }

    waypoints = {
        **{
            f'start{i}': f'{start[0]},{start[1]}'
            for i, start in enumerate(starts)
        },
        **{
            f'destination{i}': f'{destination[0]},{destination[1]}'
            for i, destination in enumerate(destinations)
        }
    }

    params = {**params, **waypoints}

    if departure:
        params['departure'] = departure
    
    with session.get(url, params=params) as response:
        data = loads(response.content.decode('utf8'))
    return data