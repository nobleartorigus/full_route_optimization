import sys
import json
import requests
import pandas as pd

#url = "https://localhost:3000/uploads"
url = "https://opensourcepyapi.herokuapp.com:443/weather/06604"
r = requests.get(url)
data = r.json()

resp = {
    "Response": 200,
    "Message": "Data from Python",
    "Data" : data
}

#prueba = pd.read_excel("./uploads/pruebas.xls", encoding = 'utf-8')

print('hello world')
print(json.dumps(resp))
#print(prueba)
prueba = pd.read_excel('./uploads/medidas_patrocinadores.xlsx', encoding = 'utf-8')

print(prueba)


sys.stdout.flush()