import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import sqlite3
import networkx as nx
import pyvis
from pyvis.network import Network
#import geopandas as gpd
import streamlit as st                            ##streamlit
import streamlit.components.v1 as components
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://api.gs.tatneft.ru/api/v2/azs/"  #(недокументированная api)
r = requests.get(url)
x = r.json()

a =[]
b =[]
c =[]
d =[]
e =[]
f =[]
g =[]
h =[]
s =[]
k = 0
for i in range(len(x["data"])):
    a.append(x['data'][i]["id"])
    b.append(x['data'][i]["lat"])
    c.append(x['data'][i]["lon"])
    d.append(x['data'][i]["region"])
    e.append(x['data'][i]["number"])
    f.append(x['data'][i]["address"])
    g.append(x['data'][i]["currency_code"])
    s.append(len(x['data'][i]["features"]))
    for j in range(len(x['data'][i]["fuel"])):
      k = 0
      if x['data'][i]["fuel"][j]['fuel_type_id'] == 10:
        h.append(x['data'][i]["fuel"][j]['price'])
        k = k+1
        break
      else:
        continue
    if  k==0:
        h.append("noData")
data ={"id":np.array(a), "lat": np.array(b),"lon":np.array(c), "region": np.array(d),"number":np.array(e), "address": np.array(f),"currency_code":np.array(g), "ДТprice":np.array(h),"features_number":np.array(s)}
df = pd.DataFrame(data)
df_2=df[df["currency_code"] == 'rub']
df__3 = df_2[df_2["ДТprice"] != "noData"]
df_3 = df__3[df__3["ДТprice"] != '0.0']

df_sql=df_3
conn = sqlite3.connect('orders.db')
c = conn.cursor()
near_azs_reg = str(st.text_input('Введите название региона:'))
st.write('Адреса заправок Татнефть в вашем регионе:')                            #SQL
def infor(near_azs):
  a = c.execute(
   """
  SELECT address FROM tabler                             
  WHERE region = ? 
  ORDER BY address
   """,  [near_azs] ).fetchall()
  return(a)
st.write(infor(near_azs_reg))


df_4 = df_3.sample(n=int(st.number_input('Посмотрим, как много заправок Татнефть одновременно схожи и по цене, и по количеству услуг для водителей. Выберем случайным образом какое-то количество заправок, а затем соединим ребрами те, которые слабо отличаются друг от друга по этим параметрам.Введите количество заправок(лучше не вводить очень большие числа, чтобы граф был читаемым:')
))

line_1=np.array(df_4["ДТprice"])
m =line_1.astype("float64")
line_2=np.array(df_4["features_number"])
g =line_2.astype("float64")
xx=np.sqrt(np.var(m))/4
yy=np.sqrt(np.var(g))/2
aa=[]
for id_1 in list(np.array(df_4["id"]).astype("int32")):
    for id_2 in list(np.array(df_4["id"]).astype("int32")):
        rr_1=abs(float(df_4[df_4["id"]==id_1]["ДТprice"].head(1).item()) - float(df_4[df_4["id"]==id_2]["ДТprice"].head(1).item()))                       #pandas
        rr_2=abs(float(df_4[df_4["id"]==id_1]["features_number"].head(1).item()) - float(df_4[df_4["id"]==id_2]["features_number"].head(1).item()))
        if rr_1 < xx and rr_2< yy:
            aa.append([id_1,id_2])
cc =[]
for obj in aa:
    cc.append(tuple(obj))
for i in range(len(cc)):
    cc[i]=(int(cc[i][0]),int(cc[i][1]))
q = []
for i in range(len(list(np.array(df_4["id"])))):
    q.append(int(list(np.array(df_4["id"]))[i]))
G = nx.Graph()
G.add_nodes_from(q)
G.add_edges_from(cc)
from pyvis.network import Network
net = Network(directed=True, notebook=True)
net.from_nx(G)                                                #visual, networkx
net.show("visualisation.html")
HtmlFile = open('visualisation.html','r',encoding='utf-8')
components.html(HtmlFile.read(),height=420)

df_3["length"] = 2*6371*np.arcsin(np.sqrt(np.square(np.sin(((df_3["lat"]-55.752004)*np.pi)/360))+np.square(np.sin(((df_3["lon"]-37.617734)*np.pi)/360))*np.cos((df_3["lat"]*np.pi)/180)*np.cos((55.752004*np.pi)/180)))
df_Mos=df_3.copy()
df_Mos=df_Mos.sort_values(by="ДТprice")
eew = st.slider("Количество ближайших к Москве заправок",1,750)
sns.set_style( "whitegrid" )
nm = sns.scatterplot(data=df_Mos[:eew], x="ДТprice", y="length", hue = "features_number")
plt.legend(labels=["Количество допуслуг"],
           fontsize = 'large')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


st.write('Теперь попробуем предсказать цену топлива на заправке по заданным координатам.  Модель взвешивает цены на нескольких соседних заправках пропорционально их расстояниям до вашей точки:')

ii = float(st.number_input('Введите широту места:')) #широта
oo = float(st.number_input('Введите долготу места:')) #долгота
df_3["length"] = 2*6371*np.arcsin(np.sqrt(np.square(np.sin(((df_3["lat"]-ii)*np.pi)/360))+np.square(np.sin(((df_3["lon"]-oo)*np.pi)/360))*np.cos((df_3["lat"]*np.pi)/180)*np.cos((ii*np.pi)/180)))           #NumpyMath

df_3_sorted=df_3.sort_values(by='length')
df_3_sorted_f=df_3_sorted[:30]
df_3_sorted_f["share"] = (df_3_sorted_f["length"].astype("float")/df_3_sorted_f["length"].astype("float").sum())
df_3_sorted_f["shareXprice"] = (df_3_sorted_f["ДТprice"].astype("float"))*(df_3_sorted_f["share"].astype("float"))                    #обучение
st.write('Ожидаемая цена топлива в этом месте:',df_3_sorted_f["shareXprice"].sum())

from shapely.geometry import Polygon
from shapely.geometry import Point, shape
with open('admin_level_4.geojson', encoding = 'utf-8') as f:
    reg_pol = json.load(f)
reg_polig=[]                                                                                        #geodata(shapely)
reg_name =[]
for a in range(len(reg_pol["features"])):
               reg_name.append(reg_pol["features"][a]["name"])
               reg_polig.append(Polygon(reg_pol["features"][a]["geometry"]["coordinates"][0][0]))
data_Rus={"reg_name":reg_name,"reg_polig":reg_polig}
df_Rus=pd.DataFrame(data_Rus)
DF=df_3.merge(df_Rus, left_on='region', right_on='reg_name')

df_3["full_coord"] = Point(list(zip(df_3.lon, df_3.lat)))

#gdf_3 = gpd.GeoDataFrame(df_3, geometry = gpd.points_from_xy(df['lon'], df['lat']))
#gdf_Rus = gpd.GeoDataFrame(df_Rus, geometry = 'reg_polig')






