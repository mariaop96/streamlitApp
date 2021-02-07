import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import numpy as np
import streamlit as st
import pydeck as pdk
import altair as alt


#STACKOVERFLOW
#Pasos previos: 
#Paso 1 - convertir xlsx a json (conversor, sin comas entre documentos!)
#Paso 2 - importar base de daots y coleccion a mongo, db: tfm; collection: ejemplo

# Conexion a mongodb local
def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)


    return conn[db]

# Cargar coleccion de mongodb y convertir en dataframe de pandas
def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df


#Method to load the dataframe
def load_data():
   data = read_mongo("tfm", "ejemplo")
   return data

#Method to add latitude and longitude for each location
def add_coordinates(df):
   conditionlist = [
       (data['LOCATION'] == 'LEZO-RENTERIA') ,
       (data['LOCATION'] == 'GRISEN'),
       (data['LOCATION'] == 'TARRAGONA PORT'),
       (data['LOCATION'] == 'SANTANDER-PUERTO'),
       (data['LOCATION'] == 'LANDABEN')]
   lat = [43.31242550318731,41.7452588,41.107681200987884,43.43025926632228,42.8029889]
   lon = [-1.9063397113438043,-1.1658454,1.2504532385014784,-3.8064118342614677,-1.7063871]
   data['lat'] = np.select(conditionlist, lat, default='Not Specified')
   data['lon'] = np.select(conditionlist, lon, default='Not Specified')
   data["lat"]=pd.to_numeric(data["lat"])
   data["lon"]=pd.to_numeric(data["lon"])


#Method to add the total number of events happened at each location
def count_event(df, event):
    df[event] = data.groupby(["EVENT_TYPE", "LOCATION"]).size().unstack(level=0)[event]

#Method to add latitude and longitude of the origin and destination of each route
def add_routes(df):
   conditionlist = [
       (df['ROUTE'] == 'GRISEN - LEZO ( PSA )') ,
       (df['ROUTE'] == 'GRISÉN - TARRAGONA PTO.  (ITALIA  -TREN LARGO)'),
       (df['ROUTE'] == 'GRISEN - LEZO'),
       (df['ROUTE'] == 'SINTAX  LANDABEN - SANTANDER PTO (UE)'),
       (df['ROUTE'] == 'SINTAX  LANDABEN - SANTANDER PTO (NO-UE)')]
 
   lat_s = [41.7452588,41.7452588,41.7452588,42.8029889,42.8029889]
   lon_s = [-1.1658454,-1.1658454,-1.1658454,-1.7063871,-1.7063871]
   lat_d = [43.31242550318731,41.107681200987884,43.31242550318731,43.43025926632228,43.43025926632228]
   lon_d = [-1.9063397113438043,1.2504532385014784,-1.9063397113438043,-3.8064118342614677,-3.8064118342614677]
   df['lat_s'] = np.select(conditionlist, lat_s, default='Not Specified')
   df['lon_s'] = np.select(conditionlist, lon_s, default='Not Specified')
   df['lat_d'] = np.select(conditionlist, lat_d, default='Not Specified')
   df['lon_d'] = np.select(conditionlist, lon_d, default='Not Specified')
   df["lat_s"]=pd.to_numeric(data["lat_s"])
   df["lon_s"]=pd.to_numeric(data["lon_s"])
   df["lon_d"]=pd.to_numeric(data["lon_d"])
   df["lat_d"]=pd.to_numeric(data["lat_d"])

#def add_path(df):
#   conditionlist = [
#       (df['ROUTE'] == 'GRISEN - LEZO ( PSA )') ,
#       (df['ROUTE'] == 'GRISÉN - TARRAGONA PTO.  (ITALIA  -TREN LARGO)'),
#       (df['ROUTE'] == 'GRISEN - LEZO'),
#       (df['ROUTE'] == 'SINTAX  LANDABEN - SANTANDER PTO (UE)'),
#       (df['ROUTE'] == 'SINTAX  LANDABEN - SANTANDER PTO (NO-UE)')]
#   color = ["#ed1c24", "#faa61a", "#ed1c24","#faa61a","#faa61a"]
#   df['color'] = np.select(conditionlist, color, default='Not Specified')
#   path = [
#      [[-1.1658454, 41.7452588], [-1.9063397113438043, 43.312425503187]],
#      [
#         [
#           -1.1658454,
#           41.7452588
#         ],
#         [
#           1.2504532385014784,
#           41.107681200987884
#         ]
#      ],
#      [[-1.1658454, 41.7452588], [-1.9063397113438043, 43.312425503187]],
#      [
#         [
#           -1.7063871,
#           42.8029889
#         ],
#         [
#           43.430259266322,
#           -3.8064118342614677
#         ]
#      ],
#      [
#         [
#           -1.7063871,
#           42.8029889
#         ],
#         [
#           43.430259266322,
#           -3.8064118342614677
#         ]
#      ]
#   ]
   #df[df['ROUTE'] == 'GRISEN - LEZO ( PSA )']['path'] = [[-1.1658454, 41.7452588], [-1.9063397113438043, 43.312425503187]]
   #df[df['ROUTE'] == 'GRISÉN - TARRAGONA PTO.  (ITALIA  -TREN LARGO)']['path']=[
   #   [
   #     -1.1658454,
   #     41.7452588
   #   ],
   #   [
   #     1.2504532385014784,
   #     41.107681200987884
   #   ]
   #]
#
#   #df[df['ROUTE'] == 'GRISEN - LEZO']['path']=[
#   #   [
#   #     -1.1658454,
#   #     41.7452588
#   #   ],
#   #   [
   #     -1.9063397113438043,
   #     43.312425503187
   #   ]
   #]
   #df[df['ROUTE'] == 'SINTAX  LANDABEN - SANTANDER PTO (NO-UE)']['path']=[
   #   [
   #     -1.7063871,
   #     42.8029889
   #   ],
   #   [
   #     43.430259266322,
   #     -3.8064118342614677
   #   ]
   #]
   #df[df['ROUTE'] == 'SINTAX  LANDABEN - SANTANDER PTO (UE)']['path']=[
   #   [
   #     -1.7063871,
   #     42.8029889
   #   ],
   #   [
   #     43.430259266322,
   #     -3.8064118342614677
   #   ]
   #]
   
   


#Load data and format dates
data = load_data()
data["DATE_EVENT"] = pd.to_datetime(data["DATE_EVENT"], format="%Y%m%d%H%M%S")
data["DATE_PLANIF"] = pd.to_datetime(data["DATE_PLANIF"], format="%Y%m%d%H%M%S")
data["DATE_PREV"] = pd.to_datetime(data["DATE_PREV"], format="%Y%m%d%H%M%S")
data["DATE_TRAIN"] = pd.to_datetime(data["DATE_TRAIN"], format="%Y%m%d%H%M%S")

add_coordinates(data)
#add_path(data)
#data.dropna(inplace=True)





#Map with locations
locations = data[['LOCATION', 'lat','lon']]
#st.write(locations)
#st.map(locations)

#LOCATIONS-EVENTS -> df: summary
summary = data.groupby("LOCATION")["EVENT_TYPE"].agg(["count"])
summary["LOCATION"] = summary.index
add_coordinates(summary)
count_event(summary, "FIN DE CARGA DE VAGONES")
count_event(summary, "EXPEDICION DE VAGONES")
count_event(summary, "SALIDA DE VAGONES")
count_event(summary, "LLEGADA A DESTINO DE VAGONES")
count_event(summary, "FIN DE DESCARGA DE VAGONES")
summary.rename(columns={"count": "TOTAL_EVENTOS","FIN DE CARGA DE VAGONES": "CARGAS","EXPEDICION DE VAGONES": "EXPEDICIONES", "SALIDA DE VAGONES": "SALIDAS", "LLEGADA A DESTINO DE VAGONES":"LLEGADAS","FIN DE DESCARGA DE VAGONES": "DESCARGAS"}, inplace=True)
summary = summary.replace(np.nan, 0)

st.header("Mando de Control")
#Visualization of data
selectbox = st.sidebar.selectbox(
    "Selecciona los datos a visualizar",
    ("No visualizar", "Dataframe", "Distribución de eventos/localización")
)
if selectbox=="Dataframe":
   st.subheader("Muestra de los datos")
   st.write(data.head(10))
if selectbox=="Distribución de eventos/localización":
   #st.write(summary)
   st.subheader("Distribución de eventos/localización")
   st.dataframe(summary.style.highlight_max(axis=0))

   #METER GRAFICO DISTRIBUCION 
   st.subheader("Eventos/distrito")
   b2 = alt.Chart(data).mark_bar(
       cornerRadiusTopLeft=3,
       cornerRadiusTopRight=3
   ).encode(
       x='EVENT_TYPE',
       y='count(EVENT_TYPE)',
       color='LOCATION'
   )
   st.altair_chart(b2, use_container_width=True)


  #mal
   b1 = alt.Chart(data).mark_bar().encode(
      x='LOCATION:O',
      y='count(EVENT_TYPE):Q',
      color='EVENT_TYPE:N',
      column=' LOCATION:N'
   )
   st.altair_chart(b1, use_container_width=True)



#SIDEBAR
#Locations map
if st.sidebar.checkbox("Mapa con localizaciones"):
   st.subheader("Mapa con localizaciones")
   st.map(locations)

#MAPA CON ETIQUETAS EVENTOS TOTALES, SALIDAS Y ENTRADAS (OK) (data completo)
#HABRIA QUE AUTOMATIZAR UN POQUITO MEJOR EL FILTRADO
filter_event = data[data["EVENT_TYPE"] == "LLEGADA A DESTINO DE VAGONES"]
filter_event2 = data[data["EVENT_TYPE"] == "SALIDA DE VAGONES"]
aux = []
sal = []
tot = []
for i in data.LOCATION:
    aux.append(filter_event[filter_event.LOCATION == i]["EVENT_TYPE"].count())
    sal.append(filter_event2[filter_event2.LOCATION == i]["EVENT_TYPE"].count())
    tot.append(data[data.LOCATION == i]["EVENT_TYPE"].count())

data["LLEGADAS"] = aux
data["SALIDAS"] = sal
data["TOTAL_EVENTOS"] = tot

if st.sidebar.checkbox("Mapa con distribución de salidas/llegadas"):
   st.subheader("Mapa con recuento de eventos")
   st.pydeck_chart(pdk.Deck(
           map_style='mapbox://styles/mapbox/light-v9',
           initial_view_state=pdk.ViewState(
               latitude=43,
               longitude=-1,
               zoom=4,
               pitch=50,
            ),
           layers=[
               pdk.Layer(
                  "ColumnLayer",
                   data=data,
                   get_position=["lon", "lat"],
                   get_elevation="TOTAL_EVENTOS",
                   elevation_range=[0, 2500],
                   elevation_scale=100,
                   radius=8000,
                   get_fill_color=[180, 0, 50],
                   pickable=True,
                   auto_highlight=True,
                   ),
                ],
           tooltip={
               "html": "<b>Eventos totales:</b> {TOTAL_EVENTOS}<br/><b>Localización:</b> {LOCATION}<br/><b>Llegadas:</b> {LLEGADAS}<br/><b>Salidas:</b> {SALIDAS}",
               "style": {"color": "white"},
           }
       ))

#DFG AND PETRI NET
#src = pd.read_excel(r'C:\Users\mariaortp\OneDrive - Universidad Politécnica de Madrid\BECA\TRANSFESA\Ejemplo.xlsx', parse_dates=True, encoding="ISO-8859-1")
src = load_data()
#PROBLEMA: CARGANDO EL DATA DE MONGO NO FUNCIONA: INDEX MUST BE UNIQUE
if st.sidebar.checkbox("DFG y Red de Petri"):
   # Index
   index = "ID_INFOFERR"
   src.drop_duplicates(index, inplace = True)
   assert src[index].is_unique, "index column must be unique"
   src[index] = src[index].unique
   src = src.set_index("ID_INFOFERR")
   # Ordering
   src = src.sort_values(["DATE_EVENT", "DATE_PLANIF"])
   
   def to_dfg(log, case, event):
       return log\
       .assign(prev_event=lambda df: df\
           .groupby(case)[event]\
           .shift(1))\
       .groupby(["prev_event", event])\
       .size()\
       .to_dict()
   
   dfg = src.pipe(to_dfg, case="LOGISTIC_OBJECT", event="EVENT_TYPE")
   from pm4py.visualization.dfg import visualizer as dfg_visualization
   
   gviz = dfg_visualization.apply(dfg)
   #dfg_visualization.view(gviz)
   st.subheader('DFG')
   st.write(gviz)
   #st.write(dfg_visualization.view(gviz))
   from pm4py.objects.conversion.dfg import converter as dfg_mining
   net, im, fm = dfg_mining.apply(dfg)
   
   from pm4py.visualization.petrinet import visualizer as pn_visualizer
   gviz2 = pn_visualizer.apply(net, im, fm)
   #pn_visualizer.view(gviz)
   st.subheader('Red de Petri')
   st.write(gviz2)


df = pd.read_json(r"C:\Users\mariaortp\OneDrive - Universidad Politécnica de Madrid\BECA\TRANSFESA\dataprueba.json")
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

if st.sidebar.checkbox("MAPA RUTAS"):
   df["color"] = df["color"].apply(hex_to_rgb)
   
   
   view_state = pdk.ViewState(latitude=42, longitude=-1.27, zoom=10)
   
   layer = pdk.Layer(
       type="PathLayer",
       data=df,
       pickable=True,
       get_color="color",
       width_scale=20,
       width_min_pixels=2,
       get_path="path",
       get_width=5,
   )
   
   r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"})
   st.pydeck_chart(r)

