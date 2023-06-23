import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import fsolve
from scipy import interpolate

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.markdown("# Balance de aguas")
st.markdown("""
            Para el diseño de Pozas, se  toma como caudal de entrada, la suma del caudal de escorrentía y el caudal de filtraciones.
            
            $$
            Q_{entrada} = Q_{escorrentía}^* + Q_{filtraciones}
            $$
            
            El caudal de escorrentia, incluirá el caudal de infiltración, para fines prácticos. Este debe de ser subido a la presente
            página en el apartado izquierdo de "Datos de Entrada" como formato .xlsx y con colummas "Time (min)" y "q_es (l/s)", 
            los cuales sería el tiempo y el caudal de escorrentia, respectivamente.
            
            $$
            Q_{escorrentía}^* = Q_{escorrentía} + Q_{infiltración}
            $$
            
            El caudal de filtraciones, será constante en todo el tiempo, y será colocado manualmente por el usuario en el menú lateral.
            
            El caudal de salida, estará definido según la capacidad máxima de la bomba. Una vez iniciada la lluvia, se estará
            bombeando la misma cantidad que el caudal de entrada, posteriormente, si se llega al máximo de la capacidad de la bomba, se bombeará 
            de forma constante a su máxima capacidad, acumulandose volumen. Una vez llegado al volumen máximo, se seguira bombeando a su máxima capacidad, 
            hasta que no quede volumen acumulado en la presa. Si se requiere reducir el tiempo de funcionamiento al máximo de la bomba, se puede 
            añadir otra bomba en el menú lateral.
            
            La descarga de la tabla de resultados, se encuentra al final.
            
            """)


##################
# Sidebar Layout #
##################

## Caudales de entrada y bomba
st.sidebar.markdown("# Datos de Entrada")

uploaded_file = st.sidebar.file_uploader("Ingrese el archivo de caudales de escorrentia .xlsx:")

if uploaded_file is not None:
    fil_data = pd.read_excel(uploaded_file)
    
q_filt = st.sidebar.number_input(
    "Caudal de filtración (l/s)",
    min_value = 0.00,
    value = 30.0,
    step = 1.0
    )

    
q_max_b = st.sidebar.number_input(
    "Caudal máximo - Bomba 1 (l/s)",
    min_value = 0.00,
    value = 100.0,
    step = 10.0
    )

bomba_extra = st.sidebar.checkbox('¿Añadir Bomba Extra?')

if bomba_extra:
    q_max_b_extra = st.sidebar.number_input(
        "Caudal máximo - Bomba 2 (l/s)",
        value = 20.0,
        step = 10.0
        )
else:
    q_max_b_extra = 0 
    
suavizar = st.sidebar.checkbox('Suavizar la curva de escorrentia')

if suavizar:
    tck = interpolate.splrep(fil_data["Time (min)"] , fil_data["q_es (l/s)"] , s=0)
    fil_data["q_es (l/s)"] = interpolate.splev(fil_data["Time (min)"], tck, der=0)
    

## Geometria de la poza
st.sidebar.markdown("# Geometría de la Poza")

b = st.sidebar.number_input(
    "Base mayor (m)",
    min_value = 0.0,
    value = 50.0,
    step = 10.0
    )

h = st.sidebar.number_input(
    "Base menor (m)",
    min_value = 0.0,
    value = 25.0,
    step = 5.0
    )

z = st.sidebar.number_input(
    "Altura (m)",
    min_value = 0.0,
    value = 5.0,
    step = 1.0
    )

s_xz1 = st.sidebar.number_input(
    "Pendiente Izquierda - Sección Transversal",
    min_value = 0.0,
    value = 1.0,
    step = 0.1
    )

s_xz2 = st.sidebar.number_input(
    "Pendiente Derecha - Sección Transversal",
    min_value = 0.0,
    value = 1.0,
    step = 0.1
    )

s_yz1 = st.sidebar.number_input(
    "Pendiente Izquierda - Sección Longitudinal",
    min_value = 0.0,
    value = 1.0,
    step = 0.1
    )

s_yz2 = st.sidebar.number_input(
    "Pendiente Derecha - Sección Longitudinal",
    min_value = 0.0,
    value = 1.0,
    step = 0.1
    )

#########################
# Procesamiento inicial #
#########################

# Iniciar columnas nuevas
fil_data["q_f (l/s)"] = q_filt
fil_data["q_entrada (l/s)"] = fil_data["q_es (l/s)"] + fil_data["q_f (l/s)"]
fil_data["q_salida (l/s)"] = 0
fil_data["q_almacenado_acc (l/s)"] = 0
fil_data["q_almacenado_acc (l/s)"][0] = 0

# Algoritmo para encontrar el rango de mayor cantidad de vlaores >= a la bomba inicial
bool_array = fil_data["q_entrada (l/s)"] >= q_max_b
start_index = -1
end_index = -1
max_start_index = -1
max_end_index = -1
max_subrange_length = 0

# Iterar a lo largo del array
for i in range(len(bool_array)):
    if bool_array[i]:
        if start_index == -1:
            start_index = i
        end_index = i
    else:
        if end_index - start_index > max_subrange_length:
            max_subrange_length = end_index - start_index
            max_start_index = start_index
            max_end_index = end_index
        start_index = -1
        end_index = -1

# Revisar si el ultimo subrango es el más alto
if end_index - start_index > max_subrange_length:
    max_subrange_length = end_index - start_index
    max_start_index = start_index
    max_end_index = end_index
# Balance de aguas

k = 0
l = 0
m = 0
inicio_bomba_extra = 0
inicio_bomba = 0
cierre_bomba = 0
cierre_bomba_predicho = False


for i in range(len(fil_data["Time (min)"])):
  # Si es menor al caudal maximo, entonces tomaría el de entrada, si no se acumularia caudal, por lo tanto cambiaría
  if k == 0:
    if fil_data["q_entrada (l/s)"][i] <= q_max_b and i != max_start_index + 1:
      fil_data["q_salida (l/s)"][i] = fil_data["q_entrada (l/s)"][i]
      k = 0
      
    else: 
        if i == max_start_index +1:
            inicio_bomba = fil_data["Time (min)"][i]
            fil_data["q_salida (l/s)"][i] = q_max_b 
            k = 1
        else:
            fil_data["q_salida (l/s)"][i] = q_max_b        

    # Asimismo calculamos los acumulados
    if i != 0:
      fil_data["q_almacenado_acc (l/s)"][i] =  fil_data["q_entrada (l/s)"][i] - fil_data["q_salida (l/s)"][i] +  fil_data["q_almacenado_acc (l/s)"][i - 1]

  # Estamos en donde se va acumular el caudal, por lo tanto debemos tomar en cuenta que se va considerar 100 hasta que vuelva a 0 el caudal acumulado
  else:
    
    if fil_data["q_almacenado_acc (l/s)"][i - 1] > 0 and l != 1:
      # Si el caudal de salida empieza a superar el de entrada entonces abririamos la bomba extra
      if fil_data["q_salida (l/s)"][i - 1] > fil_data["q_entrada (l/s)"][i - 1]:
        fil_data["q_salida (l/s)"][i] = q_max_b + q_max_b_extra
        # Corregimos, ya que la bomba debe empezar, ni bien se llegue al máximo
        if m == 0:
          inicio_bomba_extra = fil_data["Time (min)"][i-1]
          m = 1
        fil_data["q_salida (l/s)"][i - 1] = q_max_b + q_max_b_extra
        fil_data["q_almacenado_acc (l/s)"][i - 1] =  fil_data["q_entrada (l/s)"][i-1] - fil_data["q_salida (l/s)"][i-1] +  fil_data["q_almacenado_acc (l/s)"][i - 2]
      else:
        fil_data["q_salida (l/s)"][i] = q_max_b
      fil_data["q_almacenado_acc (l/s)"][i] =  fil_data["q_entrada (l/s)"][i] - fil_data["q_salida (l/s)"][i] +  fil_data["q_almacenado_acc (l/s)"][i - 1]

  # Ahora la bomba para este punto en el cual es negativo, debe compensar para llegar a 0 entonces:
    else:
      # Este caso sería para que cumpla exactamente lo necesario
      if l == 0:
        # Corregimos el caudal de salida anterior y el acumulado anterior
        fil_data["q_salida (l/s)"][i - 1] = fil_data["q_almacenado_acc (l/s)"][i - 2] + fil_data["q_entrada (l/s)"][i - 1]
        fil_data["q_almacenado_acc (l/s)"][i - 1] =  fil_data["q_entrada (l/s)"][i - 1] - fil_data["q_salida (l/s)"][i - 1] +  fil_data["q_almacenado_acc (l/s)"][i - 2]
        cierre_bomba = fil_data["Time (min)"][i-2]
        # Ahora el paso actual debe tomar el caudal de entrada
        fil_data["q_salida (l/s)"][i] = fil_data["q_entrada (l/s)"][i]
        fil_data["q_almacenado_acc (l/s)"][i] =  fil_data["q_entrada (l/s)"][i] - fil_data["q_salida (l/s)"][i] +  fil_data["q_almacenado_acc (l/s)"][i - 1]
        l = 1
      # Finalmente tomaría el caudal de entrada
      else:
        fil_data["q_salida (l/s)"][i] = fil_data["q_entrada (l/s)"][i]
        fil_data["q_almacenado_acc (l/s)"][i] =  fil_data["q_entrada (l/s)"][i] - fil_data["q_salida (l/s)"][i] +  fil_data["q_almacenado_acc (l/s)"][i - 1]


if cierre_bomba == 0:
    caudal_faltante = fil_data["q_almacenado_acc (l/s)"].iloc[-1]
    # Lo que continua del hidrograma lo consideramos como un constante (conservador)
    caudal_saliente = q_max_b + q_max_b_extra - fil_data["q_entrada (l/s)"].iloc[-1]    
    
    tiempo_faltante = caudal_faltante / caudal_saliente

    cierre_bomba_predicho = True
    cierre_bomba = round(caudal_faltante / caudal_saliente + fil_data["Time (min)"].iloc[-1] - 1, 0) 
    

fil_data["V_almacenado"] = fil_data['Time (min)'].diff().fillna(0) * fil_data["q_almacenado_acc (l/s)"] * 60 / 1000
fil_data["Delta V_almacenado"] = abs(fil_data["V_almacenado"].diff().fillna(0))


# Salida 1: Gráfico del Balance de la Poza
fig = go.Figure()


fig.add_trace(go.Scatter(x = fil_data["Time (min)"], y = fil_data["q_entrada (l/s)"], mode = 'lines', name = "Caudal de Entrada"))

fig.add_trace(go.Scatter(x = fil_data["Time (min)"], y = fil_data["q_f (l/s)"], mode = 'lines', name = "Caudal de Filtración"))

fig.add_trace(go.Scatter(x = fil_data["Time (min)"],
                                y = fil_data["q_salida (l/s)"],
                                mode = 'lines',
                         name = "Caudal de Salida"))


fig.add_trace(go.Scatter(x = [inicio_bomba],
                         y = [float(fil_data.loc[fil_data["Time (min)"] == inicio_bomba]["q_salida (l/s)"])],
                         mode = "markers",
                         name = "Inicio Bomba: "+ str(round(inicio_bomba, 2)) + " min"
  )
 )

if bomba_extra:
  fig.add_trace(go.Scatter(x = [inicio_bomba_extra],
                         y = [float(fil_data.loc[fil_data["Time (min)"] == inicio_bomba_extra]["q_salida (l/s)"])],
                         mode = "markers",
                         name = "Inicio Bomba 2: "+ str(round(inicio_bomba_extra, 2)) + " min"
  )
 )

if not cierre_bomba_predicho:
    fig.add_trace(go.Scatter(x = [cierre_bomba],
                            y = [float(fil_data.loc[fil_data["Time (min)"] == cierre_bomba]["q_salida (l/s)"])],
                            mode = "markers",
                            name = "Cierre Bomba: "+ str(round(cierre_bomba, 2)) + " min"
    ))


fig.update_layout(title='Diseño de Poza de Filtración - Balance de aguas',
                  xaxis_title='Tiempo (min)',
                  yaxis_title='Caudal (l/s)',
                  legend = dict(
                      yanchor = "bottom",
                      y = 0.65,
                      xanchor = "right",
                      x = 0.95
                  ))

tab1, tab2 = st.tabs(["Gráfica", "Tabla"])

with tab1:
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.write(fil_data)

V_dis = max(fil_data["V_almacenado"])

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

if bomba_extra:
    resultados = pd.DataFrame(
        {
            "Variable": [
                "Caudal de escorrentia máximo de entrada (l/s)",
                "Caudal total máximo de entrada (l/s)",
                "Volumen máximo de la poza (m3)",
                "Tiempo de funcionamiento al máximo de la bomba 1 (min)",
                "Tiempo de funcionamiento al máximo de la bomba 2 (min)",
                "Volumen de filtraciones (m3)"
            ],
            "Resultado": [
                max(fil_data["q_es (l/s)"]),
                max(fil_data["q_entrada (l/s)"]),
                V_dis,
                cierre_bomba - inicio_bomba,
                cierre_bomba - inicio_bomba_extra,
                cierre_bomba * q_filt
            ]
                
        }
    ) 
else:
    resultados = pd.DataFrame(
        {
            "Variable": [
                "Caudal de escorrentia máximo de entrada (l/s)",
                "Caudal total máximo de entrada (l/s)",
                "Volumen máximo de la poza (m3)",
                "Tiempo de funcionamiento al máximo de la bomba 1 (min)",
                "Volumen de filtraciones (m3)"
            ],
            "Resultado": [
                max(fil_data["q_es (l/s)"]),
                max(fil_data["q_entrada (l/s)"]),
                V_dis,
                cierre_bomba - inicio_bomba,
                cierre_bomba * q_filt
            ]
                
        }
    )

st.table(resultados)


# Cálculo de los tirantes
st.markdown("# Geometría de la Poza")
st.markdown("""
            La geometría de la poza será definida por el usuario en el menú lateral, pudiendo escoger entre una poza trapezoidal o una poza
            rectangular (Pendientes = 0). El tirante es calculado a través del volumen máximo de la poza, anteriormente calculado en el 
            balance de aguas. Se puede asimismo, observar el cambio del tirante a lo largo del evento de lluvia.
            """)

def vol_poza(x, b, h, s_xz1, s_xz2, s_yz1, s_yz2, option = 1, V_total = 0):
    
  A_1 = b * h
  A_2 = (b + s_xz1 * x + s_xz2 * x) * (h + s_yz1 * x + s_yz2 * x)
  V_poza = (x / 3) * (A_1 + A_2 + ((A_1 * A_2) ** 0.5))

  if option == 1:
    return V_poza

  elif option == 0:
    return V_total - V_poza

vol_poza_completa = vol_poza(z, b, h, s_xz1, s_xz2, s_yz1, s_yz2)

y = fsolve(vol_poza, x0=1, args=(b, h, s_xz1, s_xz2, s_yz1, s_yz2, 0, V_dis,))[0]

fil_data["Tirante (m)"] = 0

for i in range(len(fil_data["Time (min)"])):
  fil_data["Tirante (m)"][i] = fsolve(vol_poza, x0=1, args=(b, h, s_xz1, s_xz2, s_yz1, s_yz2, 0, float(fil_data["V_almacenado"][i]),))[0]


# Salida 2: Diseño de Sección de la Poza


fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"colspan": 2, 'type': 'surface'},None],
           [{}, {}]],
    row_heights=[0.7, 0.3], vertical_spacing = 0.1,
    subplot_titles=("Vista 3D","Sección Transversal", "Sección Longitudinal"))


fig.add_trace(
    go.Mesh3d(
        # 8 vertices of a cube
        x=[0, b, b, 0, -s_yz1 * z, b + s_yz2 * z, b + s_yz2 * z, -s_yz1 * z],
        y=[0, 0, h, h, -s_xz1 * z, -s_xz1 * z, h + s_xz2 * z, h + s_xz2 * z],
        z=[0, 0, 0, 0, z, z, z, z],

        i = [7, 0, 0, 0, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='gray',
        flatshading = True
    ), row=1, col=1
)

fig.add_trace(
    go.Mesh3d(
        # 8 vertices of a cube
        x=[0, b, b, 0, -s_yz1 * y, b + s_yz2 * y, b + s_yz2 * y, -s_yz1 * y],
        y=[0, 0, h, h, -s_xz1 * y, -s_xz1 * y, h + s_xz2 * y, h + s_xz2 * y],
        z=[0, 0, 0, 0, y, y, y, y],

        i = [4, 5],
        j = [5, 6],
        k = [7, 7],
        opacity=0.6,
        color='lightblue',
        flatshading = True
    ), row=1, col=1
)


fig.add_trace(
    go.Scatter(
        x = [-s_xz1 * z, 0, h, h + s_xz2 * z],
        y = [z, 0, 0, z],
        mode = 'lines', line=dict(color='gray'),
      showlegend=False
        ), row = 2, col = 1,
    )

fig.add_trace(
    go.Scatter(
        x = [-s_xz1 * y, h + s_xz2 * y],
        y = [y, y], showlegend=False,
        mode = 'lines', line=dict(color='lightblue', width = 5)
        ), row = 2, col = 1,
    )


fig.add_trace(
    go.Scatter(
        x = [-s_yz1 * z, 0, b, b + s_yz2 * z],
        y = [z, 0, 0, z], showlegend=False,
        mode = 'lines', line=dict(color='gray')
        ), row = 2, col = 2
    )

fig.add_trace(
    go.Scatter(
        x = [-s_yz1 * y, b + s_yz2 * y],
        y = [y, y], showlegend=False,
        mode = 'lines', line=dict(color='lightblue', width = 5)
        ), row = 2, col = 2,
    )

fig.add_trace(
    go.Scatter(
        x = [h/6, h/6],
        y = [0, z], showlegend=False,
        mode = 'lines+markers', line=dict(color='gray')
        ), row = 2, col = 1,
    )
fig.add_trace(
    go.Scatter(
        x = [b/6, b/6],
        y = [0, z], showlegend=False,
        mode = 'lines+markers', line=dict(color='gray')
        ), row = 2, col = 2,
    )

# Texto

## Sección Transversal
fig.add_trace(go.Scatter(
    x=[h/2],
    y=[0],
    mode="text", showlegend=False,
    text=["Base = " + str(round(h, 2)) + " m"],
    textposition="top center"), row = 2, col = 1
)

fig.add_trace(go.Scatter(
    x=[h/2],
    y=[y],
    mode="text", showlegend=False,
    text=["Tirante = " + str(round(y, 2)) + " m"],
    textposition="top center"), row = 2, col = 1
)

fig.add_trace(go.Scatter(
    x=[h/5],
    y=[z/2],
    mode="text", showlegend=False,
    text=["Altura = " + str(round(z, 2)) + " m"],
    textposition="middle right"), row = 2, col = 1
)
if s_xz1 != 0:
  fig.add_trace(go.Scatter(
      x=[-s_xz1 * z / 2],
      y=[z / 2],
      mode="text", showlegend=False,
      text=[str(round(s_xz1, 2)) + "H:1V"],
      textposition="middle right"), row = 2, col = 1
  )
if s_xz2 != 0:
  fig.add_trace(go.Scatter(
      x=[h + s_xz2 * z / 2],
      y=[z / 2],
      mode="text", showlegend=False,
      text=[str(round(s_xz2, 2)) + "H:1V"],
      textposition="middle left"), row = 2, col = 1
  )

##  Sección longitudinal
fig.add_trace(go.Scatter(
    x=[b/2],
    y=[0],
    mode="text", showlegend=False,
    text=["Base = " + str(round(b, 2)) + " m"],
    textposition="top center"), row = 2, col = 2
)

fig.add_trace(go.Scatter(
    x=[b/2],
    y=[y],
    mode="text", showlegend=False,
    text=["Tirante = " + str(round(y, 2)) + " m"],
    textposition="top center"), row = 2, col = 2
)

fig.add_trace(go.Scatter(
    x=[b/5],
    y=[z/2],
    mode="text", showlegend=False,
    text=["Altura = " + str(round(z, 2)) + " m"],
    textposition="middle right"), row = 2, col = 2
)
if s_yz1 != 0:
  fig.add_trace(go.Scatter(
      x=[-s_yz1 * z / 2],
      y=[z / 2],
      mode="text", showlegend=False,
      text=[str(round(s_yz1, 2)) + "H:1V"],
      textposition="middle right"), row = 2, col = 2
  )

if s_yz2 != 0:
  fig.add_trace(go.Scatter(
      x=[b + s_yz2 * z / 2],
      y=[z / 2],
      mode="text", showlegend=False,
      text=[str(round(s_yz2, 2)) + "H:1V"],
      textposition="middle left"), row = 2, col = 2
  )

fig.update_layout(
    autosize=True,
    width=1200,
    height=1000,)

fig.update_layout(scene_aspectmode='data')

fig.update_layout(title='Diseño de Poza de Filtración - Geometría de la Poza')

st.plotly_chart(fig, use_container_width=True)

# Salida 3: Tirante a lo largo del tiempo

fig = go.Figure()

fig.add_trace(go.Scatter(x = fil_data["Time (min)"],
                         y = fil_data["Tirante (m)"],
                         mode = 'lines', showlegend=False,))

fig.add_trace(go.Scatter(y = [max(fil_data["Tirante (m)"])],
                         x = [fil_data.loc[fil_data["Tirante (m)"] == max(fil_data["Tirante (m)"])]["Time (min)"]],
                         mode = "markers",
                         name = "Tirante máximo = " + str(round(max(fil_data["Tirante (m)"]), 2) ) + " m"
  ))

fig.update_layout(title='Tirante de la Poza a lo largo del tiempo',
                  xaxis_title='Tiempo (min)',
                  yaxis_title='Tirante (m)',
                  legend = dict(
                      yanchor = "bottom",
                      y = 0.9,
                      xanchor = "right",
                      x = 0.95
                  ))


st.plotly_chart(fig, use_container_width=True)

save_data = st.checkbox('Haga click en el botón para poder descargar la tabla de resultados en .csv')
if save_data:
    csv = convert_df(fil_data)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name= "results.csv",
        mime='text/csv',
    )
    
    