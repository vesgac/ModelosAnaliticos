# Importa la biblioteca Streamlit
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import altair as alt
import plotly.express as px
from sklearn.tree import  DecisionTreeClassifier
from PIL import Image
import graphviz
from sklearn import tree
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load
import shap
from shap import datasets, KernelExplainer, force_plot, initjs

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


# Cargar los datos
data = pd.read_csv("ObesityDataSet.csv")
data_cluster = pd.read_csv("ObesityCluster.csv")


full_form = dict({'FAVC' : "Frequent consumption of high caloric food",
                  'FCVC' : "Frequency of consumption of vegetables",
                  'NCP' :"Number of main meal",
                  'CAEC': "Consumption of food between meals",
                  'CH2O': "Consumption of water daily",
                  'SCC':  "Calories consumption monitoring",
                  'FAF': "Physical activity frequency",
                  'TUE': "Time using technology devices",
                  'CALC': "Consumption of alcohol" ,
                  'MTRANS' : "Transportation used"})


def showplot(columnname):
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    ax = ax.flatten()
    value_counts = data[columnname].value_counts()
    labels = value_counts.index.tolist()
    colors = ["#4caba4", "#d68c78", '#a3a2a2', '#ab90a0', '#e6daa3', '#6782a8', '#8ea677']

    # Donut Chart
    wedges, texts, autotexts = ax[0].pie(
        value_counts, autopct='%1.1f%%', textprops={'size': 9, 'color': 'white', 'fontweight': 'bold'},
        colors=colors, wedgeprops=dict(width=0.35), startangle=80, pctdistance=0.85)
    # circle
    centre_circle = plt.Circle((0, 0), 0.6, fc='white')
    ax[0].add_artist(centre_circle)

    # Count Plot
    sns.countplot(data=data, y=columnname, ax=ax[1], palette=colors, order=labels)
    for i, v in enumerate(value_counts):
        ax[1].text(v + 1, i, str(v), color='black', fontsize=10, va='center')
    sns.despine(left=True, bottom=True)
    plt.yticks(fontsize=9, color='black')
    ax[1].set_ylabel(None)
    plt.xlabel("")
    plt.xticks([])
    fig.suptitle("Categorias", fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Retorna la figura
    return fig


st.set_page_config(
    page_title="Predictive Health Care Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")



# Título de la aplicación
st.title("Predictive Health Care Dashboard")


with st.sidebar:
    # Define la variable objetivo y las columnas numéricas
    target = 'NObeyesdad'
    numeric_columns = data.select_dtypes(include=['float64', 'int64'])
    cat_cols = data.select_dtypes(include=['object'])

    # Sección para seleccionar la variable numérica y el género
    st.sidebar.title("Configuración del Usuario")
    selected_numeric_column = st.sidebar.selectbox("Selecciona una Variable Numérica", numeric_columns.columns)
    selected_gender = st.sidebar.selectbox("Selecciona Género", data['Gender'].unique())
    selected_cat_col = st.sidebar.selectbox("Selecciona una variable Categorica",cat_cols.columns)

    # Filtra el conjunto de datos según el género seleccionado
    filtered_data = data[data['Gender'] == selected_gender]



# Columnas en el lado izquierdo
col = st.columns((4.5, 4.5), gap='medium')


with col[0]:
        # Sección para los gráficos en el centro
        st.subheader(f"Violin Plot para {selected_numeric_column}")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.violinplot(x=target, y=selected_numeric_column, hue="Gender", data=filtered_data, split=False,  palette=['#7fb3d5'])
        if selected_numeric_column in full_form.keys():
            ax.set_ylabel(full_form[selected_numeric_column])

        # Ajusta el diseño y muestra el gráfico en Streamlit
        plt.title(f"Violin Plot para {selected_numeric_column}")
        plt.tight_layout()
        st.pyplot(fig)


with col[1]:
        # Grafico circular en la parte inferior
        st.subheader("Gráfico Circular para Categorías de Peso")
        fig = showplot("NObeyesdad")
        st.pyplot(fig)



st.subheader("Grafico de variables categoricas")
colors = ['#1f77b4', '#fc6c44', '#2b8a2b', '#fc7c7c', '#9467bd', '#4ba4ad', '#c7ad18', '#7f7f7f', '#69d108']
fig, axs = plt.subplots(1, 2, figsize=(12,4), width_ratios=[1, 4])
sns.countplot(y=selected_cat_col, data=data, palette=colors, ax=axs[0])
sns.countplot(x=selected_cat_col, data=data, hue=target, palette=colors, ax=axs[1])
if selected_cat_col in full_form.keys():
    axs[0].set_ylabel(full_form[selected_cat_col])
plt.tight_layout()
st.pyplot(fig)


# Mostrar el DataFrame en el lado izquierdo
st.write("Datos de pacientes con diferentes categorías de peso:")
st.dataframe(data.head())

# Crear un clasificador de árbol de decisión con profundidad máxima de 4
clf = DecisionTreeClassifier(max_depth=3)

# Separar las características y las etiquetas
X = data.drop(target, axis=1)  # Ajusta 'etiqueta' al nombre real de la columna de etiquetas
y = data[target]

X = pd.get_dummies(X)

# Entrenar el clasificador
clf = clf.fit(X, y)

# Columnas en el lado izquierdo
col = st.columns((4.5, 4.5), gap='medium')
with col[0]:
    # Mostrar el árbol de decisión en el dashboard de Streamlit
    st.subheader("Árbol de decisión EDA")
    fig, ax = plt.subplots(figsize=(20, 15))
    tree.plot_tree(clf, feature_names=X.columns, filled=True, fontsize=10, ax=ax)
    st.pyplot(fig)

with col[1]:
    st.subheader("Perfiles de los Cluster")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(x="Cluster", y=selected_numeric_column, data=data_cluster, palette='viridis', ax=ax)
    # Mostrar ambos gráficos inmersos en Streamlit
    plt.xlabel("Cluster", fontsize=18)
    plt.ylabel(f"{selected_numeric_column}", fontsize=18)

    # Ajustar el tamaño de la fuente en las etiquetas de los ejes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    st.pyplot(fig)

# Definir las columnas y tipos de datos
columnas = [
    "Gender", "Age", "Height", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O",
    "SCC", "FAF", "TUE", "CALC", "MTRANS"
]

tipos_de_dato = {
    "Gender": "category",
    "Age": float,
    "Height": float,
    "family_history_with_overweight": "category",
    "FAVC": "category",
    "FCVC": float,
    "NCP": float,
    "CAEC": "category",
    "SMOKE": "category",
    "CH2O": float,
    "SCC": "category",
    "FAF": float,
    "TUE": float,
    "CALC": "category",
    "MTRANS": "category"
}

# Crear un diccionario con componentes Streamlit correspondientes a cada columna
with st.expander("Herramienta de prediccion"):
      componentes_por_columna = {}
      for columna, tipo_dato in tipos_de_dato.items():
          if tipo_dato == float:
                componente = st.number_input(label=columna)
          elif tipo_dato == "category":
                valores_posibles = data_cluster[columna].unique()
                componente = st.selectbox(label=columna, options=valores_posibles)
          else:
              raise ValueError(f"Tipo de dato no soportado para la columna {columna}")
          componentes_por_columna[columna] = componente

      modelo= joblib.load("mejor_normal.joblib")
      if st.button("Generar Prediccion"):
          datos = {}
          for columna, componente in componentes_por_columna.items():
              datos[columna] = componente

          X_predict = pd.DataFrame([datos])
          columnas_numericas = X_predict.select_dtypes(include=['int64', 'float64']).columns
          columnas_categoricas = X_predict.select_dtypes(include=['object', 'category']).columns

          for columna in columnas_categoricas:
              X_predict[columna] = X_predict[columna].astype('category')

          shap.initjs()
          explainer = shap.TreeExplainer(modelo)
          pred = modelo.predict(X_predict)
          shap_values = explainer.shap_values(X_predict)

          # Mapeo de índices a nombres de clases
          class_mapping = {
              0: 'Insufficient_Weight',
              1: 'Normal_Weight',
              2: 'Overweight_Level_I',
              3: 'Overweight_Level_II',
              4: 'Obesity_Type_I',
              5: 'Obesity_Type_II',
              6: 'Obesity_Type_III'
          }

          # Obtiene el nombre de la clase predicha
          predicted_class_name = class_mapping[pred[0]]

          # Imprime el nombre de la clase predicha como un título
          st.title(f"Clase predicha: {predicted_class_name}")

          # Crear un gráfico SHAP utilizando shap.force_plot
          force_plot_fig = shap.force_plot(explainer.expected_value[pred[0]], shap_values[pred[0]][0], X_predict.values[0], feature_names=X_predict.columns, matplotlib=True)

          # Muestra el gráfico SHAP en el dashboard
          st.pyplot(force_plot_fig)

          # Opcional: Cerrar la figura de Matplotlib para evitar el conflicto
          plt.close()





