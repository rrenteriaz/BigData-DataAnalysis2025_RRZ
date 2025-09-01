import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

st.set_page_config(page_title="EDA Explicativa", layout="wide")

st.title("Plataforma de EDA y Búsqueda de Patrones Explicativos")
st.markdown("Carga un CSV, explora los datos y encuentra variables que expliquen (o predigan) una columna objetivo.")

# Sidebar: upload and options
st.sidebar.header("Datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"]) 
sample_data = st.sidebar.checkbox("Cargar ejemplo (iris/diamonds) si no subes archivo", value=True)

if uploaded_file is None and sample_data:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = list(df.columns)
    st.sidebar.info("Ejemplo: Iris cargado")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Error leyendo CSV: {e}")
        st.stop()
else:
    st.info("Sube un CSV para comenzar o activa 'Cargar ejemplo' en la barra lateral.")
    st.stop()

# Basic preview
st.subheader("Vista rápida de los datos")
with st.expander("Tabla (primeras 200 filas)"):
    st.dataframe(df.head(200))

# Data structure
st.subheader("Estructura y tipos")
col1, col2 = st.columns([2,1])
with col1:
    st.write(df.dtypes)
with col2:
    st.metric("Filas", df.shape[0])
    st.metric("Columnas", df.shape[1])

# Missing values
st.subheader("Valores faltantes")
miss = df.isnull().sum().sort_values(ascending=False)
miss = miss[miss > 0]
if not miss.empty:
    fig = px.bar(miss, x=miss.index, y=miss.values,
                 labels={'x': 'Columna', 'y': 'Valores faltantes'},
                 title="Valores faltantes por columna")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hay valores faltantes en el dataset.")

# Descriptive stats
st.subheader("Estadísticas descriptivas")
num_df = df.select_dtypes(include=[np.number])
cat_df = df.select_dtypes(exclude=[np.number])
with st.expander("Numéricas"):
    if not num_df.empty:
        st.dataframe(num_df.describe().T)
    else:
        st.info("No hay columnas numéricas para describir.")
with st.expander("Categóricas"):
    if not cat_df.empty:
        st.dataframe(cat_df.describe().T)
    else:
        st.info("No hay columnas categóricas para describir.")

# Automatic datatype suggestions
st.subheader("Sugerencias de preprocesado")
if len(cat_df.columns)>0:
    st.info(f"Columnas categóricas detectadas: {', '.join(cat_df.columns[:10])}")
if len(num_df.columns)>0:
    st.info(f"Columnas numéricas detectadas: {', '.join(num_df.columns[:10])}")

# Correlation matrix
if num_df.shape[1]>1:
    st.subheader("Matriz de correlación")
    corr = num_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlación (numéricas)")
    st.plotly_chart(fig, use_container_width=True)

# Distribuciones y outliers
st.subheader("Distribuciones")
col = st.selectbox("Selecciona una variable numérica", options=[None]+list(num_df.columns))
if col:
    fig = px.histogram(df, x=col, nbins=50, marginal="box", title=f"Distribución de {col}")
    st.plotly_chart(fig, use_container_width=True)

# Scatter / relationships
st.subheader("Relaciones entre variables")
x_var = st.selectbox("X", options=[None]+list(num_df.columns), index=0)
y_var = st.selectbox("Y", options=[None]+list(num_df.columns), index=0)
color_var = st.selectbox("Color (opcional)", options=[None]+list(df.columns), index=0)
if x_var and y_var:
    fig = px.scatter(df, x=x_var, y=y_var, color=color_var, trendline='ols', title=f"{y_var} vs {x_var}")
    st.plotly_chart(fig, use_container_width=True)

# PCA
if num_df.shape[1]>=2:
    st.subheader("PCA (2 componentes)")
    n_components = 2
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(num_df.fillna(num_df.mean()))
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    if cat_df.shape[1]>0:
        color_pca = st.selectbox('Color por (opcional)', options=[None]+list(cat_df.columns))
    else:
        color_pca = None
    fig = px.scatter(pca_df, x='PC1', y='PC2', color=(df[color_pca] if color_pca is not None else None), title='PCA 2D')
    st.plotly_chart(fig, use_container_width=True)

# Búsqueda de patrones explicativos (feature importance)
st.subheader("Búsqueda de patrones explicativos (importancia de variables)")
st.markdown("Selecciona una columna objetivo para evaluar qué variables la explican mejor usando modelos de bosque aleatorio y importancia por permutación.")

target = st.selectbox("Columna objetivo (target)", options=[None]+list(df.columns))
if target:
    task = st.radio("Tipo de tarea (si no estás seguro, elige automáticamente)", options=["Clasificación", "Regresión", "Automático"], index=2)
    if task == 'Automático':
        if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique()>20:
            task = 'Regresión'
        else:
            task = 'Clasificación'
    st.write(f"Tarea seleccionada: {task}")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Simple preprocessing: encode categoricals
    encoders = {}
    for c in X.select_dtypes(exclude=[np.number]).columns:
        X[c] = X[c].astype(str).fillna('NA')
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encoders[c] = le
    # target encoding if needed
    y_enc = None
    if task == 'Clasificación' and not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).fillna('NA')
        y_enc = LabelEncoder()
        y = y_enc.fit_transform(y)

    X = X.fillna(X.mean())

    # Train/test
    test_size = st.slider('Tamaño test (%)', 5, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    if task == 'Clasificación':
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    with st.spinner('Entrenando modelo...'):
        model.fit(X_train, y_train)

    # Performance
    y_pred = model.predict(X_test)
    if task == 'Clasificación':
        perf = accuracy_score(y_test, y_pred)
        st.metric('Accuracy (test)', f"{perf:.4f}")
    else:
        perf = r2_score(y_test, y_pred)
        st.metric('R^2 (test)', f"{perf:.4f}")

    # Feature importances (built-in)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.subheader('Importancia por el modelo (feature_importances_)')
    fig = px.bar(fi.head(30),
                 x=fi.head(30).index, y=fi.head(30).values,
                 labels={'x': 'Feature', 'y': 'Importancia'},
                 title="Importancia de variables (modelo)")
    st.plotly_chart(fig, use_container_width=True)

    # Permutation importance
    st.subheader('Importancia por permutación')
    with st.spinner('Calculando importancia por permutación (puede tardar)...'):
        r = permutation_importance(model, X_test, y_test, n_repeats=12, random_state=42, n_jobs=-1)
    perm = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    fig = px.bar(perm.head(30),
                 x=perm.head(30).index, y=perm.head(30).values,
                 labels={'x': 'Feature', 'y': 'Importancia'},
                 title="Importancia de variables (permutación)")
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar tabla de importancias
    imp_df = pd.DataFrame({'feature': X.columns, 'model_importance': fi.values, 'perm_importance': perm.values})
    imp_df = imp_df.sort_values('perm_importance', ascending=False)
    with st.expander('Tabla de importancias (ordenada por permutación)'):
        st.dataframe(imp_df)

    # Mostrar relaciones top features vs target
    topk = st.slider('Número de features a inspeccionar', 1, min(10, X.shape[1]), 3)
    top_feats = imp_df['feature'].head(topk).tolist()
    st.subheader('Relaciones de las features más importantes con el target')
    for f in top_feats:
        if pd.api.types.is_numeric_dtype(df[f]):
            fig = px.scatter(df, x=f, y=target, trendline='ols', title=f"{target} vs {f}")
        else:
            fig = px.box(df, x=f, y=target, title=f"{target} por {f}")
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Hecho con ❤️ — Plataforma simple de EDA explicativa. Ajusta y expande según tus necesidades.")

# Footer: save transformed dataset
if st.button('Descargar dataset preprocesado (CSV)'):
    st.sidebar.success('Generando CSV...')
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar CSV', data=csv, file_name='dataset_preprocesado.csv', mime='text/csv')
