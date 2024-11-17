import streamlit as st
from Metrics import show_metrics
from Predictions import show_predictions

# Estilo para ocultar el menú superior y la barra de herramientas extra
hide_menu_style = """
    <style>
    [data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Opciones de navegación en la barra lateral
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", ["Predictions", "Metrics"])

# Navegación basada en la selección
if page == "Predictions":
    show_predictions()
elif page == "Metrics":
    show_metrics()
