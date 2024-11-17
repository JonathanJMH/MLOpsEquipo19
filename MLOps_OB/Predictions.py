import streamlit as st
import pandas as pd
import requests

def show_predictions():
    st.title("Formulario de Predicciones")
    st.write("Ingresa las variables originales del dataset.")

    # Inicializar valores por defecto
    if "default_values" not in st.session_state:
        st.session_state.default_values = {
            "Gender": "Male",
            "Age": 25,
            "Height": 1.75,
            "family_history_with_overweight": "yes",
            "FAVC": "yes",
            "FCVC": 2.0,
            "NCP": 3,
            "CAEC": "no",
            "SMOKE": "no",
            "CH2O": 2.0,
            "SCC": "yes",
            "FAF": 0.5,
            "TUE": 0.5,
            "CALC": "no",
            "MTRANS": "Walking"
        }

    input_data = {
        "Gender": st.selectbox("Género", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.default_values["Gender"])),
        "Age": st.number_input("Edad", min_value=1, max_value=100, step=1, value=st.session_state.default_values["Age"]),
        "Height": st.number_input("Altura (m)", min_value=1.0, max_value=2.5, step=0.01, value=st.session_state.default_values["Height"]),
        "family_history_with_overweight": st.selectbox("Historial Familiar con Sobrepeso", ["yes", "no"], index=["yes", "no"].index(st.session_state.default_values["family_history_with_overweight"])),
        "FAVC": st.selectbox("Frecuencia de Consumo de Comida Alta en Calorías", ["yes", "no"], index=["yes", "no"].index(st.session_state.default_values["FAVC"])),
        "FCVC": st.number_input("Consumo de Verduras Frecuente", min_value=0.0, max_value=3.0, step=0.1, value=st.session_state.default_values["FCVC"]),
        "NCP": st.number_input("Número de Comidas Principales", min_value=1, max_value=5, step=1, value=st.session_state.default_values["NCP"]),
        "CAEC": st.selectbox("Consumo de Alimentos entre Comidas", ["no", "Sometimes", "Frequently", "Always"], index=["no", "Sometimes", "Frequently", "Always"].index(st.session_state.default_values["CAEC"])),
        "SMOKE": st.selectbox("Fuma", ["yes", "no"], index=["yes", "no"].index(st.session_state.default_values["SMOKE"])),
        "CH2O": st.number_input("Consumo de Agua (Litros)", min_value=0.0, max_value=3.0, step=0.1, value=st.session_state.default_values["CH2O"]),
        "SCC": st.selectbox("Monitorea las Calorías Consumidas", ["yes", "no"], index=["yes", "no"].index(st.session_state.default_values["SCC"])),
        "FAF": st.number_input("Actividad Física Frecuente (Horas)", min_value=0.0, max_value=2.0, step=0.1, value=st.session_state.default_values["FAF"]),
        "TUE": st.number_input("Uso de Dispositivos Electrónicos (Horas)", min_value=0.0, max_value=2.0, step=0.1, value=st.session_state.default_values["TUE"]),
        "CALC": st.selectbox("Frecuencia de Consumo de Alcohol", ["no", "Sometimes", "Frequently", "Always"], index=["no", "Sometimes", "Frequently", "Always"].index(st.session_state.default_values["CALC"])),
        "MTRANS": st.selectbox("Modo de Transporte", ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"], index=["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"].index(st.session_state.default_values["MTRANS"])),
    }

    # Convertir a DataFrame
    input_df = pd.DataFrame([input_data])

    # Botón para predecir
    if st.button("Predecir"):
        try:
            API_URL = "http://localhost:8000/predict"
            response = requests.post(API_URL, json={"data": input_df.to_dict(orient="records")})
            response.raise_for_status()
            
            predictions = response.json().get("predictions", [])
            
            st.write("Resultados:")
            for pred in predictions:
                st.write(f"Clase Predicha: {pred['predicted_class']}")
                st.write(f"Probabilidades: {pred['probabilities']}")
                st.write("---")
        except requests.exceptions.RequestException as e:
            st.error(f"Error al realizar la predicción: {e}")

    # Botón para reiniciar valores
    if st.button("Reiniciar a Cero"):
        # Restablecer valores predeterminados
        st.session_state.default_values = {
            "Gender": "Male",
            "Age": 25,
            "Height": 1.75,
            "family_history_with_overweight": "yes",
            "FAVC": "yes",
            "FCVC": 2.0,
            "NCP": 3,
            "CAEC": "no",
            "SMOKE": "no",
            "CH2O": 2.0,
            "SCC": "yes",
            "FAF": 0.5,
            "TUE": 0.5,
            "CALC": "no",
            "MTRANS": "Walking"
        }
        st.rerun()  # Recargar la página para aplicar los cambios
