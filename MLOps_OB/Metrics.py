import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow

def show_metrics():
    st.title("Métricas del Modelo")
    st.write("Visualiza las métricas registradas en MLflow aquí.")

    # Configuración de MLflow
    mlflow_host = "postgresql+psycopg2://mlflow2:mlflowtec2@mlflow2.cnugu40qabl8.us-east-2.rds.amazonaws.com:5432/mlflowdb"
    mlflow.set_tracking_uri(mlflow_host)
    client = mlflow.tracking.MlflowClient()

    # Listar experimentos
    experiments = client.search_experiments()
    experiment_names = [exp.name for exp in experiments]
    selected_experiment = st.selectbox("Selecciona un Experimento", experiment_names)

    if selected_experiment:
        experiment = next(exp for exp in experiments if exp.name == selected_experiment)
        runs = client.search_runs(experiment.experiment_id)
        st.subheader("Runs del Experimento")

        run_ids = [run.info.run_id for run in runs]
        selected_run_id = st.selectbox("Selecciona un Run", run_ids)

        if selected_run_id:
            selected_run = client.get_run(selected_run_id)
            model_name = selected_run.data.tags.get("mlflow.runName", "Modelo Desconocido")
            metrics = selected_run.data.metrics
            metrics_df = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])

            st.subheader("Métricas del Run")
            st.write(f"Nombre del modelo: **{model_name}**")
            st.write(f"ID del Run: **{selected_run_id}**")
            st.dataframe(metrics_df)

            # Gráficos interactivos con Plotly
            st.subheader("Gráficos de Métricas")

            # Gráfico de barras para métricas
            fig_bar = px.bar(
                metrics_df,
                x="Métrica",
                y="Valor",
                title="Métricas del Run",
                labels={"Valor": "Puntaje", "Métrica": "Tipo de Métrica"},
                text_auto=True
            )
            st.plotly_chart(fig_bar)

            # Gráfico de líneas (opcional)
            fig_line = px.line(
                metrics_df,
                x="Métrica",
                y="Valor",
                title="Tendencia de Métricas",
                markers=True
            )
            st.plotly_chart(fig_line)

# Ejemplo de cómo llamar a esta función desde Home.py


