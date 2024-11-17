from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle

# Inicializar FastAPI
app = FastAPI()

# Cargar el pipeline completo (preprocesamiento + modelo)
with open("/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_OB/Streamlit_MLOPS/ML_OPS_OB/pages/best_model.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

# Definir la estructura de entrada esperada
class InputData(BaseModel):
    data: List[dict]  # Lista de registros con las variables originales del CSV

# Endpoint para predicción
@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convertir la entrada a DataFrame
        input_df = pd.DataFrame(input_data.data)

        # Validar si todas las columnas requeridas están presentes
        expected_columns = model_pipeline.named_steps['preprocessor'].feature_names_in_
        missing_columns = set(expected_columns) - set(input_df.columns)
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Faltan columnas en la entrada: {missing_columns}")

        # Realizar predicción utilizando el pipeline completo
        predictions = model_pipeline.predict(input_df)
        probabilities = model_pipeline.predict_proba(input_df)

        # Preparar respuesta
        response = []
        for pred, prob in zip(predictions, probabilities):
            response.append({
                "predicted_class": pred,
                "probabilities": {str(i): round(p, 4) for i, p in enumerate(prob)}
            })

        return {"predictions": response}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")

# Endpoint para validar columnas esperadas
@app.get("/validate")
def validate_columns():
    try:
        # Obtener columnas originales esperadas por el preprocesador
        expected_columns = model_pipeline.named_steps['preprocessor'].feature_names_in_
        return {"expected_columns": list(expected_columns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al validar columnas: {str(e)}")

# Endpoint para probar que la API funciona correctamente
@app.get("/")
def root():
    return {"message": "API para predicciones funcionando correctamente."}
