import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_auc_score, confusion_matrix, 
                             precision_score, recall_score, f1_score, RocCurveDisplay,make_scorer)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector as selector
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Configuración de MLflow
mlflow.set_tracking_uri("postgresql+psycopg2://mlflow2:mlflowtec2@mlflow2.cnugu40qabl8.us-east-2.rds.amazonaws.com:5432/mlflowdb")
mlflow.set_experiment("ml_experiment")

# Guardar objetos como pickle
def save_as_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Objeto guardado en {filename}")

# Función para graficar y registrar la matriz de confusión
def log_confusion_matrix(y_true, y_pred, labels):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

# Función para graficar la curva ROC multiclase
def log_roc_curve_multiclass(estimator, X_test, y_test, classes):
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = estimator.predict_proba(X_test)

    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(classes):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i],
            y_score[:, i],
            name=f"ROC curve for class {class_label}",
            ax=plt.gca()
        )

    plt.title("Multiclass ROC Curve")
    plt.savefig("roc_curve_multiclass.png")
    plt.close()
    mlflow.log_artifact("roc_curve_multiclass.png")

# Entrenar modelos y registrar el mejor con MLflow
def train_and_log_model(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(),
        'KNeighbors': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }

    param_grids = {
        'Logistic Regression': {'classifier__C': [0.1, 1, 10]},
        'Gradient Boosting': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.1, 0.01]},
        'KNeighbors': {'classifier__n_neighbors': [3, 5, 7]},
        'DecisionTree': {'classifier__max_depth': [5, 10, None]},
        'AdaBoost': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.1, 1]}
    }

    auc_scorer = make_scorer(roc_auc_score, multi_class='ovo', needs_proba=True)

    best_model, best_model_name, best_score = None, "", 0

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(steps=[
                ('preprocessor', ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), selector(dtype_include=['int64', 'float64'])),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include='object'))
                    ]
                )),
                ('classifier', model)
            ])

            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=5,
                scoring=auc_scorer,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            # Evaluación en el conjunto de prueba
            y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)
            y_pred = grid_search.best_estimator_.predict(X_test)
            test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')

            # Métricas adicionales
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Log de métricas en MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("train_roc_auc_score", grid_search.best_score_)
            mlflow.log_metric("test_roc_auc_score", test_roc_auc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log confusion matrix
            log_confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

            # Log ROC curve para multiclase
            log_roc_curve_multiclass(grid_search.best_estimator_, X_test, y_test, classes=np.unique(y_test))

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_model_name = model_name

        mlflow.end_run()

    # Guardar el mejor modelo con preprocesador y transformaciones integradas
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")
    save_as_pickle(best_model, "best_model.pkl")

    return best_model_name, best_score

# Cargar y preparar datos

if __name__ == "__main__":
    data_path = '/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_OB/Streamlit_MLOPS/ML_OPS_OB/data/ObesityDataSet_raw_and_data_sinthetic.csv'
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()

    data = data.drop(columns=['Weight'], errors='ignore')
    target_column = 'NObeyesdad'
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model_name, best_score = train_and_log_model(X_train, X_test, y_train, y_test)
    print(f"Mejor modelo: {best_model_name} con ROC AUC en validación: {best_score:.2f}")
