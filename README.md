# MLOps Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<a target="_blank" href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=ffffff" alt="Python" />
</a>
<a target="_blank" href="https://dvc.org/">
    <img src="https://img.shields.io/badge/DVC-3BBF8C?logo=data-vault&logoColor=ffffff" alt="DVC" />
</a>
<a target="_blank" href="https://mlflow.org/">
    <img src="https://img.shields.io/badge/MLflow-FF4A1C?logo=mlflow&logoColor=ffffff" alt="MLflow" />
</a>
<a target="_blank" href="https://jupyter.org/">
    <img src="https://img.shields.io/badge/Jupyter-DA5B1E?logo=jupyter&logoColor=ffffff" alt="Jupyter Notebook" />
</a>
<a target="_blank" href="https://mlops.org/">
    <img src="https://img.shields.io/badge/MLOps-1C8FFF?logo=mlops&logoColor=ffffff" alt="MLOps" />
</a>

## Overview
This project is a full MLOps pipeline that uses DVC for tracking data and models, and MLflow for managing experiments and the model lifecycle. It covers everything from data preparation, model training, and evaluation to deployment.


## Project Organization

```
├── LICENSE
├── Makefile
├── params.yaml
├── README.md
├── .dvc/
├── .vscode/
├── data
│   ├── external/       
│   ├── interim/
│   ├── predictions/
│   ├── processed/      
│   └── raw/
├── docs/
├── metrics/
├── mlruns/
├── models/
├── notebooks/
├── refactoring
│   ├── obesity_refactored_v1.py
│   ├── __init__.py
│   ├── config
│   │   ├── load_params.py
│   │   └── __init__.py
│   ├── data
│   │   ├── load_data.py
│   │   └── __init_.py
│   ├── evaluation
│   │   ├── evaluate.py
│   │   └── __init__.py
│   ├── modeling
│   │   ├── train.py
│   │   └── __init__.py
│   ├── plots/
│   └── preprocessing
│       ├── data_preparation.py
│       └── __init__.py
├── references/
├── reports/
└── tests/
```

--------

## Features

- **Data and model tracking:** Implements DVC for efficient tracking.
    **Status: Completed**
- **Experiment management:** Utilizes MLflow to manage the model lifecycle and experiments.
     **Status: Completed**
- **Environment-specific configurations:** Includes configurations for development, staging, and production environments.
    **Status: Completed**
- **Flask API:** Provides an API for serving model predictions. 
    **Status: In Development**
- **Visualization with Streamlit:** Allows visualization of experiment results and model monitoring. 
    **Status: In Development**

## **Setup and Running**

### **1. Prerequisites**
- Docker & Docker Compose
- Python 3.10+ and `pip`
- DVC (`pip install dvc`)
- MLFlow (`pip install mlflow`)
- **Remote storage** set up for DVC (e.g., S3 or GDrive)

### **2. Clone the Repository**

```bash
git clone https://github.com/JonathanJMH/MLOpsEquipo19.git
cd mlops_project
```

### **3. Virtual Environment Setup**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### **4. Configure Environment Variables**

Create a folder for local configuration to set up the environment variables. This should include:

- A `.vscode` folder to store your Visual Studio Code settings and add `settings.json` file:

    ```bash
    {
        "python.envFile": "${workspaceFolder}/.env"
    }
    ```
- A `.env` file to define environment variables required by your application and replece `{workspaceFolder}` with your current workspaceFolder.

    ```bash
    PYTHONPATH={workspaceFolder}
    ```
### **5. Install the dependencies:**
```bash
pip install -r requirements.txt
```
If you have any problem with the current dependencies you can try with the old version of `requirements.txt`:
```bash
pip install -r requirements2.txt
```

### **6. Setup DVC**
    dvc pull

### **7. DVC Pipeline Management**

The DVC pipeline is defined in `dvc.yaml`, which outlines the data preparation, model training, and evaluation steps.

To run the full DVC pipeline:

```bash
dvc repro
```

This will execute the following stages:
1. **Load Data**: Load and adjust the data
2. **Data Preparation**: Load, preprocess, and split the data.
3. **Model Training**: Train the model and log it with MLflow.
4. **Model Evaluation**: Evaluate the model and track metrics.
5. **Tests**: Verify that all previous steps work correctly and track report

    You can also use:    
    ```bash
    pytest --html=reports/tests/report.html
    ```

To push the artifacts to the remote storage:
```bash
dvc push
```

### **8. MLflow Experiment Visualization**

You can monitor **MLflow experiment runs**, metrics, and models, which helps you track your production model's performance with:

```bash
mlflow ui
```
## Governance Practices
- [Project Governance](docs/docs/GOVERNANCE.md)
- [Security Policy](docs/docs/SECURITY.md)
- [Risk Assessent](docs/docs/risk_assessment)
- [Changelog](docs/docs/CHANGELOG.md)