# California Housing Price Prediction - ML Engineering Project

Proyecto completo de Machine Learning Engineering para prediccion de precios de viviendas en California utilizando el dataset clasico de scikit-learn. Enfoque en mejores practicas de MLOps.

## Descripcion General

Este proyecto implementa un pipeline end-to-end de Machine Learning que incluye:

- Entrenamiento de modelos con Random Forest
- API REST con FastAPI para predicciones en tiempo real
- Tracking de experimentos con MLflow
- Containerizacion con Docker
- Suite de pruebas automatizadas
- Notebooks interactivos para exploracion

## Arquitectura del Sistema

```
┌─────────────────┐
│  Jupyter Lab    │  Exploracion y entrenamiento
└────────┬────────┘
         │
         v
┌─────────────────┐
│    MLflow       │  Tracking de experimentos y registro de modelos
└────────┬────────┘
         │
         v
┌─────────────────┐
│  FastAPI App    │  API REST para inferencia
└────────┬────────┘
         │
         v
┌─────────────────┐
│     Docker      │  Deployment containerizado
└─────────────────┘
```

## Stack Tecnologico

**Core ML:**
- Python 3.9 - 3.12
- scikit-learn: Modelos de ML
- pandas, numpy: Procesamiento de datos
- matplotlib, seaborn: Visualizacion

**API y Serving:**
- FastAPI: Framework de API REST
- uvicorn: Servidor ASGI
- pydantic: Validacion de datos

**MLOps:**
- MLflow: Tracking de experimentos y registro de modelos
- Jupyter Lab: Exploracion interactiva

**Testing:**
- pytest: Framework de testing
- httpx: Cliente HTTP asincrono

**Deployment:**
- Docker: Containerizacion
- docker-compose: Orquestacion multi-contenedor

## Estructura del Proyecto

```
ml_engineering_project/
├── README.md                    # Este archivo - Documentacion completa
├── QUICKSTART.md                # Guia rapida de inicio en 5 minutos
├── setup.sh                     # Script de configuracion para macOS/Linux
├── setup.ps1                    # Script de configuracion para Windows
├── requirements.txt             # Dependencias Python
├── .gitignore                   # Archivos ignorados por Git
├── .env.example                 # Template de variables de entorno
│
├── app/
│   └── main.py                  # Aplicacion FastAPI completa
│
├── notebooks/
│   └── 01_california_housing_pipeline.ipynb  # Pipeline de entrenamiento
│
├── test_api.py                  # Suite de pruebas de la API
├── examples_api_usage.py        # Ejemplos de consumo de la API
│
├── Dockerfile                   # Definicion de imagen Docker
├── docker-compose.yml           # Configuracion de servicios
└── .dockerignore                # Archivos ignorados por Docker

# Carpetas generadas automaticamente:
├── venv/                        # Ambiente virtual Python
├── data/                        # Datasets procesados
├── models/                      # Modelos entrenados serializados
├── outputs/                     # Graficas y visualizaciones
├── mlruns/                      # Metadatos de MLflow
└── logs/                        # Logs de aplicacion
```

## Inicio Rapido

### Opcion 1: Setup Automatico (Recomendado)

**macOS/Linux:**
```bash
bash setup.sh
```

**Windows (PowerShell como Administrador):**
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

El script interactivo te guiara a traves de:
1. Deteccion de Python compatible (3.9-3.12)
2. Creacion de ambiente virtual
3. Instalacion de dependencias (UV o pip)
4. Configuracion de variables de entorno
5. Verificacion de la instalacion

### Opcion 2: Setup Manual

```bash
# 1. Crear ambiente virtual
python3.11 -m venv venv

# 2. Activar ambiente
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env

# 5. Verificar instalacion
python -c "import sklearn, fastapi, mlflow; print('OK')"
```

## Uso del Proyecto

### 1. Entrenamiento del Modelo

**Opcion A: Notebook Interactivo**
```bash
jupyter lab notebooks/01_california_housing_pipeline.ipynb
```

El notebook contiene 3 celdas principales:
1. Carga y exploracion de datos
2. Entrenamiento y evaluacion del modelo
3. Registro en MLflow y guardado del modelo

**Opcion B: Script Python**
```bash
python -c "
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os

# Cargar datos
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.4f}')
print(f'R2: {r2:.4f}')

# Guardar modelo
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/california_housing_model.pkl')
print('Modelo guardado en models/california_housing_model.pkl')
"
```

### 2. MLflow Tracking UI

Iniciar el servidor de MLflow:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Acceder a: http://localhost:5000

Funcionalidades:
- Comparacion de experimentos
- Visualizacion de metricas (RMSE, R2, MAE)
- Graficas de feature importance
- Registro de hiperparametros
- Versionado de modelos

### 3. API REST con FastAPI

**Iniciar servidor:**
```bash
# Desarrollo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Produccion
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Documentacion interactiva:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Endpoints disponibles:**

```
GET  /                      Informacion general de la API
GET  /health                Health check del sistema
POST /predict               Prediccion para una casa
POST /predict/batch         Prediccion para multiples casas
GET  /model/info            Informacion del modelo cargado
GET  /model/features        Lista de features requeridas
```

### 4. Ejemplos de Uso de la API

**Python:**
```python
import requests

# Prediccion simple
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.98,
        "AveBedrms": 1.02,
        "Population": 322.0,
        "AveOccup": 2.55,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
)
print(response.json())
# {'predicted_price': 4.526, 'price_category': 'high', 'confidence': 'high'}

# Prediccion batch
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "houses": [
            {"MedInc": 8.3252, "HouseAge": 41.0, ...},
            {"MedInc": 3.5214, "HouseAge": 28.0, ...}
        ]
    }
)
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322.0,
    "AveOccup": 2.55,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

**JavaScript/Fetch:**
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    MedInc: 8.3252,
    HouseAge: 41.0,
    AveRooms: 6.98,
    AveBedrms: 1.02,
    Population: 322.0,
    AveOccup: 2.55,
    Latitude: 37.88,
    Longitude: -122.23
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

### 5. Testing

**Ejecutar todas las pruebas:**
```bash
pytest test_api.py -v
```

**Ejecutar pruebas especificas:**
```bash
# Solo health check
pytest test_api.py::test_health_check -v

# Solo predicciones
pytest test_api.py::test_predict_endpoint -v

# Con coverage
pytest test_api.py --cov=app --cov-report=html
```

**Pruebas incluidas:**
- Health check y endpoint raiz
- Prediccion individual
- Prediccion batch
- Validacion de inputs
- Manejo de errores
- Info del modelo

### 6. Docker Deployment

**Opcion A: Docker Compose (Recomendado)**

```bash
# Construir e iniciar servicios
docker-compose up --build

# Modo background
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

Servicios disponibles:
- API: http://localhost:8000
- MLflow: http://localhost:5000

**Opcion B: Docker Manual**

```bash
# Construir imagen
docker build -t california-housing-api .

# Ejecutar contenedor
docker run -d \
  --name housing-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  california-housing-api

# Ver logs
docker logs -f housing-api

# Detener y eliminar
docker stop housing-api
docker rm housing-api
```

## Features del Dataset

El dataset California Housing contiene 8 features numericas:

| Feature | Descripcion | Rango Tipico |
|---------|-------------|--------------|
| MedInc | Ingreso medio del bloque (en $10k) | 0.5 - 15.0 |
| HouseAge | Edad media de las casas | 1.0 - 52.0 |
| AveRooms | Promedio de habitaciones por hogar | 1.0 - 10.0 |
| AveBedrms | Promedio de dormitorios por hogar | 0.5 - 5.0 |
| Population | Poblacion del bloque | 3.0 - 35000.0 |
| AveOccup | Promedio de ocupantes por hogar | 0.5 - 10.0 |
| Latitude | Latitud del bloque | 32.5 - 42.0 |
| Longitude | Longitud del bloque | -124.0 - -114.0 |

**Target:** MedHouseVal (Precio medio de vivienda en $100k)

## Modelo y Metricas

**Algoritmo:** Random Forest Regressor

**Hiperparametros:**
- n_estimators: 100
- max_depth: 10
- random_state: 42
- n_jobs: -1

**Metricas esperadas:**
- RMSE: ~0.50 (equivalente a $50k de error)
- R2 Score: ~0.80 (80% de varianza explicada)
- MAE: ~0.35 (error absoluto medio de $35k)

**Feature Importance (top 5):**
1. MedInc (Ingreso): ~50%
2. Latitude: ~15%
3. Longitude: ~12%
4. AveOccup: ~8%
5. HouseAge: ~7%

## Variables de Entorno

Archivo `.env` (generado desde `.env.example`):

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
MODEL_PATH=models/california_housing_model.pkl

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=california_housing

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Model Configuration
MODEL_VERSION=1.0.0
PREDICTION_THRESHOLD=0.5
```

## Troubleshooting

### Problema: Python 3.13 instalado

**Sintoma:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Solucion:**
```bash
# Instalar Python 3.11 con pyenv
pyenv install 3.11.7
pyenv local 3.11.7

# O usar conda
conda create -n mleng python=3.11
conda activate mleng
```

### Problema: Modelo no encontrado

**Sintoma:**
```
FileNotFoundError: Model not found at models/california_housing_model.pkl
```

**Solucion:**
```bash
# Entrenar modelo primero
jupyter lab notebooks/01_california_housing_pipeline.ipynb
# O ejecutar el script de entrenamiento manual
```

### Problema: Puerto 8000 ocupado

**Solucion:**
```bash
# Cambiar puerto
uvicorn app.main:app --port 8001

# O liberar puerto
# macOS/Linux:
lsof -ti:8000 | xargs kill -9
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Problema: Docker build falla

**Sintoma:**
```
ERROR: failed to solve: failed to compute cache key
```

**Solucion:**
```bash
# Limpiar cache de Docker
docker system prune -a

# Rebuild sin cache
docker-compose build --no-cache
```

### Problema: MLflow UI no accesible

**Sintoma:**
```
Connection refused at http://localhost:5000
```

**Solucion:**
```bash
# Verificar que MLflow este corriendo
ps aux | grep mlflow

# Reiniciar con logs
mlflow ui --host 0.0.0.0 --port 5000 2>&1 | tee mlflow.log
```

## Mejores Practicas Implementadas

**Codigo:**
- Type hints en todas las funciones
- Docstrings completos
- Validacion de datos con Pydantic
- Manejo robusto de errores
- Logging estructurado

**MLOps:**
- Tracking de experimentos con MLflow
- Versionado de modelos
- Reproducibilidad con seeds
- Separacion de train/test
- Metricas multiples (RMSE, R2, MAE)

**API:**
- Documentacion automatica (OpenAPI)
- Validacion de schemas
- Health checks
- Manejo de errores HTTP
- CORS habilitado

**Deployment:**
- Containerizacion con Docker
- Multi-stage builds
- Volume mounts para persistencia
- docker-compose para orquestacion
- Health checks en contenedores

**Testing:**
- Pruebas unitarias
- Pruebas de integracion
- Test fixtures
- Mocking de dependencias

## Recursos Adicionales

**Documentacion:**
- FastAPI: https://fastapi.tiangolo.com/
- MLflow: https://mlflow.org/docs/latest/
- scikit-learn: https://scikit-learn.org/stable/
- Docker: https://docs.docker.com/

**Dataset:**
- California Housing: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- Paper original: Pace, R. Kelley and Ronald Barry (1997)

**Tutoriales:**
- MLOps Zoomcamp: https://github.com/DataTalksClub/mlops-zoomcamp
- Full Stack Deep Learning: https://fullstackdeeplearning.com/

## Licencia

Este proyecto es material educativo de uso libre para fines academicos.

## Contacto y Soporte

Para preguntas sobre el proyecto:
1. Revisar la seccion de Troubleshooting
2. Consultar la documentacion de las herramientas
3. Abrir un issue en el repositorio (si aplica)

## Changelog

**v1.0.0 (2025-01-07)**
- Release inicial
- Pipeline completo de ML
- API REST con FastAPI
- Integracion con MLflow
- Containerizacion con Docker
- Suite de pruebas completa
- Documentacion completa

