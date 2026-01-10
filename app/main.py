"""
California Housing Price Prediction API

API REST para prediccion de precios de viviendas en California utilizando
Random Forest. Incluye endpoints para predicciones individuales, batch,
informacion del modelo y health checks.
"""

import os
import logging
import joblib
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd

# Configuracion de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log') if os.path.exists('logs') else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuracion de la aplicacion
app = FastAPI(
    title="California Housing Price Prediction API",
    description="API para prediccion de precios de viviendas en California",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuracion de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
MODEL = None
MODEL_INFO = {
    "loaded": False,
    "path": None,
    "load_time": None,
    "features": [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ]
}


# Modelos Pydantic para validacion
class HouseFeatures(BaseModel):
    """Schema para features de una casa individual"""
    
    MedInc: float = Field(..., description="Ingreso medio del bloque (en $10k)", ge=0.0, le=15.0)
    HouseAge: float = Field(..., description="Edad media de las casas", ge=1.0, le=52.0)
    AveRooms: float = Field(..., description="Promedio de habitaciones por hogar", ge=1.0, le=20.0)
    AveBedrms: float = Field(..., description="Promedio de dormitorios por hogar", ge=0.5, le=10.0)
    Population: float = Field(..., description="Poblacion del bloque", ge=3.0, le=40000.0)
    AveOccup: float = Field(..., description="Promedio de ocupantes por hogar", ge=0.5, le=15.0)
    Latitude: float = Field(..., description="Latitud del bloque", ge=32.0, le=42.0)
    Longitude: float = Field(..., description="Longitud del bloque", ge=-125.0, le=-114.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.98,
                "AveBedrms": 1.02,
                "Population": 322.0,
                "AveOccup": 2.55,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema para predicciones batch"""
    houses: List[HouseFeatures] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Schema para respuesta de prediccion individual"""
    predicted_price: float = Field(..., description="Precio predicho en $100k")
    price_category: str = Field(..., description="Categoria de precio")
    confidence: str = Field(..., description="Nivel de confianza")
    timestamp: str = Field(..., description="Timestamp de la prediccion")


class BatchPredictionResponse(BaseModel):
    """Schema para respuesta de prediccion batch"""
    predictions: List[PredictionResponse]
    total_predictions: int
    average_price: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Schema para informacion del modelo"""
    model_loaded: bool
    model_path: Optional[str]
    load_time: Optional[str]
    features: List[str]
    version: str


class HealthResponse(BaseModel):
    """Schema para health check"""
    status: str
    timestamp: str
    model_loaded: bool
    uptime: Optional[float]


# Funciones de utilidad
def load_model(model_path: str = "models/california_housing_model.pkl") -> bool:
    """
    Carga el modelo entrenado desde disco
    
    Args:
        model_path: Ruta al archivo del modelo
        
    Returns:
        True si el modelo se cargo exitosamente, False en caso contrario
    """
    global MODEL, MODEL_INFO
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"Modelo no encontrado en: {model_path}")
            return False
        
        MODEL = joblib.load(model_path)
        MODEL_INFO["loaded"] = True
        MODEL_INFO["path"] = model_path
        MODEL_INFO["load_time"] = datetime.now().isoformat()
        
        logger.info(f"Modelo cargado exitosamente desde: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error al cargar modelo: {str(e)}")
        MODEL_INFO["loaded"] = False
        return False


def classify_price(price: float) -> str:
    """
    Clasifica el precio en categorias
    
    Args:
        price: Precio en $100k
        
    Returns:
        Categoria del precio
    """
    if price < 1.5:
        return "low"
    elif price < 3.0:
        return "medium"
    else:
        return "high"


def get_confidence_level(price: float) -> str:
    """
    Determina el nivel de confianza de la prediccion
    
    Args:
        price: Precio predicho
        
    Returns:
        Nivel de confianza
    """
    if 0.5 <= price <= 5.0:
        return "high"
    elif 0.3 <= price < 0.5 or 5.0 < price <= 6.0:
        return "medium"
    else:
        return "low"


# Evento de startup
@app.on_event("startup")
async def startup_event():
    """Inicializacion de la aplicacion"""
    logger.info("Iniciando California Housing API...")
    
    # Crear directorio de logs si no existe
    os.makedirs("logs", exist_ok=True)
    
    # Intentar cargar el modelo
    model_path = os.getenv("MODEL_PATH", "models/california_housing_model.pkl")
    if load_model(model_path):
        logger.info("API lista para recibir solicitudes")
    else:
        logger.warning("API iniciada sin modelo cargado. Entrena un modelo primero.")


# Endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    Endpoint raiz - Informacion general de la API
    """
    return {
        "name": "California Housing Price Prediction API",
        "version": "1.0.0",
        "description": "API para prediccion de precios de viviendas en California",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "features": "/model/features"
        },
        "model_loaded": MODEL_INFO["loaded"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check del servicio
    """
    return HealthResponse(
        status="healthy" if MODEL_INFO["loaded"] else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=MODEL_INFO["loaded"],
        uptime=None
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(house: HouseFeatures):
    """
    Prediccion de precio para una casa individual
    
    Args:
        house: Features de la casa
        
    Returns:
        Prediccion con precio, categoria y confianza
        
    Raises:
        HTTPException: Si el modelo no esta cargado o hay error en la prediccion
    """
    if not MODEL_INFO["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Por favor entrena un modelo primero."
        )
    
    try:
        # Preparar datos para prediccion
        features_dict = house.model_dump()
        features_df = pd.DataFrame([features_dict])
        
        # Asegurar orden correcto de features
        features_df = features_df[MODEL_INFO["features"]]
        
        # Realizar prediccion
        prediction = MODEL.predict(features_df)[0]
        
        # Log de la prediccion
        logger.info(f"Prediccion realizada: {prediction:.4f}")
        
        return PredictionResponse(
            predicted_price=round(float(prediction), 4),
            price_category=classify_price(prediction),
            confidence=get_confidence_level(prediction),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en prediccion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar prediccion: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch(request: BatchPredictionRequest):
    """
    Prediccion de precios para multiples casas
    
    Args:
        request: Lista de casas para prediccion
        
    Returns:
        Lista de predicciones con estadisticas agregadas
        
    Raises:
        HTTPException: Si el modelo no esta cargado o hay error en la prediccion
    """
    if not MODEL_INFO["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Por favor entrena un modelo primero."
        )
    
    try:
        # Preparar datos
        houses_data = [house.model_dump() for house in request.houses]
        features_df = pd.DataFrame(houses_data)
        features_df = features_df[MODEL_INFO["features"]]
        
        # Realizar predicciones
        predictions = MODEL.predict(features_df)
        
        # Crear respuestas individuales
        prediction_responses = []
        for pred in predictions:
            prediction_responses.append(
                PredictionResponse(
                    predicted_price=round(float(pred), 4),
                    price_category=classify_price(pred),
                    confidence=get_confidence_level(pred),
                    timestamp=datetime.now().isoformat()
                )
            )
        
        logger.info(f"Prediccion batch realizada: {len(predictions)} casas")
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_predictions=len(predictions),
            average_price=round(float(np.mean(predictions)), 4),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en prediccion batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar prediccion batch: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Obtiene informacion sobre el modelo cargado
    """
    return ModelInfoResponse(
        model_loaded=MODEL_INFO["loaded"],
        model_path=MODEL_INFO["path"],
        load_time=MODEL_INFO["load_time"],
        features=MODEL_INFO["features"],
        version="1.0.0"
    )


@app.get("/model/features", response_model=Dict[str, Any])
async def get_features():
    """
    Obtiene la lista de features requeridas con sus descripciones
    """
    return {
        "features": [
            {
                "name": "MedInc",
                "description": "Ingreso medio del bloque (en $10k)",
                "range": [0.0, 15.0]
            },
            {
                "name": "HouseAge",
                "description": "Edad media de las casas",
                "range": [1.0, 52.0]
            },
            {
                "name": "AveRooms",
                "description": "Promedio de habitaciones por hogar",
                "range": [1.0, 20.0]
            },
            {
                "name": "AveBedrms",
                "description": "Promedio de dormitorios por hogar",
                "range": [0.5, 10.0]
            },
            {
                "name": "Population",
                "description": "Poblacion del bloque",
                "range": [3.0, 40000.0]
            },
            {
                "name": "AveOccup",
                "description": "Promedio de ocupantes por hogar",
                "range": [0.5, 15.0]
            },
            {
                "name": "Latitude",
                "description": "Latitud del bloque",
                "range": [32.0, 42.0]
            },
            {
                "name": "Longitude",
                "description": "Longitud del bloque",
                "range": [-125.0, -114.0]
            }
        ],
        "total_features": 8
    }


# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones"""
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true"
    )

