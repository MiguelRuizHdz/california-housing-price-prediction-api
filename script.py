import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
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
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.4f}')
print(f'R2: {r2:.4f}')

# Guardar modelo
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/california_housing_model.pkl')
print('Modelo guardado en models/california_housing_model.pkl')