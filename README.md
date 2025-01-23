# Housing Price Prediction API

API de predicción de precios de viviendas basada en un modelo de Random Forest.

## Requisitos

- Docker
- Docker Compose

## Inicio Rápido

1. Clona este repositorio:
```bash
git clone <url-del-repo>
cd <nombre-del-repo>
```

2. Inicia el servicio con Docker Compose:
```bash
docker-compose up
```

3. Abre tu navegador en http://localhost:5000

## Uso de la API

### Interfaz Web
Visita http://localhost:5000 para usar la interfaz web interactiva.

### Endpoint REST

`POST /predict`

Ejemplo de request:
```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "surface": 100,
    "bedrooms": 2,
    "restrooms": 1
}'
```

Ejemplo de respuesta:
```json
{
    "prediction": [150000],
    "graph": {...}  // Datos del gráfico en formato Plotly
}
```

## Características del Modelo

El modelo utiliza las siguientes características para hacer predicciones:
- Superficie (m²)
- Número de habitaciones
- Número de baños
- Total de habitaciones (calculado)
- Tamaño promedio por habitación (calculado)

## Rendimiento del Modelo

- Utiliza Random Forest Regressor
- Incluye eliminación de outliers
- Normalización de características
- Validación cruzada
- Métricas de rendimiento disponibles en la consola durante el entrenamiento

## Estructura del Proyecto

```
.
├── api/
│   ├── models/        # Modelo entrenado
│   ├── app.py         # API Flask
│   ├── Dockerfile     
│   └── requirements.txt
└── docker-compose.yml
```

## Notas Técnicas

- El modelo está pre-entrenado y listo para usar
- La API incluye CORS habilitado para desarrollo
- Las predicciones incluyen visualizaciones con Plotly
- El modelo está optimizado para el dataset proporcionado

## Contacto

Para cualquier duda o problema, contacta con:
[Tu nombre/contacto]