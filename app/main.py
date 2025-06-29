from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

# Ruta al modelo
model_path = os.path.join(os.path.dirname(__file__), '../model', 'final_best_model.keras')

# Cargar el modelo
model = load_model(model_path)

# Definir las clases de frutas
classes = ['freshapples', 'freshbanana', 'rottenbanana', 'rottenapples', 'freshoranges', 'rottenoranges']

# Crear la app FastAPI
app = FastAPI()

# Agregar middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las direcciones (ajustar si es necesario)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los encabezados
)

# Función para preprocesar las imágenes
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((150, 150))  # Cambiar tamaño a 150x150
    img_array = np.array(img) / 255.0  # Normalizar imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión del batch
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen subida
    img_bytes = await file.read()
    
    # Preprocesar la imagen
    img_array = preprocess_image(img_bytes)
    
    # Hacer la predicción
    preds = model.predict(img_array)[0]
    max_pred = preds.max()
    
    # Si la confianza es baja, decir que la fruta es desconocida
    if max_pred < 0.5:
        return {"message": "Fruta desconocida", "confidence": max_pred}
    else:
        predicted_class = classes[np.argmax(preds)]
        return {"predicted_class": predicted_class, "confidence": max_pred}
