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
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((150, 150))  # Cambiar tamaño a 150x150
        img_array = np.array(img) / 255.0  # Normalizar imagen
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión del batch
        return img_array
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        raise HTTPException(status_code=400, detail="Error al procesar la imagen")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_array = preprocess_image(img_bytes)
        preds = model.predict(img_array)[0]
        max_pred = preds.max()

        if max_pred < 0.5:
            return {"message": "Fruta desconocida", "confidence": max_pred}
        else:
            predicted_class = classes[np.argmax(preds)]
            return {"predicted_class": predicted_class, "confidence": max_pred}
    except Exception as e:
        print(f"Error completo: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")
