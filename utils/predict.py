import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Cargar el modelo
model = load_model("model/final_best_model.keras")

# Lista de clases
classes = [
    "freshapple", "freshbanana", "freshorange",
    "rottenapple", "rottenbanana", "rottenorange"
]

def predict_image(file):
    # Cargar la imagen
    image = load_img(file, target_size=(150, 150))  # Ajusta tamaño según tu modelo
    image = img_to_array(image)
    image = image / 255.0  # Normalización
    image = np.expand_dims(image, axis=0)

    # Predicción
    predictions = model.predict(image)
    class_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return {
        "class": classes[class_index],
        "confidence": round(confidence, 2)
    }
