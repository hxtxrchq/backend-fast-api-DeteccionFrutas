from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from utils.predict import predict_image
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O reemplaza "*" con ["https://tudominio.com"] si prefieres restringirlo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Guardar archivo temporalmente
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Realizar la predicción
    result = predict_image(file_path)

    # Eliminar archivo temporal
    os.remove(file_path)

    return JSONResponse(content=result)
