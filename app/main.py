from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import pydicom
import tempfile
import shutil
import os

from app.vit_attention import ViTAttention

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Server started running"}
# Load model
model = tf.keras.models.load_model("app/model.keras", custom_objects={'ViTAttention': ViTAttention})

# Class names (ensure order matches model training)
class_names = ['Adenocarcinoma', 'Small Cell Carcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma']

# Convert DICOM to PNG
def convert_dcm_to_png(file_path):
    try:
        dicom_image = pydicom.dcmread(file_path, force=True)
        img = Image.fromarray(dicom_image.pixel_array)

        # Convert to 8-bit grayscale or RGB
        if img.mode in ['I', 'I;16']:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        png_path = file_path.replace('.dcm', '.png')
        img.save(png_path)
        return png_path
    except Exception as e:
        print(f"Error converting DICOM: {e}")
        return None

# Preprocess the image
def load_and_prep_image(file_path, img_height=512, img_width=512):
    img = Image.open(file_path).convert("RGB").resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    # Handle DICOM conversion if necessary
    if temp_path.lower().endswith('.dcm'):
        converted_path = convert_dcm_to_png(temp_path)
        if not converted_path:
            return JSONResponse(content={"error": "Failed to convert DICOM"}, status_code=500)
    else:
        converted_path = temp_path

    # Preprocess and predict
    try:
        img_tensor = load_and_prep_image(converted_path)
        prediction = model.predict(img_tensor)
        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_index])
        predicted_label = class_names[predicted_index]

        return JSONResponse(content={
            "result": predicted_label,
            "confidence": confidence
        })
    finally:
        # Clean up temp files
        os.remove(temp_path)
        if converted_path != temp_path and os.path.exists(converted_path):
            os.remove(converted_path)
