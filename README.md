# 🫁 Lung Cancer Detection App

This is a web-based Lung Cancer Detection App using a hybrid CNN + ViT model with attention, built using **FastAPI** and optionally a **React/Tailwind frontend**.

## 🚀 Hosted on Hugging Face Spaces
This project is deployed using **FastAPI** on Hugging Face Spaces. Upload a lung CT scan image and get a prediction of the cancer type.

---

## 🧠 Model Details

- Hybrid CNN + Vision Transformer (ViT) with attention mechanism
- Trained on labeled lung CT scan images
- Achieves over 98% accuracy on test data
- Supports input classes: `[Adenocarcinoma', 'Small Cell Carcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma']`

---


### Request Format

- **Form-data**:
  - `file`: Image file (`.png`, `.jpg`, or `.dcm` converted to `.png`)

### Response Example

```json
{
  "result": "Cancer Detected - Squamous Cell Carcinoma",
  "confidence": 0.9821
}


