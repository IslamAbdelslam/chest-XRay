---
title: Chest-xRay
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Pneumonia Detection Web App

This is a web application built with FastAPI (backend) and Vanilla HTML/CSS/JS (frontend) to test the trained FastAI model.

## Setup
It is recommended to run this inside your existing virtual environment.

1. Navigate to this directory:
```bash
cd webapp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage
Once the server is running, open your web browser and go to:
http://localhost:8000/

You can drag and drop chest X-ray images, click **Analyze Image**, and the model will perform inference on it and return the prediction (NORMAL vs PNEUMONIA) and confidence probabilities.

Note: The backend automatically searches for `export.pkl` in the `artifacts/` or `model/` directory. Ensure your model is exported properly.
