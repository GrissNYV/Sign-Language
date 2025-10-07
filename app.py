# file: app.py (Cải tiến)
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
import base64
import asyncio

# Import các hàm xử lý từ services
from app.services import (
    MODEL_LOADED, 
    extract_keypoints, 
    predict_sequence,
    SEQUENCE_LENGTH
)
# Import service mới cho Gemini
from gemini_services import get_sign_explanation

app = FastAPI(title="Sign Language Recognition API")

# Phục vụ file tĩnh (HTML, CSS, JS) từ app/static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    frames: List[str] # List of base64 encoded image strings

class ExplainRequest(BaseModel):
    word: str

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Sign Language Recognition API</h1><p>Place your UI at static/index.html</p>")

@app.post("/predict")
async def predict_action(request: PredictionRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model is not available.")

    if len(request.frames) != SEQUENCE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Yêu cầu chính xác {SEQUENCE_LENGTH} frames, nhận được {len(request.frames)}.")

    keypoints_sequence = []
    for frame_b64 in request.frames:
        try:
            # Hỗ trợ cả data URI và chuỗi base64 thuần
            if ',' in frame_b64:
                frame_b64 = frame_b64.split(',', 1)[1]
            img_data = base64.b64decode(frame_b64, validate=True)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode trả về None")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            raise HTTPException(status_code=400, detail="Frame không hợp lệ, không thể giải mã base64.")

        keypoints = await asyncio.to_thread(extract_keypoints, img_rgb)
        keypoints_sequence.append(keypoints)

    label, confidence = predict_sequence(np.array(keypoints_sequence))

    return {"label": label, "confidence": confidence}

@app.post("/explain")
async def explain_sign_action(request: ExplainRequest):
    """
    New endpoint to get sign explanation from Gemini.
    """
    try:
        explanation = await get_sign_explanation(request.word)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

