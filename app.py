from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
import os
import inference
import reports
from fastapi.responses import Response

app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="API for detecting Diabetic Retinopathy from retinal images.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional, Any

class ChatRequest(BaseModel):
    message: str
    prediction_result: Optional[Any] = None
    user_role: Optional[str] = None

class ReportRequest(BaseModel):
    prediction_result: dict
    user_message: str = ""

@app.get("/")
def read_root():
    return {"message": "Welcome to DR Detection API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        contents = await file.read()
        result = inference.predict_image(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-report-pdf")
async def analyze_report_pdf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No PDF file uploaded")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    try:
        contents = await file.read()
        text = inference.extract_text_from_pdf(contents)
        analysis = inference.get_pdf_analysis(text)
        return {
            "analysis": analysis,
            "filename": file.filename,
            "char_count": len(text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(request: ChatRequest):
    response = inference.get_chat_response(request.message, request.prediction_result)
    return {"response": response}

@app.get("/hospitals")
def get_hospitals(city: str = Query(None), lat: float = Query(None), lng: float = Query(None)):
    if not city and (lat is None or lng is None):
        raise HTTPException(status_code=400, detail="City name or coordinates (lat, lng) are required")
    return inference.search_eye_hospitals(city=city, lat=lat, lng=lng)

@app.post("/download-report")
async def download_report(request: ReportRequest):
    try:
        pdf_bytes = reports.generate_pdf_report(request.prediction_result, request.user_message)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=DR_Screening_Report.pdf"}
        )
    except Exception as e:
        print(f"PDF Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
