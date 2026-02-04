from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import os
from pathlib import Path

from model import create_model
from preprocess import ImagePreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Asset Recognition & Classification API",
    description="CNN-based image classification for product/asset recognition",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
PREPROCESSOR = None
DEVICE = None
CLASS_NAMES = None

# Asset category mapping (customize based on your dataset)
CATEGORY_MAP = {
    0: "Electronics",
    1: "Clothing",
    2: "Books",
    3: "Furniture",
    4: "Toys",
    5: "Sports",
    6: "Home Appliances",
    7: "Accessories"
}

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    global MODEL, PREPROCESSOR, DEVICE
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL = create_model(num_classes=8, device=DEVICE)
    PREPROCESSOR = ImagePreprocessor()
    
    # Load pretrained model weights (if available)
    model_path = "model.pth"
    if os.path.exists(model_path):
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    MODEL.eval()
    print(f"Model loaded on {DEVICE}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Main prediction endpoint
    
    Args:
        file: Image file to classify
    
    Returns:
        JSON with predicted label and confidence score
    """
    try:
        # Read image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        img_tensor = PREPROCESSOR.preprocess_val(image)
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            logits = MODEL(img_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_idx = predicted_class.item()
        confidence_score = confidence.item()
        label = CATEGORY_MAP.get(predicted_idx, f"Class {predicted_idx}")
        
        return {
            "label": label,
            "confidence": round(confidence_score, 4),
            "class_idx": predicted_idx,
            "all_predictions": {
                CATEGORY_MAP.get(i, f"Class {i}"): round(probabilities[0][i].item(), 4)
                for i in range(len(CATEGORY_MAP))
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction endpoint for multiple images"""
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            img_tensor = PREPROCESSOR.preprocess_val(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = MODEL(img_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_idx = predicted_class.item()
            results.append({
                "filename": file.filename,
                "label": CATEGORY_MAP.get(predicted_idx, f"Class {predicted_idx}"),
                "confidence": round(confidence.item(), 4)
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    return {"results": results}

@app.get("/categories")
async def get_categories():
    """Get available asset categories"""
    return {"categories": CATEGORY_MAP}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
