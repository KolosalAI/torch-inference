from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ..core.engine import InferenceEngine
from ..core.preprocessor import ImagePreprocessor
from ..core.postprocessor import ClassificationPostprocessor
from ..utils.config import load_config

app = FastAPI()
engine = None
preprocessor = None
postprocessor = None

class InferenceRequest(BaseModel):
    image: list  # Expecting HWC format list

@app.on_event("startup")
async def startup_event():
    global engine, preprocessor, postprocessor
    config = load_config()
    
    try:
        engine = InferenceEngine(config)
        preprocessor = ImagePreprocessor(config)
        postprocessor = ClassificationPostprocessor(config)
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {str(e)}")

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        # Convert list to numpy array
        image_np = np.array(request.image, dtype=np.uint8)
        
        # Preprocess
        tensor = preprocessor.process(image_np)
        
        # Inference
        output = engine.infer(tensor)
        
        # Postprocess
        result = postprocessor.process(output)
        
        return {"success": True, "result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": engine is not None
    }