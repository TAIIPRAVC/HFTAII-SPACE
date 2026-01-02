"""
Fashion Vectorizer API
Hugging Face Spaces deployment for SentenceTransformers embeddings.
"""
import os
import time
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configuration
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
TARGET_DIM = 512
DEFAULT_BATCH_SIZE = 128
# Global model instance
model: Optional[SentenceTransformer] = None

# API Key Authentication
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Verify the API key from request header."""
    if not API_SECRET_KEY:
        raise HTTPException(status_code=500, detail="API_SECRET_KEY not configured")
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
# ============= Pydantic Models =============
class VectorizeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=256)
class VectorizeResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str
    processing_time_ms: int
class ProductInput(BaseModel):
    product_id: str
    title: str
    short_description: Optional[str] = None
    description: Optional[str] = None
    gender: str = "Unissex"
    category: Optional[Dict[str, str]] = None
    tags: Optional[Dict[str, List[str]]] = None
    scores: Optional[Dict[str, float]] = None
    palette: Optional[List[str]] = None
class ProductVectorizeRequest(BaseModel):
    products: List[ProductInput] = Field(..., min_length=1, max_length=500)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=256)
class ProductEmbedding(BaseModel):
    product_id: str
    description_embedding: List[float]
    style_dna_embedding: List[float]
class ProductVectorizeResponse(BaseModel):
    results: List[ProductEmbedding]
    dimension: int
    processing_time_ms: int
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    dimension: int
# ============= Helper Functions =============
def load_model() -> SentenceTransformer:
    """Load and return the embedding model."""
    global model
    if model is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        
        # Validate dimension
        test_emb = model.encode(["test"])
        actual_dim = len(test_emb[0])
        logger.info(f"Model loaded. Dimension: {actual_dim}")
    
    return model
def project_to_target_dim(embeddings: List[List[float]]) -> List[List[float]]:
    """Project embeddings to target dimension (truncate or pad)."""
    result = []
    for emb in embeddings:
        if len(emb) == TARGET_DIM:
            result.append(emb)
        elif len(emb) > TARGET_DIM:
            result.append(emb[:TARGET_DIM])
        else:
            result.append(emb + [0.0] * (TARGET_DIM - len(emb)))
    return result
def create_description_text(product: ProductInput) -> str:
    """Create text for description embedding."""
    parts = [product.title]
    if product.short_description:
        parts.append(product.short_description)
    if product.description:
        parts.append(product.description[:200])
    return " ".join(parts)
def create_style_dna_text(product: ProductInput) -> str:
    """Create text for style DNA embedding."""
    parts = [f"Gênero: {product.gender}"]
    
    if product.category:
        main = product.category.get("main", "")
        sub = product.category.get("sub", "")
        parts.append(f"Categoria: {main} {sub}")
    
    if product.tags:
        if product.tags.get("style"):
            parts.append(f"Estilo: {' '.join(product.tags['style'])}")
        if product.tags.get("material"):
            parts.append(f"Material: {' '.join(product.tags['material'])}")
        if product.tags.get("occasion"):
            parts.append(f"Ocasião: {' '.join(product.tags['occasion'])}")
        if product.tags.get("attribute"):
            parts.append(f"Atributos: {' '.join(product.tags['attribute'])}")
    
    if product.scores:
        high_scores = [
            f"{k}:{v:.0%}" 
            for k, v in product.scores.items() 
            if v >= 0.5
        ]
        if high_scores:
            parts.append(f"Personalidade: {' '.join(high_scores)}")
    
    if product.palette:
        parts.append(f"Cores: {' '.join(product.palette[:5])}")
    
    return " ".join(parts)
# ============= FastAPI App =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield
app = FastAPI(
    title="Fashion Vectorizer API",
    description="SentenceTransformers embedding service for fashion products",
    version="1.0.0",
    lifespan=lifespan
)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global model
    return HealthResponse(
        status="healthy" if model else "loading",
        model_loaded=model is not None,
        model_name=MODEL_NAME,
        dimension=TARGET_DIM
    )
@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_texts(request: VectorizeRequest, _: str = Depends(verify_api_key)):
    """Encode texts into embeddings."""
    start_time = time.time()
    
    try:
        m = load_model()
        
        embeddings = m.encode(
            request.texts,
            batch_size=request.batch_size,
            show_progress_bar=False
        ).tolist()
        
        embeddings = project_to_target_dim(embeddings)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return VectorizeResponse(
            embeddings=embeddings,
            dimension=TARGET_DIM,
            model=MODEL_NAME,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Vectorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/vectorize/products", response_model=ProductVectorizeResponse)
async def vectorize_products(request: ProductVectorizeRequest, _: str = Depends(verify_api_key)):
    """Generate description + style_dna embeddings for products."""
    start_time = time.time()
    
    try:
        m = load_model()
        
        # Prepare texts
        description_texts = [create_description_text(p) for p in request.products]
        style_texts = [create_style_dna_text(p) for p in request.products]
        
        # Encode
        desc_embeddings = m.encode(
            description_texts,
            batch_size=request.batch_size,
            show_progress_bar=False
        ).tolist()
        
        style_embeddings = m.encode(
            style_texts,
            batch_size=request.batch_size,
            show_progress_bar=False
        ).tolist()
        
        # Project dimensions
        desc_embeddings = project_to_target_dim(desc_embeddings)
        style_embeddings = project_to_target_dim(style_embeddings)
        
        # Build results
        results = [
            ProductEmbedding(
                product_id=p.product_id,
                description_embedding=desc_embeddings[i],
                style_dna_embedding=style_embeddings[i]
            )
            for i, p in enumerate(request.products)
        ]
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ProductVectorizeResponse(
            results=results,
            dimension=TARGET_DIM,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Product vectorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)