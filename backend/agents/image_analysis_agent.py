# agents/image_analysis_agent.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import base64
from PIL import Image
import io
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Configuration & API Key Check ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in the environment. The agent cannot start.")

# --- Groq Client Initialization ---
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Image Analysis Agent")

# --- Pydantic Models ---
# **CHANGE**: The agent now accepts an image_path
class AnalyzeImageRequest(BaseModel):
    image_path: str = Field(..., description="The local file path to the image to be analyzed.")
    query: str = Field(..., description="The user's question about the image.")

class AnalyzeImageResponse(BaseModel):
    answer: str

# --- Helper Functions ---
def read_compress_and_encode_image(image_path: str, max_size_kb: int = 200, quality: int = 85) -> str:
    """
    Reads an image from a file path, compresses it, and returns a new,
    compressed base64 string in JPEG format.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at path: {image_path}")

        with open(image_path, "rb") as image_file:
            image = Image.open(image_file)

            # Convert to RGB if it has an alpha channel (like PNGs)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            # Compress the image
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG", quality=quality, optimize=True)

            # Iteratively reduce quality if the image is still too large
            while img_buffer.tell() / 1024 > max_size_kb and quality > 10:
                quality -= 10
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="JPEG", quality=quality, optimize=True)

            return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format or data: {e}")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Image Analysis Agent is running and ready to analyze images."}

@app.post("/analyze", response_model=AnalyzeImageResponse)
async def analyze_image(request: AnalyzeImageRequest):
    """
    Analyzes an image from a file path using Groq's LLaMA 4 vision model.
    """
    try:
        # **CHANGE**: The agent now performs the compression and encoding
        compressed_base64 = read_compress_and_encode_image(request.image_path)
        
        # Create the multimodal message payload for the Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{compressed_base64}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )

        answer = chat_completion.choices[0].message.content
        return AnalyzeImageResponse(answer=answer)
        
    except Exception as e:
        # Catch potential HTTPExceptions from the helper and other errors
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to analyze image with Groq API: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("image_analysis_agent:app", host="127.0.0.1", port=8060, reload=True)