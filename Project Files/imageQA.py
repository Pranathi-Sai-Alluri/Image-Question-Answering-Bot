from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

from transformers import BlipProcessor
from transformers import BlipForQuestionAnswering

from PIL import Image

import torch
import io

# Create FastAPI app
app = FastAPI()

# Load BLIP model
model_name = "Salesforce/blip-vqa-base"

processor = BlipProcessor.from_pretrained(model_name)

model = BlipForQuestionAnswering.from_pretrained(model_name)

# Open frontend
@app.get("/")
def home():

    return FileResponse("index.html")

# AI API
@app.post("/ask")
async def ask_question(

    question: str = Form(...),

    image: UploadFile = File(...)

):

    # Read image
    image_bytes = await image.read()

    # Convert image
    raw_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Prepare inputs
    inputs = processor(
        raw_image,
        question,
        return_tensors="pt"
    )

    # Predict answer
    with torch.no_grad():

        output = model.generate(**inputs)

    # Decode answer
    answer = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return {
        "question":question,
        "answer": answer
    }
