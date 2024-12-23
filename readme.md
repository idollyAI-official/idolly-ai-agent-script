# Idolly AI - **Extended Image Backend Server** (Simplified Skeleton)

This repository demonstrates more of the **Idolly AI** image backend code structure in a **simplified** and **expanded** format.  
We have **intentionally removed or reduced internal logic**, showing only function skeletons and essential architecture to illustrate how the system works.

> **Disclaimer**:  
> The code snippets below **do not** reflect the final production-level code. They’re simplified skeletons intended for demonstration during the **Solana AI Hackathon**. Certain lines, dependencies, and security measures are omitted or abbreviated.

---

## Table of Contents

1. [Overview](#overview)  
2. [Architecture](#architecture)  
3. [Directory Structure](#directory-structure)  
4. [Core Components](#core-components)
    - [config.py](#configpy)
    - [logging_config.py](#logging_configpy)
    - [main.py](#mainpy)
    - [api/\_\_init\_\_.py](#api__init__py)
    - [api/api.py](#apiapipy)
    - [api/image/txt2img.py](#apiimagetxt2imgpy)
    - [api/image/reface.py](#apiimagerefacepy)
    - [api/image/scoring.py](#apiimagescoringpy)
    - [prompt/\_\_init\_\_.py](#prompt__init__py)
    - [prompt/comfy\_prompt\_generator.py](#promptcomfy_prompt_generatorpy)
    - [prompt/comfy/workflow\_t2i.py](#promptcomfyworkflow_t2ipy)
    - [service/\_\_init\_\_.py](#service__init__py)
    - [service/server\_address\_manager.py](#serviceserver_address_managerpy)
    - [utils/\_\_init\_\_.py](#utils__init__py)
    - [utils/image\_utils.py](#utilsimage_utilspy)
    - [utils/comfy\_utils.py](#utilscomfy_utilspy)
5. [Usage](#usage)
6. [Extended Explanation and Future Plans](#extended-explanation-and-future-plans)

---

## Overview

**Idolly AI** is a platform that helps users generate AI-based digital assets—3D Avatars, AI idols, text-to-video clips—and mint them as **compressed NFTs (cNFTs)** on **Solana** with near-zero minting fees. This server is the **image backend**, receiving requests from the front-end & blockchain backend, then routing them to various GPU servers for processing.

Key features include:

- **Text-to-Image Generation** (Stable Diffusion + ComfyUI, FLUX “schnell” model)  
- **Face Swap (ReFace)**  
- **Face Cropping**  
- **Recursive Enhancement Pipeline** for advanced post-processing  
- **Prompt Scoring & Rarity** (OpenAI embeddings + MongoDB VectorSearch)

The code samples here provide **skeleton-level** glimpses of how our solution is orchestrated.

---

## Architecture

```
(1) Front-End (Web or Mobile)
   |
   v
(2) Solana & Wallet Backend
   |
   v
(3) Image Backend (FastAPI)
   |
   v
(4) GPU Servers (ComfyUI, FLUX, Stable Diffusion, etc.)
```

**Key Points**:
- This repository corresponds to (3) **Idolly AI - Image Backend**.  
- We rely on GPU servers to run heavy AI tasks (text-to-image, face-swapping, face inpainting, etc.).  
- Communication is primarily done via **HTTP** or **WebSockets** (FastAPI + aiohttp).

---

## Directory Structure

A simplified overview:

```bash
.
├── app/
│   ├── api/
│   │   ├── image/
│   │   │   ├── txt2img.py
│   │   │   ├── reface.py
│   │   │   ├── scoring.py
│   │   │   └── face_crop.py
│   │   └── utils.py
│   ├── prompt/
│   │   ├── comfy/
│   │   │   └── workflow_t2i.py
│   │   └── comfy_prompt_generator.py
│   ├── scoring/
│   ├── service/
│   │   └── server_address_manager.py
│   ├── schema/
│   ├── utils/
│   ├── logging_config.py
│   └── main.py
├── config.py
├── requirements.txt
└── README.md
```

---

## Core Components

Below are **expanded** but **simplified** skeletons of the primary files in this backend.  
We preserve top-level structures, docstrings, and some placeholder logic, while removing or abbreviating certain internal processes.

---

### config.py

```python
import os
from dotenv import load_dotenv

# Simplified skeleton: we just show placeholders for environment variables.
load_dotenv()

SERVER_KEY = os.environ.get("SERVER_KEY", "placeholder-server-key")
SEED_KEY = os.environ.get("SEED_KEY", "placeholder-seed-key")
AWS_X_API_KEY = os.environ.get("AWS_X_API_KEY", "placeholder-aws-key")
AWS_LAMBDA_ADDRESS = os.environ.get("AWS_LAMBDA_ADDRESS", "placeholder-lambda-address")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
PROMPT_COLLECTION = os.environ.get("PROMPT_COLLECTION", "prompt_collection")


# Additional placeholders or configurations...
```

**Notes**:  
- Loads environment variables (Fernet keys, MongoDB URIs, etc.).  
- Simplified to only show a few key lines.

---

### logging_config.py

```python
import os
import logging
from logging.handlers import RotatingFileHandler
from pytz import timezone
from datetime import datetime

def setup_logging():
    """Set up rotating file handlers and console logs."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/outputs", exist_ok=True)

    logging.getLogger().handlers = []
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = KSTFormatter("[%(asctime)s] %(levelname)s %(message)s")

    file_handler = RotatingFileHandler("logs/log.txt", maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
```

**Notes**:  
- Creates rotating logs in a `logs/` folder.  
- Injects a custom formatter with KST (Korea Standard Time).

---

### main.py

```python
import asyncio
from fastapi import FastAPI
from app.api import api_router
from app.logging_config import setup_logging
from app.service.server_address_manager import server_address_manager
from app.utils import http_exception_handler, add_exception_handlers

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()
    application = FastAPI(title="Idolly AI - Image Backend")

    @application.on_event("startup")
    async def on_startup():
        await server_address_manager.fetch_server_addresses()
        _ = asyncio.create_task(server_address_manager.update_server_addresses())

    application.include_router(api_router)
    http_exception_handler(application)
    add_exception_handlers(application)

    return application

app = create_app()
```

**Notes**:  
- Defines `app` (FastAPI) with custom startup events, logging, and router inclusion.  
- Skeleton for server initialization.

---

### api/__init__.py

```python
from fastapi import APIRouter
from .api import api_router

__all__ = ["api_router"]
```

**Notes**:  
- Simple aggregator for the `api` subpackage.  
- Exports `api_router` to be consumed in `main.py`.

---

### api/api.py

```python
from fastapi import APIRouter
# Import all sub-routers
from app.api.image import txt2img, reface, scoring, face_crop
from app.api import utils

api_router = APIRouter()

# Add child routers
api_router.include_router(utils.router)
api_router.include_router(face_crop)
api_router.include_router(txt2img)
api_router.include_router(reface)
api_router.include_router(scoring)
```

**Notes**:  
- Central place where we mount image-related routers and utilities.  
- Each sub-router handles a different feature (e.g., /v1/txt2img).

---

### api/image/txt2img.py

```python
import asyncio
from fastapi import APIRouter, Body, HTTPException

from app.schema import UserRequest, UserResponse
from app.service.request_server.comfy_t2i import generate_comfy_image
from app.service.flux.workflow_flux import flux_prompt_generator, create_flux_image
from app.service.server_address_manager import server_address_manager
from app.utils import add_watermark, base64_to_jpg

router = APIRouter()

@router.post("/v1/txt2img", response_model=UserResponse)
async def text2image(user_request: UserRequest = Body(...)):
    """
    Simplified route for text-to-image generation via COMFY or FLUX.
    """
    try:
        # Example: check style, call flux or comfy
        if user_request.style.upper() in ["FLUX_SCHENLL", "AVATAR"]:
            # Build FLUX workflow
            workflow = await flux_prompt_generator(user_request)
            # Generate image
            flux_image_base64 = await create_flux_image(workflow)
            # Optional watermark
            # ...
            return UserResponse(status="SUCCESS", data={...})
        else:
            # Use COMFY server
            server_addr = server_address_manager.get_server_address(user_request.server.upper())
            if not server_addr:
                raise HTTPException(status_code=500, detail="No ComfyUI server available.")
            return await generate_comfy_image(user_request, server_addr)
    except Exception as e:
        raise HTTPException(status_code=500, detail="INTERNAL_SERVER_ERROR")
```

**Notes**:  
- Skeleton that decides whether to run FLUX or COMFY pipeline based on user input (`style`).  
- Uses `UserRequest` and returns a `UserResponse`.

---

### api/image/reface.py

```python
from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse
from app.schema import ReFaceRequest
from app.service.reface import request_reface
from app.service.server_address_manager import server_address_manager

router = APIRouter()

@router.post("/v1/reface")
async def reface_image(request: ReFaceRequest):
    """
    Simplified route to handle face-swapping (ReFace).
    """
    try:
        comfy_address = server_address_manager.get_server_address("COMFY")
        if not comfy_address:
            raise HTTPException(status_code=500, detail="NO_AVAILABLE_COMFY_SERVER")

        json_content, image_content = await request_reface(request, comfy_address)

        def stream_files():
            yield b"--boundary\r\n"
            yield b'Content-Disposition: form-data; name="json"\r\n\r\n'
            yield json_content
            # ... more boundary code ...
            yield image_content
            yield b"\r\n--boundary--\r\n"

        return StreamingResponse(stream_files(), media_type="multipart/form-data; boundary=boundary")

    except Exception:
        raise HTTPException(status_code=500, detail="INTERNAL_SERVER_ERROR")
```

**Notes**:  
- Streams a **multipart** response with the JSON info and swapped face image.  
- The actual logic is delegated to `request_reface`.

---

### api/image/scoring.py

```python
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.config import prompt_collection
from app.scoring import compute_embedding
from scipy.stats import norm

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/v1/prompt_scoring")
async def compute_prompt_similarity(request: PromptRequest):
    """
    Simplified prompt scoring (embeddings + vector search).
    """
    try:
        user_embedding = compute_embedding(request.prompt)
        similar_prompts = prompt_collection.aggregate([...])  # Simplified
        # ... scoring logic ...
        return {"status": "SUCCESS", "score": 0.72}
    except Exception:
        raise HTTPException(status_code=500, detail="ERROR")
```

**Notes**:  
- High-level flow for computing embedding-based similarity.  
- The real code does vector search, z-score normalization, etc.

---

### prompt/__init__.py

```python
# Typically empty or re-exports
from .comfy_prompt_generator import comfy_prompt_generator
```

**Notes**:  
- Re-exports or aggregates prompt-related modules.

---

### prompt/comfy_prompt_generator.py

```python
import spacy
from app.utils import bad_words
from app.schema import UserRequest, UserFluxRequest

nlp = spacy.load("en_core_web_sm")

async def comfy_prompt_generator(user_request):
    """
    Generate final and negative prompts for the ComfyUI pipeline.
    Simplified version for demonstration.
    """
    style = user_request.style.upper()
    doc = nlp(user_request.prompt)
    filtered_tokens = []

    if user_request.isAdult:
        # minimal filtering
        filtered_tokens = [t.text for t in doc]
    else:
        # remove bad words, punctuation
        filtered_tokens = [t.text.lower() for t in doc if t.text.lower() not in bad_words]

    # Combine into final prompts
    # ...
    return "final_prompt_example", "negative_prompt_example"
```

**Notes**:  
- Simplified prompt generator with placeholders.  
- Real code merges tags, templates, LORA references, etc.

---

### prompt/comfy/workflow_t2i.py

```python
import copy
from random import randint
from cryptography.fernet import Fernet

from app.config import SEED_KEY
from app.prompt import comfy_prompt_generator
from app.schema import UserRequest

BASE_WORKFLOW = {
    # A large dictionary of node definitions and links
}

async def generate_t2i_workflow(user_request: UserRequest) -> dict:
    """
    Builds the T2I (text-to-image) workflow for ComfyUI.
    """
    workflow = copy.deepcopy(BASE_WORKFLOW)
    final_prompt, negative_prompt = await comfy_prompt_generator(user_request)
    
    # Insert prompts into the workflow
    # ...
    # Decrypt or generate seed
    # ...
    # Configure aspect ratio, batch size, file saving
    return workflow
```

**Notes**:  
- Outlines how we combine the user’s prompt + negative prompt + seed + model references.  
- The actual `BASE_WORKFLOW` is a large dictionary with ComfyUI node definitions.

---

### service/__init__.py

```python
from .request_server.comfy_t2i import generate_comfy_image
# Other imports for reface, flux...
```

**Notes**:  
- Re-exports convenience functions from the submodules.

---

### service/server_address_manager.py

```python
import asyncio
import aiohttp
import ssl
from cryptography.fernet import Fernet
from app.config import SERVER_KEY, AWS_LAMBDA_ADDRESS, AWS_X_API_KEY

class ServerAddressManager:
    """
    Periodically fetches GPU server addresses from AWS Lambda.
    Simplified logic for demonstration.
    """
    def __init__(self):
        self.fernet = Fernet(SERVER_KEY)
        self.server_address = {}

    async def fetch_server_addresses(self):
        async with aiohttp.ClientSession() as session:
            # GET from AWS_LAMBDA_ADDRESS
            # Decrypt addresses
            # Store in self.server_address

    async def update_server_addresses(self):
        while True:
            await self.fetch_server_addresses()
            await asyncio.sleep(120)  # 2 minutes

    def get_server_address(self, style: str):
        # Round-robin logic
        # ...
        pass

server_address_manager = ServerAddressManager()
```

**Notes**:  
- Manages the pool of GPU server endpoints (ComfyUI, FLUX, etc.).  
- Simplified to highlight the approach without showing full error handling.

---

### utils/__init__.py

```python
# Re-exports from submodules
from .error_handler import http_exception_handler, add_exception_handlers
from .image_utils import (
    save_image, crop_face, add_watermark, base64_to_jpg, image_bytes_to_base64
)
# Other utility placeholders
```

**Notes**:  
- Consolidates utility imports to one place.

---

### utils/image_utils.py

```python
import base64
from PIL import Image
from io import BytesIO
import dlib
import numpy as np
import os
from datetime import datetime

def save_image(image_data: bytes, suffix: str) -> str:
    """Saves image bytes to logs/outputs with a timestamp-based filename."""
    dir_path = "logs/outputs/"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}.jpg"
    file_path = os.path.join(dir_path, filename)
    # ...
    return file_path

def crop_face(image_contents: bytes) -> Image:
    """Detect and crop the first face using dlib."""
    # Minimal placeholders
    pass

def add_watermark(image_data: bytes):
    """Apply watermark to the image."""
    pass

def base64_to_jpg(image_base64: str):
    """Convert base64 string to JPG-encoded base64."""
    pass

def image_bytes_to_base64(image_bytes: bytes):
    """Convert bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')
```

**Notes**:  
- Contains skeletal face detection, watermarking, and base64 functions.  
- Real code uses face detection from dlib, merges logos via PIL, etc.

---

### utils/comfy_utils.py

```python
import asyncio
import ssl
import aiohttp
from fastapi import HTTPException

async def process_comfy_request(comfy_address: str, workflow: dict):
    """
    Send workflow to the ComfyUI server and retrieve generated images.
    """
    # 1) POST the workflow
    # 2) Check status
    # 3) Download images
    # ...
    pass
```

**Notes**:  
- Orchestrates the entire request/response cycle with ComfyUI.  
- In reality, checks for errors, partial results, etc.

---



## Extended Explanation and Future Plans

1. **3D Avatars & text-to-video**  
   - Next steps include integrating local or cloud-based text-to-video (e.g., Hunyuan) and 3D avatar generation.  
   - Planned pipeline to incorporate local stable diffusion and LoRA-based training.

2. **Web3 Social Integration**  
   - The platform includes a social feed where cNFT owners can share and trade minted images.  
   - Future expansion to integrate influencer “AI Idol” livestreaming on X (Twitter).

3. **Rarity Evaluation**  
   - Currently uses a basic z-score approach with vector search.  
   - Will move toward a more advanced “on-chain verified” rarity logic tied to cNFT metadata.

4. **Scalability & Performance**  
   - This skeleton omits advanced caching strategies and concurrency management.  
   - Production environment would involve container orchestration (Kubernetes, ECS, etc.), multi-region load balancing, and deeper GPU resource management.

5. **Token Plans**  
   - Potentially introduce a $DOLLIZ utility token for in-app payments, tipping, and staking.  
   - However, the hackathon version focuses primarily on cNFT functionalities.

6. **Team & Contact**  
   - For more details, contact [dev@sendai.fun](mailto:dev@sendai.fun) or check the official Idolly AI project deck.

---

**Thank you for reviewing our extended skeleton code!**  

We hope this helps demonstrate how **Idolly AI** orchestrates text-to-image workflows, manages GPU endpoints, and integrates Solana cNFTs.  
For any additional inquiries, feel free to reach out to the **Idolly AI** team.  

**- Team Idolly AI**  
