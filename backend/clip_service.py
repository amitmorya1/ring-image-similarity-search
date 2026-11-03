import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import numpy as np
import faiss
import json

from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel


# --------------------------------
# 1. Define project paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"

INDEX_PATH = ARTIFACTS_DIR / "ring_faiss.index"
IMAGE_PATHS_FILE = ARTIFACTS_DIR / "image_paths.json"


# --------------------------------
# 2. Select device
# --------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# --------------------------------
# 3. Load CLIP model
# --------------------------------

model_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_name)

model = CLIPModel.from_pretrained(model_name)

model = model.to(device)

model.eval()

print("CLIP model loaded successfully!")


# --------------------------------
# 4. Load saved FAISS index
# --------------------------------

index = faiss.read_index(str(INDEX_PATH))

print("FAISS index loaded successfully!")
print("Total images:", index.ntotal)


# --------------------------------
# 5. Load image paths
# --------------------------------

# with open(IMAGE_PATHS_FILE, "r") as f:
#     image_paths = json.load(f)

with open(IMAGE_PATHS_FILE, "r") as f:
    stored_paths = json.load(f)

image_paths = [
    str(BASE_DIR / "data" / Path(p).name)
    for p in stored_paths
]

print("Image paths loaded successfully!")


# --------------------------------
# 6. Convert image into vector
# --------------------------------

def image_to_vector(image):

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        features = model.get_image_features(**inputs)

        if hasattr(features, "image_embeds"):
            features = features.image_embeds

        elif hasattr(features, "pooler_output"):
            features = features.pooler_output

        elif hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state[:, 0, :]

        # Normalize embedding
        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=-1
        )

    return features.cpu().numpy().astype("float32")


# --------------------------------
# 7. Search similar images
# --------------------------------

def search_similar_images(query_image, top_k=5):

    query_vector = image_to_vector(query_image)

    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for score, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "image_id": int(idx),
            "image_path": image_paths[idx],
            "similarity_score": float(score)
        })

    return results


print("CLIP + FAISS service initialized successfully!")