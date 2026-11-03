from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
from fastapi.responses import FileResponse
from pathlib import Path
from backend.clip_service import image_paths

from backend.clip_service import search_similar_images

app = FastAPI(title="Ring Image Similarity Search API")


@app.get("/")
def home():
    return {
        "message": "Ring Image Similarity Search API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "CLIP and FAISS are ready"
    }


@app.post("/search")
async def search(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()

        query_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        results = search_similar_images(
            query_image,
            top_k=5
        )

        # return {
        #     "total_results": len(results),
        #     "results": results
        # }
        for result in results:
            result["image_url"] = (
                f"/images/{result['image_id']}"
            )

        return {
            "total_results": len(results),
            "results": results
        }


#
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/images/{image_id}")
def get_image(image_id: int):

    if image_id < 0 or image_id >= len(image_paths):
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    image_path = Path(image_paths[image_id])

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image file does not exist"
        )

    return FileResponse(image_path)