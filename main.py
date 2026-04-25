import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Collection, Photo
from similarity import compute_color_histogram, compute_embedding, find_or_create_collection

UPLOAD_DIR = Path("uploads")
THUMB_DIR = Path("thumbnails")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
THUMB_SIZE = (300, 300)

# Create storage dirs at import time so StaticFiles mounts don't raise
UPLOAD_DIR.mkdir(exist_ok=True)
THUMB_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Photo Organizer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")

templates = Jinja2Templates(directory="templates")
templates.env.filters["replace_ext"] = lambda s, ext: Path(s).stem + ext


def _render_index(request: Request, db: Session, error: str = None, status_code: int = 200):
    collections = db.query(Collection).order_by(Collection.created_at.desc()).all()
    return templates.TemplateResponse(
        request,
        "index.html",
        context={"collections": collections, "error": error},
        status_code=status_code,
    )


@app.get("/")
def index(request: Request, db: Session = Depends(get_db)):
    return _render_index(request, db)


@app.post("/upload")
async def upload_photo(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _render_index(
            request, db,
            error=f"Unsupported file type '{ext or '(none)'}'. Use JPEG, PNG, WEBP, or GIF.",
            status_code=400,
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        return _render_index(request, db, error="File too large. Maximum size is 20 MB.", status_code=400)

    stored_name = f"{uuid.uuid4()}{ext}"
    upload_path = UPLOAD_DIR / stored_name
    upload_path.write_bytes(contents)

    # Verify it's actually a valid image
    try:
        with Image.open(upload_path) as img:
            img.verify()
    except (UnidentifiedImageError, Exception):
        upload_path.unlink(missing_ok=True)
        return _render_index(request, db, error="File could not be read as an image.", status_code=400)

    # Generate thumbnail (re-open after verify() which closes the file handle)
    thumb_name = Path(stored_name).stem + ".jpg"
    thumb_path = THUMB_DIR / thumb_name
    try:
        with Image.open(upload_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)
            img.save(thumb_path, "JPEG", quality=85, optimize=True)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        return _render_index(request, db, error=f"Could not generate thumbnail: {exc}", status_code=500)

    import json
    embedding       = compute_embedding(str(upload_path))
    color_histogram = compute_color_histogram(str(upload_path))
    collection_id, _ = find_or_create_collection(db, embedding, color_histogram)

    photo = Photo(
        filename=file.filename,
        stored_name=stored_name,
        embedding=json.dumps(embedding),
        color_histogram=json.dumps(color_histogram),
        collection_id=collection_id,
    )
    db.add(photo)
    db.flush()

    collection = db.query(Collection).filter_by(id=collection_id).first()
    if collection.cover_photo_id is None:
        collection.cover_photo_id = photo.id

    db.commit()
    return RedirectResponse(url=f"/collections/{collection_id}", status_code=303)


@app.get("/collections/{collection_id}")
def collection_detail(collection_id: int, request: Request, db: Session = Depends(get_db)):
    collection = db.query(Collection).filter_by(id=collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    return templates.TemplateResponse(
        request,
        "collection.html",
        context={"collection": collection, "photos": collection.photos},
    )


class RenameBody(BaseModel):
    name: str


@app.patch("/collections/{collection_id}")
def rename_collection(collection_id: int, body: RenameBody, db: Session = Depends(get_db)):
    collection = db.query(Collection).filter_by(id=collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    collection.name = name
    db.commit()
    return JSONResponse({"ok": True, "name": collection.name})


@app.delete("/collections/{collection_id}")
def delete_collection(collection_id: int, db: Session = Depends(get_db)):
    collection = db.query(Collection).filter_by(id=collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    for photo in list(collection.photos):
        (UPLOAD_DIR / photo.stored_name).unlink(missing_ok=True)
        (THUMB_DIR / (Path(photo.stored_name).stem + ".jpg")).unlink(missing_ok=True)
        db.delete(photo)
    db.delete(collection)
    db.commit()
    return JSONResponse({"ok": True})


@app.delete("/photos/{photo_id}")
def delete_photo(photo_id: int, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter_by(id=photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    collection = photo.collection
    (UPLOAD_DIR / photo.stored_name).unlink(missing_ok=True)
    (THUMB_DIR / (Path(photo.stored_name).stem + ".jpg")).unlink(missing_ok=True)
    db.delete(photo)
    db.flush()

    # Reassign cover if needed
    remaining = db.query(Photo).filter_by(collection_id=collection.id).all()
    if not remaining:
        db.delete(collection)
        db.commit()
        return JSONResponse({"ok": True, "collection_deleted": True})

    if collection.cover_photo_id == photo_id:
        collection.cover_photo_id = remaining[0].id

    db.commit()
    return JSONResponse({"ok": True, "collection_deleted": False})


def run():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
