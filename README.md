# Photo Organizer

A local web app that automatically groups similar photos into collections using AI-powered image similarity. Upload photos via drag-and-drop, and visually similar ones are clustered together based on both shape and colour.

## How It Works

Each uploaded photo is analysed by two signals:

- **MobileNetV3 CNN embeddings** (70%) — captures shape, texture, and semantic content (e.g. "this is a flower")
- **HSV colour histogram** (30%) — captures dominant hue and saturation so differently-coloured subjects are kept separate

The two scores are blended and compared against a threshold (`0.73`). Photos that score above the threshold are placed in the same collection; otherwise a new collection is created.

## Features

- Drag-and-drop or click-to-browse photo upload (JPEG, PNG, WEBP, GIF · max 20 MB)
- Automatic grouping of similar photos into collections
- Responsive grid layout for collections and photos
- Rename collections (click the name to edit inline)
- Delete individual photos or entire collections

## Tech Stack

| Layer | Library |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Templates | Jinja2 |
| Image processing | Pillow |
| CNN embeddings | PyTorch + TorchVision (MobileNetV3) |
| Colour analysis | NumPy (HSV histogram) |
| Database | SQLite via SQLAlchemy |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Start the server
uv run uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

> The MobileNetV3 model (~10 MB) is downloaded automatically from PyTorch Hub on first run and cached in `~/.cache/torch/`.

## Project Structure

```
photo-organizer/
├── main.py          # FastAPI app and all route handlers
├── models.py        # SQLAlchemy ORM models (Photo, Collection)
├── database.py      # DB engine, session, and Base
├── similarity.py    # CNN embeddings + colour histogram + grouping logic
├── templates/       # Jinja2 HTML templates
├── static/          # CSS
├── uploads/         # Uploaded original photos (git-ignored)
├── thumbnails/      # Auto-generated 300×300 thumbnails (git-ignored)
└── photos.db        # SQLite database (git-ignored)
```

## Tuning Similarity

Edit the constants at the top of `similarity.py`:

| Constant | Default | Effect |
|---|---|---|
| `CNN_WEIGHT` | `0.70` | Higher → shape/subject matters more |
| `COLOR_WEIGHT` | `0.30` | Higher → colour matching is stricter |
| `SIMILARITY_THRESHOLD` | `0.73` | Lower → more photos grouped together |

## Reset / Cleanup

To wipe all uploaded photos and start fresh:

```bash
# Stop the server
pkill -f "uvicorn main:app"

# Delete data
rm -f photos.db uploads/* thumbnails/*

# Restart (DB is recreated automatically)
uv run uvicorn main:app --reload --port 8000
```
