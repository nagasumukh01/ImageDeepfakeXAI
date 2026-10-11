import os
import sys
import logging
import sqlite3
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ImageGuardAI")

# Ensure required workspace directories exist
for folder in ["static", "static/css", "static/xai", "templates", "uploads"]:
    os.makedirs(folder, exist_ok=True)

# Initialize FastAPI application
app = FastAPI(
    title="Image Guard AI - Deepfake & Authenticity Forensics",
    description="Research-Oriented Deepfake Detection and Image Authenticity Verification with Explainable AI (XAI)",
    version="2.0.0"
)

# Mount static asset routes
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# Initialize SQLite database with extended schema for XAI & forensic metrics
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            status TEXT,
            score REAL,
            fake_probability REAL,
            reason TEXT,
            lat REAL,
            lon REAL,
            timestamp TEXT,
            device TEXT,
            depth REAL,
            cluster INTEGER,
            ela_score REAL,
            fft_ratio REAL,
            noise_variance REAL,
            gradcam_url TEXT,
            dct_score REAL,
            lbp_anomaly REAL,
            chroma_anomaly REAL,
            copy_move_score REAL,
            ghost_anomaly REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Automatically add any missing columns if table pre-existed
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(submissions)").fetchall()]
    new_cols = [
        ("fake_probability", "REAL"),
        ("ela_score", "REAL"),
        ("fft_ratio", "REAL"),
        ("noise_variance", "REAL"),
        ("gradcam_url", "TEXT"),
        ("dct_score", "REAL"),
        ("lbp_anomaly", "REAL"),
        ("chroma_anomaly", "REAL"),
        ("copy_move_score", "REAL"),
        ("ghost_anomaly", "REAL"),
    ]
    for col, col_type in new_cols:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE submissions ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()

init_db()

# Import verification orchestrator
from verification import verify_photos


@app.on_event("startup")
async def startup_event():
    logger.info("Image Guard AI 2.0 with XAI Forensics initialized successfully.")


@app.get("/", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    """Renders main upload and analysis interface."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/verify_boards/", response_class=HTMLResponse)
async def verify_boards(request: Request, files: List[UploadFile] = File(...)):
    """
    Main web form handler: Accepts 1 to 10 photos, executes verification & XAI forensics,
    stores records, and renders the interactive Explainable AI dashboard.
    """
    if not files or len(files) == 0:
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "error": "Please upload at least 1 photo for authenticity & deepfake forensic verification.",
                "results": [],
                "map_url": None
            }
        )

    saved_paths = []
    try:
        # Save uploaded files into uploads folder
        for file in files:
            if not file.filename:
                continue
            safe_filename = os.path.basename(file.filename)
            file_path = os.path.join("uploads", safe_filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_paths.append(file_path)

        if not saved_paths:
            return templates.TemplateResponse(
                request=request,
                name="results.html",
                context={
                    "error": "No valid files received.",
                    "results": [],
                    "map_url": None
                }
            )

        # Run verification and Explainable AI forensics
        verification_results, map_url = verify_photos(saved_paths)

        # Persist results in SQLite
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        for res in verification_results:
            meta = res.get("metadata", {})
            xai = res.get("xai", {})
            c.execute("""
                INSERT INTO submissions (
                    filename, status, score, fake_probability, reason,
                    lat, lon, timestamp, device, depth, cluster,
                    ela_score, fft_ratio, noise_variance, gradcam_url,
                    dct_score, lbp_anomaly, chroma_anomaly, copy_move_score, ghost_anomaly
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                res.get("file", ""),
                res.get("status", "Unknown"),
                res.get("score", 0.0),
                xai.get("fake_probability", 0.0),
                "; ".join(res.get("reason", [])),
                meta.get("lat"),
                meta.get("lon"),
                meta.get("timestamp").isoformat() if meta.get("timestamp") else None,
                meta.get("device"),
                res.get("depth", 0.0),
                res.get("cluster", -1),
                xai.get("ela_score", 0.0),
                xai.get("fft_ratio", 0.0),
                xai.get("noise_variance", 0.0),
                xai.get("gradcam_url", ""),
                xai.get("dct_score", 1.0),
                xai.get("lbp_anomaly", 0.0),
                xai.get("chroma_anomaly", 0.0),
                xai.get("copy_move_score", 0.0),
                xai.get("ghost_anomaly", 0.0)
            ))
        conn.commit()
        conn.close()

        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "results": verification_results,
                "map_url": map_url,
                "error": None
            }
        )

    except Exception as e:
        logger.error(f"Error processing verification request: {e}", exc_info=True)
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "error": f"Verification error: {str(e)}",
                "results": [],
                "map_url": None
            }
        )


@app.get("/results", response_class=HTMLResponse)
async def get_latest_results(request: Request):
    """Retrieve and display recent verification submissions."""
    try:
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("""
            SELECT filename, status, score, fake_probability, reason,
                   lat, lon, timestamp, device, depth, cluster,
                   ela_score, fft_ratio, noise_variance, gradcam_url,
                   dct_score, lbp_anomaly, chroma_anomaly, copy_move_score, ghost_anomaly
            FROM submissions ORDER BY id DESC LIMIT 5
        """)
        rows = c.fetchall()
        conn.close()

        if not rows:
            return templates.TemplateResponse(
                request=request,
                name="results.html",
                context={"results": [], "map_url": None, "error": "No submissions found yet. Please upload photos."}
            )

        results = []
        for row in rows:
            filename = row[0]
            base_name = os.path.splitext(filename)[0]
            results.append({
                "file": filename,
                "image_url": f"/uploads/{filename}",
                "status": row[1],
                "score": row[2] or 0.0,
                "reason": [r.strip() for r in (row[4] or "").split(";") if r.strip()],
                "metadata": {
                    "lat": row[5],
                    "lon": row[6],
                    "timestamp": row[7],
                    "device": row[8]
                },
                "depth": row[9] or 0.0,
                "cluster": row[10] if row[10] is not None else -1,
                "location_method": "EXIF GPS Geotag" if row[5] else "None",
                "xai": {
                    "authenticity_score": row[2] or 0.0,
                    "fake_probability": row[3] or 0.0,
                    "verdict": "Authentic / Real" if (row[2] or 0) >= 0.7 else ("Suspicious" if (row[2] or 0) >= 0.45 else "Tampered"),
                    "ela_score": row[11] or 0.0,
                    "ela_url": f"/static/xai/ela_{base_name}.jpg",
                    "fft_ratio": row[12] or 0.0,
                    "fft_url": f"/static/xai/fft_{base_name}.png",
                    "noise_variance": row[13] or 0.0,
                    "noise_url": f"/static/xai/noise_{base_name}.png",
                    "gradcam_url": row[14] or f"/static/xai/gradcam_{base_name}.png",
                    "dct_score": row[15] or 1.0,
                    "dct_url": f"/static/xai/dct_{base_name}.png",
                    "lbp_anomaly": row[16] or 0.0,
                    "lbp_url": f"/static/xai/lbp_{base_name}.png",
                    "chroma_anomaly": row[17] or 0.0,
                    "chroma_url": f"/static/xai/chroma_{base_name}.png",
                    "copy_move_score": row[18] or 0.0,
                    "copy_move_url": f"/static/xai/copymove_{base_name}.png",
                    "ghost_anomaly": row[19] or 0.0,
                    "ghost_url": f"/static/xai/ghost_{base_name}.png"
                }
            })

        map_url = "/static/map.html" if os.path.exists("static/map.html") else None
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={"results": results, "map_url": map_url, "error": None}
        )
    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={"error": f"Error: {e}", "results": [], "map_url": None}
        )


@app.post("/api/verify")
async def api_verify(files: List[UploadFile] = File(...)):
    """
    Programmatic REST API endpoint for research benchmarks and integration.
    Returns structured JSON with all XAI and metadata forensic attributes.
    """
    try:
        saved_paths = []
        for file in files:
            if not file.filename:
                continue
            safe_filename = os.path.basename(file.filename)
            file_path = os.path.join("uploads", safe_filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_paths.append(file_path)

        verification_results, map_url = verify_photos(saved_paths)
        
        # Serialize datetime fields for clean JSON response
        serialized_results = []
        for res in verification_results:
            item = dict(res)
            meta = dict(item.get("metadata", {}))
            if meta.get("timestamp") and hasattr(meta["timestamp"], "isoformat"):
                meta["timestamp"] = meta["timestamp"].isoformat()
            item["metadata"] = meta
            serialized_results.append(item)

        return JSONResponse(content={
            "success": True,
            "total_images": len(serialized_results),
            "results": serialized_results,
            "map_url": map_url
        })
    except Exception as e:
        logger.error(f"API verification error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.py:app", host="127.0.0.1", port=8000, reload=False)
