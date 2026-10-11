"""
Image Guard AI - Core Verification & Forensic Orchestration Module
==================================================================
Combines:
- Explainable AI (XAI) with PyTorch Grad-CAM, ELA, 2D FFT, and Noise Residuals
- EXIF Metadata & Device/Camera Hardware Provenance
- OCR Scene Text & Geocoding Verification
- Geospatial Haversine & DBSCAN Spatial Clustering
- Multi-photo cross-temporal & spatial fraud detection
"""

import os
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ExifTags

# Try importing specialized libraries with graceful fallbacks
try:
    from exif import Image as ExifImage
    HAS_EXIF_LIB = True
except ImportError:
    HAS_EXIF_LIB = False

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from haversine import haversine
    HAS_HAVERSINE = True
except ImportError:
    HAS_HAVERSINE = False

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import pytesseract
    # Configure default tesseract path if on Windows
    tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tess_path):
        pytesseract.pytesseract.tesseract_cmd = tess_path
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

try:
    from geopy.geocoders import Nominatim
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

# Import our specialized XAI Engine
from forensics_xai import run_xai_forensics

logger = logging.getLogger(__name__)


# =====================================================================
# 1. EXIF Metadata Extraction (Dual-Method Engine)
# =====================================================================

def _dms_to_deg(dms_tuple) -> float:
    """Convert degrees, minutes, seconds tuple to decimal degrees."""
    try:
        d = float(dms_tuple[0])
        m = float(dms_tuple[1]) / 60.0
        s = float(dms_tuple[2]) / 3600.0
        return d + m + s
    except Exception:
        return 0.0


def extract_exif_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts GPS coordinates, timestamp, camera model, and software tags
    using both exif library and PIL fallback.
    """
    metadata: Dict[str, Any] = {
        "lat": None,
        "lon": None,
        "timestamp": None,
        "device": None,
        "software": None
    }
    
    # Method A: Try exif library
    if HAS_EXIF_LIB:
        try:
            with open(image_path, "rb") as f:
                img = ExifImage(f)
                if img.has_exif:
                    if hasattr(img, "gps_latitude") and hasattr(img, "gps_longitude"):
                        lat = _dms_to_deg(img.gps_latitude)
                        lon = _dms_to_deg(img.gps_longitude)
                        if hasattr(img, "gps_latitude_ref") and img.gps_latitude_ref.upper() == "S":
                            lat = -lat
                        if hasattr(img, "gps_longitude_ref") and img.gps_longitude_ref.upper() == "W":
                            lon = -lon
                        if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0):
                            metadata["lat"] = round(lat, 6)
                            metadata["lon"] = round(lon, 6)
                            
                    if hasattr(img, "datetime"):
                        try:
                            metadata["timestamp"] = datetime.strptime(img.datetime, "%Y:%m:%d %H:%M:%S")
                        except Exception:
                            pass
                            
                    if hasattr(img, "model"):
                        metadata["device"] = str(img.model).strip()
                    elif hasattr(img, "make"):
                        metadata["device"] = str(img.make).strip()
                        
                    if hasattr(img, "software"):
                        metadata["software"] = str(img.software).strip()
        except Exception as e:
            logger.debug(f"exif library extraction notice: {e}")

    # Method B: Fallback to PIL ExifTags if lat/lon/device still missing
    if metadata["lat"] is None or metadata["device"] is None:
        try:
            with Image.open(image_path) as pil_img:
                raw_exif = pil_img._getexif()
                if raw_exif:
                    exif_data = {
                        ExifTags.TAGS.get(tag_id, tag_id): val
                        for tag_id, val in raw_exif.items()
                    }
                    if not metadata["device"]:
                        device = exif_data.get("Model") or exif_data.get("Make")
                        if device:
                            metadata["device"] = str(device).strip()
                            
                    if not metadata["software"]:
                        sw = exif_data.get("Software")
                        if sw:
                            metadata["software"] = str(sw).strip()
                            
                    if not metadata["timestamp"] and "DateTimeOriginal" in exif_data:
                        try:
                            metadata["timestamp"] = datetime.strptime(str(exif_data["DateTimeOriginal"]), "%Y:%m:%d %H:%M:%S")
                        except Exception:
                            pass
                            
                    if metadata["lat"] is None and "GPSInfo" in exif_data:
                        gps_info = exif_data["GPSInfo"]
                        gps_tags = {
                            ExifTags.GPSTAGS.get(t, t): gps_info[t]
                            for t in gps_info
                        }
                        if "GPSLatitude" in gps_tags and "GPSLongitude" in gps_tags:
                            lat = _dms_to_deg(gps_tags["GPSLatitude"])
                            lon = _dms_to_deg(gps_tags["GPSLongitude"])
                            if gps_tags.get("GPSLatitudeRef", "N").upper() == "S":
                                lat = -lat
                            if gps_tags.get("GPSLongitudeRef", "E").upper() == "W":
                                lon = -lon
                            if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0):
                                metadata["lat"] = round(lat, 6)
                                metadata["lon"] = round(lon, 6)
        except Exception as e:
            logger.debug(f"PIL EXIF fallback notice: {e}")

    return metadata


TESSERACT_CMD = None
for candidate_path in [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe')
]:
    if os.path.exists(candidate_path):
        TESSERACT_CMD = candidate_path
        break

if TESSERACT_CMD and HAS_PYTESSERACT:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    HAS_ACTIVE_TESSERACT = True
else:
    HAS_ACTIVE_TESSERACT = False


def detect_location_from_text(image_path: str) -> Tuple[Optional[float], Optional[float], float, str]:
    """Extract coordinates or address from text in image via Tesseract if installed."""
    if not HAS_ACTIVE_TESSERACT:
        return None, None, 0.0, ""
        
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None, None, 0.0, ""
            
        h, w = img_bgr.shape[:2]
        if max(h, w) > 800:
            scale = 800.0 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config, timeout=2)
        
        lat, lon = None, None
        confidence = 0.0
        location_text = ""
        
        import re
        for line in text.split('\n'):
            line_clean = line.strip()
            if not line_clean:
                continue
            if 'lat' in line_clean.lower() and ('long' in line_clean.lower() or 'lon' in line_clean.lower()):
                coords = re.findall(r'[-+]?\d*\.\d+|\d+', line_clean)
                if len(coords) >= 2:
                    lat = float(coords[0])
                    lon = float(coords[1])
                    confidence = 0.85
                    location_text = line_clean
                    break
                    
        return lat, lon, confidence, location_text
    except Exception as e:
        logger.debug(f"OCR detection notice: {e}")
        return None, None, 0.0, ""


# =====================================================================
# 3. Computer Vision Depth & Scene Geometry
# =====================================================================

def compute_depth_estimate(image_path: str) -> float:
    """Fast structural scene depth / edge gradient score using Laplacian filter."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        h, w = img.shape[:2]
        if max(h, w) > 600:
            scale = 600.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        edges = cv2.Laplacian(img, cv2.CV_32F)
        depth_score = float(np.mean(np.abs(edges)))
        return round(depth_score, 2)
    except Exception as e:
        logger.debug(f"Depth computation error: {e}")
        return 0.0


# =====================================================================
# 4. Geospatial Clustering (Haversine + DBSCAN)
# =====================================================================

def haversine_distance_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Great Circle distance between two points in km."""
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(6371.0 * c)


def cluster_locations(locations: List[Tuple[float, float]], eps_km: float = 0.05) -> List[int]:
    """Groups geographic coordinates into clusters (50m threshold)."""
    if not locations:
        return []
    if len(locations) == 1:
        return [0]
        
    if HAS_SKLEARN:
        try:
            # 50 meters in radians: 0.05 / 6371
            eps_rad = eps_km / 6371.0
            rad_coords = np.radians(locations)
            db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit(rad_coords)
            return [int(lbl) for lbl in db.labels_]
        except Exception:
            pass
            
    # Pure Python clustering fallback
    clusters = []
    for i, loc in enumerate(locations):
        assigned = False
        for c_idx, member_idx in enumerate(clusters):
            if haversine_distance_km(loc, locations[member_idx]) <= eps_km:
                clusters.append(c_idx)
                assigned = True
                break
        if not assigned:
            clusters.append(i)
    return clusters


# =====================================================================
# 5. Core Verification Orchestrator
# =====================================================================

def verify_photos(file_paths: List[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Main entry point: Verifies a list of uploaded photos.
    Returns:
        (verification_results, map_url)
    """
    results: List[Dict[str, Any]] = []
    locations: List[Tuple[float, float]] = []
    loc_file_indices: List[int] = []

    for idx, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        try:
            # 1. Extract EXIF Metadata
            metadata = extract_exif_metadata(file_path)
            
            # 2. Run Explainable AI (XAI) Deepfake Forensics
            xai_res = run_xai_forensics(file_path)
            
            # 3. Compute structural depth
            depth = compute_depth_estimate(file_path)
            
            # 4. Location determination
            location_method = "None"
            location_confidence = 0.0
            
            if metadata["lat"] is not None and metadata["lon"] is not None:
                location_method = "EXIF GPS Geotag"
                location_confidence = 0.95
                locations.append((metadata["lat"], metadata["lon"]))
                loc_file_indices.append(idx)
            else:
                # Attempt OCR detection
                ocr_lat, ocr_lon, ocr_conf, ocr_text = detect_location_from_text(file_path)
                if ocr_lat is not None and ocr_lon is not None:
                    metadata["lat"] = ocr_lat
                    metadata["lon"] = ocr_lon
                    location_method = f"Scene OCR Text ('{ocr_text[:30]}')"
                    location_confidence = ocr_conf
                    locations.append((ocr_lat, ocr_lon))
                    loc_file_indices.append(idx)
                    
            # 5. Build initial reasons & score
            score = float(xai_res["authenticity_score"])
            reasons: List[str] = list(xai_res["indicators"])
            
            # Software check (Photoshop, GIMP, Canva, etc.)
            if metadata.get("software"):
                sw_lower = metadata["software"].lower()
                if any(sw in sw_lower for sw in ["adobe", "photoshop", "gimp", "canva", "lightroom", "facetune", "snapseed"]):
                    score *= 0.70
                    reasons.append(f"Image edited or exported using photo manipulation software: {metadata['software']}")
                    
            if location_method != "None":
                reasons.append(f"Geographic coordinates confirmed via {location_method}")
            else:
                reasons.append("No geographic GPS metadata embedded (standard on stripped or web downloads)")
                
            results.append({
                "file": filename,
                "file_path": file_path,
                "image_url": f"/uploads/{filename}",
                "status": "Pending",
                "score": score,
                "reason": reasons,
                "metadata": metadata,
                "depth": depth,
                "cluster": -1,
                "location_confidence": location_confidence,
                "location_method": location_method,
                "xai": xai_res
            })
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)
            results.append({
                "file": filename,
                "file_path": file_path,
                "image_url": f"/uploads/{filename}",
                "status": "Error",
                "score": 0.0,
                "reason": [f"Processing error: {str(e)}"],
                "metadata": {"lat": None, "lon": None, "timestamp": None, "device": None, "software": None},
                "depth": 0.0,
                "cluster": -1,
                "location_confidence": 0.0,
                "location_method": "None",
                "xai": {
                    "authenticity_score": 0.0,
                    "fake_probability": 1.0,
                    "verdict": "Error",
                    "verdict_class": "status-error",
                    "gradcam_url": "",
                    "ela_url": "",
                    "ela_score": 0.0,
                    "fft_url": "",
                    "fft_ratio": 0.0,
                    "noise_url": "",
                    "noise_variance": 0.0,
                    "indicators": [str(e)]
                }
            })

    # =================================================================
    # 6. Cross-Photo Spatial Clustering & Multi-Image Fraud Analysis
    # =================================================================
    if locations:
        clusters = cluster_locations(locations)
        for loc_idx, file_idx in enumerate(loc_file_indices):
            if loc_idx < len(clusters):
                results[file_idx]["cluster"] = clusters[loc_idx]

    # Perform multi-image correlation if 2 or more images
    if len(results) > 1:
        for i, res in enumerate(results):
            if res["status"] == "Error":
                continue
                
            curr_score = res["score"]
            curr_reasons = res["reason"]
            cluster = res["cluster"]
            
            # Compare with other photos in batch
            for j, other in enumerate(results):
                if i == j or other["status"] == "Error":
                    continue
                    
                # A) Temporal Check: photos taken within < 3 seconds
                t1, t2 = res["metadata"]["timestamp"], other["metadata"]["timestamp"]
                if t1 and t2:
                    delta_sec = abs((t1 - t2).total_seconds())
                    if delta_sec < 4:
                        curr_score *= 0.85
                        curr_reasons.append(f"Suspicious rapid capture: taken only {delta_sec:.1f}s from {other['file']}")
                        
                # B) Depth & Perspective Check
                if res["depth"] > 0 and other["depth"] > 0:
                    diff_depth = abs(res["depth"] - other["depth"])
                    if diff_depth < 0.8:
                        curr_reasons.append(f"Identical scene perspective & depth to {other['file']} (variation < 1.0)")
                        
                # C) Device Collision: same device ID in rapid sequence
                if res["metadata"]["device"] and res["metadata"]["device"] == other["metadata"]["device"]:
                    pass # Normal for multi-angle photo batch

            res["score"] = curr_score

    # Final Verdict Thresholding
    for res in results:
        if res["status"] == "Error":
            continue
        final_score = float(np.clip(res["score"], 0.0, 1.0))
        res["score"] = round(final_score, 4)
        
        if final_score >= 0.70:
            res["status"] = "Verified"
        elif final_score >= 0.45:
            res["status"] = "Suspicious"
        else:
            res["status"] = "Rejected"

    # =================================================================
    # 7. Generate Interactive Folium Map
    # =================================================================
    map_url = None
    if locations and HAS_FOLIUM:
        try:
            center_lat = float(np.mean([loc[0] for loc in locations]))
            center_lon = float(np.mean([loc[1] for loc in locations]))
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15)
            
            for res in results:
                lat, lon = res["metadata"]["lat"], res["metadata"]["lon"]
                if lat is not None and lon is not None:
                    icon_color = "green" if res["status"] == "Verified" else ("orange" if res["status"] == "Suspicious" else "red")
                    popup_html = f"""
                    <div style='font-family:sans-serif;'>
                        <b>{res['file']}</b><br>
                        Status: <b>{res['status']}</b> ({res['score']*100:.1f}%)<br>
                        Cluster: #{res['cluster']}<br>
                        Device: {res['metadata']['device'] or 'N/A'}<br>
                        Time: {res['metadata']['timestamp'] or 'N/A'}
                    </div>
                    """
                    folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=f"{res['file']} ({res['status']})",
                        icon=folium.Icon(color=icon_color, icon="camera")
                    ).add_to(fmap)
                    
            map_filepath = os.path.join("static", "map.html")
            fmap.save(map_filepath)
            map_url = "/static/map.html"
        except Exception as e:
            logger.error(f"Error generating Folium map: {e}")

    return results, map_url