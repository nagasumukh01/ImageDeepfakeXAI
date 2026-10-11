---
title: Image Guard AI - Deepfake & Authenticity Forensics
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🛡️ Image Guard AI — Forensic & Explainable AI (XAI) Suite v3.0

> **Research-Oriented Deepfake, AI-Generated Image & Tampering Forensics Engine**

Image Guard AI is a high-performance, web-based forensic verification system that exposes deepfakes, AI image generation artifacts (Stable Diffusion, Midjourney, Flux, GANs), and digital manipulations.

---

## 🚀 Free 1-Click Cloud Deployment Options (100% Free Forever)

### Option 1: Hugging Face Spaces (Recommended for AI Projects)
**Specs:** 16 GB RAM • 2 vCPUs • 100% Free • No Credit Card Required
1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**
2. Set Space Name: `ImageGuardAI`
3. License: `MIT`
4. Select Space SDK: **Docker** (Blank)
5. Under Space Hardware: Keep the default **CPU Basic (Free - 2 vCPU, 16GB RAM)**
6. Click **Create Space**
7. Connect your GitHub repository:
   - Go to your Space's **Settings** > **Repository** > Connect to GitHub: `nagasumukh01/ImageDeepfakeXAI`
   - *Or push via git*:
     ```bash
     git remote add hf https://huggingface.co/spaces/<YOUR-USERNAME>/ImageGuardAI
     git push -u hf main
     ```
8. Hugging Face will automatically build the `Dockerfile` and give you a live HTTPS public URL (`https://<username>-imageguardai.hf.space`)!

---

### Option 2: Render.com (Free Web Service)
**Specs:** Free Tier Web Service
1. Go to **[dashboard.render.com](https://dashboard.render.com)** and sign in with GitHub.
2. Click **New +** > **Web Service**.
3. Select your repository: `nagasumukh01/ImageDeepfakeXAI`.
4. Render will automatically detect the settings from `render.yaml`:
   - **Environment:** `Python` (or `Docker`)
   - **Build Command:** `pip install -U pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service** (Free). Your site will be live at `https://image-guard-ai.onrender.com`.

---

### Option 3: Local or Self-Hosted Docker
Run anywhere with Docker installed:
```bash
docker build -t image-guard-ai .
docker run -p 8000:7860 image-guard-ai
```
Visit `http://localhost:8000`.

---

## 🔬 10-Layer Explainable AI (XAI) Forensics Suite

| Module | Method | Scientific Principle | Tampering Indicator |
|---|---|---|---|
| **1. Grad-CAM** | Gradient-Weighted Class Activation Mapping | CNN layer activations | Highlights manipulated spatial regions |
| **2. Error Level Analysis (ELA)** | Differential Re-compression Analysis | JPEG compression error delta | Bright zones indicate non-matching compression |
| **3. 2D FFT Spectrum** | 2D Fast Fourier Transform | Frequency domain artifact analysis | Grid stars and abnormal high-frequency peaks |
| **4. Noise Variance (PRNU)** | Sensor Pattern Noise Homogeneity | Camera sensor noise consistency | Noise discontinuity between foreground/background |
| **5. DCT Block Grid** | 8×8 JPEG Discrete Cosine Transform | Grid-aligned gradient variance | Spliced regions lack matching JPEG grid boundaries |
| **6. LBP Texture Anomaly** | Local Binary Patterns Micro-texture | Texture entropy distribution | AI generators create unnatural micro-texture smoothing |
| **7. Chroma Divergence** | YCbCr Color Inconsistency | Chrominance channel cross-ratio | AI diffusion models exhibit chrominance bleeding |
| **8. Copy-Move Forgery** | SIFT / ORB Keypoint Block Matching | Feature descriptor clustering | Duplicate cloned elements within the same image |
| **9. GHOST Forensics** | Generalized Higher-Order Statistics | Multi-quality JPEG residual difference | Reveals original compression quality of spliced regions |
| **10. Provenance & Metadata** | EXIF, GPS & OCR Verification | Hardware signature & temporal data | Device tags, GPS coordinate mapping & timestamp validation |

---

## 🛠️ Local Development & Running

1. **Clone repository:**
   ```bash
   git clone https://github.com/nagasumukh01/ImageDeepfakeXAI.git
   cd ImageDeepfakeXAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the server:**
   ```bash
   uvicorn app:app --reload --port 8000
   ```
   Open `http://127.0.0.1:8000` in your browser.
