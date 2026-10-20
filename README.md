# Tvara R3ACTR - Flask Backend (Vercel & Cloud Deployable)

This folder contains a standalone, fully functional Python Flask backend for the **Tvara R3ACTR Specialist Model** using the exact `tvara_r3actr_fp16.pt` weights.

---

## 📁 Folder Structure

```
tvara_flask_backend/
├── api/
│   └── index.py            # Vercel Serverless WSGI entry point
├── model_files/
│   ├── config.json         # Tvara Transformer hyperparameter configuration
│   ├── tvara_tokenizer.json# Tokenizer vocabulary and BPE merges
│   └── tvara_r3actr_fp16.pt# FP16 Model Weights (216 MB)
├── app.py                  # Flask Application & inference routing with CORS
├── model.py                # TvaraLM architecture, RoPE & KV-Cache implementation
├── requirements.txt        # Python dependency manifest
├── vercel.json             # Vercel deployment and routing configuration
├── .gitignore              # Ignores bytecode, virtual environments, .vercel
└── README.md               # Deployment and usage instructions
```

---

## 🚀 Quickstart: Running Locally

### 1. Install dependencies
```bash
cd tvara_flask_backend
pip install -r requirements.txt
```

### 2. Start the Flask server
```bash
python app.py
```
The server will boot up on `http://localhost:5000`.

---

## 📡 API Endpoints

### 1. Health Check
- **Route**: `GET /` or `GET /health`
- **Response**:
```json
{
  "status": "online",
  "service": "Tvara R3ACTR Flask Backend",
  "model_info": "Tvara R3ACTR FP16 Specialist (216.08 MB)",
  "device": "cpu",
  "timestamp": 1792502020.12
}
```

### 2. Chat / Inference
- **Route**: `POST /chat` (or `POST /reply`)
- **Headers**: `Content-Type: application/json`
- **Request Body (Single message)**:
```json
{
  "message": "What is R3ACTR?",
  "max_new_tokens": 150,
  "temperature": 0.4,
  "repetition_penalty": 1.12
}
```
- **Request Body (Multi-turn conversation)**:
```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! I am Tvara, dedicated to R3ACTR."},
    {"role": "user", "content": "What is PaperLab?"}
  ],
  "max_new_tokens": 200
}
```
- **Response**:
```json
{
  "reply": "PaperLab is a flagship research initiative by R3ACTR...",
  "tokens_generated": 48,
  "elapsed_seconds": 1.05,
  "tokens_per_second": 45.71,
  "model": "Tvara R3ACTR FP16 Specialist (216.08 MB)"
}
```

---

## 🚀 Deploying to Render (Recommended for PyTorch)

Render is ideal for hosting PyTorch models because it provides dedicated persistent containers without the 250 MB serverless lambda size limit.

### Method 1: Blueprint Deployment (One-Click)
1. Push your repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com/), click **New +** -> **Blueprint**.
3. Connect your repository (`tvara-r3actr`). Render will automatically detect [render.yaml](file:///a:/tvara_flask_backend/render.yaml) and configure the service.
4. Click **Apply**.

### Method 2: Manual Web Service Setup
1. In Render Dashboard, click **New +** -> **Web Service**.
2. Connect your GitHub repository.
3. Configure settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Plan**: `Free` or higher (minimum 512MB RAM recommended)
4. Environment Variables (optional, defaults are auto-configured in `app.py`):
   - `TVARA_CONFIG_PATH`: `model_files/config.json`
   - `TVARA_TOKENIZER_PATH`: `model_files/tvara_tokenizer.json`
   - `TVARA_MODEL_PATH`: `model_files/tvara_r3actr_fp16.pt`
5. Click **Create Web Service**.

---

## ☁️ Hosting on Vercel

### Important Vercel Architecture Consideration:
Vercel's standard Serverless Function limits are:
- **Max Unzipped Function Bundle Size**: 250 MB.
- PyTorch + CUDA/CPU binaries combined with a 216 MB model file exceed the direct static zip bundle limit of standard serverless lambdas.

### Supported Vercel Deployment Approaches:

#### Option A: Serverless with Remote Weights (`TVARA_MODEL_URL`)
1. Upload `tvara_r3actr_fp16.pt` to Cloudflare R2, AWS S3, or Hugging Face Hub (as a direct download link).
2. Set the environment variable in your Vercel Project Dashboard:
   - `TVARA_MODEL_URL` = `https://your-storage-bucket.com/tvara_r3actr_fp16.pt`
3. Exclude `model_files/*.pt` from git (so git push stays under upload limits).
4. Deploy using Vercel CLI:
   ```bash
   vercel
   ```
   The backend automatically downloads the weights to `/tmp` on serverless cold-start and caches it in memory.

#### Option B: Deploying on Render / Railway / Fly.io / Docker
If you want persistent local weight execution without bundle size caps:
1. Dockerfile or direct git repo push contains `tvara_r3actr_fp16.pt` intact.
2. Start command: `python app.py` or `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`.
