import os
import sys
import time
import json
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import torch
from tokenizers import Tokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import TvaraConfig, TvaraLM

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_files"

CONFIG_PATH = Path(os.environ.get("TVARA_CONFIG_PATH", MODEL_DIR / "config.json"))
TOKENIZER_PATH = Path(os.environ.get("TVARA_TOKENIZER_PATH", MODEL_DIR / "tvara_tokenizer.json"))
MODEL_PATH = Path(os.environ.get("TVARA_MODEL_PATH", MODEL_DIR / "tvara_r3actr_fp16.pt"))
MODEL_URL = os.environ.get("TVARA_MODEL_URL", "").strip()

SYSTEM_PROMPT = (
    "You are Tvara, a specialized AI model dedicated exclusively to r3actr.com, its products, architecture, research, team, and services. "
    "You provide clear, accurate, and authoritative information about R3ACTR, PaperLab, Open Source Quest, and technical initiatives. "
    "Respond warmly to casual greetings and state your identity as Tvara. "
    "If asked about topics outside R3ACTR or questions you do not have verified knowledge about, simply say: 'Sorry, I cannot help with that. I only answer questions related to r3actr.com.'"
)

# -----------------------------------------------------------------------------
# Global Cached Model State
# -----------------------------------------------------------------------------
_model: Optional[TvaraLM] = None
_tokenizer: Optional[Tokenizer] = None
_device: Optional[torch.device] = None
_model_info: str = "Uninitialized"
_im_end_id: Optional[int] = None
_init_error: Optional[str] = None


def download_model_if_needed(target_path: Path, url: str) -> None:
    """Download model weights from remote storage if not present locally."""
    if target_path.exists() and target_path.stat().st_size > 0:
        return

    if not url:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".tmp")
    print(f"Downloading model checkpoint from {url} to {target_path}...")
    urllib.request.urlretrieve(url, str(temp_target))
    temp_target.replace(target_path)
    print("Download completed.")


def get_model_and_tokenizer() -> Tuple[TvaraLM, Tokenizer, torch.device, int]:
    """Lazy-load the model and tokenizer on first call or warm startup."""
    global _model, _tokenizer, _device, _model_info, _im_end_id, _init_error

    if _model is not None and _tokenizer is not None and _device is not None:
        return _model, _tokenizer, _device, _im_end_id

    if _init_error:
        raise RuntimeError(_init_error)

    try:
        # Check remote download fallback if weight file is missing
        if not MODEL_PATH.exists() and MODEL_URL:
            # On serverless platforms, writable storage is in /tmp
            writable_path = Path("/tmp") / "tvara_r3actr_fp16.pt" if Path("/tmp").exists() else MODEL_PATH
            download_model_if_needed(writable_path, MODEL_URL)
            load_target = writable_path
        else:
            load_target = MODEL_PATH

        if not load_target.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{load_target}'. "
                "Ensure 'tvara_r3actr_fp16.pt' is placed in model_files/ or set the TVARA_MODEL_URL environment variable."
            )

        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(f"Tokenizer not found at '{TOKENIZER_PATH}'.")

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Tokenizer
        _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        _im_end_id = _tokenizer.token_to_id("<|im_end|>")

        # Load Model Config
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            config = TvaraConfig(**cfg_data)
        else:
            config = TvaraConfig()

        print(f"Loading Tvara FP16 weights on {_device} from {load_target}...")
        ckpt = torch.load(str(load_target), map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            if "config" in ckpt and isinstance(ckpt["config"], dict):
                config = TvaraConfig(**ckpt["config"])
        elif isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError("Unsupported checkpoint format.")

        # Instantiate Model
        model = TvaraLM(config)
        # Convert FP16 weights to float32 for CPU stability or float16 on CUDA
        weight_dtype = torch.float16 if _device.type == "cuda" else torch.float32
        model.to(dtype=weight_dtype)

        prepared_sd = {}
        for k, v in state_dict.items():
            prepared_sd[k] = v.to(dtype=weight_dtype)

        # Tie embeddings if lm_head is missing
        if "lm_head.weight" not in prepared_sd and "tok_embeddings.weight" in prepared_sd:
            prepared_sd["lm_head.weight"] = prepared_sd["tok_embeddings.weight"]

        model.load_state_dict(prepared_sd, strict=False)
        model.to(_device)
        model.eval()

        _model = model
        size_mb = load_target.stat().st_size / (1024**2)
        _model_info = f"Tvara R3ACTR FP16 Specialist ({size_mb:.2f} MB)"
        print(f"Model successfully loaded: {_model_info}")

        return _model, _tokenizer, _device, _im_end_id

    except Exception as e:
        _init_error = f"Model load failed: {str(e)}"
        print(f"Error initializing model: {_init_error}", file=sys.stderr)
        raise RuntimeError(_init_error)


# -----------------------------------------------------------------------------
# Flask Application
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Tvara R3ACTR Flask Backend",
        "model_info": _model_info,
        "device": str(_device) if _device else "auto",
        "timestamp": time.time()
    }), 200


@app.route("/chat", methods=["POST"])
@app.route("/reply", methods=["POST"])
def chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON with 'Content-Type: application/json'"}), 400

    data = request.get_json(silent=True) or {}
    message = data.get("message")
    messages = data.get("messages")
    max_new_tokens = int(data.get("max_new_tokens", 220))
    temperature = float(data.get("temperature", 0.4))
    repetition_penalty = float(data.get("repetition_penalty", 1.12))

    max_new_tokens = max(1, min(max_new_tokens, 512))
    temperature = max(0.0, min(temperature, 2.0))
    repetition_penalty = max(1.0, min(repetition_penalty, 2.0))

    # Construct Prompt
    if messages and isinstance(messages, list) and len(messages) > 0:
        prompt_text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt_text += "<|im_start|>assistant\n"
    elif message and isinstance(message, str) and message.strip():
        prompt_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{message.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return jsonify({"error": "Either 'message' (string) or 'messages' (list) must be provided."}), 400

    try:
        model, tokenizer, device, im_end_id = get_model_and_tokenizer()
    except Exception as e:
        return jsonify({"error": f"Failed to initialize model: {str(e)}"}), 503

    start_time = time.time()
    try:
        input_ids = tokenizer.encode(prompt_text).ids
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)

        stop_tokens = [im_end_id] if im_end_id is not None else []
        output_tokens = model.generate_cached(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stop_ids=stop_tokens
        )

        gen_tokens = output_tokens[0, len(input_ids):].tolist()
        reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        elapsed = time.time() - start_time
        num_generated = len(gen_tokens)
        tps = round(num_generated / max(elapsed, 1e-6), 2)

        return jsonify({
            "reply": reply,
            "tokens_generated": num_generated,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": tps,
            "model": _model_info
        }), 200

    except Exception as e:
        return jsonify({"error": f"Inference execution failed: {str(e)}"}), 500


# Warm up on local start if run directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Tvara Flask Server on port {port}...")
    try:
        get_model_and_tokenizer()
    except Exception as e:
        print(f"Warning: Initial model pre-load had an issue: {e}")
    app.run(host="0.0.0.0", port=port, debug=False)
