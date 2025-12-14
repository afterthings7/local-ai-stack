#!/usr/bin/env python3
"""
Simple Web UI for Local AI Stack
Provides a clean interface for SDXL image generation via ComfyUI
"""

import json
import uuid
import time
import urllib.request
import urllib.error
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_OUTPUT = Path.home() / "local-ai-stack/ComfyUI/output"

def queue_prompt(prompt_workflow):
    """Submit a workflow to ComfyUI and return the prompt_id"""
    data = json.dumps({"prompt": prompt_workflow}).encode('utf-8')
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())

def get_history(prompt_id):
    """Get the execution history for a prompt"""
    with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response:
        return json.loads(response.read())

def build_workflow(positive_prompt, negative_prompt, width=1024, height=1024, steps=20, cfg=7.5, seed=None):
    """Build the SDXL workflow"""
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": positive_prompt}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": negative_prompt}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "webui"}
        }
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    positive = data.get("prompt", "a beautiful landscape")
    negative = data.get("negative", "blurry, low quality, distorted")
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 20))
    cfg = float(data.get("cfg", 7.5))

    # Build and queue the workflow
    workflow = build_workflow(positive, negative, width, height, steps, cfg)

    try:
        result = queue_prompt(workflow)
        prompt_id = result.get("prompt_id")

        if not prompt_id:
            return jsonify({"error": "Failed to queue prompt"}), 500

        return jsonify({"prompt_id": prompt_id, "status": "queued"})

    except urllib.error.URLError as e:
        return jsonify({"error": f"ComfyUI not responding: {str(e)}"}), 503

@app.route("/status/<prompt_id>")
def status(prompt_id):
    try:
        history = get_history(prompt_id)

        if prompt_id not in history:
            return jsonify({"status": "processing"})

        outputs = history[prompt_id].get("outputs", {})

        # Find the SaveImage node output
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                images = node_output["images"]
                if images:
                    filename = images[0]["filename"]
                    return jsonify({
                        "status": "complete",
                        "filename": filename
                    })

        return jsonify({"status": "processing"})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route("/image/<filename>")
def get_image(filename):
    image_path = COMFYUI_OUTPUT / filename
    if image_path.exists():
        return send_file(image_path, mimetype="image/png")
    return "Image not found", 404

@app.route("/health")
def health():
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5) as response:
            return jsonify({"status": "ok", "comfyui": "connected"})
    except:
        return jsonify({"status": "ok", "comfyui": "disconnected"})

if __name__ == "__main__":
    print("\n🎨 Simple Image Generator UI")
    print("   http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
