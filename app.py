#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from PIL import Image
import easyocr
from pdf2image import convert_from_path
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ============================================================
# MODEL DOWNLOAD FROM HUGGING FACE HUB
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "indic_news_model")
os.makedirs(MODEL_DIR, exist_ok=True)

required_files = ["config.json", "model.pth", "label2id.json", "id2label.json"]

missing_files = [
    f for f in required_files if not os.path.exists(os.path.join(MODEL_DIR, f))
]

if missing_files:
    print(f"Missing: {missing_files}. Downloading...")
    for file in required_files:
        try:
            hf_hub_download(
                repo_id="Sathvik2954/hindi-news-model",
                filename=file,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded: {file}")
        except Exception as e:
            print(f"Failed: {file} - {e}")
    print("Model download complete.")
else:
    print("All model files found locally.")

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ============================================================
# EASYOCR
# ============================================================
print("Loading EasyOCR...")
reader = easyocr.Reader(["hi", "en"], gpu=False)
print("EasyOCR ready.")

# ============================================================
# LOAD TOKENIZER (from original model, NOT from local folder)
# ============================================================
print("Loading tokenizer from original IndicBERTv2...")
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
print("Tokenizer ready.")

# ============================================================
# LOAD LABEL MAPPINGS AND MODEL WEIGHTS
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

with open(os.path.join(MODEL_DIR, "label2id.json"), "r", encoding="utf-8") as f:
    label2id = json.load(f)
with open(os.path.join(MODEL_DIR, "id2label.json"), "r", encoding="utf-8") as f:
    id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}

# Load model config from local config.json
config = AutoConfig.from_pretrained(MODEL_DIR)
config.num_labels = len(label2id)
config.label2id = label2id
config.id2label = id2label

# Create model with config, then load state_dict
model = AutoModelForSequenceClassification.from_config(config)
state_dict = torch.load(os.path.join(MODEL_DIR, "model.pth"), map_location="cpu")
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("MODEL READY. LABELS:", list(id2label.values()))


# ============================================================
# CLASSIFICATION
# ============================================================
def preprocess_text(text):
    text = " ".join(text.split())
    text = ".".join(text.split(".")[:2])
    return text


def classify_with_confidence(text):
    cleaned = preprocess_text(text)
    inputs = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, pred_id = torch.max(probs, dim=1)
    pred_label = id2label[pred_id.item()]
    return pred_label, round(confidence.item() * 100, 2)


# ============================================================
# WEB SCRAPER (use your existing function)
# ============================================================
def scrape_hindi_news():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    all_headlines = []
    # Amar Ujala
    try:
        url = "https://www.amarujala.com/"
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.content, "html.parser")
        for h in soup.find_all(["h2", "h3"], limit=15):
            link = h.find("a")
            title = link.get_text(strip=True) if link else h.get_text(strip=True)
            if title and len(title) > 20 and title not in ["उत्तर प्रदेश", "मध्य प्रदेश"]:
                all_headlines.append(("Amar Ujala", title))
    except Exception as e:
        print(f"Amar Ujala error: {e}")
    # Dainik Jagran
    try:
        url = "https://www.jagran.com/"
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.content, "html.parser")
        for h in soup.find_all(["h2", "h3"], limit=15):
            link = h.find("a")
            title = link.get_text(strip=True) if link else h.get_text(strip=True)
            if title and len(title) > 20:
                all_headlines.append(("Dainik Jagran", title))
    except Exception as e:
        print(f"Dainik Jagran error: {e}")
    # Navbharat Times
    try:
        url = "https://navbharattimes.indiatimes.com/"
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.content, "html.parser")
        exclude = ["उत्तर प्रदेश", "मध्य प्रदेश", "बिज़नेस", "एंटरटेनमेंट"]
        for h in soup.find_all(["h2", "h3", "h4"], limit=15):
            link = h.find("a")
            title = link.get_text(strip=True) if link else h.get_text(strip=True)
            if (
                title
                and len(title) > 20
                and any("\u0900" <= c <= "\u097f" for c in title)
                and title not in exclude
            ):
                all_headlines.append(("Navbharat Times", title))
    except Exception as e:
        print(f"Navbharat Times error: {e}")
    # BBC Hindi
    try:
        url = "https://www.bbc.com/hindi"
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.content, "html.parser")
        for h in soup.find_all(["h2", "h3"], limit=15):
            link = h.find("a")
            title = link.get_text(strip=True) if link else h.get_text(strip=True)
            if title and len(title) > 20:
                all_headlines.append(("BBC Hindi", title))
    except Exception as e:
        print(f"BBC Hindi error: {e}")
    # Remove duplicates
    unique = []
    seen = set()
    for src, hl in all_headlines:
        if hl not in seen:
            seen.add(hl)
            unique.append((src, hl))
    return unique


# ============================================================
# OCR FUNCTIONS
# ============================================================
def ocr_image(image):
    img_np = np.array(image)
    result = reader.readtext(img_np, detail=0)
    return " ".join(result)


def extract_headings_from_text(text):
    lines = text.split("\n")
    headings = []
    for line in lines:
        line = line.strip()
        if 3 < len(line) < 80 and (line.isupper() or line.istitle()):
            headings.append(line)
    if not headings:
        headings = [l for l in lines if len(l.strip()) > 5][:5]
    return headings


# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/text")
def text_page():
    return render_template("text.html")


@app.route("/scrape")
def scrape_page():
    return render_template("scrape.html")


@app.route("/ocr")
def ocr_page():
    return render_template("ocr.html")


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    label, confidence = classify_with_confidence(text)
    return jsonify({"category": label, "confidence": confidence})


@app.route("/api/scrape", methods=["GET"])
def api_scrape():
    requested_label = request.args.get("label", "").strip().lower()
    headlines = scrape_hindi_news()
    results = []
    for source, hl in headlines:
        label, conf = classify_with_confidence(hl)
        if requested_label and requested_label != label:
            continue
        results.append(
            {"source": source, "headline": hl, "category": label, "confidence": conf}
        )
    return jsonify(
        {"results": results, "timestamp": datetime.now().strftime("%d %B %Y, %I:%M %p")}
    )


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            pages = convert_from_path(filepath, dpi=200)
            all_text = ""
            for page in pages:
                all_text += ocr_image(page) + "\n"
        else:
            img = Image.open(filepath)
            all_text = ocr_image(img)
        headings = extract_headings_from_text(all_text)
        results = []
        for h in headings:
            label, conf = classify_with_confidence(h)
            results.append({"heading": h, "category": label, "confidence": conf})
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
