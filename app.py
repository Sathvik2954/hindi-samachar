#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ============================================================
# PORTABLE TESSERACT PATH (adjust if needed)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESSERACT_PATH = os.path.join(BASE_DIR, "tesseract_portable", "tesseract.exe")
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"  # fallback to system PATH

# ============================================================
# MODEL LOADING
# ============================================================
MODEL_DIR = os.path.join(BASE_DIR, "indic_news_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128


def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    with open(os.path.join(model_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    with open(os.path.join(model_dir, "id2label.json"), "r", encoding="utf-8") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
    config = AutoConfig.from_pretrained(model_dir)
    config.num_labels = len(label2id)
    config.label2id = label2id
    config.id2label = id2label
    model = AutoModelForSequenceClassification.from_config(config)
    state_dict = torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, id2label


print("LOADING MODEL...")
model, tokenizer, id2label = load_model_and_tokenizer(MODEL_DIR)
print("MODEL READY. LABELS:", list(id2label.values()))


# ============================================================
# CLASSIFICATION WITH CONFIDENCE SCORE
# ============================================================
def preprocess_text(text):
    text = " ".join(text.split())
    text = ".".join(text.split(".")[:2])
    return text


def classify_with_confidence(text):
    """Returns (predicted_label, confidence_percentage)"""
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
    return pred_label, round(confidence.item() * 100, 2)  # return as percentage


# ============================================================
# WEB SCRAPER (all headlines, no filters)
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
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(gray, lang="hin+eng")
    return text


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
# FLASK ROUTES (PAGES)
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


# ============================================================
# API ENDPOINTS (with confidence)
# ============================================================
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
