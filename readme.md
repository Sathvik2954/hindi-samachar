# 📰 Hindi News Classification System

A full-stack machine learning application that classifies Hindi news headlines into predefined categories using a fine-tuned **IndicBERTv2** model. The system integrates **text classification, real-time web scraping, and OCR-based extraction**, providing a complete end-to-end NLP solution.

---

## 📌 Overview

This project demonstrates how transformer-based models can be applied to **Indic language processing**, specifically for Hindi news classification. It combines:

* Deep Learning (IndicBERTv2)
* Backend API (Flask)
* OCR (EasyOCR)
* Web Scraping (BeautifulSoup)
* Frontend UI (HTML/CSS)

The application allows users to:

* Classify manually entered text
* Fetch and classify live news headlines
* Extract and classify text from images or PDFs

---

## 🚀 Live Demo

🌐 **Deployed Web App:**
https://huggingface.co/spaces/Sathvik2954/hindi-samachar-2

---

## 📂 Dataset

📊 **Dataset Used:**
https://huggingface.co/datasets/ai4bharat/indic_glue

---

## 📦 Model Download

The trained model is hosted on Hugging Face.
To upload or manage the model, you can use:

```python
from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path=".", repo_id="Sathvik2954/hindi-news-model", repo_type="model")
```

> ✅ The model is automatically loaded from Hugging Face during runtime.

---

## 🚀 Features

### 🧠 1. Text Classification

* Input any Hindi headline
* Returns:

  * Predicted category
  * Confidence score (%)
* Fast and lightweight inference

---

### 🌐 2. Web Scraper + Classification

* Scrapes live headlines from:

  * Amar Ujala
  * Dainik Jagran
  * Navbharat Times
  * BBC Hindi
* Automatically classifies each headline
* Allows filtering by category
* Removes duplicate headlines

---

### 🖼️ 3. OCR + Classification

* Upload:

  * Images (.jpg, .png)
  * PDFs
* Extracts Hindi text using EasyOCR
* Identifies headings
* Classifies each extracted heading

---

## 🧠 Model Details

### 📊 Dataset

* Source: IndicGLUE (bbca.hi subset)
* Original categories: 14+

### 🏷️ Selected Categories

* News
* Entertainment
* Sport
* Science
* International

---

### ⚙️ Preprocessing

* Text normalization (remove extra spaces)
* Sentence truncation (first 1–2 sentences)
* Label normalization

---

### ⚖️ Data Balancing

* Hybrid approach:

  * Undersampling large classes
  * Oversampling smaller classes
* Ensures balanced training distribution

---

### 🏆 Model Performance

| Model               | Accuracy   |
| ------------------- | ---------- |
| BERT (Multilingual) | 73.98%     |
| XLM-RoBERTa         | 78.06%     |
| **IndicBERTv2**     | **79.57%** |

👉 IndicBERTv2 performed best due to its specialization in Indic languages.

---

## 🛠️ Tech Stack

### 🔧 Backend

* Flask
* PyTorch
* Transformers (HuggingFace)

### 🧠 Model

* IndicBERTv2

### 📄 OCR

* EasyOCR
* PIL

### 🌐 Web Scraping

* Requests
* BeautifulSoup

### 🎨 Frontend

* HTML5
* CSS3 (Brutalist Design)
* Vanilla JavaScript

### 🚀 Deployment

* Hugging Face Spaces (Docker)

---

## 📌 Key Highlights

* ✅ End-to-end ML pipeline (training → deployment)
* ✅ Real-time news classification
* ✅ Multi-modal input (text + web + OCR)
* ✅ Confidence-based predictions
* ✅ Lightweight and responsive UI
* ✅ Modular and scalable architecture

---

## 🔮 Future Improvements

* 🔹 Batch inference for faster processing
* 🔹 Improve OCR heading detection
* 🔹 Add multilingual support (Hindi + English)
* 🔹 Add confidence visualization (charts/bars)
* 🔹 Model comparison feature (BERT vs IndicBERT)

---

## 👨‍🏫 Project Guide

**Mr. Panigrahi Srikanth**

Assistant Professor
Department of AIML
Chaitanya Bharathi Institute of Technology (Autonomous)
Gandipet – 500075

---

## 👨‍💻 Authors

* Parin Uday
* Parna Revanth Kumar
* Peesari Sathvik Reddy

---

## 📄 License

This project is developed for **academic and educational purposes only**.

---

## ⭐ Acknowledgements

* IndicGLUE Dataset (AI4Bharat)
* HuggingFace Transformers
* EasyOCR
