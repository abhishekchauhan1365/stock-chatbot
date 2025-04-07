import os
import random
import string
import logging
import pytesseract
import cv2
import numpy as np
import requests
import yfinance as yf
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from textblob import TextBlob
from fuzzywuzzy import process
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)

# Constants
NEWS_API_KEY = "ed0d7f95c4524f3bbaac2ece0ba3e48b"
STOCKS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN", "Google": "GOOGL",
    "Meta": "META", "Tesla": "TSLA", "Netflix": "NFLX", "Intel": "INTC", "Adobe": "ADBE",
    "AMD": "AMD", "Paypal": "PYPL", "Reliance": "RELIANCE.NS", "Tata Motors": "TATAMOTORS.NS",
    "Infosys": "INFY.NS", "HDFC Bank": "HDFCBANK.NS", "Wipro": "WIPRO.NS", "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS", "Bharti Airtel": "BHARTIARTL.NS"
}

# ---------- Utility Functions ---------- #

def random_string(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def resolve_symbol(user_input):
    names = list(STOCKS.keys())
    match, score = process.extractOne(user_input.title(), names)
    if score > 70:
        return STOCKS[match], f"Ohh, are you referring to {match}?"
    return user_input.upper(), None

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5d")
        if hist.empty:
            return None
        info = stock.info
        return {
            "name": info.get("longName", symbol),
            "price": hist["Close"].iloc[-1],
            "change": hist["Close"].iloc[-1] - hist["Close"].iloc[-2],
            "percent": ((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100,
            "history": hist
        }
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        return None

def create_chart(hist, symbol):
    try:
        plt.figure(figsize=(6, 4))
        hist["Close"].plot(title=f"{symbol} - Last 5 Days", grid=True)
        plt.ylabel("Price ($)")
        plt.tight_layout()
        filename = f"chart_{random_string()}.png"
        chart_path = os.path.join("static", filename)
        plt.savefig(chart_path)
        plt.close()
        return f"/static/{filename}"
    except Exception as e:
        logging.error(f"Chart generation failed: {e}")
        return ""

def get_news(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url)
        data = res.json()
        articles = data.get("articles", [])[:5]
        result = []
        for article in articles:
            title = article.get("title", "")
            desc = article.get("description", "")
            sentiment = analyze_sentiment(f"{title} {desc}")
            result.append({"text": f"{title}\n\n{desc}".strip(), "sentiment": sentiment})
        return result
    except Exception as e:
        logging.warning(f"News fetch failed: {e}")
        return []

def analyze_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "🟢 Positive"
    elif score < 0:
        return "🔴 Negative"
    else:
        return "⚪ Neutral"

def preprocess_image_for_ocr(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("L").filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        return img
    except Exception as e:
        logging.error(f"OCR preprocessing failed: {e}")
        return Image.open(image_path)

def analyze_chart_trend(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=10)
        if lines is None:
            return "🤔 Couldn't detect a clear trend."
        slopes = [(y2 - y1) / (x2 - x1) for [[x1, y1, x2, y2]] in lines if x2 != x1]
        avg_slope = np.mean(slopes)
        if avg_slope < -0.2:
            return "📉 This chart shows a *downtrend*. Be cautious or consider shorting."
        elif avg_slope > 0.2:
            return "📈 This chart shows an *uptrend*. Possible buying opportunity."
        else:
            return "🔄 The chart indicates a *sideways or consolidating* trend."
    except Exception as e:
        return f"⚠️ Chart analysis failed: {e}"

# ---------- Routes ---------- #

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_data", methods=["POST"])
def get_data():
    user_input = request.json.get("symbol", "").strip()
    if not user_input:
        return jsonify({"error": "Symbol is required"}), 400

    symbol, suggestion = resolve_symbol(user_input)
    stock_data = get_stock_data(symbol)
    if not stock_data:
        return jsonify({"error": "Invalid stock symbol"}), 404

    chart_url = create_chart(stock_data["history"], symbol)
    news = get_news(symbol)

    return jsonify({
        "name": stock_data["name"],
        "price": f"${stock_data['price']:.2f}",
        "change": f"{stock_data['change']:+.2f}",
        "percent": f"{stock_data['percent']:+.2f}%",
        "news": news,
        "chart": chart_url,
        "suggestion": suggestion
    })

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = f"{random_string()}_{secure_filename(image.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(filepath)

    try:
        processed_img = preprocess_image_for_ocr(filepath)
        extracted_text = pytesseract.image_to_string(processed_img)
        sentiment = analyze_sentiment(extracted_text)
        chart_trend = analyze_chart_trend(filepath)

        return jsonify({
            "extracted_text": extracted_text.strip(),
            "sentiment": sentiment,
            "trend": chart_trend,
            "image_path": f"/{filepath}"
        })
    except Exception as e:
        logging.error(f"OCR/Trend analysis failed: {e}")
        return jsonify({"error": str(e)}), 500

# ---------- Run ---------- #

if __name__ == "__main__":
    app.run(debug=True)
