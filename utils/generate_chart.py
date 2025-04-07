import cv2
import numpy as np
from pytesseract import image_to_string
from textblob import TextBlob

def extract_text_from_chart(image_path):
    try:
        text = image_to_string(image_path)
        return text.strip()
    except Exception as e:
        return "Error reading image."

def analyze_chart_pattern(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, threshold1=30, threshold2=100)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=10)
        if lines is None:
            return "🤔 Could not detect a clear trend in the chart."

        slopes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)

        avg_slope = np.mean(slopes)
        if avg_slope < -0.2:
            return "📉 This chart shows a *downtrend*. Be cautious or consider shorting."
        elif avg_slope > 0.2:
            return "📈 This chart shows an *uptrend*. Possible buying opportunity."
        else:
            return "🔄 The chart indicates a *sideways or consolidating* trend."
    except Exception as e:
        return "⚠️ Error analyzing chart trend."
