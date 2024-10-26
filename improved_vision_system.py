import cv2
import torch
import numpy as np
import pytesseract
from datetime import datetime
import re
import pandas as pd
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image):
    results = model(image)
    return results.pandas().xyxy[0]



def preprocess_for_ocr(image):
    """Preprocess image to improve OCR accuracy"""
    # Resize image to improve OCR
    height = 1000
    aspect_ratio = height / image.shape[0]
    width = int(image.shape[1] * aspect_ratio)
    image = cv2.resize(image, (width, height))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get black text on white background
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Apply adaptive thresholding as backup
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return thresh, adaptive_thresh


def extract_expiry_date(crop):
    """Extract expiry date with improved preprocessing and pattern matching"""
    # Configure tesseract to look for digits and common date formats
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/-'

    # Preprocess image
    thresh, adaptive_thresh = preprocess_for_ocr(crop)

    # Try different preprocessed images
    text = ''
    for img in [crop, thresh, adaptive_thresh]:
        text += pytesseract.image_to_string(img, config=custom_config) + ' '

    # Common date patterns
    date_patterns = [
        r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',  # DD/MM/YYYY or MM/DD/YYYY
        r'\b(?:\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b',  # YYYY/MM/DD
        r'(?:EXP|BEST BEFORE|BB).*?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # Dates with prefixes
        r'(?:EXPIRY|USE BY|BEST BY).*?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'  # More prefixes
    ]

    # Try each pattern
    for pattern in date_patterns:
        dates = re.findall(pattern, text, re.IGNORECASE)
        if dates:
            return dates[0]

    return 'Not detected'


def detect_text(crop):
    """General text detection function for product information"""
    # Preprocess image
    thresh, adaptive_thresh = preprocess_for_ocr(crop)

    # Configure tesseract for general text
    custom_config = r'--oem 3 --psm 6'

    # Try different preprocessed images
    all_text = []
    for img in [crop, thresh, adaptive_thresh]:
        text = pytesseract.image_to_string(img, config=custom_config)
        all_text.append(text.strip())

    # Combine results, removing duplicates
    combined_text = ' '.join(set(filter(None, all_text)))
    return combined_text if combined_text else 'No text detected'

def assess_damage(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_pixel_percentage = (thresh > 0).mean() * 100
    return dark_pixel_percentage > 10

def assess_freshness(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_lower = np.array([25, 52, 72])
    green_upper = np.array([102, 255, 255])
    mask = cv2.inRange(hsv, green_lower, green_upper)
    green_percentage = (mask > 0).mean() * 100
    if green_percentage > 50:
        return "Fresh"
    elif green_percentage > 20:
        return "Moderately Fresh"
    else:
        return "Not Fresh"

def process_image(image):
    detections = detect_objects(image)
    results = []
    product_count = {}

    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
        crop = image[y1:y2, x1:x2]

        product = detection['name']
        confidence = detection['confidence']
        expiry_date = extract_expiry_date(crop)
        is_damaged = assess_damage(crop)
        freshness = assess_freshness(crop)

        if product in product_count:
            product_count[product] += 1
        else:
            product_count[product] = 1

        result = {
            'Product': product,
            'Confidence': f"{confidence:.2f}",
            'Expiry Date': expiry_date,
            'Is Damaged': 'Yes' if is_damaged else 'No',
            'Freshness': freshness,
            'Count': product_count[product]
        }
        results.append(result)

        # Draw bounding box
        color = (0, 255, 0) if not is_damaged else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{product} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image, results

def save_to_excel(results):
    df = pd.DataFrame(results)
    df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = f"product_inspection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    file_path = os.path.join('results', filename)
    df.to_excel(file_path, index=False)

    return filename