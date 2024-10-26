from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from improved_vision_system import process_image, save_to_excel
import os
import base64
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        # Process the image
        image = cv2.imread(file_path)
        processed_image, results = process_image(image)

        # Save results to Excel
        excel_file = save_to_excel(results)

        # Save processed image
        processed_filename = f"processed_{unique_filename}"
        processed_file_path = os.path.join(RESULTS_FOLDER, processed_filename)
        cv2.imwrite(processed_file_path, processed_image)

        # Convert processed image to base64 for display
        _, buffer = cv2.imencode('.jpg', processed_image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'message': 'Image processed successfully',
            'processed_image': img_str,
            'results': results,
            'excel_file': excel_file
        })

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('results', filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)