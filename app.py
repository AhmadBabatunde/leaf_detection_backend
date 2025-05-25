from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)


CORS(app)

# Load YOLO model
model = YOLO('best_0.34.pt', task='detect')
model.eval()

# Get class names from model (replace with your actual class names)
if hasattr(model, 'names'):
    id2class = model.names
else:

    # Class mapping (update with your actual classes)
    id2class = {'Corn_Cercospora_Leaf_Spot': 0, 'Corn_Common_Rust': 1, 'Corn_Healthy': 2, 
                'Corn_Northern_Leaf_Blight': 3, 'Corn_Streak': 4, 'Pepper_Bacterial_Spot': 5, 
                'Pepper_Cercospora': 6, 'Pepper_Early_Blight': 7, 'Pepper_Fusarium': 8, 
                'Pepper_Healthy': 9, 'Pepper_Late_Blight': 10, 'Pepper_Leaf_Blight': 11, 
                'Pepper_Leaf_Curl': 12, 'Pepper_Leaf_Mosaic': 13, 'Pepper_Septoria': 14, 
                'Tomato_Bacterial_Spot': 15, 'Tomato_Early_Blight': 16, 'Tomato_Fusarium': 17, 
                'Tomato_Healthy': 18, 'Tomato_Late_Blight': 19, 
                'Tomato_Leaf_Curl': 20, 'Tomato_Mosaic': 21, 'Tomato_Septoria': 22}

@app.route('/analyze', methods=['POST'])
@cross_origin(origin='*')
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read and process image
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    results = model(img, imgsz=640, conf=0.3, iou=0.4)[0]
    
    # Process results
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()

    # Draw bounding boxes
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        label = f"{id2class[cls]} {conf:.2f}"
        color = (0, 255, 0) if id2class[cls] == 'Healthy' else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Determine final result
    detected_classes = [id2class[cls] for cls in classes]
    has_defect = any(cls != 'Healthy' for cls in detected_classes)
    result = ', '.join(detected_classes) if has_defect else 'Healthy Leaf 🍃'

    return jsonify({
        'result': result,
        'image': img_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)