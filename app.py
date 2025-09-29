import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model


app = Flask(__name__)


MODEL_FILE = 'flower_recognition_model.h5' 


IMG_HEIGHT = 64
IMG_WIDTH = 64


CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

VIETNAMESE_NAMES = {
    'daisy': 'daisy',
    'dandelion': 'dandelion',
    'rose': 'rose',
    'sunflower': 'sunflower',
    'tulip': 'tulip'
}


print(f"Đang tải mô hình từ file '{MODEL_FILE}'...")
try:
    model = load_model(MODEL_FILE)
    print("Tải mô hình thành công.")
except Exception as e:
    print(f"Lỗi: Không thể tải được file mô hình. Chi tiết: {e}")
    model = None


def predict_flower(image_bytes):
   
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
   
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
   
    img_ready = np.expand_dims(img_resized, axis=0).astype('float32')
    
    # Thực hiện dự đoán
    preds = model.predict(img_ready)
    predicted_class_index = np.argmax(preds[0])
    
    # Lấy tên tiếng Anh từ chỉ số dự đoán được
    english_name = CLASS_NAMES[predicted_class_index]
    
    # Dịch sang tiếng Việt
    vietnamese_name = VIETNAMESE_NAMES.get(english_name, english_name.capitalize())
    
    return vietnamese_name

# --- CÁC ĐƯỜNG DẪN (ROUTES) CỦA WEB ---

@app.route('/', methods=['GET'])
def index():
    # Trả về file HTML giao diện chính
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Mô hình chưa được tải, vui lòng kiểm tra lỗi server.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file nào được gửi lên.'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file nào.'}), 400
        
    try:
        # Đọc file ảnh dưới dạng byte
        img_bytes = file.read()
        
        # Gọi hàm dự đoán
        prediction = predict_flower(img_bytes)
        
        # Trả về kết quả dưới dạng JSON
        return jsonify({
            'prediction': prediction
        })

    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        return jsonify({'error': 'Đã xảy ra lỗi trong quá trình xử lý ảnh.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
