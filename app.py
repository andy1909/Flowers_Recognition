# app.py
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model

# --- KHỞI TẠO ỨNG DỤNG FLASK VÀ CÁC CẤU HÌNH ---
app = Flask(__name__)

# <<< THAY ĐỔI 1: Cập nhật tên file model cho đúng với file train.py >>>
# Tên file hơi khó hiểu (face_recognition) nhưng ta dùng đúng tên đã lưu
MODEL_FILE = 'face_recognition_model.h5' 

# <<< THAY ĐỔI 2: Cập nhật kích thước ảnh cho khớp với lúc huấn luyện >>>
IMG_HEIGHT = 64
IMG_WIDTH = 64

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# <<< THAY ĐỔI (Cải thiện): Dịch tên hoa sang tiếng Việt >>>
VIETNAMESE_NAMES = {
    'daisy': 'Hoa cúc dại',
    'dandelion': 'Hoa bồ công anh',
    'rose': 'Hoa hồng',
    'sunflower': 'Hoa hướng dương',
    'tulip': 'Hoa tulip'
}

# Tải mô hình AI một lần duy nhất khi server khởi động
print(f"Đang tải mô hình từ file '{MODEL_FILE}'...")
try:
    model = load_model(MODEL_FILE)
    print("Tải mô hình thành công.")
except Exception as e:
    print(f"Lỗi: Không thể tải được file mô hình. Chi tiết: {e}")
    model = None

# --- HÀM XỬ LÝ DỰ ĐOÁN ---
def predict_flower(image_bytes):
    # Chuyển byte ảnh thành numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Chuyển từ BGR (OpenCV) sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh về đúng kích thước mô hình yêu cầu
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    
    # Mở rộng chiều để tạo thành một batch (batch size = 1)
    # <<< THAY ĐỔI 3: Bỏ đi việc chia cho 255.0 vì model đã có lớp Rescaling >>>
    img_ready = np.expand_dims(img_resized, axis=0).astype('float32')
    
    # Thực hiện dự đoán
    preds = model.predict(img_ready)
    predicted_class_index = np.argmax(preds[0])
    
    # Lấy tên tiếng Anh từ chỉ số dự đoán được
    english_name = CLASS_NAMES[predicted_class_index]
    
    # Dịch sang tiếng Việt (nếu không tìm thấy thì trả về tên gốc)
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

# --- CHẠY SERVER ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)