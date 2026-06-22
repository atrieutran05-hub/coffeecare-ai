import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Tắt warning của TensorFlow để giao diện sạch hơn
tf.keras.backend.clear_session()

# Load model và labels một lần duy nhất
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('keras_model.h5')
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_model()

# Giao diện chính
st.title("CoffeeCare AI - Chuyên gia Cà phê Đắk Lắk ☕")
st.header("Nhận diện bệnh trên lá cà phê")

camera_image = st.file_uploader("Tải ảnh lá cà phê tại đây", type=["jpg", "png", "jpeg"])

if camera_image is not None:
    # 1. Mở và resize ảnh
    image = Image.open(camera_image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 2. Hiển thị ảnh đã tải lên
    st.image(image, caption="Ảnh bạn vừa tải lên", use_container_width=True)
    
    # 3. Chuẩn hóa ảnh để AI đọc được
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # 4. Tạo mảng dữ liệu
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # 5. Cho AI dự đoán
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # 6. Hiển thị kết quả
    disease_name = class_name[2:].strip()
    st.success(f"Kết quả dự đoán: {disease_name}") 
    st.info(f"Độ tin cậy: {confidence_score * 100:.2f}%")
    
    # 7. Logic cảnh báo bệnh
    if "rust" in disease_name.lower():
        st.warning("⚠️ Cảnh báo: Bệnh Gỉ Sắt (Coffee Leaf Rust). Cần kiểm tra vườn ngay!")
    elif "healthy" in disease_name.lower():
        st.success("✅ Cây trông có vẻ bình thường. Tiếp tục chăm sóc tốt nhé!")
    else:
        st.info("Kết quả hiện tại: " + disease_name)
