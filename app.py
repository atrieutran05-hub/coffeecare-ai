import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Cấu hình app
st.set_page_config(page_title="CoffeeCare AI", page_icon="☕")
st.title("CoffeeCare AI - Chẩn đoán bệnh cà phê ☕")

# Load model (Sử dụng cache để không load lại nhiều lần)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('keras_model.h5')
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_model()

# Dictionary chứa biện pháp xử lý
treatment_data = {
    "rust": {
        "title": "Bệnh Gỉ Sắt",
        "cure": "Phun thuốc gốc Đồng (Copper Hydroxide) hoặc Azoxystrobin. Cắt tỉa cành bị bệnh nặng để tránh lây lan.",
        "symptom": "Các đốm bột màu vàng cam hoặc đỏ gạch trên mặt dưới lá."
    },
    "healthy": {
        "title": "Cây Khỏe Mạnh",
        "cure": "Tiếp tục chế độ chăm sóc hiện tại. Duy trì tưới nước và bón phân cân đối.",
        "symptom": "Lá xanh, bóng, không đốm lạ."
    },
    "cercospora": {
        "title": "Bệnh Đốm Mắt Cua",
        "cure": "Sử dụng thuốc trừ nấm gốc Đồng hoặc Carbendazim. Cải thiện độ thông thoáng cho vườn.",
        "symptom": "Đốm nâu sẫm, hình tròn với viền vàng bao quanh."
    }
}

# CHỨC NĂNG CHỤP ẢNH TRỰC TIẾP
camera_image = st.camera_input("Chụp ảnh lá cà phê của ní tại đây")

if camera_image is not None:
    # Xử lý ảnh
    image = Image.open(camera_image).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Dự đoán
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    disease_name = class_names[index][2:].strip().lower()

    # HIỂN THỊ KẾT QUẢ VỚI BỘ LỌC TIN CẬY
    # Nếu AI tự tin dưới 75%, cảnh báo ní chụp lại cho rõ
    if confidence_score < 0.75:
        st.warning("⚠️ AI chưa nhận diện rõ. Vui lòng chụp gần và lấy nét hơn!")
        st.info(f"Độ tự tin hiện tại: {confidence_score * 100:.1f}%")
    else:
        st.success(f"Kết quả: **{disease_name.upper()}**")
        st.write(f"Độ tự tin: {confidence_score * 100:.1f}%")

        # Tra cứu biện pháp từ Dictionary
        found = False
        for key in treatment_data:
            if key in disease_name:
                data_info = treatment_data[key]
                st.subheader(f"✅ Biện pháp cho: {data_info['title']}")
                st.markdown(f"**Triệu chứng:** {data_info['symptom']}")
                st.markdown(f"**Cách xử lý:** {data_info['cure']}")
                found = True
                break
        
        if not found:
            st.info("Hiện tại chưa có hướng dẫn chi tiết cho bệnh này trong kho dữ liệu.")
