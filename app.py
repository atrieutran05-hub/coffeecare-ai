import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Cấu hình giao diện
st.set_page_config(page_title="CoffeeCare AI", layout="centered")
st.title("☕ CoffeeCare AI - Chuyên gia Cà phê Đắk Lắk")
st.write("Sử dụng camera điện thoại để quét và nhận diện bệnh trên cây cà phê.")

# 1. Load mô hình Teachable Machine
@st.cache_resource
def load_ai_model():
    model = load_model("keras_model.h5")
    # Đọc file labels
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_ai_model()

# 2. Kích hoạt Camera điện thoại
#camera_image = st.camera_input("📸 Hướng camera vào lá hoặc cành cà phê cần kiểm tra:")
camera_image = st.file_uploader("Chọn ảnh", type=["jpg", "png"])

# 1. Tạo một mảng trống đúng định dạng của Teachable Machine
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 2. Đưa ảnh đã chuẩn hóa vào mảng
data[0] = normalized_image_array

# 3. Cho AI dự đoán
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# 4. Hiển thị kết quả chi tiết
st.success(f"Kết quả: {class_name[2:]}") # [2:] để cắt bỏ số thứ tự trong file labels
st.info(f"Độ tin cậy (Confidence): {confidence_score * 100:.2f}%")

if camera_image is not None:
        # 1. Mở và resize ảnh
        image = Image.open(camera_image).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # 2. Chuyển sang mảng
        image_array = np.asarray(image)
        
        # 3. CHỖ NÀY LÀ QUAN TRỌNG: Phải tạo ra biến này trước khi dùng nó
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # 4. Tạo thùng chứa và đổ dữ liệu vào
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # 5. Cho AI dự đoán
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # 6. Hiển thị kết quả
        st.success(f"Kết quả: {class_name[2:]}") 
        st.info(f"Độ tin cậy: {confidence_score * 100:.2f}%")
    # 5. Đưa ra giải pháp
    if disease_name.lower() == "rust":
        st.error("**Phát hiện: Bệnh Gỉ sắt (Coffee Leaf Rust)**")
        st.markdown("- **Triệu chứng:** Mặt dưới lá xuất hiện các đốm bột màu vàng cam. Lá bệnh sẽ rụng sớm.")
        st.markdown("- **Xử lý:** Cắt tỉa cành vô hiệu. Phun các loại thuốc gốc Đồng (Copper Hydroxide) để phòng trị.")
        
    elif disease_name.lower() == "red_spider_mite" or disease_name.lower() == "red spider mite":
        st.warning("**Phát hiện: Nhện đỏ gây hại (Red Spider Mite)**")
        st.markdown("- **Triệu chứng:** Lá mất màu xanh bóng, chuyển sang màu xám bạc hoặc lấm tấm vàng.")
        st.markdown("- **Xử lý:** Tưới nước phun mưa áp lực mạnh để rửa trôi. Sử dụng tinh dầu Neem sinh học.")
        
    elif disease_name.lower() == "healthy":
        st.success("**🎉 Không phát hiện mầm bệnh! Lá cà phê khỏe mạnh.**")
        st.markdown("- **Khuyến nghị:** Cây đang phát triển tốt. Tiếp tục duy trì chế độ chăm sóc hiện tại.")
        
    else:
        st.info("Hệ thống chưa nhận diện rõ. Vui lòng thử chụp lại gần và rõ nét hơn.")
