import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Cấu hình giao diện app
st.set_page_config(page_title="CoffeeCare AI", layout="centered")
st.title("☕ CoffeeCare AI - Chuyên gia Cà phê Đắk Lắk")
st.write("Sử dụng camera điện thoại hoặc tải ảnh lên để nhận diện bệnh trên cây cà phê.")

# 1. Hàm Load mô hình AI và nhãn labels
@st.cache_resource
def load_ai_model():
    # Load model từ file h5
    model = tf.keras.models.load_model("keras_model.h5")
    # Đọc tên các lớp từ file labels.txt
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_ai_model()

# 2. Vùng upload ảnh hoặc chụp từ camera
camera_image = st.file_uploader("Chọn ảnh lá cà phê cần kiểm tra", type=["jpg", "jpeg", "png"])

# TOÀN BỘ LOGIC XỬ LÝ CHỈ CHẠY KHI NGƯỜI DÙNG ĐÃ TẢI ẢNH LÊN
if camera_image is not None:
    # Hiển thị ảnh cho người dùng xem
    st.image(camera_image, caption="Ảnh đã tải lên", use_container_width=True)
    
    # Xử lý và resize ảnh về chuẩn 224x224 như lúc train trên Teachable Machine
    image = Image.open(camera_image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Chuyển ảnh thành mảng numpy và chuẩn hóa dữ liệu
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Tạo mảng chứa dữ liệu đầu vào cho model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Tiến hành dự đoán bệnh
    with st.spinner("AI đang phân tích bức ảnh..."):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        
        # Cắt chuỗi để lấy tên bệnh sạch (Ví dụ: "0 Rust" -> lấy chữ "rust")
        raw_class_name = class_names[index]
        disease_name = raw_class_name.split(" ", 1)[-1].strip().lower()
        confidence_score = prediction[0][index]

    # Hiển thị độ tin cậy của AI
    st.info(f"🧬 Độ tin cậy (Confidence score): {confidence_score * 100:.2f}%")

    # 3. Phân loại và đưa ra giải pháp dựa trên 4 lớp của AI
    if "rust" in disease_name:
        st.error("⚠️ **Phát hiện: Bệnh Gỉ Sắt (Coffee Leaf Rust)**")
        st.markdown("- **Triệu chứng:** Mặt dưới lá xuất hiện các vết bệnh lốm đốm bột màu vàng cam.")
        st.markdown("- **Biện pháp xử lý:** Cắt tỉa cành thông thoáng, gom lá bệnh đem tiêu hủy. Phun thuốc gốc Đồng để phòng trị.")

    elif "spider" in disease_name or "mite" in disease_name:
        st.warning("⚠️ **Phát hiện: Nhện Đỏ gây hại (Red Spider Mite)**")
        st.markdown("- **Triệu chứng:** Lá bị mất màu xanh bóng ban đầu, chuyển sang màu rám bạc hoặc lấm tấm vàng nhỏ.")
        st.markdown("- **Biện pháp xử lý:** Phun nước áp lực mạnh để rửa trôi nhện, sử dụng các loại tinh dầu hữu cơ hoặc thuốc đặc trị nhện.")

    elif "healthy" in disease_name:
        st.success("✅ **Kết quả: Lá cà phê Khỏe Mạnh**")
        st.markdown("- **Đánh giá:** Cây đang ở trạng thái tốt, không phát hiện dấu hiệu sâu bệnh nguy hiểm.")
        st.markdown("- **Khuyên nghị:** Tiếp tục duy trì chế độ bón phân và tưới nước định kỳ.")

    elif "background" in disease_name:
        st.info("🤖 **Hệ thống: Chưa phát hiện rõ lá cà phê**")
        st.markdown("- **Lý do:** Bạn đang chụp cảnh vật, mặt người, đồ vật linh tinh hoặc góc chụp quá xa.")
        st.markdown("- **Khuyên nghị:** Hãy đưa camera **lại gần hơn**, chụp rõ nét bề mặt của duy nhất một chiếc lá cà phê cần kiểm tra.")
    
    else:
        # Đề phòng trường hợp tên nhãn bị gõ sai trên Teachable Machine
        st.write(f"Kết quả phân tích: **{raw_class_name}**")

else:
    # Thông báo khi app đang ở trạng thái chờ ảnh
    st.write("👋 Đang đợi bạn tải ảnh lên để bắt đầu đó!")
