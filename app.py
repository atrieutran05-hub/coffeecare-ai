import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="CoffeeCare AI", layout="centered")
st.title("☕ CoffeeCare AI - Chuyên gia Cà phê Phú Yên")
st.write("Bật camera chụp trực tiếp lá cà phê để quét và nhận diện bệnh ngay lập tức.")

@st.cache_resource
def load_ai_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_ai_model()

camera_image = st.camera_input("Hãy đưa lá cà phê vào chính giữa khung hình và bấm chụp")

if camera_image is not None:
    
    image = Image.open(camera_image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.BICUBIC)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    with st.spinner("AI đang phân tích bức ảnh bạn vừa chụp..."):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        
        raw_class_name = class_names[index]
        disease_name = raw_class_name.split(" ", 1)[-1].strip().lower()
        confidence_score = prediction[0][index]

    st.info(f"🧬 Độ tin cậy (Confidence score): {confidence_score * 100:.2f}%")

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
        st.markdown("- **Khuyến nghị:** Tiếp tục duy trì chế độ bón phân và tưới nước định kỳ.")

    elif "background" in disease_name:
        st.info("🤖 **Hệ thống: Chưa phát hiện rõ lá cà phê**")
        st.markdown("- **Lý do:** Bạn đang chụp cảnh vật, mặt người, đồ vật linh tinh hoặc góc chụp quá xa.")
        st.markdown("- **Khuyến nghị:** Hãy đưa camera **lại gần hơn**, chụp rõ nét bề mặt của duy nhất một chiếc lá cà phê cần kiểm tra.")
    
    else:
        st.write(f"Kết quả phân tích: **{raw_class_name}**")

else:
    st.write("👋 Đang đợi bạn bấm nút chụp ảnh trên màn hình để bắt đầu phân tích đó!")
