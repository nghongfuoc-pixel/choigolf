import json
import tempfile

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============ TẢI MODEL VÀ CONFIG (chỉ chạy 1 lần, cache lại) ============
@st.cache_resource
def load_model_and_config():
    model = tf.keras.models.load_model(
        "golf_swing_mobilenetv2.keras",
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input},
    )

    with open("class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)

    with open("preprocess_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    return model, class_names, config


def extract_frames(video_path, num_frames, img_size):
    """Trích num_frames frame cách đều nhau từ video, resize về img_size x img_size."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    target_set = set(frame_indices.tolist())

    frames = []
    current_idx = 0
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        current_idx += 1
    cap.release()

    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    if len(frames) == 0:
        return None

    return np.array(frames, dtype=np.uint8)


# ============ GIAO DIỆN ỨNG DỤNG ============
st.set_page_config(page_title="Nhận diện video Golf Swing", page_icon="🏌️")
st.title("🏌️ Nhận diện video Golf Swing")
st.write("Upload 1 video ngắn để kiểm tra xem đây có phải cảnh đánh golf hay không.")

model, class_names, config = load_model_and_config()
NUM_FRAMES = config["NUM_FRAMES"]
IMG_SIZE = config["IMG_SIZE"]

uploaded_file = st.file_uploader("Chọn file video (.avi, .mp4, .mov)", type=["avi", "mp4", "mov"])

if uploaded_file is not None:
    # Lưu file tạm để OpenCV đọc được (OpenCV cần đường dẫn file, không đọc trực tiếp từ bytes)
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.video(uploaded_file)

    with st.spinner("Đang phân tích video..."):
        frames = extract_frames(tmp_path, NUM_FRAMES, IMG_SIZE)

    if frames is None:
        st.error("Không đọc được video này. Thử file khác.")
    else:
        input_tensor = np.expand_dims(frames, axis=0)
        prob = model.predict(input_tensor, verbose=0)[0][0]

        predicted_label = class_names[1] if prob > 0.5 else class_names[0]
        confidence = prob if prob > 0.5 else 1 - prob

        st.subheader("Kết quả:")
        if predicted_label == "golf_swing":
            st.success(f"✅ Đây LÀ cảnh đánh golf (độ tin cậy: {confidence*100:.1f}%)")
        else:
            st.info(f"❌ Đây KHÔNG phải cảnh đánh golf (độ tin cậy: {confidence*100:.1f}%)")

        st.progress(float(confidence))
