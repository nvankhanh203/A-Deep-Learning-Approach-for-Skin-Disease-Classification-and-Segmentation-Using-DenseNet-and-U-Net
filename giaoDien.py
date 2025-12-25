import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import DenseUNetClassifier  # Đảm bảo 'model.py' chứa class này
import matplotlib.pyplot as plt
from st_aggrid.shared import JsCode 

# ========== CẤU HÌNH ==========
st.set_page_config(page_title="Phân loại da", layout="centered")
st.title("🔬 Phân loại và phân vùng tổn thương da bằng mô hình học sâu DenseNet121 kết hợp U-net")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= LOAD MÔ HÌNH ========
@st.cache_resource
@st.cache_resource
def load_model():
    model = DenseUNetClassifier(n_classes_seg=3, n_classes_cls=4)
    model.load_state_dict(torch.load("C:\\Users\\TGDD\\Downloads\\benhngoaida\\best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ======= LABEL MAPPING =======
label_names = ['BKL', 'MEL', 'NV', 'NORMAL']

# ======= TIỀN XỬ LÝ ẢNH =======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========== TẢI ẢNH ==========
uploaded_file = st.file_uploader("📁 Tải ảnh tổn thương da", type=["jpg", "jpeg", "png","webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã chọn")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_out, cls_out = model(input_tensor)
        probs = F.softmax(cls_out, dim=1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()
        pred_label = label_names[pred_idx]
        confidence = probs.max()

    # ======= HIỂN THỊ DỰ ĐOÁN =======
    # 🎯 Badge kết quả
    st.markdown(
        f"<h3 style='display:inline'>🔎 Nhãn dự đoán: </h3> "
        f"<span style='background-color:#e6f4ea; color:#137333; font-weight:bold; "
        f"padding:5px 10px; border-radius:8px;'>{pred_label}</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"**Độ tin cậy dự đoán:** `{confidence*100:.2f}%`")
    st.progress(float(confidence)) 
    
    # === CẢNH BÁO NẾU CONFIDENCE THẤP ===
    if confidence < 0.4:
        st.error("⚠️ Mô hình không tự tin vào dự đoán này!")
    elif confidence < 0.7:
        st.warning("⚠️ Mô hình chưa chắc chắn")
    else:
        st.success("✅ Mô hình khá tự tin với dự đoán này")

    # === HIỂN THỊ XÁC SUẤT TỪNG LỚP ===
    st.markdown("### 📊 Xác suất dự đoán:")

    for i, prob in enumerate(probs):
        label = label_names[i]
        color = "#0f172a"  # màu xám đậm, đẹp trên nền sáng

        st.markdown(
            f"<div style='font-weight:600; font-size:15px; color:{color}; margin-top:10px;'>{label}</div>",
            unsafe_allow_html=True
        )


        bar_html = f"""
        <div style='
        background-color: #eee;
        border-radius: 8px;
        overflow: hidden;
        margin: 5px 0;
        height: 24px;
        '>
        <div style='
        width: {prob*100:.2f}%;
        background: linear-gradient(to right, #4ade80, #22c55e);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding-left: 10px;
        color: black;
        font-weight: bold;
        font-size: 14px;
        '>
            {prob*100:.2f}%
        </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)




    # === SEGMENTATION NẾU KHÔNG PHẢI NORMAL ===
    if pred_label != 'NORMAL':
        st.markdown("### 🧩 Vùng tổn thương được phân đoạn:")
        seg_mask = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(image.resize((224, 224)))
        ax[0].set_title("Ảnh gốc")
        ax[0].axis('off')

        ax[1].imshow(image.resize((224, 224)), alpha=0.6)
        ax[1].imshow(seg_mask, cmap='jet', alpha=0.4)
        ax[1].set_title("Overlay phân đoạn")
        ax[1].axis('off')

        st.pyplot(fig)