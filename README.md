# Skin Disease Detection using DenseNet and U-Net

This project applies Deep Learning techniques to **classify and segment skin diseases** from dermatology images using **DenseNet** and **U-Net**.

The approach combines **image classification** and **lesion segmentation** to support automated analysis of skin diseases.

---

## 📌 Models
- **DenseNet**: Skin disease classification  
- **U-Net**: Skin lesion segmentation  

---

## 📂 Dataset
- Dermoscopy images from publicly available dermatology datasets.
- Ground truth masks are provided for lesion segmentation.

⚠️ **Note:**  
> In this codebase, **only a subset of skin diseases is used**, and **not all disease categories available in the original dataset are included**.  
>  
> The dataset is intentionally filtered to:
> - Simplify experiments  
> - Reduce class imbalance  
> - Focus on representative skin diseases  

You are welcome to **extend this project by using the full dataset** or adding more disease classes to train models capable of recognizing **a wider range of skin diseases**.

---

## ⚙️ Training
- **Framework:** PyTorch  
- **Loss Functions:**
  - Classification: Cross Entropy Loss  
  - Segmentation: Dice Loss / Binary Cross Entropy (BCE) + Dice Loss  

---

## 📊 Evaluation Metrics
- **Classification:** Accuracy, Precision, Recall, F1-score  
- **Segmentation:** Dice Score, Intersection over Union (IoU)  

---

## 🎯 Objectives
- Apply DenseNet for skin disease classification  
- Use U-Net for accurate lesion segmentation  
- Evaluate model performance using standard metrics  
- Provide a baseline for further research in dermatological image analysis  

---

## 🧠 Author
**Nguyễn Văn Khánh**

---

✨ *You can use this dataset and codebase to train models that recognize more skin diseases.  
Wishing you success in your research and experiments!* 🚀
