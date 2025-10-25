# 🧠 Violence Detection using Qwen2-VL + MobileNet-LSTM
> 🎥 **CCTV 영상 속 폭력 장면을 자동 인식하는 인공지능 모델**
>
> 본 프로젝트는 **다중 프레임 분석 기반 폭력 감지 시스템**으로,
> MobileNet + LSTM 모델로 학습된 폭력 분류 모델과
> Qwen2-VL 비전 언어 모델을 이용해 **사건 요약 문장 자동 생성**까지 수행합니다.
---
## 🚀 Project Overview
| 항목 | 설명 |
|------|------|
| **프로젝트명** | Violence Detection |
| **목표** | CCTV 또는 일반 영상에서 폭력 장면을 자동으로 탐지하고 사건 개요를 요약 |
| **핵심 기술** | MobileNetV2 + BiLSTM, Qwen2.5-VL-7B, HuggingFace Inference API |
| **프레임 단위 입력** | (16, 64, 64, 3) → 16장의 연속 프레임 입력 |
| **데이터셋** | [Real-Life Violence Situations Dataset (Kaggle)](https://www.kaggle.com/mohamedmustafa/real-life-violence-situations-dataset) |
---
## 🧩 System Pipeline
🎥 CCTV / Video Input
↓
🧱 Frame Extraction (16 frames per sequence)
↓
👁️ MobileNetV2 → Feature Extraction
↓
🧠 BiLSTM → Temporal Motion Analysis
↓
⚖️ Dense Layers → Violence / NonViolence Classification
↓
🗣️ Qwen2-VL → 사건 요약문 자동 생성

---

## ⚙️ Model Structure

| Stage | Layer | Output Shape | Description |
|--------|-------|---------------|--------------|
| Input | Frames (16, 64, 64, 3) | - | 16프레임 RGB 영상 입력 |
| CNN | MobileNetV2 (pretrained) | (16, 7×7×1280) | 프레임별 특징 추출 |
| Flatten + Dropout | - | (16, 1280) | 특징 벡터 정규화 |
| BiLSTM | 32 units × 2 | (64,) | 시간 흐름에 따른 움직임 학습 |
| Dense Layers | 256 → 128 → 64 → 32 | - | 고차원 특징 압축 |
| Output | Softmax(2) | (2,) | Violence / NonViolence 확률 |

---

## 🧠 Qwen2-VL 기반 사건 요약

모델은 폭력 감지 후, 프레임 이미지를 기반으로  
**6하원칙(누가, 어디서, 무엇을, 어떻게, 왜)** 형태의 사건 요약문을 자동 생성합니다.

예시👇  
👀 Input: 8장 CCTV 연속 프레임
🧾 Output:
“한 남성이 도로변에서 상대방을 밀치며 폭행하는 장면으로 보이며,
주변은 외부 도로 환경으로 추정됩니다. 사건은 즉시 신고가 필요합니다.”

- **사용 모델:** `Qwen/Qwen2.5-VL-7B-Instruct`
- **API:** HuggingFace Inference via `openai` compatible endpoint  
- **Base64 Encoding:** 로컬 이미지를 직접 인코딩하여 API 호출  

---

## 📊 Training Performance

| Metric | Score |
|--------|-------|
| **Train Accuracy** | 99.9% |
| **Validation Accuracy** | 93.9% |
| **Test Accuracy** | 96.0% |
| **Best Val Loss** | 0.35 |

✅ **Loss Function:** Categorical Crossentropy  
✅ **Optimizer:** SGD  
✅ **Batch Size:** 8  
✅ **Epochs:** 50  

---

---

## 🔍 Example Results

| Input 영상 | Predicted | Confidence |
|--------------|------------|-------------|
| `V_341.mp4` | Violence | 0.99998 |
| `NV_112.mp4` | NonViolence | 0.99993 |

📄 **Qwen2-VL 사건 요약 결과**
한 남성이 도로에서 상대방을 손으로 밀치며 폭행하는 모습이 포착되었습니다.
현장은 도로변 외부로 보이며, 즉각적인 제지가 필요해 보입니다.

---

## 🧰 Tech Stack

- **Python 3.9**
- **TensorFlow / Keras**
- **MobileNetV2 (ImageNet pretrained)**
- **HuggingFace Hub + Qwen2-VL**
- **OpenCV, NumPy, Matplotlib**
- **Scikit-learn**

---

## 📦 Installation & Run

```bash
# Clone the repo
git clone https://github.com/fortuneSeven/Violence-Detection.git
cd Violence-Detection

# Create environment
conda create -n violence python=3.9
conda activate violence

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook

🧭 Future Work
	•	🔁 Real-time inference via CCTV live feed
	•	💡 Add attention layer (Temporal or Spatial)
	•	🧩 Integrate TFLite for edge deployment
	•	🗣️ Multilingual event description generation

🧑‍💻 Author

Jeong Jihoon (정지훈)
Department of Artificial Intelligence, Kyonggi University
📧 GitHub: fortuneSeven
