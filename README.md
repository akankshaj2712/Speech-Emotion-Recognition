
# 🎙️ Speech Emotion Recognition (SER) using CNN-BiLSTM
**🔍 Project Overview**
This project focuses on classifying emotions from speech signals using a hybrid deep learning model that combines Convolutional Neural Networks (CNN) with Bidirectional LSTM (BiLSTM). Recognizing emotions like Anger, Happiness, Sadness, and Neutrality from vocal tone is crucial for applications in healthcare, customer service, virtual assistants, and more.

**📈 Business Context**

Understanding emotions in voice can enhance:

🧑‍⚕️ Mental health diagnostics

☎️ Customer support personalization

🕹️ Game character interactions

🧠 Human-computer interfaces

**📁 Dataset**

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset):

7,442 audio clips from 91 actors (48 male, 43 female)

Emotions: Angry, Disgust, Fear, Happy, Neutral, Sad

**🧪 Data Preprocessing & Feature Engineering**

1. Cleaning:

Validated audio readability

Filtered out noisy or low-quality samples

2. Feature Extraction:
Extracted multiple spectral and temporal features using Librosa:

MFCCs (40 Coefficients) – Captures timbral texture

Chroma – Highlights harmonic content

Mel Spectrogram – Time-frequency energy representation

Zero Crossing Rate (ZCR) – Indicates signal noisiness

RMSE (Energy) – Reflects vocal intensity

3. Augmentation Techniques:

Gaussian Noise Injection

Time Shifting

Pitch Alteration

Spectral Padding to 216 frames

4. Normalization: Scaled all features to [0, 1] range for stable learning.

**📊 Exploratory Data Analysis**

Class Imbalance: Balanced dataset with slight variations across emotions.

Correlation Heatmaps: Most MFCCs showed unique contributions, justifying deep model complexity.

Duration Analysis: Normalized input length to ensure consistent model input dimensions.

**🧠 Model Architecture**

CNN + BiLSTM Hybrid Model

Conv2D (64 filters, 3x3) + ReLU

MaxPooling2D (2x2)

BiLSTM (128 units) – captures bidirectional temporal dependencies

Dense + Softmax – multi-class output

**✅ Performance Metrics**

Metric	Value
Training Accuracy	94%
Validation Accuracy	88.3%
Avg Test Accuracy	92%
Validation Loss	0.42

High precision/recall for Angry and Happy

Confusion observed for Fear and Disgust due to overlapping spectral features

**🔍 Key Insights**

MFCCs, Chroma, and Mel features were highly effective for capturing emotional nuances.

CNN layers efficiently extracted spatial audio patterns, while BiLSTM captured temporal emotional trends.

Model overfitting was addressed through data augmentation and padding strategies.
![image](https://github.com/user-attachments/assets/7ea7f130-d3b3-45e0-ba6c-6bfe639ca1fd)

**🚀 Future Improvements**

Implement transfer learning (e.g., Wav2Vec)

Explore attention mechanisms for better emotion context

Apply hyperparameter tuning and dropout for regularization

**🛠️ Tech Stack**

Python, Keras, TensorFlow, Librosa, Scikit-learn

Matplotlib, Seaborn for visualization

CREMA-D dataset

**References**

HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

https://arxiv.org/abs/2106.07447

Efficient Emotion Recognition from Speech Using Deep Learning on Spectrograms

https://ieeexplore.ieee.org/document/7952153

Real-time speech emotion recognition using deep learning and data augmentation

https://link.springer.com/article/10.1007/s10462-024-11065-x

Real-time speech emotion recognition using deep learning and data augmentation

https://ieeexplore.ieee.org/document/10522204
