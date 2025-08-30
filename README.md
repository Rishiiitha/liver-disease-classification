# 🩺 Liver Disease Classification App  

A **Streamlit web app** that predicts whether a person is at **risk of liver disease** based on clinical parameters. The model is trained on the [Indian Liver Patient Dataset (ILPD)](https://archive.ics.uci.edu/dataset/60/indian+liver+patient+dataset) and evaluated using multiple machine learning algorithms.  

---

## 🚀 Features
- Exploratory Data Analysis (EDA) with visualizations (Seaborn/Matplotlib).  
- Data preprocessing: handling imbalance with **SMOTE**, scaling features.  
- Compared ML models: Logistic Regression, Random Forest, XGBoost, SVM, Naive Bayes, Gradient Boosting, AdaBoost, etc.  
- Model selection based on **Recall Score** (prioritizing detection of diseased cases).  
- Final trained model + scaler saved in `.pkl` file.  
- **Streamlit frontend** for user-friendly predictions.  
- Risk levels: **Low / Borderline / High**.  

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost)  
- **Streamlit** for web app  
- **Matplotlib / Seaborn** for visualization  

---

## 📂 Project Structure
├── data/
│ └── liver.csv # original dataset
├── notebooks/
│ └── liver_disease_analysis.ipynb
├── models/
│ └── liver_model.pkl # trained ML model
│ └── scaler.pkl # feature scaler
├── app.py # Streamlit app
├── README.md # project readme
└── requirements.txt # dependencies

---

## ⚙️ Installation & Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/liver-disease-classification.git
   cd liver-disease-classification

2.Create virtual environment & install dependencies:
  ```pip install -r requirements.txt
3. Run the app:
  ```streamlit run app.py

📊 Model Performance

## Evaluation metric: Recall (to minimize false negatives).

Best model selected: ( XGBoost)

## 🌐 Deployment
The app is deployed on **Streamlit Cloud**.  

👉 [Click here to use the Liver Disease Classification App](https://your-streamlit-link.streamlit.app)
