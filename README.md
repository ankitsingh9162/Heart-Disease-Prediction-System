🫀 Heart Disease Prediction System
Machine Learning | Data Science | Streamlit Application | UCI Dataset
📌 Overview

This project is an end-to-end Heart Disease Prediction System built using supervised machine learning algorithms.
It analyzes key clinical factors such as age, cholesterol, resting BP, chest pain type, ECG results, heart rate, and more to determine whether a person is at risk of heart disease.

The system includes:
✔ Data preprocessing & EDA
✔ Model training & tuning (Logistic Regression, KNN, Random Forest)
✔ Performance comparison
✔ Deployment-ready Streamlit application (dark UI)
✔ Real-time and bulk CSV predictions

🚀 Key Features
🔍 Machine Learning Models Implemented

Logistic Regression – interpretable baseline

K-Nearest Neighbors (Tuned) – similarity-based learning

Random Forest (Tuned) – ⭐ highest accuracy & AUC

🖥️ Interactive Streamlit UI

Modern dark theme interface

Single patient prediction form

Bulk CSV upload for multi-record predictions

Probability visualization charts

Instant model inference using saved .pkl files

📊 Data Analysis Highlights

Heatmap for correlation analysis

Pairplot for feature relationships

Feature scaling using StandardScaler

GridSearchCV tuning for KNN & Random Forest

📂 Project Structure
📦 Heart-Disease-Prediction
├── app.py
├── scaler.pkl
├── logistic_model.pkl
├── knn_model.pkl
├── rf_model.pkl
├── heart_disease_10000_rows.csv
├── README.md
└── requirements.txt

🧠 Machine Learning Workflow

1️⃣ Load & explore dataset
2️⃣ Perform EDA (correlations, pairplot, distributions)
3️⃣ Split data into train & test sets
4️⃣ Feature scaling
5️⃣ Train ML models
6️⃣ Hyperparameter tuning
7️⃣ Compare model performance
8️⃣ Save best models
9️⃣ Build Streamlit UI for prediction

📊 Model Performance Summary
Model	Accuracy	AUC Score	Remarks
Logistic Regression	Good	Good	Strong baseline model
KNN (Tuned)	Moderate	Good	Works well with scaling
Random Forest	⭐ Highest	⭐ Highest	Best performing model

🏆 Final Recommended Model → Random Forest Classifier

▶️ How to Run Locally
🔧 Install dependencies
pip install -r requirements.txt

▶️ Start Streamlit App
streamlit run app.py

📁 Upload CSV for bulk prediction

The CSV must include all 13 clinical features in the correct order.

📥 Dataset

The dataset used is approx 3,000-record Heart Disease dataset inspired by the UCI Machine Learning Repository.

🛠️ Tech Stack

Python

NumPy, Pandas

Scikit-learn

Seaborn, Matplotlib

Streamlit

Joblib

📸 Screenshots (Optional)

You can include UI images here after uploading to GitHub.
Example:

![App UI](images/ui.png)

🌟 Future Enhancements

Deploy model on cloud (AWS / Azure)

Add deep learning models (ANN, CNN)

Integration with smart IoT health devices

Improved clinical interpretability

PDF Health Report Generator

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a pull request.

⭐ Support

If you like this project, please star this repository to encourage more work like this!

👨‍💻 Author

Developed by Ankit Kumar
