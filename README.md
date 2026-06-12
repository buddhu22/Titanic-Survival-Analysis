# Titanic Survival Analysis: Data Storytelling with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 📌 Project Overview
This project is an end-to-end Data Science and Machine Learning portfolio piece based on the infamous Titanic dataset. The goal is to analyze the demographics of the passengers, discover what factors most heavily influenced survival (Data Storytelling), and build a predictive Machine Learning model to determine survival chances.

## 🎯 Objectives
- **Data Cleaning & Preprocessing:** Handle missing values, outliers, and categorical encoding.
- **Exploratory Data Analysis (EDA):** Use Seaborn and Matplotlib to uncover patterns in survival rates across Gender, Class, Age, and Family Size.
- **Data Storytelling:** Translate statistical findings into clear, business-focused insights.
- **Machine Learning:** Train, evaluate, and compare multiple classification models (Logistic Regression, Decision Tree, Random Forest) to predict survival.
- **Interactive Dashboard:** Build a Streamlit web application to allow users to interact with the data and the prediction model.

## 📊 Dataset Information
The dataset used is the well-known Titanic dataset, which includes the following key features:
- `Survived`: 0 = No, 1 = Yes
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex`: Gender
- `Age`: Age in years
- `SibSp`: # of siblings / spouses aboard the Titanic
- `Parch`: # of parents / children aboard the Titanic
- `Fare`: Passenger fare
- `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🛠️ Technologies Used
- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Random Forest, Decision Tree, Logistic Regression)
- **Web Application:** Streamlit
- **Environment:** Jupyter Notebook

## 📈 Key Results & Insights
1. **Gender was the strongest predictor:** Women had a ~75% survival rate compared to ~20% for men.
2. **Socio-economic class mattered:** 1st class passengers survived at more than double the rate of 3rd class passengers.
3. **Random Forest performed best:** The Random Forest Classifier achieved the highest accuracy and F1 score among the tested models.
4. **Feature Importance:** The model confirmed that Gender, Fare, and Age were the most critical factors determining survival.

## 🚀 How to Run the Project

### 1. Clone the repository (or download the folder)
```bash
cd Titanic_Survival_Project
```

### 2. Install dependencies
Ensure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
To view the data storytelling and model training process:
```bash
jupyter notebook notebooks/Titanic_Data_Storytelling_and_ML.ipynb
```

### 4. Run the Streamlit Dashboard
To interact with the visualizations and test the predictive model:
```bash
streamlit run app/app.py
```

## 📁 Project Structure
```
Titanic_Survival_Project/
│
├── app/                  # Streamlit dashboard application
│   └── app.py
├── notebooks/            # Jupyter notebooks for EDA and ML
│   └── Titanic_Data_Storytelling_and_ML.ipynb
├── models/               # Saved machine learning models
│   └── model.pkl
├── reports/              # Career materials and summaries
│   └── career_materials.md
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```
