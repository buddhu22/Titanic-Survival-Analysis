import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Titanic Survival Dashboard", page_icon="🚢", layout="wide")

# --- LOAD DATA & TRAIN MODEL ---
@st.cache_data
def load_and_prep_data():
    # Load dataset
    df = sns.load_dataset('titanic')
    
    # Cleaning
    cols_to_drop = ['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive']
    df = df.drop(columns=cols_to_drop)
    
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df = df.drop_duplicates()
    
    # Feature Engineering
    df['FamilySize'] = df['sibsp'] + df['parch'] + 1
    df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
    
    return df

@st.cache_resource
def train_model(df):
    # Prepare data for modeling
    model_df = df.copy()
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    model_df['sex'] = le_sex.fit_transform(model_df['sex'])
    model_df['embarked'] = le_embarked.fit_transform(model_df['embarked'])
    
    features = ['pclass', 'sex', 'age', 'fare', 'embarked', 'FamilySize', 'IsAlone']
    X = model_df[features]
    y = model_df['survived']
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    
    return rf, le_sex, le_embarked, features

df = load_and_prep_data()
rf_model, le_sex, le_embarked, features = train_model(df)

# --- HEADER ---
st.title("🚢 Titanic Survival Analysis & Prediction Dashboard")
st.markdown("Explore the demographics of the Titanic passengers and use Machine Learning to predict survival chances.")

# --- KPI CARDS ---
st.header("Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

total_passengers = len(df)
total_survivors = df['survived'].sum()
survival_rate = (total_survivors / total_passengers) * 100
avg_fare = df['fare'].mean()

col1.metric("Total Passengers", f"{total_passengers}")
col2.metric("Total Survivors", f"{total_survivors}")
col3.metric("Overall Survival Rate", f"{survival_rate:.1f}%")
col4.metric("Average Ticket Fare", f"${avg_fare:.2f}")

st.markdown("---")

# --- INTERACTIVE DASHBOARD ---
st.header("Interactive Data Exploration")

# Filters
st.sidebar.header("Filter Data")
selected_class = st.sidebar.multiselect("Select Passenger Class", options=[1, 2, 3], default=[1, 2, 3])
selected_gender = st.sidebar.multiselect("Select Gender", options=['male', 'female'], default=['male', 'female'])

# Apply filters
filtered_df = df[(df['pclass'].isin(selected_class)) & (df['sex'].isin(selected_gender))]

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Survival Rate by Class & Gender")
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=filtered_df, x='pclass', y='survived', hue='sex', ci=None, palette="Set2", ax=ax)
        ax.set_ylabel("Survival Rate")
        st.pyplot(fig)
    else:
        st.warning("No data matches the selected filters.")

with col_b:
    st.subheader("Age Distribution of Passengers")
    if not filtered_df.empty:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.histplot(data=filtered_df, x='age', hue='survived', multiple='stack', bins=20, palette="viridis", ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("No data matches the selected filters.")

st.markdown("---")

# --- PREDICTION SECTION ---
st.header("🔮 Survival Predictor (Machine Learning)")
st.markdown("Enter your details below to see if the Random Forest model predicts you would have survived the Titanic disaster.")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    p_class = st.selectbox("Passenger Class", [1, 2, 3])
    p_sex = st.selectbox("Gender", ["male", "female"])
    p_age = st.slider("Age", 0, 100, 30)

with pred_col2:
    p_sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    p_parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    p_fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=30.0)

with pred_col3:
    p_embarked = st.selectbox("Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])

if st.button("Predict Survival", type="primary"):
    # Preprocess input
    sex_encoded = le_sex.transform([p_sex])[0]
    port_map = {"C (Cherbourg)": "C", "Q (Queenstown)": "Q", "S (Southampton)": "S"}
    embarked_encoded = le_embarked.transform([port_map[p_embarked]])[0]
    
    family_size = p_sibsp + p_parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    input_data = pd.DataFrame([[p_class, sex_encoded, p_age, p_fare, embarked_encoded, family_size, is_alone]], 
                              columns=features)
    
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][1] * 100
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"🎉 **You Survived!** (Estimated Probability: {probability:.1f}%)")
    else:
        st.error(f"💀 **You Did Not Survive.** (Estimated Probability of Survival: {probability:.1f}%)")
