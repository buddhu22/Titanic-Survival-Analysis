import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import pickle
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Titanic Survival Prediction Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN STYLE ---
st.markdown("""
<style>
    .main {
        background-color: #f9fbfd;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
    }
    .css-1r6g72y {
        background-color: #1e293b;
    }
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    div[data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    .badge-survived {
        background-color: #22c55e;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 20px;
    }
    .badge-died {
        background-color: #ef4444;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 20px;
    }
</style>
""", unsafe_style:="True")

# --- HELPER FUNCTIONS ---
@st.cache_data
def get_clean_data():
    df = sns.load_dataset('titanic')
    # Basic cleaning matching our notebook analysis
    cols_to_drop = ['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df = df.drop_duplicates()
    return df

def load_ml_model():
    model_path = os.path.join('models', 'model.pkl')
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure `models/model.pkl` is generated and located correctly.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

# Load data and model
df = get_clean_data()
model_data = load_ml_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚢 Titanic Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate Project",
    ["Project Overview", "Data Exploration", "Visualizations", "Survival Prediction", "Key Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **About this app:**\n"
    "This web application presents the exploratory data analysis and a Random Forest prediction tool built using the historical Titanic dataset."
)

# ==========================================
# 1. PROJECT OVERVIEW PAGE
# ==========================================
if page == "Project Overview":
    st.title("🚢 Titanic Survival Prediction Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Project Objectives & Background")
        st.write(
            "The sinking of the RMS Titanic on April 15, 1912, is one of the most famous maritime disasters in history. "
            "Of the estimated 2,224 passengers and crew aboard, more than 1,500 died because there were not enough lifeboats. "
            "While survival contained a degree of luck, demographic factors played an overwhelming role in who lived and who perished."
        )
        st.write(
            "**Goal:** Analyze passenger demographics to find key patterns in survival rates, and build a Machine Learning model "
            "to predict if a passenger would survive the disaster based on features like class, age, gender, and family size."
        )
        
        st.subheader("Model Overview")
        st.write(
            "We trained a **Random Forest Classifier** on the cleaned passenger list. "
            "Random Forest is an ensemble learning method that builds multiple decision trees and merges them together "
            "to get a more accurate and stable prediction."
        )
        
    with col2:
        st.subheader("Model Performance")
        st.metric("Algorithm Used", "Random Forest")
        st.metric("Model Accuracy", "81.4%")
        st.metric("Primary Features", "Gender, Fare, Pclass, Age")

# ==========================================
# 2. DATA EXPLORATION PAGE
# ==========================================
elif page == "Data Exploration":
    st.title("🔍 Data Exploration")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passengers (Cleaned)", f"{df.shape[0]}")
    col2.metric("Total Features", f"{df.shape[1]}")
    col3.metric("Survival Rate", f"{(df['survived'].mean() * 100):.1f}%")
    
    st.subheader("Sample Passenger Dataset")
    st.dataframe(df.head(50), use_container_width=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Missing Values Summary")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum(),
            'Percentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
        }).reset_index(drop=True)
        st.table(missing_df)
        
    with col_right:
        st.subheader("Statistical Summary (Numerical Columns)")
        st.dataframe(df.describe().T, use_container_width=True)

# ==========================================
# 3. VISUALIZATIONS PAGE
# ==========================================
elif page == "Visualizations":
    st.title("📊 Interactive Visualizations")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Demographics & Survival", "Distributions", "Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Survival Count Distribution")
            survival_count = df['survived'].value_counts().reset_index()
            survival_count.columns = ['status', 'count']
            survival_count['status'] = survival_count['status'].map({0: 'Perished', 1: 'Survived'})
            fig = px.pie(survival_count, values='count', names='status', color='status',
                         color_discrete_map={'Perished': '#ef4444', 'Survived': '#22c55e'},
                         hole=0.4, title="Overall Survival Ratio")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Survival Rate by Gender")
            gender_survival = df.groupby('sex')['survived'].mean().reset_index()
            gender_survival['survived'] = gender_survival['survived'] * 100
            fig = px.bar(gender_survival, x='sex', y='survived', text_auto='.1f',
                         labels={'sex': 'Gender', 'survived': 'Survival Rate (%)'},
                         color='sex', color_discrete_sequence=['#ff7f0e', '#1f77b4'],
                         title="Survival Rate (%) by Gender")
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Survival Rate by Passenger Class")
        class_survival = df.groupby('pclass')['survived'].mean().reset_index()
        class_survival['survived'] = class_survival['survived'] * 100
        class_survival['pclass'] = class_survival['pclass'].astype(str)
        fig = px.bar(class_survival, x='pclass', y='survived', text_auto='.1f',
                     labels={'pclass': 'Passenger Class (1st, 2nd, 3rd)', 'survived': 'Survival Rate (%)'},
                     color='pclass', color_discrete_sequence=px.colors.sequential.Teal,
                     title="Survival Rate by Ticket Class")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution by Survival Status")
            df_copy = df.copy()
            df_copy['survived_status'] = df_copy['survived'].map({0: 'Perished', 1: 'Survived'})
            fig = px.histogram(df_copy, x='age', color='survived_status', barmode='overlay',
                               color_discrete_map={'Perished': '#ef4444', 'Survived': '#22c55e'},
                               labels={'age': 'Age', 'survived_status': 'Status'},
                               title="Histogram of Passenger Ages")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Fare Distribution by Ticket Class")
            fig = px.box(df, x='pclass', y='fare', color='pclass',
                         labels={'pclass': 'Passenger Class', 'fare': 'Fare Paid ($)'},
                         title="Fare Boxplot per Ticket Class")
            st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        st.subheader("Correlation Heatmap")
        # Compute correlation on numerical features
        corr_df = df.select_dtypes(include=[np.number]).corr().round(2)
        
        fig = px.imshow(corr_df, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Pearson Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. SURVIVAL PREDICTION PAGE
# ==========================================
elif page == "Survival Prediction":
    st.title("🔮 Titanic Survival Predictor")
    st.markdown("---")
    
    if model_data is None:
        st.warning("Please train the model first by running the setup script.")
    else:
        st.subheader("Enter Passenger Information")
        st.markdown("Provide demographic details below to predict survival likelihood:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            p_class = st.selectbox("Passenger Ticket Class", [1, 2, 3], index=2, 
                                   help="1st Class = Luxury/Upper, 2nd Class = Middle, 3rd Class = Economy/Lower")
            p_sex = st.selectbox("Gender / Sex", ["Female", "Male"], index=0)
            p_age = st.slider("Age (in years)", min_value=1, max_value=80, value=28)
            p_fare = st.slider("Fare Paid ($)", min_value=0.0, max_value=512.0, value=25.0, step=1.0)
            
        with col2:
            p_sibsp = st.number_input("Number of Siblings / Spouses Aboard", min_value=0, max_value=8, value=0)
            p_parch = st.number_input("Number of Parents / Children Aboard", min_value=0, max_value=6, value=0)
            p_embarked = st.selectbox("Port of Embarkation", ["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"])
            
        st.markdown("---")
        
        if st.button("Predict Passenger Survival Status", type="primary"):
            try:
                # Load components from pickle
                model = model_data['model']
                le_sex = model_data['le_sex']
                le_embarked = model_data['le_embarked']
                features = model_data['features']
                
                # Format inputs to match trained model expectations
                sex_val = p_sex.lower()
                sex_encoded = le_sex.transform([sex_val])[0]
                
                emb_val = p_embarked[0] # Take 'S', 'C', or 'Q'
                emb_encoded = le_embarked.transform([emb_val])[0]
                
                # Feature arrangement: ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch', 'embarked']
                input_df = pd.DataFrame([[p_class, sex_encoded, p_age, p_fare, p_sibsp, p_parch, emb_encoded]], 
                                        columns=features)
                
                # Perform real-time prediction
                prediction = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                survival_prob = probs[1]
                death_prob = probs[0]
                
                st.subheader("Prediction Outcome:")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if prediction == 1:
                        st.markdown('<span class="badge-survived">🎉 PREDICTED TO SURVIVE</span>', unsafe_allow_html=True)
                        st.metric("Survival Probability", f"{survival_prob * 100:.1f}%")
                    else:
                        st.markdown('<span class="badge-died">💀 PREDICTED TO PERISH</span>', unsafe_allow_html=True)
                        st.metric("Survival Probability", f"{survival_prob * 100:.1f}%")
                
                with res_col2:
                    confidence = max(survival_prob, death_prob) * 100
                    st.metric("Model Confidence Score", f"{confidence:.1f}%")
                    
                    if confidence >= 80:
                        st.success("🎯 **High Confidence:** The model is very confident in this outcome.")
                    else:
                        st.warning("⚠️ **Moderate Confidence:** The details represent a borderline classification case.")
                        
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# ==========================================
# 5. KEY INSIGHTS PAGE
# ==========================================
elif page == "Key Insights":
    st.title("💡 Key Findings & Storytelling Insights")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👩‍💼 1. The Power of Gender")
        st.write(
            "**Finding:** Females survived at a rate of approximately **75%**, whereas males had a survival rate "
            "of only **20%**."
        )
        st.info("💡 **Insight:** The historic maritime protocol of 'Women and children first' was heavily enforced, making gender the single most critical factor in survival.")
        
        st.markdown("### 💰 2. Class Division & Wealth Impact")
        st.write(
            "**Finding:** Over **60%** of First-Class passengers survived the disaster, compared to around "
            "**25%** of Third-Class passengers."
        )
        st.info("💡 **Insight:** First-class cabins were positioned closer to the boat decks, and wealthy individuals had direct line-of-sight and better access to the evacuation coordinates.")

    with col2:
        st.markdown("### 👶 3. Prioritizing Children")
        st.write(
            "**Finding:** Children under the age of 12 had survival rates above **50%**, far higher than teenagers "
            "and working-age adults."
        )
        st.info("💡 **Insight:** Age analysis confirms that young children were heavily prioritized during lifeboat boarding, regardless of class in many instances.")
        
        st.markdown("### 👨‍👩‍👧‍👦 4. Family Size Sweet Spot")
        st.write(
            "**Finding:** Passengers traveling in small families (2-4 members) had a much higher survival rate "
            "than individuals traveling completely alone or in large family groups (5+)."
        )
        st.info("💡 **Insight:** Small groups could coordinate and secure spots together, while large families got separated in the chaos and solo travelers lacked support networks.")
