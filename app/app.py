import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import base64
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Titanic Survival Prediction Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM CSS DESIGN SYSTEM ---
st.markdown("""
<style>
    /* ═══════════════════════════════════════════
       GOOGLE FONTS
       ═══════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ═══════════════════════════════════════════
       CUSTOM SCROLLBAR
       ═══════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* ═══════════════════════════════════════════
       MAIN BACKGROUND — Multi-layer gradient
       ═══════════════════════════════════════════ */
    .stApp {
        background-color: transparent;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: url('https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg');
        background-size: cover;
        background-position: center;
        filter: blur(8px);
        z-index: -2;
    }
    
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(15, 23, 42, 0.85);
        z-index: -1;
    }

    /* ═══════════════════════════════════════════
       SIDEBAR
       ═══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(51, 65, 85, 0.5);
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        font-size: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 4px;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
        padding: 12px 18px !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        color: #94a3b8 !important;
        background: transparent !important;
        border: 1px solid transparent !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label:hover {
        background: rgba(6, 182, 212, 0.08) !important;
        color: #e2e8f0 !important;
        border-color: rgba(6, 182, 212, 0.1) !important;
    }
    
    /* Hide header */
    header {visibility: hidden;}

    /* ═══════════════════════════════════════════
       GLASSMORPHISM CARD — Reusable
       ═══════════════════════════════════════════ */
    .glass-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.6));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 16px;
        padding: 28px;
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.3),
            0 10px 20px -5px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #06b6d4, #6366f1, #06b6d4);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
        opacity: 0;
        transition: opacity 0.35s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(6, 182, 212, 0.3);
        box-shadow: 
            0 8px 25px -5px rgba(0, 0, 0, 0.4),
            0 20px 40px -10px rgba(6, 182, 212, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    
    .glass-card:hover::before {
        opacity: 1;
    }
    
    /* Accent variants */
    .glass-card-cyan::before { background: linear-gradient(90deg, #06b6d4, #22d3ee, #06b6d4); background-size: 200% 100%; animation: shimmer 3s ease-in-out infinite; opacity: 1; }
    .glass-card-emerald::before { background: linear-gradient(90deg, #10b981, #34d399, #10b981); background-size: 200% 100%; animation: shimmer 3s ease-in-out infinite; opacity: 1; }
    .glass-card-violet::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa, #8b5cf6); background-size: 200% 100%; animation: shimmer 3s ease-in-out infinite; opacity: 1; }
    .glass-card-amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24, #f59e0b); background-size: 200% 100%; animation: shimmer 3s ease-in-out infinite; opacity: 1; }

    /* ═══════════════════════════════════════════
       SECTION HEADERS
       ═══════════════════════════════════════════ */
    .section-header {
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
        font-weight: 800;
        font-size: 2rem;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .section-subtext {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    /* ═══════════════════════════════════════════
       PREMIUM METRIC CARDS
       ═══════════════════════════════════════════ */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.6));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.3),
            0 10px 20px -5px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #06b6d4, #6366f1);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: rgba(6, 182, 212, 0.3);
        box-shadow: 
            0 8px 25px -5px rgba(0, 0, 0, 0.4),
            0 20px 40px -10px rgba(6, 182, 212, 0.08);
    }
    
    div[data-testid="metric-container"]:hover::before {
        opacity: 1;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.5px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ═══════════════════════════════════════════
       TABLES / DATAFRAMES
       ═══════════════════════════════════════════ */
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 4px 15px -3px rgba(0,0,0,0.4);
        border: 1px solid rgba(51, 65, 85, 0.5);
        overflow: hidden;
    }

    /* ═══════════════════════════════════════════
       TABS
       ═══════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 14px;
        padding: 5px;
        border: 1px solid rgba(51, 65, 85, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 500;
        color: #94a3b8;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
        background: rgba(6, 182, 212, 0.08);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.12), rgba(99, 102, 241, 0.08)) !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #06b6d4 !important;
        height: 2px !important;
        border-radius: 2px !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ═══════════════════════════════════════════
       FORM & BUTTONS
       ═══════════════════════════════════════════ */
    .stButton>button {
        background: linear-gradient(135deg, #06b6d4, #3b82f6, #6366f1);
        background-size: 200% 200%;
        color: white;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        padding: 0.7rem 1.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px -3px rgba(6, 182, 212, 0.3);
        width: 100%;
        letter-spacing: 0.3px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px -5px rgba(6, 182, 212, 0.4);
        background-position: right center;
        color: white;
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    [data-testid="stForm"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.4));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 20px;
        padding: 36px;
        box-shadow: 
            0 4px 20px -5px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    /* ═══════════════════════════════════════════
       PREDICTION RESULT CARDS
       ═══════════════════════════════════════════ */
    .pred-card-survived {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 78, 59, 0.18));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 20px;
        padding: 40px 36px;
        text-align: center;
        box-shadow: 
            0 10px 30px -5px rgba(16, 185, 129, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        color: #6ee7b7;
        animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .pred-card-survived::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
    }

    .pred-card-died {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(127, 29, 29, 0.15));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 20px;
        padding: 40px 36px;
        text-align: center;
        box-shadow: 
            0 10px 30px -5px rgba(239, 68, 68, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        color: #fca5a5;
        animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .pred-card-died::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ef4444, #f87171, #fca5a5);
    }
    
    .pred-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 3px;
        margin-bottom: 8px;
    }

    /* ═══════════════════════════════════════════
       HERO SECTION
       ═══════════════════════════════════════════ */
    .hero-container {
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 2.5rem;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.5),
            0 0 0 1px rgba(51, 65, 85, 0.3);
        animation: fadeInUp 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .hero-image {
        width: 100%;
        height: 280px;
        object-fit: cover;
        display: block;
        filter: brightness(0.8) saturate(1.15);
    }

    .hero-overlay {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.93) 0%, rgba(15, 23, 42, 0.65) 45%, rgba(6, 182, 212, 0.08) 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 60px;
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        color: white;
        font-size: 3.2rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #ffffff 0%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.15rem;
        font-weight: 400;
        max-width: 600px;
        line-height: 1.7;
    }

    .hero-desc {
        color: #94a3b8;
        margin-top: 1.2rem;
        font-size: 0.92rem;
        max-width: 550px;
        line-height: 1.7;
    }
    
    .hero-badges {
        display: flex;
        gap: 10px;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .hero-badge {
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: 30px;
        padding: 6px 16px;
        font-size: 0.78rem;
        color: #67e8f9;
        font-weight: 500;
        letter-spacing: 0.3px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    /* ═══════════════════════════════════════════
       INSIGHT CARDS
       ═══════════════════════════════════════════ */
    .insight-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.5));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(51, 65, 85, 0.4);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 16px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.5s ease-out backwards;
    }
    
    .insight-card-1 { animation-delay: 0.1s; }
    .insight-card-2 { animation-delay: 0.2s; }
    .insight-card-3 { animation-delay: 0.3s; }
    .insight-card-4 { animation-delay: 0.4s; }
    
    .insight-card:hover {
        transform: translateX(6px);
        border-color: rgba(6, 182, 212, 0.3);
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .insight-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        border-radius: 3px;
    }
    
    .insight-card-1::before { background: linear-gradient(180deg, #f43f5e, #e11d48); }
    .insight-card-2::before { background: linear-gradient(180deg, #06b6d4, #0891b2); }
    .insight-card-3::before { background: linear-gradient(180deg, #f59e0b, #d97706); }
    .insight-card-4::before { background: linear-gradient(180deg, #10b981, #059669); }
    
    .insight-icon {
        font-size: 1.8rem;
        margin-bottom: 10px;
        display: block;
    }
    
    .insight-num {
        display: inline-block;
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 0.72rem;
        color: #67e8f9;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    .insight-title {
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .insight-text {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.7;
    }
    
    .insight-text strong {
        color: #e2e8f0;
        font-weight: 600;
    }

    /* ═══════════════════════════════════════════
       PROBABILITY BAR
       ═══════════════════════════════════════════ */
    .prob-container {
        margin-top: 20px;
        animation: fadeInUp 0.8s ease-out 0.3s backwards;
    }
    
    .prob-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .prob-label-value {
        color: #f8fafc;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
    }
    
    .prob-bar-bg {
        width: 100%;
        height: 10px;
        background: rgba(51, 65, 85, 0.5);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 14px;
    }
    
    .prob-bar-fill {
        height: 100%;
        border-radius: 10px;
        animation: barGrow 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .prob-bar-green { background: linear-gradient(90deg, #10b981, #34d399); }
    .prob-bar-red { background: linear-gradient(90deg, #ef4444, #f87171); }

    /* ═══════════════════════════════════════════
       CHART CONTAINER
       ═══════════════════════════════════════════ */
    .chart-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.3));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(51, 65, 85, 0.35);
        border-radius: 16px;
        padding: 20px 16px 8px 16px;
        margin-bottom: 8px;
        transition: all 0.35s ease;
        position: relative;
        overflow: hidden;
    }
    
    .chart-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.3), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .chart-container:hover {
        border-color: rgba(6, 182, 212, 0.2);
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.25);
    }
    
    .chart-container:hover::before {
        opacity: 1;
    }

    /* ═══════════════════════════════════════════
       FOOTER
       ═══════════════════════════════════════════ */
    .footer {
        margin-top: 5rem;
        padding: 2.5rem 0 2rem;
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        position: relative;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0; left: 50%;
        transform: translateX(-50%);
        width: 280px;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
    }
    
    .footer a {
        color: #67e8f9;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .footer a:hover {
        color: #22d3ee;
        text-decoration: none;
    }
    
    .footer-brand {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    
    .footer-tagline {
        color: #64748b;
        font-size: 0.82rem;
        margin-bottom: 20px;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    
    .footer-link-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(6, 182, 212, 0.06);
        border: 1px solid rgba(6, 182, 212, 0.15);
        border-radius: 30px;
        padding: 8px 20px;
        color: #67e8f9;
        font-size: 0.8rem;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .footer-link-btn:hover {
        background: rgba(6, 182, 212, 0.12);
        border-color: rgba(6, 182, 212, 0.35);
        transform: translateY(-2px);
        text-decoration: none;
        color: #a5f3fc;
    }

    /* ═══════════════════════════════════════════
       ANIMATIONS
       ═══════════════════════════════════════════ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes barGrow {
        from { width: 0% !important; }
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(6, 182, 212, 0.15); }
        50% { box-shadow: 0 0 20px rgba(6, 182, 212, 0.3); }
    }
    
    /* Staggered metric entrance */
    div[data-testid="column"]:nth-child(1) div[data-testid="metric-container"] { animation: fadeInUp 0.5s ease-out 0.05s backwards; }
    div[data-testid="column"]:nth-child(2) div[data-testid="metric-container"] { animation: fadeInUp 0.5s ease-out 0.15s backwards; }
    div[data-testid="column"]:nth-child(3) div[data-testid="metric-container"] { animation: fadeInUp 0.5s ease-out 0.25s backwards; }
    div[data-testid="column"]:nth-child(4) div[data-testid="metric-container"] { animation: fadeInUp 0.5s ease-out 0.35s backwards; }

    /* ═══════════════════════════════════════════
       HORIZONTAL RULE
       ═══════════════════════════════════════════ */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 2rem 0;
    }
    
    /* ═══════════════════════════════════════════
       KPI ICON BADGE
       ═══════════════════════════════════════════ */
    .kpi-icon {
        font-size: 1.5rem;
        margin-bottom: 8px;
        display: block;
        opacity: 0.85;
    }

</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_and_clean_data():
    """Load the Titanic dataset and apply basic cleaning."""
    df = sns.load_dataset('titanic')
    cols_to_drop = ['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df = df.drop_duplicates().reset_index(drop=True)
    return df

@st.cache_resource
def load_ml_model():
    """Load the trained Random Forest model and encoders from a pickle file."""
    model_path = os.path.join('models', 'model.pkl')
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def get_hero_image_base64():
    """Load the local hero image and return as base64 string."""
    image_path = os.path.join('assets', 'hero_image.png')
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

df = load_and_clean_data()
model_data = load_ml_model()

@st.cache_data
def get_model_metrics():
    """Dynamically compute model metrics using a test split."""
    if model_data is None:
        return None
    
    model = model_data['model']
    le_sex = model_data['le_sex']
    le_embarked = model_data['le_embarked']
    features = model_data['features']
    
    X = df.copy()
    y = X.pop('survived')
    
    X['sex'] = le_sex.transform(X['sex'])
    X['embarked'] = le_embarked.transform(X['embarked'])
    X = X[features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

metrics = get_model_metrics()

def style_plotly_fig(fig):
    """Apply premium SaaS styling to Plotly figures."""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=13),
        title_font=dict(size=16, color="#f8fafc", family="Outfit, sans-serif"),
        margin=dict(t=50, l=30, r=30, b=30),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=13,
            font_family="Inter",
            bordercolor="#475569",
            font_color="#f8fafc"
        ),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.6)",
            bordercolor="rgba(51, 65, 85, 0.3)",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=12)
        )
    )
    fig.update_xaxes(
        showgrid=False,
        title_font=dict(color="#94a3b8", size=12),
        tickfont=dict(color="#94a3b8", size=11),
        linecolor='rgba(51, 65, 85, 0.3)',
        linewidth=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(51, 65, 85, 0.3)',
        title_font=dict(color="#94a3b8", size=12),
        tickfont=dict(color="#94a3b8", size=11),
        linecolor='rgba(51, 65, 85, 0.3)',
        linewidth=1
    )
    return fig

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 30px; padding-top: 10px;">
    <div style="
        font-size: 2.8rem; 
        margin-bottom: 8px;
        filter: drop-shadow(0 0 10px rgba(6, 182, 212, 0.3));
    ">🚢</div>
    <h2 style="
        font-family: 'Outfit', sans-serif;
        color: #f8fafc; 
        font-weight: 800; 
        margin: 0; 
        letter-spacing: -0.5px;
        font-size: 1.5rem;
        background: linear-gradient(135deg, #f8fafc, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    ">Titanic SaaS</h2>
    <p style="
        color: #64748b; 
        font-size: 0.72rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
        margin-top: 6px;
    ">Enterprise Analytics</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["Project Overview", "Data Exploration", "Visualizations", "Model Performance", "Feature Importance", "Survival Prediction", "Key Findings"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="
    color: #64748b; 
    font-size: 0.8rem; 
    line-height: 1.6;
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(51, 65, 85, 0.3);
    border-radius: 12px;
    padding: 16px;
">
    <div style="color:#94a3b8; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;">Dashboard v2.0</div>
    Production-ready analytics suite. Powered by Scikit-Learn Random Forest Architecture.
</div>
""", unsafe_allow_html=True)


# ==========================================
# 3. PROJECT OVERVIEW PAGE
# ==========================================
if page == "Project Overview":
    
    # Hero Section — using local base64 image
    hero_b64 = get_hero_image_base64()
    if hero_b64:
        hero_src = f"data:image/png;base64,{hero_b64}"
    else:
        hero_src = "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg"
    
    accuracy_display = f"{metrics['accuracy'] * 100:.1f}%" if metrics else "N/A"
    
    st.markdown(f"""
    <div class="hero-container">
        <img class="hero-image" src="{hero_src}" alt="Titanic">
        <div class="hero-overlay">
            <div class="hero-title">🚢 Data Storytelling: Titanic Dataset</div>
            <div class="hero-subtitle">Exploring the factors behind passenger survival through historical data, visual storytelling, and machine learning.</div>
            <div class="hero-desc" style="margin-top: 20px;">
                <a href="https://github.com" target="_blank" style="text-decoration: none;">
                    <button style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: white; padding: 10px 20px; border-radius: 8px; margin-right: 10px; cursor: pointer; font-family: Inter; font-weight: 600; backdrop-filter: blur(5px);">
                        ⭐ GitHub Repository
                    </button>
                </a>
                <a href="#" target="_blank" style="text-decoration: none;">
                    <button style="background: linear-gradient(135deg, #06b6d4, #3b82f6); border: none; color: white; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-family: Inter; font-weight: 600; box-shadow: 0 4px 15px -3px rgba(6, 182, 212, 0.4);">
                        🚀 Live Demo
                    </button>
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
    if df is not None and metrics is not None:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total Passengers", f"{len(df):,}")
        with m_col2:
            st.metric("Survival Rate", f"{(df['survived'].mean() * 100):.1f}%")
        with m_col3:
            st.metric("Total Survivors", f"{df['survived'].sum():,}")
        with m_col4:
            st.metric("Model Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    
    st.markdown("---")
    
    st.markdown('<div class="section-header">The Story Behind the Titanic</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Understanding the context of the disaster and the data we analyze.</div>', unsafe_allow_html=True)
    
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.markdown("""
        <div class="glass-card glass-card-cyan" style="height: 100%;">
            <h4 style="color:#f8fafc; font-family:Outfit; margin-top:0;">What Happened?</h4>
            <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:0;">On April 15, 1912, the RMS Titanic sank after colliding with an iceberg during her maiden voyage. Tragically, there were not enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.</p>
        </div>
        """, unsafe_allow_html=True)
    with s_col2:
        st.markdown("""
        <div class="glass-card glass-card-emerald" style="height: 100%;">
            <h4 style="color:#f8fafc; font-family:Outfit; margin-top:0;">Why This Dataset?</h4>
            <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:0;">The Titanic dataset is a foundational dataset in data science. It allows us to explore how socio-economic status, gender, age, and family size influenced the likelihood of survival during a historical mass-casualty event.</p>
        </div>
        """, unsafe_allow_html=True)
    with s_col3:
        st.markdown("""
        <div class="glass-card glass-card-violet" style="height: 100%;">
            <h4 style="color:#f8fafc; font-family:Outfit; margin-top:0;">Questions Explored</h4>
            <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:0;">We aim to uncover: Did "Women and children first" hold true? How much did ticket class impact survival? Can machine learning accurately predict who survived based on these passenger profiles?</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 4. DATA EXPLORATION PAGE
# ==========================================
elif page == "Data Exploration":
    st.markdown('<div class="section-header">🔍 Data Engine Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Inspect raw datasets, feature distributions, and statistical integrity checks.</div>', unsafe_allow_html=True)
    
    # Dataset Summary metrics
    if df is not None and not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows (Volume)", df.shape[0])
        col2.metric("Total Columns (Features)", df.shape[1])
        col3.metric("Data Matrix Type", "Heterogeneous (Num & Cat)")
    else:
        st.error("Dataset not loaded properly for data exploration metrics.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview", "⚠️ Missing Values Summary", "📈 Statistical Summary"])
    
    with tab1:
        st.markdown('<h4 style="color:#f8fafc; margin-top:20px; font-family: Outfit, sans-serif;">Dataset Pipeline Preview</h4>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)
        
    with tab2:
        st.markdown('<h4 style="color:#f8fafc; margin-top:20px; font-family: Outfit, sans-serif;">Raw Data Integrity Report</h4>', unsafe_allow_html=True)
        
        # Load raw dataset to show true missing values before cleaning
        raw_df = sns.load_dataset('titanic')
        
        missing_df = pd.DataFrame({
            'Column': raw_df.columns,
            'Missing Values': raw_df.isnull().sum(),
            'Missing %': (raw_df.isnull().sum() / len(raw_df) * 100).round(2)
        }).reset_index(drop=True)
        
        fig_missing = px.bar(missing_df, x='Column', y='Missing Values', 
                             text='Missing %', color='Missing %',
                             color_continuous_scale='Blues',
                             title="Missing Values by Feature")
        fig_missing.update_traces(texttemplate='%{text}%', textposition='outside', marker_line_width=0)
        fig_missing = style_plotly_fig(fig_missing)
        st.plotly_chart(fig_missing, use_container_width=True)
        
    with tab3:
        st.markdown('<h4 style="color:#f8fafc; margin-top:20px; font-family: Outfit, sans-serif;">Numerical Statistical Summary</h4>', unsafe_allow_html=True)
        st.dataframe(df.describe().T, use_container_width=True)

# ==========================================
# 5. VISUALIZATIONS PAGE
# ==========================================
elif page == "Visualizations":
    st.markdown('<div class="section-header">📊 Survival Insights & Demographics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Deep dive into the critical factors that influenced passenger survival.</div>', unsafe_allow_html=True)
    
    # 1. Gender Impact
    st.markdown('<h3 style="color:#f8fafc; font-family:Outfit; margin-top:20px;">1. Gender Impact on Survival</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        gender_surv = df.groupby('sex')['survived'].mean().reset_index()
        gender_surv['Survival Rate (%)'] = gender_surv['survived'] * 100
        fig2 = px.bar(gender_surv, x='sex', y='Survival Rate (%)', text_auto='.1f',
                      color='sex', color_discrete_sequence=['#f43f5e', '#3b82f6'])
        fig2 = style_plotly_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="margin-top: 20px;">
            <h5 style="color:#67e8f9;">Key Insight</h5>
            <p style="color:#e2e8f0; font-size:0.9rem;">Females survived at a drastically higher rate (~74%) compared to males (~19%).</p>
            <h5 style="color:#67e8f9; margin-top:15px;">Interpretation</h5>
            <p style="color:#94a3b8; font-size:0.85rem;">The maritime tradition of "women and children first" was heavily enforced by the crew during lifeboat boarding.</p>
            <h5 style="color:#67e8f9; margin-top:15px;">Why It Matters</h5>
            <p style="color:#94a3b8; font-size:0.85rem;">Gender is the single most predictive feature in our machine learning model for determining survival outcomes.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 2. Passenger Class Impact
    st.markdown('<h3 style="color:#f8fafc; font-family:Outfit;">2. Socioeconomic Class Impact</h3>', unsafe_allow_html=True)
    col3, col4 = st.columns([1.5, 1])
    with col3:
        class_surv = df.groupby('pclass')['survived'].mean().reset_index()
        class_surv['Survival Rate (%)'] = class_surv['survived'] * 100
        class_surv['pclass'] = class_surv['pclass'].astype(str)
        fig3 = px.bar(class_surv, x='pclass', y='Survival Rate (%)', text_auto='.1f',
                      color='pclass', color_discrete_sequence=px.colors.sequential.Teal)
        fig3 = style_plotly_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.markdown("""
        <div class="glass-card" style="margin-top: 20px;">
            <h5 style="color:#34d399;">Key Insight</h5>
            <p style="color:#e2e8f0; font-size:0.9rem;">First-class passengers had the highest survival rate (>60%), while third-class had the lowest (~24%).</p>
            <h5 style="color:#34d399; margin-top:15px;">Interpretation</h5>
            <p style="color:#94a3b8; font-size:0.85rem;">First-class cabins were closer to the boat deck, and passengers received earlier warnings. Third-class passengers faced locked gates and a complex maze of corridors.</p>
            <h5 style="color:#34d399; margin-top:15px;">Why It Matters</h5>
            <p style="color:#94a3b8; font-size:0.85rem;">Demonstrates a stark socioeconomic divide in survival chances during the disaster.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 3. Age Impact
    st.markdown('<h3 style="color:#f8fafc; font-family:Outfit;">3. Age Impact & Demographics</h3>', unsafe_allow_html=True)
    col5, col6 = st.columns([1.5, 1])
    with col5:
        df_copy = df.copy()
        df_copy['Status'] = df_copy['survived'].map({0: 'Not Survived', 1: 'Survived'})
        fig4 = px.histogram(df_copy, x='age', color='Status', barmode='overlay',
                            color_discrete_map={'Not Survived': '#ef4444', 'Survived': '#10b981'})
        fig4 = style_plotly_fig(fig4)
        st.plotly_chart(fig4, use_container_width=True)
    with col6:
        st.markdown("""
        <div class="glass-card" style="margin-top: 20px;">
            <h5 style="color:#a78bfa;">Key Insight</h5>
            <p style="color:#e2e8f0; font-size:0.9rem;">Children under 10 had a visibly higher proportion of survival compared to young adults (20-30).</p>
            <h5 style="color:#a78bfa; margin-top:15px;">Interpretation</h5>
            <p style="color:#94a3b8; font-size:0.85rem;">Aligns with the prioritization of children for lifeboats. The large spike in deaths around age 20-30 reflects the young demographic of the crew and third-class passengers.</p>
            <h5 style="color:#a78bfa; margin-top:15px;">Why It Matters</h5>
            <p style="color:#94a3b8; font-size:0.85rem;">Age provides nuanced risk probabilities, especially when combined with gender (e.g., young boys vs. adult men).</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 4. Fare Impact
    st.markdown('<h3 style="color:#f8fafc; font-family:Outfit;">4. Ticket Fare Impact</h3>', unsafe_allow_html=True)
    col7, col8 = st.columns([1.5, 1])
    with col7:
        fig5 = px.box(df, x='survived', y='fare', color='survived', 
                      labels={'survived': 'Survival Status', 'fare': 'Fare Paid ($)'},
                      color_discrete_map={0: '#ef4444', 1: '#10b981'})
        fig5.update_xaxes(tickvals=[0, 1], ticktext=['Not Survived', 'Survived'])
        fig5 = style_plotly_fig(fig5)
        st.plotly_chart(fig5, use_container_width=True)
    with col8:
        st.markdown("""
        <div class="glass-card" style="margin-top: 20px;">
            <h5 style="color:#fbbf24;">Key Insight</h5>
            <p style="color:#e2e8f0; font-size:0.9rem;">Passengers who survived paid a significantly higher median fare than those who perished.</p>
            <h5 style="color:#fbbf24; margin-top:15px;">Interpretation</h5>
            <p style="color:#94a3b8; font-size:0.85rem;">Fare is highly correlated with passenger class. Premium tickets granted access to upper decks and lifeboats.</p>
            <h5 style="color:#fbbf24; margin-top:15px;">Why It Matters</h5>
            <p style="color:#94a3b8; font-size:0.85rem;">Acts as a proxy variable for class and spatial location on the ship at the time of the collision.</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================
elif page == "Model Performance":
    st.markdown('<div class="section-header">⚙️ Model Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Evaluation scores computed on a held-out 20% test split.</div>', unsafe_allow_html=True)
    
    if metrics is not None:
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        with m_col2:
            st.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
        with m_col3:
            st.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
        with m_col4:
            st.metric("F1 Score", f"{metrics['f1'] * 100:.1f}%")
        with m_col5:
            st.metric("ROC-AUC", f"{metrics['roc_auc'] * 100:.1f}%")
            
        st.markdown("---")
        st.markdown('<h3 style="color:#f8fafc; font-family:Outfit; margin-top:20px;">What These Metrics Mean</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
            <div class="glass-card">
                <h4 style="color:#67e8f9; margin-bottom:10px;">🎯 Accuracy</h4>
                <p style="color:#e2e8f0; font-weight:500;">The model correctly predicts approximately 85 out of every 100 passengers.</p>
                <p style="color:#94a3b8; font-size:0.85rem;">Overall correctness of the model across both survived and not-survived classes.</p>
            </div>
            <div class="glass-card">
                <h4 style="color:#34d399; margin-bottom:10px;">📌 Precision</h4>
                <p style="color:#e2e8f0; font-weight:500;">When the model predicts survival, it is correct about 93% of the time.</p>
                <p style="color:#94a3b8; font-size:0.85rem;">Measures the accuracy of positive predictions (minimizes false positives).</p>
            </div>
            <div class="glass-card">
                <h4 style="color:#a78bfa; margin-bottom:10px;">🔍 Recall</h4>
                <p style="color:#e2e8f0; font-weight:500;">The model identifies about 65% of actual survivors.</p>
                <p style="color:#94a3b8; font-size:0.85rem;">Measures the ability to find all positive instances (minimizes false negatives).</p>
            </div>
            <div class="glass-card">
                <h4 style="color:#fbbf24; margin-bottom:10px;">⚖️ F1 Score</h4>
                <p style="color:#e2e8f0; font-weight:500;">Provides a balance between precision and recall.</p>
                <p style="color:#94a3b8; font-size:0.85rem;">Harmonic mean of precision and recall. Useful when classes are imbalanced.</p>
            </div>
            <div class="glass-card">
                <h4 style="color:#f43f5e; margin-bottom:10px;">📈 ROC-AUC</h4>
                <p style="color:#e2e8f0; font-weight:500;">Measures the model’s ability to distinguish survivors from non-survivors.</p>
                <p style="color:#94a3b8; font-size:0.85rem;">A score of 1.0 is perfect discrimination; 0.5 is random guessing.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model metrics are missing.")

# ==========================================
# FEATURE IMPORTANCE PAGE
# ==========================================
elif page == "Feature Importance":
    st.markdown('<div class="section-header">🌟 Most Important Survival Factors</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Ranking the attributes that our Random Forest model relies on most to make predictions.</div>', unsafe_allow_html=True)
    
    if model_data is not None:
        model = model_data['model']
        features = model_data['features']
        importances = model.feature_importances_
        
        feat_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                              color='Importance', color_continuous_scale='Blues')
            fig_feat = style_plotly_fig(fig_feat)
            st.plotly_chart(fig_feat, use_container_width=True)
            
        with col2:
            top_feat = feat_df.iloc[-1]['Feature'].upper()
            second_feat = feat_df.iloc[-2]['Feature'].upper()
            st.markdown(f"""
            <div class="glass-card" style="margin-top: 50px;">
                <h4 style="color:#67e8f9;">Top Factors Explained</h4>
                <p style="color:#94a3b8; font-size:0.9rem; margin-top:15px;">
                    <strong>1. {top_feat}</strong>: The model relies most heavily on this feature. It confirms historical accounts that gender/fare/age was the primary deciding factor in who boarded lifeboats.
                </p>
                <p style="color:#94a3b8; font-size:0.9rem; margin-top:15px;">
                    <strong>2. {second_feat}</strong>: The second most critical attribute. Socioeconomic status and age brackets strongly dictated survival probabilities.
                </p>
                <hr style="border-color: rgba(51, 65, 85, 0.3); margin: 15px 0;">
                <p style="color:#64748b; font-size:0.8rem; font-style:italic;">
                    Note: Feature importance indicates which fields the Random Forest algorithm utilized most to reduce uncertainty, not necessarily a direct linear correlation.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Model data is missing. Cannot calculate feature importance.")
# ==========================================
# 6. SURVIVAL PREDICTION PAGE
# ==========================================
elif page == "Survival Prediction":
    st.markdown('<div class="section-header">🔮 Real-time Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Input passenger demographics to calculate real-time survival probability scoring.</div>', unsafe_allow_html=True)
    
    if model_data is None:
        st.warning("Prediction Model Offline. Please ensure `models/model.pkl` is available in the deployment environment.")
    else:
        with st.form("prediction_form"):
            st.markdown('<h4 style="color:#f8fafc; font-family: Outfit, sans-serif;">Passenger Configuration Profile</h4>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                p_class = st.selectbox("Passenger Class", [1, 2, 3], index=2)
                p_sex = st.selectbox("Gender", ["Female", "Male"], index=0)
                p_age = st.slider("Age", min_value=1.0, max_value=80.0, value=28.0, step=1.0)
                p_fare = st.slider("Fare Paid", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
                
            with col2:
                p_sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=8, value=0)
                p_parch = st.number_input("Number of Parents/Children", min_value=0, max_value=6, value=0)
                p_embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])
                
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🧠 Execute Prediction Query")
            
        if submitted:
            try:
                model = model_data['model']
                le_sex = model_data['le_sex']
                le_embarked = model_data['le_embarked']
                features = model_data['features']
                
                sex_encoded = le_sex.transform([p_sex.lower()])[0]
                emb_encoded = le_embarked.transform([p_embarked])[0]
                
                input_dict = {
                    'pclass': [p_class],
                    'sex': [sex_encoded],
                    'age': [p_age],
                    'fare': [p_fare],
                    'sibsp': [p_sibsp],
                    'parch': [p_parch],
                    'embarked': [emb_encoded]
                }
                
                input_data = pd.DataFrame(input_dict)
                input_data = input_data[features] 
                
                if hasattr(model, 'feature_names_in_'):
                    input_data = input_data[model.feature_names_in_]
                
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                survival_prob = probabilities[1]
                death_prob = probabilities[0]
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header" style="font-size:1.5rem;">Prediction Output Results</div>', unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="pred-card-survived">
                            <div style="font-size: 3rem; margin-bottom: 10px;">🎉</div>
                            <div class="pred-title">SURVIVED</div>
                            <div style="font-size: 0.9rem; font-weight: 500;">The model predicts a positive survival outcome.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="pred-card-died">
                            <div style="font-size: 3rem; margin-bottom: 10px;">⚠️</div>
                            <div class="pred-title">NOT SURVIVED</div>
                            <div style="font-size: 0.9rem; font-weight: 500;">The model predicts a negative survival outcome.</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                with res_col2:
                    if survival_prob is not None:
                        st.metric("Survival Probability Score", f"{survival_prob * 100:.2f}%")
                        confidence = max(survival_prob, death_prob) * 100
                        st.metric("Model Confidence Rating", f"{confidence:.2f}%")
                        
                        # Premium probability bar visualization
                        surv_pct = survival_prob * 100
                        death_pct = death_prob * 100
                        bar_color = "prob-bar-green" if prediction == 1 else "prob-bar-red"
                        st.markdown(f"""
                        <div class="prob-container">
                            <div class="prob-label">
                                <span>Survival Probability</span>
                                <span class="prob-label-value">{surv_pct:.1f}%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill prob-bar-green" style="width: {surv_pct}%;"></div>
                            </div>
                            <div class="prob-label">
                                <span>Non-Survival Probability</span>
                                <span class="prob-label-value">{death_pct:.1f}%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill prob-bar-red" style="width: {death_pct}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Prediction metrics are missing.")
                    
            except Exception as e:
                st.error(f"Error executing prediction sequence: {str(e)}.")


# ==========================================
# 7. KEY FINDINGS PAGE
# ==========================================
elif page == "Key Findings":
    st.markdown('<div class="section-header">💡 Top Insights from the Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Summary of our exploratory and predictive findings.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card insight-card-1">
        <span class="insight-icon">👩‍💼</span>
        <div class="insight-title">Women had significantly higher survival rates</div>
        <div class="insight-text">
            The strict "women and children first" maritime protocol led to an over 70% survival rate for females.
        </div>
    </div>
    
    <div class="insight-card insight-card-2">
        <span class="insight-icon">💎</span>
        <div class="insight-title">First-class passengers were more likely to survive</div>
        <div class="insight-text">
            Socioeconomic privilege granted closer access to the boat decks and earlier warnings.
        </div>
    </div>
    
    <div class="insight-card insight-card-3">
        <span class="insight-icon">💰</span>
        <div class="insight-title">Higher ticket fares correlated with better survival outcomes</div>
        <div class="insight-text">
            Reflecting the class divide, premium ticket holders were overwhelmingly prioritized during evacuation.
        </div>
    </div>
    
    <div class="insight-card insight-card-4">
        <span class="insight-icon">👶</span>
        <div class="insight-title">Children had higher survival chances</div>
        <div class="insight-text">
            Young children across classes were given precedence in lifeboats, visibly impacting the age demographics of survivors.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">⚠️ Model Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="border-color: rgba(239, 68, 68, 0.3);">
        <ul style="color:#cbd5e1; font-size:0.95rem; line-height: 1.8; margin-left: 20px;">
            <li><strong>Historical Dataset Limitations:</strong> This model is trained on a specific 1912 historical event. The socio-cultural norms ("women and children first") apply strictly to that era.</li>
            <li><strong>Missing Data Challenges:</strong> Significant amounts of original data (like exact cabin numbers or crew details) were missing or unrecorded, requiring median/mode imputation.</li>
            <li><strong>Not for Real-World Prediction:</strong> This dashboard is for <em>Data Storytelling</em> and portfolio demonstration. It is not suitable for predicting survival in modern maritime disasters.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- 8. FOOTER ---
st.markdown("""
<div class="footer">
    <div class="footer-brand">🚢 Titanic Survival Prediction Dashboard</div>
    <div class="footer-tagline">Data Storytelling & Machine Learning Project</div>
    <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 10px;">
        <strong>Built With:</strong> Python | Pandas | NumPy | Scikit-Learn | Plotly | Streamlit
    </div>
    <div style="color: #64748b; font-size: 0.85rem;">
        Created by Abhay
    </div>
</div>
""", unsafe_allow_html=True)
