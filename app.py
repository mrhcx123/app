import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# ============================================================
# 1. 页面配置与“全方位紧凑” CSS
# ============================================================
st.set_page_config(page_title="Adsorption Prediction System", layout="wide")

st.markdown("""
    <style>
    /* 1. 强制全局字体：英文新罗马，中文宋体 */
    * {
        font-family: "Times New Roman", "SimSun", serif !important;
    }

    /* 2. 背景与容器：锁定 1400px，压缩顶部间距 */
    .stApp {
        background-color: #B8C6D4;
    }
    .block-container {
        max-width: 1400px !important;
        padding-top: 3rem !important; /* 显著减少顶部留白 */
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: auto;
    }

    /* 3. 标题样式：仅保留英文，减小下边距 */
    .main-title-en {
        text-align: center;
        font-weight: 900;
        font-size: 2.2rem !important;
        color: #000;
        margin-bottom: 2.5rem;
    }

    /* 4. 输入框标签：全黑加粗，强制不换行，压缩间距 */
    [data-testid="stWidgetLabel"] p {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #000 !important;
        white-space: nowrap !important; /* 核心：禁止换行对齐 */
        margin-bottom: -10px !important; /* 压缩标签与输入框的距离 */
    }

    /* 5. 左右面板：增加间距，同步压缩内部留白 */
    .left-panel {
        border-right: 2px solid #64748B;
        padding-right: 60px;
    }
    .right-panel {
        padding-left: 60px;
    }

    /* 6. 结果卡片：高度压缩 */
    .result-card {
        background: linear-gradient(135deg, #000 0%, #1E3A8A 100%);
        color: white !important;
        padding: 15px 25px;
        border-radius: 10px;
        text-align: center;
        margin: 20px auto; /* 减小上下边距 */
        max-width: 500px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }
    .result-card * { color: white !important; }

    /* 7. 预测按钮：紧凑设计 */
    .stButton { display: flex; justify-content: center; margin-top: 15px; }
    .stButton>button {
        background: #0F172A;
        color: white !important;
        padding: 10px 80px !important;
        font-size: 1.2rem !important;
        font-weight: bold;
        border-radius: 6px;
    }

    /* 8. 强制所有数字输入框居中 */
    input {
        text-align: center !important;
        font-size: 1.05rem !important;
    }

    /* 隐藏多余组件 */
    footer {visibility: hidden;}
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. 资源加载
# ============================================================
@st.cache_resource
def load_assets():
    # 模拟加载，实际请确保文件存在
    model = joblib.load('final_xgb_model_linear.pkl')
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_names.json', 'r') as f:
        features = json.load(f)
    return model, imputer, scaler, features

physchem_list = ['C', 'H', 'N', 'O', 'Surface Area (m2/g)', 'Pyrolysis temp(°C)']
experimental_list = ['Temp(°C)', 'C0(mg/L)', 'pH', 'Dosage(g/L)', 'Time(h)']

# 标准学术标签：全称 (简写, 单位)
input_labels = {
    'C': 'Carbon Content (C, %)',
    'H': 'Hydrogen Content (H, %)',
    'N': 'Nitrogen Content (N, %)',
    'O': 'Oxygen Content (O, %)',
    'Surface Area (m2/g)': 'Surface Area (Sвєт, m²/g)',
    'Pyrolysis temp(°C)': 'Pyrolysis Temperature (Tₚᵧᵣₒ, °C)',
    'Temp(°C)': 'Experimental Temperature (Tₑₓₚ, °C)',
    'C0(mg/L)': 'Initial Concentration (C₀, mg/L)',
    'pH': 'Solution pH (pH, -)',
    'Dosage(g/L)': 'Adsorbent Dosage (Dosage, g/L)',
    'Time(h)': 'Pyrolysis Time (tₚ, h)'
}

# 历史表格标签
table_labels = {
    'Pyrolysis temp(°C)': 'Tₚᵧᵣₒ', 'Temp(°C)': 'Tₑₓₚ',
    'C0(mg/L)': 'C₀', 'pH': 'pH', 'Dosage(g/L)': 'Dosage',
    'Time(h)': 'tₚ', 'Surface Area (m2/g)': 'Sвєт',
    'C': 'C', 'H': 'H', 'N': 'N', 'O': 'O'
}

try:
    model, imputer, scaler, all_features = load_assets()
except Exception as e:
    st.error(f"Asset Error: {e}")
    st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================================
# 3. 核心布局
# ============================================================
st.markdown('<div class="main-title-en">Ammonia Adsorption Prediction</div>', unsafe_allow_html=True)

input_dict = {}
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 1.2rem; font-weight: bold; color: #000; text-align: center; border-bottom: 2px solid #64748B; margin-bottom: 20px; padding-bottom: 5px;">Physicochemical Properties</div>', unsafe_allow_html=True)
    sub_l1, sub_l2 = st.columns(2)
    for i, name in enumerate(physchem_list):
        target_sub = sub_l1 if i % 2 == 0 else sub_l2
        with target_sub:
            input_dict[name] = st.number_input(label=input_labels.get(name, name), value=0.000, step=0.001, format="%.3f", key=f"pc_{name}")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 1.2rem; font-weight: bold; color: #000; text-align: center; border-bottom: 2px solid #64748B; margin-bottom: 20px; padding-bottom: 5px;">Experimental Parameters</div>', unsafe_allow_html=True)
    sub_r1, sub_r2 = st.columns(2)
    for i, name in enumerate(experimental_list):
        target_sub = sub_r1 if i % 2 == 0 else sub_r2
        with target_sub:
            input_dict[name] = st.number_input(label=input_labels.get(name, name), value=0.000, step=0.001, format="%.3f", key=f"exp_{name}")
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("RUN PREDICTION"):
    df_input = pd.DataFrame([input_dict])[all_features]
    input_imputed = imputer.transform(df_input)
    input_scaled = scaler.transform(input_imputed)
    qe_result = max(0, model.predict(input_scaled)[0])

    st.markdown(f"""
        <div class="result-card">
            <span style="font-size: 1.1rem; opacity: 0.9;">Predicted Capacity: </span>
            <span style="font-size: 2.6rem; font-weight: bold; margin-left: 15px;">
                Q<sub>e</sub> = {qe_result:.4f} <span style="font-size: 1rem;">mg/g</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

    hist_entry = {table_labels.get(k, k): f"{v:.3f}" for k, v in input_dict.items()}
    hist_entry["Qₑ (mg/g)"] = f"{qe_result:.4f}"
    hist_entry["Time"] = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.insert(0, hist_entry)

# ============================================================
# 4. 历史表格：HTML 强制居中 & 字体锁定
# ============================================================
if st.session_state.history:
    st.markdown('<div style="font-size: 1.2rem; font-weight: bold; color: #000; text-align: center; margin-top: 30px; margin-bottom: 15px;">Prediction History</div>', unsafe_allow_html=True)

    headers = list(st.session_state.history[0].keys())
    header_html = "".join([
        f'<th style="font-family: \'Times New Roman\' !important; font-weight: 900; color: #000; text-align: center; padding: 12px; border: 1px solid #64748B; background: rgba(0,0,0,0.05); font-size: 1.1rem;">{h}</th>'
        for h in headers
    ])

    rows_html = ""
    for entry in st.session_state.history[:8]:
        row = "".join([
            f'<td style="font-family: \'Times New Roman\' !important; color: #000; text-align: center; padding: 8px; border: 1px solid #64748B; font-size: 1rem;">{entry[h]}</td>'
            for h in headers
        ])
        rows_html += f"<tr>{row}</tr>"

    table_html = f"""
    <div style="width: 100%; overflow-x: auto; background-color: rgba(255, 255, 255, 0.98); border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <table style="width: 100%; border-collapse: collapse;">
            <thead><tr style="border-bottom: 2px solid #000;">{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; margin-top: 15px;'>", unsafe_allow_html=True)
    if st.button("Clear All"):
        st.session_state.history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #334155; font-size: 0.85rem; margin-top: 40px;'>© 2026 Ammonia Adsorption Prediction|System </p>", unsafe_allow_html=True)