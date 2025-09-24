import pickle
import cloudpickle
import pandas as pd
import xgboost as xgb
import streamlit as st
import traceback

# ===============================
# Load Training Data (for dropdowns)
# ===============================
try:
    train = pd.read_csv("train.csv")
    X_train = train.drop(columns="price")
except FileNotFoundError:
    st.error("Training data (train.csv) not found. Please ensure it's in the same directory as the app.")
    st.stop()

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="✈ Flight Price Predictor",
    page_icon="💸",
    layout="wide"
)

# ===============================
# Custom CSS for Styling
# ===============================
st.markdown(
    """
    <style>
    /* Whole app background */
    .stApp {
        background-color: #f9fbfd;
    }

    /* Headings */
    h1, h2, h3 {
        color: #003366;
    }

    /* Buttons */
    .stButton button {
        background-color: grey !important;
        color: black !important;   /* 🔹 Button label text black */
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #FFC125 !important;
        color: black !important;   /* Keep black on hover */
    }

    /* Dropdown main box */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
        border-radius: 6px;
        padding: 4px;
    }

    /* Dropdown arrow */
    div[data-baseweb="select"] svg {
        fill: black !important;
    }

    /* Dropdown options list (the menu) */
    div[role="listbox"] {
        background-color: white !important;
        color: black !important;
    }

    /* Dropdown options text */
    div[role="option"] {
        background-color: white !important;
        color: black !important;
    }

    /* Dropdown option when hovered/selected */
    div[role="option"][aria-selected="true"],
    div[role="option"]:hover {
        background-color: #f0f0f0 !important;
        color: black !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Header Section
# ===============================
st.title("💸 Flight Price Prediction Dashboard")
st.caption("Built with *XGBoost + Streamlit* | Predict Flight Ticket Prices in INR")
st.markdown("---")

# ===============================
# Sidebar Section
# ===============================
st.sidebar.title("ℹ About")
st.sidebar.info(
    "This app predicts flight prices using a machine learning model trained with XGBoost. "
    "Enter your flight details in the form and click *Predict* to estimate the price."
)

# ===============================
# User Input Form
# ===============================
st.header("📌 Enter Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("✈ Airline", options=X_train.airline.unique())
    source = st.selectbox("🛫 Source", options=X_train.source.unique())
    dep_time = st.time_input("🕐 Departure Time")
    duration = st.number_input("⏱ Duration (minutes)", step=1)

with col2:
    destination = st.selectbox("🛬 Destination", options=X_train.destination.unique())
    total_stops = st.number_input("🛑 Total Stops", step=1, min_value=0)
    arrival_time = st.time_input("🕒 Arrival Time")
    additional_info = st.selectbox("ℹ Additional Info", options=X_train.additional_info.unique())

# Journey date (full width)
doj = st.date_input("📅 Date of Journey")

st.markdown("---")

# ===============================
# Data Preparation
# ===============================
x_new = pd.DataFrame(dict(
    airline=[airline],
    date_of_journey=[doj],
    source=[source],
    destination=[destination],
    dep_time=[dep_time],
    arrival_time=[arrival_time],
    duration=[duration],
    total_stops=[total_stops],
    additional_info=[additional_info]
)).astype({
    col: "str" for col in ["date_of_journey", "dep_time", "arrival_time"]
})

# ===============================
# Prediction
# ===============================
if st.button("🔮 Predict Flight Price"):
    try:
        # Load preprocessor
        with open("preprocessor.joblib", "rb") as f:
            saved_preprocessor = cloudpickle.load(f)

        st.write("📊 Raw Input Data:", x_new)
        x_new_pre = saved_preprocessor.transform(x_new)
        st.write("✅ After Preprocessing:", x_new_pre.shape)

        # Load model
        with open("xgboost-model", "rb") as f:
            model = pickle.load(f)

        # Make prediction
        x_new_xgb = xgb.DMatrix(x_new_pre)
        pred = model.predict(x_new_xgb)[0]

        # Styled output
        st.markdown(
            f"""
            <div style='
                background-color:#003366;
                color:white;
                padding:15px;
                border-radius:10px;
                font-size:20px;
                font-weight:bold;
                text-align:center;
            '>
                💵 Estimated Flight Price: ₹{pred:,.0f}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(label="Predicted Price (INR)", value=f"{pred:,.0f}")

    except Exception as e:
        st.error(f"⚠ Error while predicting: {e}")
        st.text("🔎 Full Traceback:")
        st.text("".join(traceback.format_exception(type(e), e, e.__traceback__)))