
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="Life Expectancy Prediction",
    page_icon="🌍",
    layout="wide"
)

# UI styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef5ff, #f8fafc);
    color: #1e293b;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label,
[data-testid="stWidgetLabel"] {
    color: #1e293b !important;
}
.main-title {
    text-align: center;
    color: #173b70;
    font-size: 36px;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #526581;
    margin-bottom: 28px;
}
.section-title {
    background: #e1ecff;
    padding: 12px;
    border-radius: 10px;
    color: #173b70 !important;
    font-size: 19px;
    font-weight: 700;
    margin: 8px 0 16px 0;
}
div[data-testid="stForm"] {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #d9e3f2;
}
div[data-testid="stNumberInput"] input {
    background-color: white;
    color: #1e293b !important;
}
div[data-baseweb="select"] * {
    color: #1e293b !important;
}
.stFormSubmitButton button {
    background: #2563eb;
    color: white !important;
    border-radius: 10px;
    font-weight: 700;
    width: 100%;
}
.prediction-card {
    background: #e4f7eb;
    border: 1px solid #9ddbb5;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-top: 24px;
}
.prediction-value {
    color: #166534;
    font-size: 38px;
    font-weight: 800;
}
.help-card {
    background: #eff6ff;
    border-left: 5px solid #3b82f6;
    padding: 16px;
    border-radius: 8px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🌍 Life Expectancy Prediction</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Estimate life expectancy using health, '
    'economic and social indicators.</div>',
    unsafe_allow_html=True
)

# Load model and configuration
folder = Path(__file__).parent / "phase3_artifacts"
model_path = folder / "best_life_expectancy_model.joblib"
config_path = folder / "feature_config.json"

if not model_path.exists() or not config_path.exists():
    st.error(
        "Model files are missing. Ensure the model and "
        "feature_config.json are inside phase3_artifacts."
    )
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

with open(config_path, encoding="utf-8") as file:
    config = json.load(file)

model = load_model(str(model_path))

# Organize features into two balanced columns
country_features = ["Country", "Status"]

economy_features = [
    "Year", "GDP", "Population", "Alcohol",
    "percentage expenditure", "Total expenditure",
    "Income composition of resources", "Schooling"
]

left_features = []
right_features = []

# Put Country and Status first, then distribute remaining features
for feature in country_features:
    if feature in config:
        left_features.append(feature)

remaining = [
    feature for feature in config
    if feature not in left_features
]

# Distribute remaining features between columns
for i, feature in enumerate(remaining):
    if len(left_features) <= len(right_features):
        left_features.append(feature)
    else:
        right_features.append(feature)

def display_feature(feature, inputs):
    settings = config[feature]

    if settings["type"] == "numeric":
        inputs[feature] = st.number_input(
            feature,
            value=float(settings["default"]),
            key=f"input_{feature}"
        )
    else:
        options = settings["options"]
        default = settings["default"]
        index = options.index(default) if default in options else 0

        inputs[feature] = st.selectbox(
            feature,
            options,
            index=index,
            key=f"input_{feature}"
        )

with st.form("prediction_form"):
    inputs = {}
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(
            '<div class="section-title">🌍 Country and Health Indicators</div>',
            unsafe_allow_html=True
        )
        for feature in left_features:
            display_feature(feature, inputs)

    with right_col:
        st.markdown(
            '<div class="section-title">📊 Economic and Social Indicators</div>',
            unsafe_allow_html=True
        )
        for feature in right_features:
            display_feature(feature, inputs)

    submitted = st.form_submit_button(
        "🔍 Predict Life Expectancy",
        use_container_width=True
    )

if submitted:
    try:
        # Preserve the model's expected feature order
        input_data = pd.DataFrame([inputs])[list(config.keys())]
        prediction = float(model.predict(input_data)[0])

        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="font-size:20px;color:#166534;">
                    Estimated Life Expectancy
                </div>
                <div class="prediction-value">
                    {prediction:.2f} years
                </div>
                <div style="color:#166534;">
                    Estimate generated by your trained Random Forest model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="help-card">
                <b>How to understand this result</b><br>
                The model uses the country, health, economic and social
                indicators entered above to estimate life expectancy.
                Different input values may produce a different estimate.
                This is a machine-learning prediction, not a guaranteed
                outcome or an official public-health statistic.
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.caption(
    "Academic ML Project | Random Forest Regression | "
    "Predictions depend on the training data."
)
