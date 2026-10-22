
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="LifeExpect | Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f8fc 0%, #edf4f8 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        background: linear-gradient(120deg, #123c69, #1d7874);
        padding: 35px;
        border-radius: 22px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(18, 60, 105, 0.15);
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #e7f4f5;
        font-size: 17px;
        margin-bottom: 0;
    }

    .section-title {
        color: #123c69;
        font-size: 23px;
        font-weight: 700;
        margin: 12px 0 5px 0;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e1eaf0;
        box-shadow: 0 5px 18px rgba(20, 50, 80, 0.06);
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(100deg, #123c69, #1d7874);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 700;
        min-height: 45px;
        width: 100%;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        color: white;
        border: 1px solid #1d7874;
        opacity: 0.9;
    }

    .result-card {
        background: white;
        padding: 28px;
        border-radius: 18px;
        border-left: 6px solid #1d7874;
        box-shadow: 0 5px 20px rgba(20, 50, 80, 0.08);
        margin-top: 22px;
    }

    .result-label {
        color: #587080;
        font-size: 16px;
    }

    .result-value {
        color: #123c69;
        font-size: 42px;
        font-weight: 800;
        margin: 5px 0;
    }

    [data-testid="stSidebar"] {
        background-color: #eaf1f6;
    }

    footer {
        visibility: hidden;
    }
    
/* Fix text visibility on the light background */
.stApp,
.stApp p,
.stApp label,
.stApp h2,
.stApp h3,
.stApp li {
    color: #243447;
}

/* Sidebar text */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label {
    color: #243447 !important;
}

/* Keep the header text white */
.hero,
.hero h1,
.hero p {
    color: #ffffff !important;
}

/* Keep input labels readable */
.stApp [data-testid="stWidgetLabel"] p {
    color: #243447 !important;
}

/* Improve the small helper text */
.stApp [data-testid="stCaptionContainer"] p {
    color: #526779 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL AND CONFIG ----------------
folder = Path(__file__).parent / "phase3_artifacts"
model_path = folder / "best_life_expectancy_model.joblib"
config_path = folder / "feature_config.json"

if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

if not config_path.exists():
    st.error(f"Feature configuration not found: {config_path}")
    st.stop()


@st.cache_resource
def load_model(path):
    return joblib.load(path)


with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

model = load_model(str(model_path))

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🌍 LifeExpect")
    st.caption("Life Expectancy Prediction System")
    st.divider()

    st.markdown("### About this project")
    st.write(
        "This machine learning application estimates "
        "life expectancy using country, health, and "
        "socioeconomic indicators."
    )

    st.markdown("### How to use")
    st.write("1. Enter the required details.")
    st.write("2. Click the prediction button.")
    st.write("3. View your estimated result.")

    st.divider()
    st.caption("Machine Learning Mini Project")
    st.caption("Built with Python and Streamlit")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌍 Life Expectancy Predictor</h1>
    <p>
        Explore how health, education, and socioeconomic
        indicators relate to estimated life expectancy.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- INTRODUCTION ----------------
st.markdown("### Welcome to your prediction dashboard")
st.write(
    "Enter the country and indicator values below. "
    "The trained machine learning model will generate "
    "an estimate based on the information provided."
)

st.info(
    "For a meaningful estimate, use values consistent "
    "with the dataset used to train your model."
)

# ---------------- INPUT FORM ----------------
st.markdown(
    '<div class="section-title">📋 Enter Prediction Details</div>',
    unsafe_allow_html=True
)
st.caption("Fill in the available indicators and submit the form.")

with st.form("prediction_form"):
    inputs = {}
    features = list(config.keys())

    # Two-column layout for easier data entry
    left_col, right_col = st.columns(2)

    for i, feature in enumerate(features):
        settings = config[feature]
        target_col = left_col if i % 2 == 0 else right_col

        with target_col:
            if settings["type"] == "numeric":
                default = float(settings["default"])

                inputs[feature] = st.number_input(
                    feature,
                    value=default,
                    key=f"input_{i}",
                    help=f"Enter a value for {feature}."
                )
            else:
                options = settings.get("options", [])
                default = settings.get("default", "")

                if not options:
                    st.warning(f"No options available for {feature}.")
                    inputs[feature] = default
                else:
                    index = (
                        options.index(default)
                        if default in options else 0
                    )

                    inputs[feature] = st.selectbox(
                        feature,
                        options=options,
                        index=index,
                        key=f"input_{i}"
                    )

    submitted = st.form_submit_button(
        "🔍 Predict Life Expectancy",
        use_container_width=True
    )

# ---------------- PREDICTION RESULT ----------------
if submitted:
    try:
        input_data = pd.DataFrame([inputs])

        # Preserve the same feature order as the config
        input_data = input_data[features]

        prediction = float(model.predict(input_data)[0])

        st.markdown("---")
        st.markdown("## 📊 Prediction Result")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">
                Estimated Life Expectancy
            </div>
            <div class="result-value">
                {prediction:.2f} years
            </div>
            <div class="result-label">
                Generated using your trained machine learning model
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.success("Prediction generated successfully.")

        with st.expander("View submitted input details"):
            st.dataframe(
                input_data.T.rename(columns={0: "Input Value"}),
                use_container_width=True
            )

        st.caption(
            "This is a model-based estimate, not an individual "
            "medical or lifespan prediction."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.exception(error)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<center style='color:#587080;'>"
    "LifeExpect · Machine Learning Mini Project"
    "</center>",
    unsafe_allow_html=True
)
