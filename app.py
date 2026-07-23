import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Civil Mix Strength Predictor", page_icon="🏗️", layout="wide")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background: linear-gradient(180deg, #07111d 0%, #0f1722 100%);
        color: #e5edf7;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .section-card {
        background: #111b2a;
        border: 1px solid #243244;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    }
    .stMetric { background: #0d1624; border-radius: 10px; padding: 0.6rem; }
    div[data-testid="stNumberInput"] > div > div > input {
        background-color: #0d1624;
        color: #f4f8ff;
    }
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #0d1624;
        color: #f4f8ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("best_compressive_strength_model.joblib")


def main():
    st.title("Civil Engineering Mix Strength Predictor")
    st.caption("A practical tool for estimating compressive strength from basic mix proportions.")

    model = load_model()
    feature_names = ["CNT_kg", "RCA_kg", "Water_kg", "Gum_Arabic_kg"]

    data = pd.read_csv("comprehensive_strength.csv")
    target = data["Comp Strength 1 (N/mm2) "]

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Study data points", f"{len(data):,}")
    with col2:
        st.metric("Observed strength range", f"{target.min():.1f} - {target.max():.1f} N/mm²")
    with col3:
        st.metric("Average strength", f"{target.mean():.2f} N/mm²")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    left_col, right_col = st.columns([1.15, 0.85])

    with left_col:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Input values")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                cnt = st.number_input("CNT (kg)", min_value=0.0, value=120.0, step=1.0)
                rca = st.number_input("RCA (kg)", min_value=0.0, value=230.0, step=1.0)
            with c2:
                water = st.number_input("Water (kg)", min_value=0.0, value=160.0, step=1.0)
                gum = st.number_input("Gum Arabic (kg)", min_value=0.0, value=18.0, step=1.0)
            submitted = st.form_submit_button("Estimate strength")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Result")
        if submitted:
            input_df = pd.DataFrame([[cnt, rca, water, gum]], columns=feature_names)
            prediction = model.predict(input_df)[0]
            st.metric("Predicted compressive strength", f"{prediction:.2f} N/mm²")
            st.info("This estimate is based on the trained model and should be checked against laboratory testing for design use.")
        else:
            st.info("Enter the mix values on the left to get a prediction.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Briefing for the analysis")
    st.markdown(
        """
        - The model estimates compressive strength from four practical mix inputs: cement content, recycled aggregate content, water content, and gum arabic content.
        - In civil engineering practice, water content and aggregate balance strongly influence the final strength, so the result should be treated as a planning estimate rather than a replacement for lab testing.
        - The chart section below provides the supporting interpretation of the data trends and model fit.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Chart explanations")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.image("eda_corr.png", caption="Correlation plot: shows how the mix variables relate to strength and to each other.")
        st.image("feature_importance.png", caption="Feature importance: highlights which input variables most influence the model output.")
    with chart_col2:
        st.image("eda_scatter.png", caption="Scatter view: helps compare each material input against the observed strength trend.")
        st.image("predicted_vs_actual.png", caption="Prediction fit: shows how closely the model follows the measured compressive strength values.")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
