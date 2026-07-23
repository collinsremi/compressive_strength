import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Concrete Strength Predictor", page_icon="📊", layout="centered")

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }
    .stApp {
        background-color: #f7f7f7;
        color: #1f1f1f;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("best_compressive_strength_model.joblib")


def main():
    st.title("Concrete Strength Predictor")
    st.caption("Estimate compressive strength from the mix values below.")

    model = load_model()
    feature_names = ["CNT_kg", "RCA_kg", "Water_kg", "Gum_Arabic_kg"]

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            cnt = st.number_input("CNT (kg)", min_value=0.0, value=100.0, step=1.0)
            rca = st.number_input("RCA (kg)", min_value=0.0, value=250.0, step=1.0)

        with col2:
            water = st.number_input("Water (kg)", min_value=0.0, value=150.0, step=1.0)
            gum = st.number_input("Gum Arabic (kg)", min_value=0.0, value=20.0, step=1.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame(
            [[cnt, rca, water, gum]],
            columns=feature_names,
        )
        prediction = model.predict(input_df)[0]

        st.subheader("Result")
        st.metric("Predicted compressive strength", f"{prediction:.2f} N/mm²")
        st.info("This uses the saved model from the project folder.")


if __name__ == "__main__":
    main()
