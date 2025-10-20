import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("random_forest_model.pkl")
df = pd.read_csv("master_dataset_cleaned.csv")

st.title("📊 Predictive Analytics Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to:", ["Make Prediction", "Past Data", "Annual Forecast", "Top 10 Case"])

# --- 1️⃣ Make Prediction ---
if page == "Make Prediction":
    st.header("🔮 Predict New Case Value")

    # Example input
    feature1 = st.number_input("Enter Feature 1 Value")
    feature2 = st.number_input("Enter Feature 2 Value")
    feature3 = st.number_input("Enter Feature 3 Value")

    if st.button("Predict"):
        X_new = pd.DataFrame([[feature1, feature2, feature3]], columns=["Feature1", "Feature2", "Feature3"])
        prediction = model.predict(X_new)[0]
        st.success(f"Predicted Value: {prediction:.2f}")

# --- 2️⃣ View Past Data ---
elif page == "Past Data":
    st.header("📂 Past Data Overview")
    st.dataframe(df.tail(20))

    st.subheader("Historical Trend")
    st.line_chart(df[['Year', 'Case']].set_index('Year'))

# --- 3️⃣ Annual Forecast (2024) ---
elif page == "Annual Forecast":
    st.header("📈 Predicted 2024 vs Actual 2024")

    df_2024 = df[df['Year'] == 2024]
    X_2024 = df_2024.drop(columns=['Case'])
    y_pred_2024 = model.predict(X_2024)

    plt.figure(figsize=(8, 4))
    plt.plot(df_2024['Case'].values, label='Actual 2024', marker='o')
    plt.plot(y_pred_2024, label='Predicted 2024', marker='x')
    plt.title("Predicted vs Actual (2024)")
    plt.legend()
    st.pyplot(plt)

# --- 4️⃣ Top 10 Case per Month ---
elif page == "Top 10 Case":
    st.header("🏆 Top 10 Case per Month")

    month = st.selectbox("Select Month", sorted(df['Month'].unique()))
    top10 = df[df['Month'] == month].nlargest(10, 'Case')

    st.write(top10)
    st.bar_chart(top10[['Case']].set_index(top10['area']))
