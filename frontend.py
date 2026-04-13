import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px

# --- Configuration & Layout ---
st.set_page_config(page_title="CTR Prediction Dashboard", layout="wide", page_icon="📈")

st.title("Click-Through Rate (CTR) Prediction Dashboard 📈")
st.markdown("""
Welcome to the CTR predictive analytics platform. Use the **Prediction Engine** to score the likelihood of engagement for a specific ad placement, 
or navigate to **Business Analytics** to explore historical trends from our machine learning data pipeline.
""")

tab1, tab2 = st.tabs(["Prediction Engine 🚀", "Business Analytics 📊"])

with tab1:
    st.header("Ad Engagement Predictor")
    
    # --- UI Layout: Columns and Sidebar configuration ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Target Profile & Context")
        with st.form("prediction_form"):
            user_age = st.slider("User Age", min_value=18, max_value=100, value=25)
            user_gender = st.selectbox("User Gender", ["Male", "Female", "Other"])
            user_income = st.selectbox("User Income Level", ["Low", "Medium", "High"])
            
            st.markdown("---")
            
            device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
            time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
            day_of_week = st.slider("Day of the Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=3)
            
            st.markdown("---")
            
            ad_category = st.selectbox("Ad Category", ["Electronics", "Fashion", "Finance", "Automotive", "Entertainment"])
            ad_placement = st.selectbox("Ad Placement", ["Sidebar", "Header", "Footer", "In-feed"])
            
            submit_button = st.form_submit_button(label="Run Prediction ⚡")
            
    with col2:
        st.subheader("Prediction Results")
        if submit_button:
            # Map inputs strictly to the Flask API payload
            payload = {
                "user_age": user_age,
                "user_gender": user_gender,
                "user_income": user_income,
                "device_type": device_type,
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "ad_category": ad_category,
                "ad_placement": ad_placement
            }
            
            with st.spinner("Analyzing target configuration with Machine Learning pipeline..."):
                try:
                    # Make a request to the live Flask API
                    response = requests.post("predictor-ctr.streamlit.app", json=payload, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        ctr_prob = data.get("predicted_ctr_probability", 0) * 100
                        pred_class = data.get("prediction_class", 0)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Massive Metric Display
                        st.metric(label="Predicted Click Probability", value=f"{ctr_prob:.2f}%")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # In digital marketing, anything over an 8-10% CTR is usually fantastic.
                        if ctr_prob >= 10.0: 
                            st.success("💡 Recommendation: High likelihood of engagement. Proceed with ad placement.")
                        else:
                            st.warning("⚠️ Recommendation: Low Engagement Likelihood! — Reconsider targeting parameters or creative.")
                            
                        with st.expander("View Raw API Response"):
                            st.json(data)
                            
                    else:
                        st.error(f"Prediction failed with status code {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the Backend API. Please ensure `python app.py` is running on localhost:5000.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
        else:
            st.info("👈 Configure the target audience and ad features on the left and click **Run Prediction** to see real-time AI inference.")

with tab2:
    st.header("Historical Engagement Analytics")
    try:
        # Load the generated dataset
        df = pd.read_csv("ctr_data.csv")
        
        st.markdown("Explore historical ad performance across different segmentations to refine your targeting strategies.")
        
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            # Chart 1: Average CTR by Ad Category
            category_ctr = df.groupby("ad_category")["is_click"].mean().reset_index()
            category_ctr["is_click"] = category_ctr["is_click"] * 100  # convert to percentage
            fig_cat = px.bar(
                category_ctr, 
                x="ad_category", 
                y="is_click", 
                title="Average CTR by Ad Category",
                labels={"ad_category": "Ad Category", "is_click": "Average CTR (%)"},
                color="ad_category"
            )
            st.plotly_chart(fig_cat, use_container_width=True)
            
        with row1_col2:
            # Chart 2: Average CTR by Time of Day
            time_ctr = df.groupby("time_of_day")["is_click"].mean().reset_index()
            # Sort time naturally
            order = ["Morning", "Afternoon", "Evening", "Night"]
            time_ctr["time_of_day"] = pd.Categorical(time_ctr["time_of_day"], categories=order, ordered=True)
            time_ctr = time_ctr.sort_values("time_of_day")
            time_ctr["is_click"] = time_ctr["is_click"] * 100
            
            fig_time = px.line(
                time_ctr, 
                x="time_of_day", 
                y="is_click", 
                markers=True,
                title="Performance Trend across the Day",
                labels={"time_of_day": "Time of Day", "is_click": "Average CTR (%)"}
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
        # Chart 3: Device Type vs Placement matrix
        st.subheader("Device & Placement Effectiveness")
        heat_data = df.groupby(["device_type", "ad_placement"])["is_click"].mean().reset_index()
        heat_data["is_click"] = heat_data["is_click"] * 100
        heat_pivot = heat_data.pivot(index="device_type", columns="ad_placement", values="is_click")
        
        fig_heat = px.imshow(
            heat_pivot, 
            text_auto=".1f",
            title="CTR Heatmap: Device Type vs Ad Placement (%)",
            labels=dict(x="Ad Placement", y="Device Type", color="CTR (%)"),
            aspect="auto"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Dataset `ctr_data.csv` not found. Please ensure `python generate_data.py` has been executed to generate the historical data.")
