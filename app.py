import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

# Load encoders and scaler
venue_encoder = pd.read_pickle("venue_encoder.pkl")
bat_encoder = pd.read_pickle("bat_encoder.pkl")
bowl_encoder = pd.read_pickle("bowl_encoder.pkl")
striker_encoder = pd.read_pickle("striker_encoder.pkl")
bowler_encoder = pd.read_pickle("bowler_encoder.pkl")
scaler = pd.read_pickle("scaler.pkl")

# Load trained model
model = load_model("ipl_model.h5")

# Sample dropdown values (you can also load from CSV if needed)
venues = venue_encoder.classes_.tolist()
bat_teams = bat_encoder.classes_.tolist()
bowl_teams = bowl_encoder.classes_.tolist()
strikers = striker_encoder.classes_.tolist()
bowlers = bowler_encoder.classes_.tolist()

st.title("🏏 IPL Final Score Predictor")

venue = st.selectbox("Select Venue", venues)
bat_team = st.selectbox("Select Batting Team", bat_teams)
bowl_team = st.selectbox("Select Bowling Team", bowl_teams)
striker = st.selectbox("Select Striker", strikers)
bowler = st.selectbox("Select Bowler", bowlers)

if st.button("Predict Score"):
    try:
        encoded = [
            venue_encoder.transform([venue])[0],
            bat_encoder.transform([bat_team])[0],
            bowl_encoder.transform([bowl_team])[0],
            striker_encoder.transform([striker])[0],
            bowler_encoder.transform([bowler])[0]
        ]

        input_array = np.array(encoded).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0][0]

        st.success(f"🎯 Predicted Final Score: {int(prediction)} runs")
    except Exception as e:
        st.error(f"Error: {e}")