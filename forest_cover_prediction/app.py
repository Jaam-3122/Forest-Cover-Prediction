import streamlit as st
import pandas as pd
import pickle

# Load your trained model (update with your model filename)
model = pickle.load(open("xgboost_forest_model.pkl", "rb"))

st.title("Forest Cover Type Prediction")
st.write("Enter the environmental data to predict the forest cover type.")

# Example inputs - replace/add your features here
elevation = st.number_input("Elevation", min_value=0, max_value=5000, value=2000)
slope = st.number_input("Slope", min_value=0, max_value=90, value=15)
horizontal_distance_to_water = st.number_input("Horizontal Distance to Water", min_value=0, max_value=5000, value=500)
vertical_distance_to_water = st.number_input("Vertical Distance to Water", min_value=-100, max_value=1000, value=30)
horizontal_distance_to_roadways = st.number_input("Horizontal Distance to Roadways", min_value=0, max_value=10000, value=1000)

# Add more inputs here based on your dataset

if st.button("Predict"):
    # Convert inputs to dataframe
    input_data = pd.DataFrame([[elevation, slope, horizontal_distance_to_water, vertical_distance_to_water, horizontal_distance_to_roadways]],
                              columns=["Elevation", "Slope", "Horizontal_Distance_To_Water", "Vertical_Distance_To_Water", "Horizontal_Distance_To_Roadways"])
    
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Forest Cover Type: {prediction}")
