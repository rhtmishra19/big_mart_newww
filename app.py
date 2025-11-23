import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Big Mart Sales Prediction App")

# Input fields for user
item_weight = st.number_input("Item Weight")
item_visibility = st.number_input("Item Visibility")
item_mrp = st.number_input("Item MRP")
outlet_age = st.number_input("Outlet Age")

# Predict button
if st.button("Predict Sales"):
    input_data = np.array([[item_weight, item_visibility, item_mrp, outlet_age]])
    result = model.predict(input_data)[0]
    st.success(f"Predicted Sales: {round(result,2)}")
