import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load trained model
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Big Mart Sales Prediction App")

# Input fields for user
item_weight = st.number_input("Item Weight", min_value=0.0, format="%.2f")
item_visibility = st.number_input("Item Visibility", min_value=0.0, format="%.4f")
item_mrp = st.number_input("Item MRP", min_value=0.0, format="%.2f")
outlet_age = st.number_input("Outlet Age", min_value=0, step=1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Sales"):
    # Build DataFrame with same feature names as training
    input_df = pd.DataFrame([{
        "Item_Weight": item_weight,
        "Item_Visibility": item_visibility,
        "Item_MRP": item_mrp,
        "Outlet_Age": outlet_age
    }])

    # Debugging (optional)
    st.write("Input shape:", input_df.shape)
    st.write("Input data:", input_df)

    # Predict
    try:
        result = model.predict(input_df)[0]
        st.success(f"Predicted Sales: {round(result, 2)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

