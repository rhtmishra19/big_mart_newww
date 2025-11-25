import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Big Mart Sales Prediction App")

# Numeric inputs
item_weight = st.number_input("Item Weight", min_value=0.0, format="%.2f")
item_visibility = st.number_input("Item Visibility", min_value=0.0, format="%.4f")
item_mrp = st.number_input("Item MRP", min_value=0.0, format="%.2f")
outlet_est_year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2025, step=1)

# Categorical inputs (must match training preprocessing)
item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
item_type = st.selectbox(
    'Item Type',
    ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods',
     'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks',
     'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood']
)
outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])

if st.button("Predict Sales"):
    # Build DataFrame with all required features
    input_df = pd.DataFrame([{
        "Item_Weight": item_weight,
        "Item_Fat_Content": item_fat_content,
        "Item_Visibility": item_visibility,
        "Item_Type": item_type,
        "Item_MRP": item_mrp,
        "Outlet_Establishment_Year": outlet_est_year,
        "Outlet_Size": outlet_size,
        "Outlet_Location_Type": outlet_location_type,
        "Outlet_Type": outlet_type
    }])
    input_df['Item_Fat_Content'] = input_df['Item_Fat_Content'].astype('category')
    input_df['Item_Type'] = input_df['Item_Type'].astype('category')
    input_df['Outlet_Size'] = input_df['Outlet_Size'].astype('category')
    input_df['Outlet_Location_Type'] = input_df['Outlet_Location_Type'].astype('category')
    input_df['Outlet_Type'] = input_df['Outlet_Type'].astype('category')
    try:
        result = model.predict(input_df)[0]
        st.success(f"Predicted Sales: {round(result, 2)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
