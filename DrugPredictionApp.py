import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load pre-trained XGBoost model (you should replace this with your trained model file path)
#@st.cache_resource
#def load_model():
    # Replace 'xgboost_model.json' with your model's actual file path
    #model = xgb.XGBRegressor()
    #model.load_model("xgboost_model.json")
    #return model

#model = load_model()

# Sidebar Information
st.sidebar.title("Pharmacy Inventory Management")
st.sidebar.info(
    """
    **Purpose**: Simplify inventory counting for oral solid dosage forms using weights.
    """
)

# App Title
st.title("Pharmacy Inventory Management with Machine Learning")

# Section: Barcode and Drug Information
st.subheader("Step 1: Enter Drug Information")
barcode = st.text_input("Scan Drug Bin Barcode:")
drug_name = st.text_input("Drug Name:")
strength = st.number_input("Drug Strength (mg):", min_value=0.0, format="%.2f")
if st.button("Retrieve Info"):
    # Simulate database retrieval
    st.success(f"Information retrieved for: {drug_name} (Barcode: {barcode})")

# Section: Enter Packaging Details
st.subheader("Step 2: Enter Packaging and Weight Details")
# Allow multiple packaging types to be selected
packaging_types = st.multiselect(
    "Select Packaging Type(s):",
    options=["Full Box", "Strip", "Loose Cut"],
    default=["Full Box"]  # Set a default selection if needed
)

# Display the selected packaging types
if packaging_types:
    st.write(f"Selected Packaging Types: {', '.join(packaging_types)}")
else:
    st.warning("Please select at least one packaging type.")
weights = {}
for packaging in packaging_types:
    weights[packaging] = st.number_input(
        f"Enter weight for {packaging} (grams):", min_value=0.0, format="%.2f"
    )

# Display weights entered
if weights:
    st.write("Weights entered for each packaging type:")
    st.write(weights)
weight = st.number_input(f"Enter weight for {packaging_types} (grams):", min_value=0.0, format="%.2f")
if st.button("Add Weight Details"):
    st.session_state['input_data'] = {
        "Weight": weight,
        "Strength": strength,
       
    }
    st.success(f"Details recorded: {packaging_types}, {weight} grams")

# Section: Machine Learning Prediction
st.subheader("Step 3: Predict Drug Quantity")
if "input_data" in st.session_state and st.button("Predict Quantity"):
    # Prepare input data for prediction
    input_data = pd.DataFrame([st.session_state['input_data']])
    
    # Perform feature scaling or transformations if required (example here is placeholder)
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)
    
    # Predict using the loaded model
    #prediction = model.predict(input_scaled)
    #predicted_quantity = np.round(prediction[0])  # Round to nearest whole number
    
    #st.success(f"Predicted quantity: {int(predicted_quantity)} tablets.")

# Section: Save and Confirm
st.subheader("Step 4: Confirm and Save Data")
if st.button("Save Entry"):
    saved_data = {
        "Drug Name": drug_name,
        "Barcode": barcode,
        #"Predicted Quantity": predicted_quantity,
        "Weight": weight,
        
    }
    # Save the data (e.g., append to a CSV or database)
    st.success("Data saved successfully!")
if st.button("Clear Form"):
    st.session_state['input_data'] = {}
    st.success("Form cleared.")

# Footer: Additional Notes
st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    - Step 1: Scan the drug barcode and retrieve basic drug information.
    - Step 2: Enter weight and packaging details.
    - Step 3: Use the model to predict the drug quantity.
    - Step 4: Confirm and save the predicted quantity to the database.
    """
)
with st.form(key='drug_entry_form'):
    barcode = st.text_input("Scan Drug Bin Barcode:")
    drug_name = st.text_input("Drug Name:")
    weight = st.number_input("Enter Weight (grams):", min_value=0.0, format="%.2f")
    strength = st.number_input("Drug Strength (mg):", min_value=0.0, format="%.2f")
  

    # Submit button
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.success("Form submitted!")

tab1, tab2, tab3 = st.tabs(["Input Details", "Predictions", "Saved Data"])

with tab1:
    st.subheader("Step 1: Enter Drug Details")
    # Inputs for drug details

with tab2:
    st.subheader("Step 2: Model Prediction")
    # Prediction interface

with tab3:
    st.subheader("Step 3: View Saved Entries")
    # Display saved data

if st.session_state.get('saved_data', []):
    st.dataframe(pd.DataFrame(st.session_state['saved_data']))
else:
    st.info("No data to display yet.")

uploaded_file = st.file_uploader("Upload a CSV file for bulk processing:")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)  # Display uploaded data

