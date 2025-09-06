import streamlit as st
import pandas as pd
import joblib
import os

# Configure page
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

@st.cache_resource
def load_model_and_features():
    """Load model and feature names with error handling"""
    try:
        MODEL_PATH = "models/laptop_price_model.pkl"
        FEATURES_PATH = "models/feature_names.pkl"
        
        if not os.path.exists(MODEL_PATH):
            st.error("Model files not found. Please ensure model files are in the correct directory.")
            return None, None
        if not os.path.exists(FEATURES_PATH):
            st.error("Feature files not found. Please ensure feature files are in the correct directory.")
            return None, None
            
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and features
model, feature_names = load_model_and_features()

if model is None or feature_names is None:
    st.stop()

# Header
st.title("💻 Laptop Price Prediction")
st.markdown("---")
st.write("Enter your laptop specifications below to get an estimated price prediction.")

# Create columns for better layout
col1, col2 = st.columns(2)

# Define options
companies = ["Dell", "Apple", "Hp", "Lenovo", "Acer", "Asus", "Msi"]
types = ["Notebook", "Ultrabook", "Gaming", "2 In 1 Convertible", "Workstation"]
oss = ["Windows 10", "Windows 7", "Macos", "Linux", "No Os"]

gpus = [
    "Intel HD Graphics 400",
    "Intel HD Graphics 620", 
    "Intel Iris Plus Graphics 640",
    "AMD Radeon R5 M330",
    "AMD Radeon 530",
    "Nvidia GeForce 940MX",
    "Nvidia GeForce GTX 960M",
    "Nvidia GeForce GTX 1050 Ti",
    "Nvidia GeForce GTX 1060",
    "Nvidia GeForce GTX 1070"
]

cpus = [
    "Intel Core i7 7700HQ",
    "Intel Core i7 7500U", 
    "AMD A9-Series 9420",
    "AMD A12-Series 9720P",
    "Intel Core i5 7200U",
    "Intel Core i5 8250U",
    "Intel Core i3 6006U",
    "Intel Core i3 7100U",
    "Intel Celeron Dual Core N3060",
    "Intel Pentium Quad Core N4200"
]

# Input fields in columns
with col1:
    st.subheader("🏢 Basic Specifications")
    company = st.selectbox("Company", companies)
    type_name = st.selectbox("Type", types)
    inches = st.slider("Screen Size (inches)", 10.0, 20.0, 15.6, 0.1)
    ram = st.slider("RAM (GB)", 2, 128, 8, 2)
    weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)
    opsys = st.selectbox("Operating System", oss)

with col2:
    st.subheader("💾 Storage & Hardware")
    ssd = st.number_input("SSD Size (GB)", min_value=0, value=256, step=128)
    hdd = st.number_input("HDD Size (GB)", min_value=0, value=0, step=500)
    hybrid = st.number_input("Hybrid Storage (GB)", min_value=0, value=0, step=128)
    flash_storage = st.number_input("Flash Storage (GB)", min_value=0, value=0, step=32)
    cpu = st.selectbox("CPU", cpus)
    gpu = st.selectbox("GPU", gpus)

# Storage validation
total_storage = ssd + hdd + hybrid + flash_storage
if total_storage == 0:
    st.warning("⚠️ Please specify at least one type of storage.")

# Create input dictionary
input_dict = {
    "Company": company,
    "TypeName": type_name,
    "Inches": inches,
    "Ram": ram,
    "Weight": weight,
    "OpSys": opsys,
    "SSD": ssd,
    "HDD": hdd,
    "Hybrid": hybrid,
    "Flash_Storage": flash_storage,
    "Cpu": cpu,
    "Gpu": gpu,
    
}

def predict_price(sample_dict):
    """Predict laptop price with error handling"""
    try:
        sample = pd.DataFrame([sample_dict])
        sample_encoded = pd.get_dummies(sample)
        sample_encoded = sample_encoded.reindex(columns=feature_names, fill_value=0)
        prediction = model.predict(sample_encoded)[0]
        return round(prediction, 2)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Prediction section
st.markdown("---")
col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])

with col_pred2:
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        if total_storage == 0:
            st.error("❌ Cannot predict price without storage specification.")
        elif not cpu or not gpu:
            st.error("❌ Please specify both CPU and GPU.")
        else:
            with st.spinner("Calculating price..."):
                price = predict_price(input_dict)
                if price is not None:
                    st.success(f"💰 **Predicted Laptop Price: €{price:,.2f}**")
                    
                    # Additional info
                    st.info(f"""
                    **Configuration Summary:**
                    - {company} {type_name}
                    - {inches}" display, {ram}GB RAM
                    - Storage: {total_storage}GB total
                    - {cpu} + {gpu}
                    - {cpu} ({speed} GHz) + {gpu}
                    """)

# Display current configuration
with st.expander("📋 Current Configuration Details"):
    config_df = pd.DataFrame(list(input_dict.items()), columns=['Specification', 'Value'])
    config_df["Value"] = config_df["Value"].astype(str)
    st.dataframe(config_df, width='stretch')

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    💡 Tip: Adjust the specifications above to see how they affect the predicted price.<br>
    This prediction is based on historical laptop data and market trends.
    </div>
    """, 
    unsafe_allow_html=True
)