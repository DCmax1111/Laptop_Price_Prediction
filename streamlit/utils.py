import re
import os
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

# Paths
MODEL_PATH = "models/best_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"
LOG_FILE = "logs/input_errors.log"

# Load model + features (fail gracefully if missing)
try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    # st.write("Loaded feature names count:", len(feature_names))
    # st.write("First few features:", feature_names[:10])

except Exception as e:
    # Delay hard failure until prediction time; keep CLI usable for now.
    st.error(f"Error loading feature_names.")
    model = None
    feature_names = None

# Canonical choices that match your training data casing
VALID_COMPANIES = ["Dell", "Apple", "Hp", "Lenovo", "Acer", "Asus", "Msi"]
VALID_TYPES     = ["Notebook", "Ultrabook", "Gaming", "2 In 1 Convertible", "Workstation"]
VALID_OSS       = ["Windows 10", "Windows 7", "Macos", "Linux", "No Os"]

# Aliases to improve UX (lowercase keys)
OPSYS_ALIASES = {
    "windows 10": ["10", "win10", "windows10", "windows 10", "windows 10 home", "windows 10 pro"],
    "windows 7":  ["7", "win7", "windows7", "windows 7"],
    "macos":      ["macos", "mac os", "osx", "os x", "mac os x"],
    "linux":      ["linux", "ubuntu", "debian", "fedora", "mint"],
    "no os":      ["no os", "none", "no operating system", "without os", "freedos", "dos"]
}
# Map canonical (lower) -> dataset casing
OPSYS_CANON_TO_DATA = {
    "windows 10": "Windows 10",
    "windows 7":  "Windows 7",
    "macos":      "Macos",
    "linux":      "Linux",
    "no os":      "No Os"
}

COMPANY_ALIASES = {
    "hp": "Hp",
    "hewlett packard": "Hp",
    "HP": "Hp",
    "msi": "Msi",
    "micro star": "Msi",
}

TYPE_ALIASES = {
    "2 in 1": "2 In 1 Convertible",
    "2-in-1": "2 In 1 Convertible",
    "convertible": "2 In 1 Convertible",
    "gaming laptop": "Gaming",
    "ultrabook laptop": "Ultrabook",
    "work station": "Workstation",
}

def log_event(kind, field, value, message):
    """Log invalid input or auto-corrections to a file."""
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {kind.upper()} | Field: {field} | Value: '{value}' | {message}\n")
    except Exception as e:
        # Never crash on logging, but show debug info in console.
        st.error(f"Logging event failed.")

# ---------- Smart parsing helpers ----------

NUM_WORDS = {
    "zero": 0, "none": 0, "nil": 0, "no": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twelve": 12, "sixteen": 16, "twenty": 20, "thirty two": 32, "thirty-two": 32,
    "sixty four": 64, "sixty-four": 64, "one hundred": 100, "one hundred twenty eight": 128,
}

def _coerce_numeric_token(raw: str):
    """Try hard to get a number out of a messy string (e.g., '512GB', '1,024', 'zero')."""
    s = raw.strip().lower()
    s = s.replace(",", " ")
    s = s.replace("gb", " ").replace("g", " ").replace("kg", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Exact word matches
    if s in NUM_WORDS:
        return NUM_WORDS[s]

    # Extract first number (int or float)
    m = re.search(r"\d+(\.\d+)?", s)
    if m:
        token = m.group(0)
        # Return float if it had a decimal point, else int
        return float(token) if "." in token else int(token)

    return None

def safe_numeric(prompt, cast_type=float, default=None, min_val=None, max_val=None, field=""):
    """Numeric input with coercion, warnings, and bounds checking."""
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default

        coerced = _coerce_numeric_token(raw)
        if coerced is not None:
            # Cast to desired type
            try:
                val = cast_type(coerced)
            except Exception:
                print(f"⚠️ Please enter a valid {cast_type.__name__}.")
                log_event("error", field, raw, f"Cast to {cast_type.__name__} failed after coercion")
                continue

            # Bounds
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                if min_val is not None and val < min_val:
                    print(f"⚠️ Value too low. Minimum allowed is {min_val}. Try again.")
                    log_event("error", field, raw, f"Below min ({min_val}) after coercion -> {val}")
                else:
                    print(f"⚠️ Value too high. Maximum allowed is {max_val}. Try again.")
                    log_event("error", field, raw, f"Above max ({max_val}) after coercion -> {val}")
                continue

            # Warn if auto-coerced from a non-pure number
            if str(raw).strip() != str(val):
                print(f"⚠️ Interpreted '{raw}' as {val}.")
                log_event("warn", field, raw, f"Auto-coerced to {val}")

            return val

        # Could not coerce anything numeric
        print(f"⚠️ Please enter a valid {cast_type.__name__}.")
        log_event("error", field, raw, "Unparseable numeric input")

def _normalize_from_aliases(raw_lower: str, alias_map: dict):
    """Return canonical lower key if any alias matches (by equality or containment)."""
    for canonical, aliases in alias_map.items():
        if raw_lower == canonical:
            return canonical
        for a in aliases:
            if raw_lower == a or a in raw_lower:
                return canonical
    return None

def normalize_opsys(raw: str):
    s = raw.strip().lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    canonical = _normalize_from_aliases(s, OPSYS_ALIASES)
    if canonical:
        return OPSYS_CANON_TO_DATA[canonical]  # match dataset casing
    return None

def normalize_company(raw: str):
    s = raw.strip().lower()
    if s in [c.lower() for c in VALID_COMPANIES]:
        # Return dataset casing
        for c in VALID_COMPANIES:
            if s == c.lower():
                return c
    if s in COMPANY_ALIASES:
        fixed = COMPANY_ALIASES[s]
        print(f"⚠️ Interpreted '{raw}' as '{fixed}'.")
        log_event("warn", "Company", raw, f"Auto-corrected to {fixed}")
        return fixed
    return None

def normalize_type(raw: str):
    s = raw.strip().lower()
    if s in [t.lower() for t in VALID_TYPES]:
        for t in VALID_TYPES:
            if s == t.lower():
                return t
    for k, v in TYPE_ALIASES.items():
        if k in s:
            print(f"⚠️ Interpreted '{raw}' as '{v}'.")
            log_event("warn", "TypeName", raw, f"Auto-corrected to {v}")
            return v
    return None

def safe_choice_normalized(prompt, field, normalizer, options_list, default=None):
    """Choice input with normalization + warnings; never crashes."""
    options_str = ", ".join(options_list)
    while True:
        raw = input(f"{prompt} ({options_str}): ").strip()
        if raw == "" and default is not None:
            return default
        fixed = normalizer(raw)
        if fixed and fixed in options_list:
            # Warn if normalization changed the value
            if fixed.lower() != raw.strip().lower():
                print(f"⚠️ Interpreting '{raw}' as '{fixed}'.")
                log_event("warn", field, raw, f"Normalized to {fixed}")
            return fixed
        print("⚠️ Invalid choice. Please pick from the list (type the full name).")
        log_event("error", field, raw, "Invalid choice")

# ---------- Public API used by predict.py ----------

def get_user_inputs():
    """Collect laptop specifications safely with validations."""
    print("Laptop Price Prediction Demo")
    print("Please enter your laptop specifications.\n")

    company = safe_choice_normalized("Company", "Company", normalize_company, VALID_COMPANIES, default="Dell")
    typename = safe_choice_normalized("Type", "TypeName", normalize_type, VALID_TYPES, default="Notebook")
    inches = safe_numeric("Screen Size (inches): ", float, default=15.6, min_val=10, max_val=20, field="Inches")
    ram = safe_numeric("RAM (GB): ", int, default=8, min_val=2, max_val=128, field="Ram")
    weight = safe_numeric("Weight (kg): ", float, default=2.0, min_val=0.5, max_val=5.0, field="Weight")
    opsys = None
    while opsys is None:
        raw_os = input(f"Operating System (e.g., Windows 10, MacOS, Linux) ({', '.join(VALID_OSS)}): ").strip()
        if raw_os == "":
            opsys = "Windows 10"
            break
        fixed = normalize_opsys(raw_os)
        if fixed:
            if fixed.lower() != raw_os.strip().lower():
                print(f"⚠️ Interpreting '{raw_os}' as '{fixed}'.")
                log_event("warn", "OpSys", raw_os, f"Normalized to {fixed}")
            opsys = fixed
        else:
            print("⚠️ Invalid OS. Please type a full name like 'Windows 10', 'MacOS', 'Linux', or 'No OS'.")
            log_event("error", "OpSys", raw_os, "Invalid OS")

    ssd = safe_numeric("SSD size (GB, 0 if none): ", int, default=0, min_val=0, field="SSD")
    hdd = safe_numeric("HDD size (GB, 0 if none): ", int, default=0, min_val=0, field="HDD")
    hybrid = safe_numeric("Hybrid storage (GB, 0 if none): ", int, default=0, min_val=0, field="Hybrid")
    flash = safe_numeric("Flash storage (GB, 0 if none): ", int, default=0, min_val=0, field="Flash_Storage")

    cpu = input("CPU (e.g., Intel Core i5, AMD Ryzen 7): ").strip()
    gpu = input("GPU (e.g., Nvidia GeForce GTX 1050, Intel HD Graphics): ").strip()

    return {
        "Company": company,
        "TypeName": typename,
        "Inches": inches,
        "Ram": ram,
        "Weight": weight,
        "OpSys": opsys,
        "SSD": ssd,
        "HDD": hdd,
        "Hybrid": hybrid,
        "Flash_Storage": flash,
        "Cpu": cpu,
        "Gpu": gpu
    }
 
def preprocess_input(sample_dict, feature_names):
    """Convert user inputs into the same format as training data."""
    if feature_names is None:
        print("⚠️ Feature files not found. Please run the training notebook to generate models first.")
        return None
    
    # Convert dict -> DataFrame
    sample = pd.DataFrame([sample_dict])

    # Normalize field names to training-time keys
    if "Cpu" in sample.columns:
        sample = sample.rename(columns={"Cpu": "Cpu_brand"})
    if "Gpu" in sample.columns:
        sample = sample.rename(columns={"Gpu": "Gpu_brand"})
    if "OpSys" in sample.columns:
        sample = sample.rename(columns={"OpSys": "OpSys"})

    # One-hot encode & align
    sample_encoded = pd.get_dummies(sample)
    sample_encoded = sample_encoded.reindex(columns=feature_names, fill_value=0)

    # numeric_columns = sample_encoded.select_dtypes(include=["float64", "int64"]).columns
    # sample_encoded[numeric_columns] = scaler.transform(sample_encoded[numeric_columns])
    return sample_encoded

def predict_price(model, sample_encoded):
    """Predict laptop price given specs; logs issues if any."""
    # Safety Checks
    if model is None:
        st.warning("⚠️ Model not loaded. Please train first.")
        return None
    
    try:
        pred = model.predict(sample_encoded)[0]
        pred = round(float(pred), 4)

        # Log outliers (dataset: ~€100–€6990)
        if pred < 100 or pred > 6990:
            log_event("warn", "PREDICT", str(pred), f"Unrealistic prediction generated: €{pred}.")

        return pred
    
    except Exception as e:
        st.warning("⚠️ Prediction failed. Please check preprocessing.")
        log_event("error", "PREDICT", str(sample_encoded), f"Prediction failure: {e}")
        return None

def final_price(pred, company, typename, touch):
    """
    Adjust predicted laptop price based on range and touchscreen feature.
    
    Parameters:
        pred (float): initial predicted price
        touch (int): 1 if touchscreen, 0 if not
        
    Returns:
        f_prediction: adjusted price"""
    # Base adjustment by price range
    if 100 < pred < 400:     # Low-end.
        final_pred = pred * 1.05   # +5%
        if touch == "No":
            st.info(f"Approximate Price for the {str(company)} {str(typename)}: €{pred:.2f}")
            st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")
        else:  # Touchscreen price bump (+50)
            pred += 50
            final_pred += 50
            st.info(f"Approximate Price for the {str(company)} {str(typename)}: €{pred:.2f}")
            st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")

    elif 400 < pred < 800:  # Mid-range.
        final_pred = pred * 0.98   # -2%
        # Touchscreen price bump (+100)
        if touch == "No":
            st.info(f"Approximate Price for the {str(company)} {str(typename)}: €{pred:.2f}")
            st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")
        else:
            pred += 100
            final_pred += 100
            st.info(f"Approximate Price for the {str(company)} {str(typename)}: €{pred:.2f}")
            st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")

    elif pred >= 800:     # High-end. So we keep it as is.
        # Touchscreen price bump (+150)
        if touch == "No":
            st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")
        else:
            final_pred += 150
            st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")

    else:
        st.warning("⚠️ Prediction failed. Please check preprocessing.")
        log_event("error", "PREDICT", str(pred), f"Prediction failure")
        return None
