import os
import sys
from flask import Flask, render_template, request, redirect, url_for, jsonify
import joblib
import pandas as pd

# Resolve project root and ensure src can be imported cleanly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import feature definitions and transformer required for unpickling the model pipeline
from src.features import FeatureEngineer, FEATURE_COLUMNS

app = Flask(__name__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_web_model.pkl")

# Load the trained website model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Error loading model from {MODEL_PATH}: {e}")

FEATURE_NAMES = FEATURE_COLUMNS

# Standard feature defaults for initial load or sample presets
DEFAULT_INPUTS = {
    "OverallQual": "7",
    "GrLivArea": "1750",
    "FullBath": "2",
    "BedroomAbvGr": "3",
    "YearBuilt": "2008",
    "GarageCars": "2",
    "TotalBsmtSF": "850"
}


def calculate_metrics(prediction, gr_liv_area, overall_qual):
    """Calculate supplementary real-estate insights."""
    # Clamping negative predictions if extreme out-of-distribution values are entered
    price = max(10000.0, float(prediction))

    # Price per square foot
    price_per_sqft = price / max(1.0, float(gr_liv_area))

    # Estimated 30-year fixed mortgage monthly payment (approx 6.5% interest, 20% down)
    loan_amount = price * 0.80
    monthly_interest_rate = 0.065 / 12
    num_payments = 360
    factor = (1 + monthly_interest_rate) ** num_payments
    monthly_mortgage = (
        loan_amount
        * (monthly_interest_rate * factor)
        / max(1e-6, factor - 1)
    )

    # Quality rating tier description
    if overall_qual >= 9:
        quality_tier = "Luxury / Exceptional"
        badge_class = "badge-luxury"
    elif overall_qual >= 7:
        quality_tier = "High Quality / Modern"
        badge_class = "badge-high"
    elif overall_qual >= 5:
        quality_tier = "Standard / Moderate"
        badge_class = "badge-standard"
    else:
        quality_tier = "Economy / Fixer-Upper"
        badge_class = "badge-economy"

    return {
        "price": price,
        "formatted_price": f"${price:,.2f}",
        "price_per_sqft": f"${price_per_sqft:,.2f}",
        "monthly_mortgage": f"${monthly_mortgage:,.0f}/mo",
        "quality_tier": quality_tier,
        "badge_class": badge_class
    }


def validate_and_extract_inputs(form_data):
    """Validate input parameters and enforce realistic bounds with friendly error messages."""
    field_labels = {
        "OverallQual": "Overall Quality (1-10)",
        "GrLivArea": "Above Ground Living Area (sq ft)",
        "FullBath": "Number of Full Bathrooms",
        "BedroomAbvGr": "Number of Bedrooms",
        "YearBuilt": "Year Built",
        "GarageCars": "Garage Capacity (cars)",
        "TotalBsmtSF": "Basement Area (sq ft)"
    }

    # Verify presence of all required fields
    for field, label in field_labels.items():
        val = form_data.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            return None, f"Missing required field: Please enter a value for {label}."

    try:
        raw_qual = form_data.get("OverallQual")
        overall_qual = int(float(raw_qual))
        if not (1 <= overall_qual <= 10):
            return None, "Overall Quality must be an integer between 1 (Very Poor) and 10 (Very Excellent)."

        raw_area = form_data.get("GrLivArea")
        gr_liv_area = float(raw_area)
        if gr_liv_area <= 0:
            return None, "Living Area must be greater than 0 square feet."
        if gr_liv_area > 20000:
            return None, "Living Area exceeds realistic residential limits (max 20,000 sq ft)."

        raw_bath = form_data.get("FullBath")
        full_bath = float(raw_bath)
        if full_bath < 0 or full_bath > 15:
            return None, "Number of Full Bathrooms must be between 0 and 15."

        raw_bed = form_data.get("BedroomAbvGr")
        bedroom = float(raw_bed)
        if bedroom < 0 or bedroom > 20:
            return None, "Number of Bedrooms must be between 0 and 20."

        raw_year = form_data.get("YearBuilt")
        year_built = int(float(raw_year))
        if not (1800 <= year_built <= 2030):
            return None, "Year Built must be between 1800 and 2030."

        raw_cars = form_data.get("GarageCars")
        garage_cars = float(raw_cars)
        if garage_cars < 0 or garage_cars > 10:
            return None, "Garage Capacity must be between 0 and 10 cars."

        raw_bsmt = form_data.get("TotalBsmtSF")
        total_bsmt = float(raw_bsmt)
        if total_bsmt < 0 or total_bsmt > 15000:
            return None, "Basement Area must be between 0 and 15,000 square feet."

        input_data = pd.DataFrame(
            [[
                overall_qual,
                gr_liv_area,
                full_bath,
                bedroom,
                year_built,
                garage_cars,
                total_bsmt
            ]],
            columns=FEATURE_NAMES
        )

        extracted = {
            "OverallQual": str(overall_qual),
            "GrLivArea": str(int(gr_liv_area) if gr_liv_area.is_integer() else gr_liv_area),
            "FullBath": str(int(full_bath) if full_bath.is_integer() else full_bath),
            "BedroomAbvGr": str(int(bedroom) if bedroom.is_integer() else bedroom),
            "YearBuilt": str(year_built),
            "GarageCars": str(int(garage_cars) if garage_cars.is_integer() else garage_cars),
            "TotalBsmtSF": str(int(total_bsmt) if total_bsmt.is_integer() else total_bsmt)
        }

        return (input_data, extracted, overall_qual, gr_liv_area), None

    except (ValueError, TypeError) as err:
        return None, f"Invalid number entered: please ensure all inputs are numeric. Details: {err}"



@app.route("/")
def home():
    return render_template(
        "index.html",
        inputs=DEFAULT_INPUTS,
        prediction=None,
        metrics=None,
        error=None
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Gracefully redirect direct GET visits back to home
    if request.method == "GET":
        return redirect(url_for("home"))

    if model is None:
        return render_template(
            "index.html",
            inputs=request.form,
            error="Prediction model is not available. Please verify model file exists."
        )

    parsed_result, error_msg = validate_and_extract_inputs(request.form)
    if error_msg:
        return render_template(
            "index.html",
            inputs=request.form,
            error=error_msg
        )

    input_df, inputs_dict, overall_qual, gr_liv_area = parsed_result

    try:
        raw_prediction = model.predict(input_df)[0]
        metrics = calculate_metrics(raw_prediction, gr_liv_area, overall_qual)

        return render_template(
            "index.html",
            inputs=inputs_dict,
            prediction=metrics["formatted_price"],
            metrics=metrics,
            error=None
        )
    except Exception as e:
        return render_template(
            "index.html",
            inputs=request.form,
            error=f"Prediction failed: {str(e)}"
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for seamless client-side predictions."""
    if model is None:
        return jsonify({"success": False, "error": "Model file not loaded on server."}), 500

    data = request.get_json(silent=True) or request.form
    parsed_result, error_msg = validate_and_extract_inputs(data)

    if error_msg:
        return jsonify({"success": False, "error": error_msg}), 400

    input_df, inputs_dict, overall_qual, gr_liv_area = parsed_result

    try:
        raw_prediction = model.predict(input_df)[0]
        metrics = calculate_metrics(raw_prediction, gr_liv_area, overall_qual)
        return jsonify({
            "success": True,
            "prediction": metrics["price"],
            "formatted_price": metrics["formatted_price"],
            "metrics": metrics,
            "inputs": inputs_dict
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)