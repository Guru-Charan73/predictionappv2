from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model and feature column order
model = joblib.load('xgb_sales_model.pkl')
feature_columns = joblib.load('xgb_feature_columns.pkl')

# Label Encoders with predefined classes
fat_content_encoder = LabelEncoder()
fat_content_encoder.classes_ = np.array(['Low Fat', 'Regular'])

outlet_size_encoder = LabelEncoder()
outlet_size_encoder.classes_ = np.array(['High', 'Medium', 'Small'])

location_type_encoder = LabelEncoder()
location_type_encoder.classes_ = np.array(['Tier 1', 'Tier 2', 'Tier 3'])

outlet_type_encoder = LabelEncoder()
outlet_type_encoder.classes_ = np.array([
    'Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'
])

outlet_id_encoder = LabelEncoder()
outlet_id_encoder.classes_ = np.array([
    'OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019',
    'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'
])

item_type_encoder = LabelEncoder()
item_type_encoder.classes_ = np.array([
    'Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods',
    'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household',
    'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'
])

@app.route('/predict', methods=['POST'])
def predict_sales():
    try:
        input_data = request.get_json()

        # ❌ Reject if input is a list
        if isinstance(input_data, list):
            return jsonify({'error': 'Only single record supported. Do not send a list.'}), 400

        # ✅ Ensure input is a dictionary
        if not isinstance(input_data, dict):
            return jsonify({'error': 'Invalid input format: must be a single JSON object'}), 400

        # Wrap in a list to use DataFrame as before
        df = pd.DataFrame([input_data])

        # Standardize Fat Content values
        df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)

        # Fill missing values
        df['Item_Weight'].fillna(12.8576451841, inplace=True)
        df['Outlet_Size'].fillna('Medium', inplace=True)

        # Feature engineering
        df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']

        # Label Encoding
        df['Item_Fat_Content'] = fat_content_encoder.transform(df['Item_Fat_Content'])
        df['Outlet_Size'] = outlet_size_encoder.transform(df['Outlet_Size'])
        df['Outlet_Location_Type'] = location_type_encoder.transform(df['Outlet_Location_Type'])
        df['Outlet_Type'] = outlet_type_encoder.transform(df['Outlet_Type'])
        df['Outlet_Identifier'] = outlet_id_encoder.transform(df['Outlet_Identifier'])
        df['Item_Type'] = item_type_encoder.transform(df['Item_Type'])

        # Drop unused fields
        df.drop(['Item_Identifier', 'Outlet_Establishment_Year'], axis=1, inplace=True)

        # Convert numeric columns to float
        numeric_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add missing columns
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns
        df = df[feature_columns]

        # Predict
        prediction = model.predict(df)[0]
        prediction = round(float(prediction), 2)

        return jsonify({'predicted_sales': prediction})

    except Exception as e:
        return jsonify({'error': f"Sales prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
