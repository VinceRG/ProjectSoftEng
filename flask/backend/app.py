from flask import Flask, jsonify, render_template, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import itertools
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Config
CSV_PATH = 'master_dataset_cleaned.csv'
CASE_JSON = 'case_dictionary.json'
MODEL_FILE = 'random_forest_model.pkl'
HOLIDAY_MONTHS = [1, 11, 12]
CONSULTATION_MAP = {1: "Consultation", 2: "Diagnosis", 3: "Mortality"}

# Globals
df = None
case_map = {}
model = None
X_train_columns = None

def load_data():
    global df, case_map
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found.")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} rows from CSV")
    print(f"✓ Columns: {df.columns.tolist()}")
    print(f"✓ Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"✓ Consultation Types: {sorted(df['Consultation_Type'].unique())}")

    if os.path.exists(CASE_JSON):
        with open(CASE_JSON, 'r') as f:
            case_map_str_keys = json.load(f)
            case_map = {int(v): k for k, v in case_map_str_keys.items()}
        print(f"✓ Loaded {len(case_map)} case mappings")
    else:
        case_map = {}
        print("⚠ No case dictionary found")

def train_or_load_model():
    global model, X_train_columns
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        X_train_columns = model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else None
        print(f"✓ Loaded existing model with {len(X_train_columns) if X_train_columns else 0} features")
        return

    print("Training new model...")
    X = df.drop(columns=['Total'])
    y = df['Total']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_columns = X_train.columns.tolist()

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

    y_pred = model.predict(X_test)
    y_pred[y_pred < 0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✓ Model trained. MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "data_loaded": df is not None,
        "model_loaded": model is not None,
        "total_rows": len(df) if df is not None else 0,
        "years": [int(df['Year'].min()), int(df['Year'].max())] if df is not None else [],
        "consultation_types": sorted(df['Consultation_Type'].unique().tolist()) if df is not None else []
    })

@app.route('/api/holiday_comparison')
def api_holiday_comparison():
    try:
        monthly_totals = df.groupby(['Year','Month'])['Total'].sum().reset_index()
        monthly_totals['is_major_holiday'] = monthly_totals['Month'].isin(HOLIDAY_MONTHS).astype(int)
        holiday_avg = float(monthly_totals[monthly_totals['is_major_holiday']==1]['Total'].mean())
        non_holiday_avg = float(monthly_totals[monthly_totals['is_major_holiday']==0]['Total'].mean())
        return jsonify({
            "holiday_months": HOLIDAY_MONTHS,
            "holiday_avg": holiday_avg,
            "non_holiday_avg": non_holiday_avg
        })
    except Exception as e:
        print(f"Error in holiday_comparison: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/monthly_totals')
def api_monthly_totals():
    try:
        consult_type = request.args.get('consult_type', default=None, type=int)
        year = request.args.get('year', default=None, type=int)
        print(f"API call: monthly_totals, consult_type={consult_type}, year={year}")
        
        df_copy = df.copy()
        if consult_type is not None:
            df_copy = df_copy[df_copy['Consultation_Type'] == consult_type]
            print(f"  Filtered to {len(df_copy)} rows for type {consult_type}")

        if year is not None:
            df_copy = df_copy[df_copy['Year'] == year]
            print(f"  Filtered to {len(df_copy)} rows for year {year}")
        
        monthly = df_copy.groupby(['Year', 'Month'])['Total'].sum().reset_index()
        monthly['label'] = monthly.apply(lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}", axis=1)
        
        result = monthly[['label','Year','Month','Total']].to_dict(orient='records')
        print(f"  Returning {len(result)} monthly records")
        return jsonify(result)
    except Exception as e:
        print(f"Error in monthly_totals: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/month_cases')
def api_month_cases():
    try:
        year = request.args.get('year', type=int)
        month = request.args.get('month', type=int)
        consult_type = request.args.get('consult_type', type=int)
        
        print(f"API call: month_cases, year={year}, month={month}, type={consult_type}")
        
        if year is None or month is None or consult_type is None:
            return jsonify({"error":"provide year, month, consult_type as query params"}), 400
        
        cropped = df[(df['Year']==year)&(df['Month']==month)&(df['Consultation_Type']==consult_type)]
        print(f"  Found {len(cropped)} rows")
        
        if cropped.empty:
            return jsonify([])
        
        agg = cropped.groupby('Case')['Total'].sum().reset_index().sort_values('Total', ascending=False).head(20)
        agg['Case_Name'] = agg['Case'].map(case_map).fillna(agg['Case'].apply(lambda x: f'Case {x} (No Name)'))
        
        result = agg.to_dict(orient='records')
        print(f"  Returning {len(result)} cases")
        return jsonify(result)
    except Exception as e:
        print(f"Error in month_cases: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/top_predictions')
def api_top_predictions():
    try:
        last_year = int(df['Year'].max())
        last_month = int(df[df['Year']==last_year]['Month'].max())
        
        if last_month == 12:
            next_month = 1
            next_year = last_year + 1
        else:
            next_month = last_month + 1
            next_year = last_year

        print(f"API call: top_predictions for {next_year}-{next_month:02d}")

        all_cases = df['Case'].unique()
        all_consult_types = df['Consultation_Type'].unique()
        all_sexes = df['Sex'].unique()
        all_age_ranges = df['Age_range'].unique()

        future_combinations = list(itertools.product(all_consult_types, all_cases, all_sexes, all_age_ranges))
        X_future = pd.DataFrame(future_combinations, columns=['Consultation_Type', 'Case', 'Sex', 'Age_range'])
        X_future['Year'] = next_year
        X_future['Month'] = next_month
        X_future['is_major_holiday'] = 1 if next_month in HOLIDAY_MONTHS else 0

        if X_train_columns is not None:
            for c in X_train_columns:
                if c not in X_future.columns:
                    X_future[c] = 0
            X_future = X_future[X_train_columns]

        preds = model.predict(X_future)
        preds[preds < 0] = 0
        X_future['Predicted_Total'] = preds

        future_case_totals = X_future.groupby(['Consultation_Type', 'Case'])['Predicted_Total'].sum().reset_index()
        future_case_totals['Predicted_Total'] = future_case_totals['Predicted_Total'].round(0).astype(int)

        top_5 = future_case_totals.sort_values('Predicted_Total', ascending=False).groupby('Consultation_Type').head(5)
        top_5['Consultation_Type_Name'] = top_5['Consultation_Type'].map(CONSULTATION_MAP)
        top_5['Case_Name'] = top_5['Case'].map(case_map).fillna(top_5['Case'].apply(lambda x: f'Case {x} (No Name)'))
        payload = top_5[['Consultation_Type_Name','Case_Name','Predicted_Total']].to_dict(orient='records')

        print(f"  Generated {len(payload)} predictions")
        return jsonify({"predicted_for": f"{next_year}-{next_month:02d}", "top_predictions": payload})
    except Exception as e:
        print(f"Error in top_predictions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_filtered')
def api_predict_filtered():
    try:
        consult_types_str = request.args.get('consult_types', default=None)
        predict_year = request.args.get('predict_year', type=int)

        if consult_types_str:
            consult_types = [int(x) for x in consult_types_str.split(',') if x.strip().isdigit()]
        else:
            consult_types = sorted(df['Consultation_Type'].unique().tolist())

        if predict_year is None:
            predict_year = int(df['Year'].max()) + 1

        print(f"API call: predict_filtered, types={consult_types}, year={predict_year}")

        all_cases = df['Case'].unique()
        all_sexes = df['Sex'].unique()
        all_age_ranges = df['Age_range'].unique()

        combinations = list(itertools.product(consult_types, all_cases, all_sexes, all_age_ranges))
        X_future = pd.DataFrame(combinations, columns=['Consultation_Type', 'Case', 'Sex', 'Age_range'])
        X_future['Year'] = predict_year
        X_future['Month'] = 1
        X_future['is_major_holiday'] = 1 if 1 in HOLIDAY_MONTHS else 0

        if X_train_columns is not None:
            for c in X_train_columns:
                if c not in X_future.columns:
                    X_future[c] = 0
            X_future = X_future[X_train_columns]

        preds = model.predict(X_future)
        preds[preds < 0] = 0
        X_future['Predicted_Total'] = preds

        grouped = X_future.groupby(['Consultation_Type', 'Case'])['Predicted_Total'].sum().reset_index()
        grouped['Predicted_Total'] = grouped['Predicted_Total'].round(0).astype(int)
        grouped['Consultation_Type_Name'] = grouped['Consultation_Type'].map(CONSULTATION_MAP)
        grouped['Case_Name'] = grouped['Case'].map(case_map).fillna(grouped['Case'].apply(lambda x: f'Case {x}'))

        top_5 = grouped.sort_values('Predicted_Total', ascending=False).groupby('Consultation_Type').head(5)
        payload = top_5[['Consultation_Type_Name', 'Case_Name', 'Predicted_Total']].to_dict(orient='records')

        print(f"  Generated {len(payload)} filtered predictions")
        return jsonify({
            "predicted_for": predict_year,
            "consultation_types": consult_types,
            "results": payload
        })
    except Exception as e:
        print(f"Error in predict_filtered: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Monthly Volume Prediction Dashboard")
    print("=" * 60)
    
    try:
        print("\n📊 Loading data...")
        load_data()
        
        print("\n🤖 Training or loading model...")
        train_or_load_model()
        
        print("\n" + "=" * 60)
        print("✓ Server ready at http://127.0.0.1:5000")
        print("✓ Health check: http://127.0.0.1:5000/api/health")
        print("=" * 60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"\n❌ Failed to start server: {str(e)}")
        raise
