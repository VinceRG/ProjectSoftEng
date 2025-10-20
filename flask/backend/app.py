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
        year = request.args.get('year', default=None, type=int)
        df_copy = df.copy()
        if year:
            df_copy = df_copy[df_copy['Year'] == year]

        if df_copy.empty:
            return jsonify({
                "holiday_months": HOLIDAY_MONTHS,
                "holiday_avg": 0,
                "non_holiday_avg": 0
            })

        monthly_totals = df_copy.groupby(['Year','Month'])['Total'].sum().reset_index()
        monthly_totals['is_major_holiday'] = monthly_totals['Month'].isin(HOLIDAY_MONTHS).astype(int)
        
        holiday_data = monthly_totals[monthly_totals['is_major_holiday']==1]
        non_holiday_data = monthly_totals[monthly_totals['is_major_holiday']==0]

        holiday_avg = float(holiday_data['Total'].mean()) if not holiday_data.empty else 0
        non_holiday_avg = float(non_holiday_data['Total'].mean()) if not non_holiday_data.empty else 0
        
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
        # Use current date instead of dataset's last date
        from datetime import datetime
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        # Calculate next month
        if current_month == 12:
            next_month = 1
            next_year = current_year + 1
        else:
            next_month = current_month + 1
            next_year = current_year

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

        top_10 = future_case_totals.sort_values('Predicted_Total', ascending=False).groupby('Consultation_Type').head(10)
        top_10['Consultation_Type_Name'] = top_10['Consultation_Type'].map(CONSULTATION_MAP)
        top_10['Case_Name'] = top_10['Case'].map(case_map).fillna(top_10['Case'].apply(lambda x: f'Case {x} (No Name)'))
        payload = top_10[['Consultation_Type_Name','Case_Name','Predicted_Total']].to_dict(orient='records')

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
        predict_month_str = request.args.get('predict_month', default='all')

        if consult_types_str:
            consult_types = [int(x) for x in consult_types_str.split(',') if x.strip().isdigit()]
        else:
            consult_types = sorted(df['Consultation_Type'].unique().tolist())

        if predict_year is None:
            predict_year = int(df['Year'].max()) + 1

        print(f"API call: predict_filtered, types={consult_types}, year={predict_year}, month={predict_month_str}")

        months_to_predict = []
        if predict_month_str and predict_month_str.isdigit():
            predict_month = int(predict_month_str)
            if 1 <= predict_month <= 12:
                months_to_predict = [predict_month]
        
        if not months_to_predict:
            months_to_predict = range(1, 13)

        all_cases = df['Case'].unique()
        all_sexes = df['Sex'].unique()
        all_age_ranges = df['Age_range'].unique()
        
        all_predictions_df = pd.DataFrame()

        for month in months_to_predict:
            combinations = list(itertools.product(consult_types, all_cases, all_sexes, all_age_ranges))
            X_future_month = pd.DataFrame(combinations, columns=['Consultation_Type', 'Case', 'Sex', 'Age_range'])
            X_future_month['Year'] = predict_year
            X_future_month['Month'] = month
            X_future_month['is_major_holiday'] = 1 if month in HOLIDAY_MONTHS else 0

            if X_train_columns is not None:
                for c in X_train_columns:
                    if c not in X_future_month.columns:
                        X_future_month[c] = 0
                X_future_month = X_future_month[X_train_columns]
            
            preds = model.predict(X_future_month)
            preds[preds < 0] = 0
            X_future_month['Predicted_Total'] = preds
            all_predictions_df = pd.concat([all_predictions_df, X_future_month], ignore_index=True)

        grouped = all_predictions_df.groupby(['Consultation_Type', 'Case'])['Predicted_Total'].sum().reset_index()
        grouped['Predicted_Total'] = grouped['Predicted_Total'].round(0).astype(int)
        grouped['Consultation_Type_Name'] = grouped['Consultation_Type'].map(CONSULTATION_MAP)
        grouped['Case_Name'] = grouped['Case'].map(case_map).fillna(grouped['Case'].apply(lambda x: f'Case {x}'))

        top_10 = grouped.sort_values('Predicted_Total', ascending=False).groupby('Consultation_Type').head(5)
        payload = top_10[['Consultation_Type_Name', 'Case_Name', 'Predicted_Total']].to_dict(orient='records')

        print(f"  Generated {len(payload)} filtered predictions")
        return jsonify({
            "predicted_for": {"year": predict_year, "month": predict_month_str},
            "consultation_types": consult_types,
            "results": payload
        })
    except Exception as e:
        print(f"Error in predict_filtered: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_volume_timeline')
def api_predict_volume_timeline():
    """Predict total patient volumes for future months and years"""
    try:
        start_year = request.args.get('start_year', type=int)
        start_month = request.args.get('start_month', type=int, default=1)
        num_months = request.args.get('num_months', type=int, default=12)
        
        # Handle consult_type properly - can be None, int, or string 'all'
        consult_type_param = request.args.get('consult_type', default=None)
        consult_type = None
        if consult_type_param is not None and consult_type_param != 'all':
            try:
                consult_type = int(consult_type_param)
            except (ValueError, TypeError):
                consult_type = None

        if start_year is None:
            from datetime import datetime
            current_date = datetime.now()
            start_year = current_date.year
            start_month = current_date.month

        print(f"API call: predict_volume_timeline, start={start_year}-{start_month}, months={num_months}, type={consult_type}")

        # Validate inputs
        if not (1 <= start_month <= 12):
            return jsonify({"error": "Invalid start_month. Must be 1-12"}), 400
        
        if num_months < 1 or num_months > 120:  # Max 10 years
            return jsonify({"error": "Invalid num_months. Must be 1-120"}), 400

        all_cases = df['Case'].unique()
        all_sexes = df['Sex'].unique()
        all_age_ranges = df['Age_range'].unique()
        
        # Determine consultation types to include
        if consult_type is not None:
            consult_types = [consult_type]
        else:
            consult_types = df['Consultation_Type'].unique()

        print(f"  Using consultation types: {consult_types}")

        timeline_data = []

        # Generate predictions for each month
        for i in range(num_months):
            current_month = start_month + i
            current_year = start_year
            
            # Handle year rollover
            while current_month > 12:
                current_month -= 12
                current_year += 1

            # Create all combinations for this month
            combinations = list(itertools.product(consult_types, all_cases, all_sexes, all_age_ranges))
            X_future = pd.DataFrame(combinations, columns=['Consultation_Type', 'Case', 'Sex', 'Age_range'])
            X_future['Year'] = current_year
            X_future['Month'] = current_month
            X_future['is_major_holiday'] = 1 if current_month in HOLIDAY_MONTHS else 0

            # Align columns with training data
            if X_train_columns is not None:
                for c in X_train_columns:
                    if c not in X_future.columns:
                        X_future[c] = 0
                X_future = X_future[X_train_columns]
            
            # Predict
            preds = model.predict(X_future)
            preds[preds < 0] = 0
            
            # Sum all predictions for this month
            total_volume = int(preds.sum())
            
            timeline_data.append({
                'year': int(current_year),
                'month': int(current_month),
                'label': f"{current_year}-{current_month:02d}",
                'predicted_volume': total_volume
            })

        print(f"  Generated {len(timeline_data)} timeline predictions")
        print(f"  Sample data: {timeline_data[:3] if len(timeline_data) >= 3 else timeline_data}")
        
        return jsonify({
            "timeline": timeline_data,
            "start": {"year": start_year, "month": start_month},
            "num_months": num_months,
            "consultation_type": consult_type
        })
    except Exception as e:
        print(f"Error in predict_volume_timeline: {str(e)}")
        import traceback
        traceback.print_exc()
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