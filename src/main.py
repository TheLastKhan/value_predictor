from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import plotly.graph_objs as go
import plotly.utils
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
CORS(app)  # CORS support

class ValueTransferAPI:
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path, dayfirst=True)
        self.prepare_data()
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self):
        """Prepare the data"""
        self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True)
        self.data.set_index('Date', inplace=True)
        self.data = self.data.ffill().bfill()
        self.numeric_columns = self.data.columns.tolist()
        
    def get_correlation_data(self):
        """Return correlation data"""
        correlation_matrix = self.data.corr()
        
        # Find correlation pairs
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                asset1, asset2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                correlation_pairs.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'correlation': round(corr_value, 3),
                    'relationship': 'Positive' if corr_value > 0 else 'Negative',
                    'strength': 'Strong' if abs(corr_value) > 0.7 else 'Medium' if abs(corr_value) > 0.4 else 'Weak'
                })
        
        # Sort from strongest to weakest
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'matrix': correlation_matrix.to_dict(),
            'pairs': correlation_pairs[:255],  # Top 15 strongest relationships
            'assets': self.numeric_columns
        }
    
    def create_features(self, target_asset):
        """Feature engineering"""
        features_df = self.data.copy()
        
        # Ratio relationships
        for col in self.numeric_columns:
            if col != target_asset:
                features_df[f'{col}_ratio'] = features_df[col] / features_df[target_asset]
        
        # Moving averages
        for col in self.numeric_columns:
            features_df[f'{col}_ma7'] = features_df[col].rolling(window=7).mean()
            features_df[f'{col}_ma30'] = features_df[col].rolling(window=30).mean()
        
        # Price changes
        for col in self.numeric_columns:
            features_df[f'{col}_pct'] = features_df[col].pct_change()
        
        return features_df.dropna()
    
    def train_and_predict(self, target_asset):
        """Train model and make predictions"""
        features_df = self.create_features(target_asset)
        
        y = features_df[target_asset]
        X = features_df.drop(columns=[target_asset])
        
        # Select most important features
        target_corr = features_df.corr()[target_asset].abs().sort_values(ascending=False)
        important_features = target_corr.head(15).index.tolist()
        if target_asset in important_features:
            important_features.remove(target_asset)
        
        X_selected = X[important_features]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, shuffle=False
        )
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_score = r2_score(y_test, lr_pred)
        
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_score = r2_score(y_test, rf_pred)
        
        # Future prediction
        last_features = X_selected.iloc[-1:] 
        last_features_scaled = scaler.transform(last_features)
        
        lr_future = lr_model.predict(last_features_scaled)[0]
        rf_future = rf_model.predict(last_features_scaled)[0]
        current_value = self.data[target_asset].iloc[-1]
        
        return {
            'current_value': round(current_value, 2),
            'linear_regression': {
                'prediction': round(lr_future, 2),
                'change_percent': round(((lr_future/current_value-1)*100), 2),
                'r2_score': round(lr_score, 3)
            },
            'random_forest': {
                'prediction': round(rf_future, 2),
                'change_percent': round(((rf_future/current_value-1)*100), 2),
                'r2_score': round(rf_score, 3)
            },
            'important_features': important_features[:5]
        }
    
    def get_price_history(self, assets=None):
        """Return price history data"""
        if assets is None:
            assets = ['Bitcoin', 'Gold', 'Silver', 'Copper', 'Platinum', 'Palladium', 'Aluminium', 'Lead', 'Zinc', 'Nickel', 'Tin', 'BrentOil', 'CrudeOil', 'HeatingOil', 'Gasoline', 'NaturalGas', 'US30', 'US500', 'SP500', 'DowJones', 'NASDAQ', 'CBOE', 'USDollarIndex'
    ]
        
        history_data = {}
        for asset in assets:
            if asset in self.data.columns:
                history_data[asset] = {
                    'dates': self.data.index.strftime('%Y-%m-%d').tolist(),
                    'values': self.data[asset].tolist()
                }
        
        return history_data

# Global variable
api = None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/assets')
def get_assets():
    """Return available assets"""
    return jsonify({
        'assets': api.numeric_columns,
        'total_records': len(api.data),
        'date_range': {
            'start': api.data.index.min().strftime('%d-%m-%Y'),
            'end': api.data.index.max().strftime('%d-%m-%Y')
        }
    })

@app.route('/api/correlations')
def get_correlations():
    """Return correlation analysis"""
    return jsonify(api.get_correlation_data())

@app.route('/api/predict/<asset>')
def predict_asset(asset):
    """Make prediction for a specific asset"""
    if asset not in api.numeric_columns:
        return jsonify({'error': 'Asset not found'}), 404
    
    try:
        prediction_data = api.train_and_predict(asset)
        return jsonify(prediction_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    """Return price history"""
    assets = request.args.getlist('assets')
    if not assets:
        assets = ['Bitcoin', 'Gold', 'Silver', 'Copper', 'Platinum', 'Palladium', 'Aluminium', 'Lead', 'Zinc', 'Nickel', 'Tin', 'BrentOil', 'CrudeOil', 'HeatingOil', 'Gasoline', 'NaturalGas', 'US30', 'US500', 'SP500', 'DowJones', 'NASDAQ', 'CBOE', 'USDollarIndex'
    ]
    
    history_data = api.get_price_history(assets)
    return jsonify(history_data)

@app.route('/api/compare')
def compare_assets():
    """Compare two assets"""
    asset1 = request.args.get('asset1', 'Bitcoin')
    asset2 = request.args.get('asset2', 'Gold')
    
    if asset1 not in api.numeric_columns or asset2 not in api.numeric_columns:
        return jsonify({'error': 'Asset not found'}), 404
    
    # Calculate correlation
    correlation = api.data[asset1].corr(api.data[asset2])
    
    # Last 30 days performance
    recent_data = api.data.tail(30)
    asset1_change = ((recent_data[asset1].iloc[-1] / recent_data[asset1].iloc[0]) - 1) * 100
    asset2_change = ((recent_data[asset2].iloc[-1] / recent_data[asset2].iloc[0]) - 1) * 100
    
    return jsonify({
        'asset1': asset1,
        'asset2': asset2,
        'correlation': round(correlation, 3),
        'relationship': 'Positive' if correlation > 0 else 'Negative',
        'strength': 'Strong' if abs(correlation) > 0.7 else 'Medium' if abs(correlation) > 0.4 else 'Weak',
        'recent_performance': {
            asset1: round(asset1_change, 2),
            asset2: round(asset2_change, 2)
        },
        'value_transfer_analysis': {
            'description': f"Correlation between {asset1} and {asset2} is {correlation:.3f}",
            'interpretation': 'One rises while the other falls' if correlation < -0.3 else 'They move together' if correlation > 0.3 else 'Weak relationship exists'
        }
    })

if __name__ == '__main__':
    # Write your CSV file path here
    CSV_FILE_PATH = "../tests/index.csv"  # Write your own file path
    
    try:
        api = ValueTransferAPI(CSV_FILE_PATH)
        print("API started!")
        print(f"Total data: {len(api.data)} records")
        print(f"Assets: {', '.join(api.numeric_columns)}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except FileNotFoundError:
        print("CSV file not found! Please check the file path.")
    except Exception as e:
        print(f"Error occurred: {e}")