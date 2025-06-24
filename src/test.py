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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class ValueTransferAPI:
    def __init__(self, csv_file_path):
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        self.data = pd.read_csv(csv_file_path, dayfirst=True)
        self.prepare_data()
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self):
        """Prepare and clean the data"""
        try:
            # Try different date formats
            self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True, errors='coerce')
            if self.data['Date'].isnull().any():
                self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            
            # Remove rows with invalid dates
            self.data = self.data.dropna(subset=['Date'])
            self.data.set_index('Date', inplace=True)
            self.data.sort_index(inplace=True)
            
            # Only keep numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.data = self.data[numeric_cols]
            
            # Forward fill then backward fill for missing values
            self.data = self.data.ffill().bfill()
            
            # Remove columns with too many missing values (>50%)
            missing_threshold = len(self.data) * 0.5
            self.data = self.data.dropna(thresh=missing_threshold, axis=1)
            
            self.numeric_columns = self.data.columns.tolist()
            
            if len(self.numeric_columns) == 0:
                raise ValueError("No numeric columns found in the data")
                
        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")
        
    def get_correlation_data(self):
        """Return correlation data with better analysis"""
        if len(self.data) < 30:
            return {'error': 'Not enough data for correlation analysis'}
        
        correlation_matrix = self.data.corr()
        
        # Find correlation pairs
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                asset1, asset2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if not np.isnan(corr_value):
                    correlation_pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'correlation': round(corr_value, 3),
                        'relationship': 'Positive' if corr_value > 0 else 'Negative',
                        'strength': self._get_correlation_strength(abs(corr_value))
                    })
        
        # Sort by absolute correlation value
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'matrix': correlation_matrix.to_dict(),
            'pairs': correlation_pairs[:15],  # Top 15 correlations
            'assets': self.numeric_columns,
            'total_pairs': len(correlation_pairs)
        }
    
    def _get_correlation_strength(self, abs_corr):
        """Classify correlation strength"""
        if abs_corr > 0.8:
            return 'Very Strong'
        elif abs_corr > 0.6:
            return 'Strong'
        elif abs_corr > 0.4:
            return 'Moderate'
        elif abs_corr > 0.2:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def create_time_series_features(self, target_asset, lookback_days=30):
        """Create proper time series features without data leakage"""
        features_df = pd.DataFrame(index=self.data.index)
        
        # Lag features (previous values)
        for lag in [1, 3, 7, 14, 30]:
            if len(self.data) > lag:
                for col in self.numeric_columns:
                    features_df[f'{col}_lag_{lag}'] = self.data[col].shift(lag)
        
        # Moving averages (using past data only)
        for window in [7, 14, 30]:
            if len(self.data) > window:
                for col in self.numeric_columns:
                    features_df[f'{col}_ma_{window}'] = self.data[col].rolling(window=window).mean().shift(1)
        
        # Price changes (using past data)
        for period in [1, 7, 30]:
            if len(self.data) > period:
                for col in self.numeric_columns:
                    features_df[f'{col}_pct_{period}'] = self.data[col].pct_change(periods=period).shift(1)
        
        # Volatility features
        for window in [7, 30]:
            if len(self.data) > window:
                for col in self.numeric_columns:
                    features_df[f'{col}_vol_{window}'] = self.data[col].rolling(window=window).std().shift(1)
        
        # Relative strength compared to other assets (avoid target leakage)
        for col in self.numeric_columns:
            if col != target_asset:
                # Use lagged values to avoid leakage
                lagged_target = self.data[target_asset].shift(1)
                lagged_col = self.data[col].shift(1)
                features_df[f'{col}_relative_to_{target_asset}'] = lagged_col / lagged_target
        
        # Add target variable
        features_df['target'] = self.data[target_asset]
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def train_and_predict(self, target_asset, test_size=0.2):
        """Train models with proper time series validation"""
        if target_asset not in self.numeric_columns:
            return {'error': f'Asset {target_asset} not found'}
        
        if len(self.data) < 100:
            return {'error': 'Not enough data for prediction (minimum 100 records required)'}
        
        try:
            # Create features
            features_df = self.create_time_series_features(target_asset)
            
            if len(features_df) < 50:
                return {'error': 'Not enough data after feature engineering'}
            
            # Separate features and target
            y = features_df['target']
            X = features_df.drop(columns=['target'])
            
            # Feature selection based on correlation with target
            target_corr = X.corrwith(y).abs().sort_values(ascending=False)
            # Select top features, but limit to avoid overfitting
            n_features = min(15, len(target_corr))
            important_features = target_corr.head(n_features).index.tolist()
            X_selected = X[important_features]
            
            # Time series split for validation
            test_split_point = int(len(X_selected) * (1 - test_size))
            X_train = X_selected.iloc[:test_split_point]
            X_test = X_selected.iloc[test_split_point:]
            y_train = y.iloc[:test_split_point]
            y_test = y.iloc[test_split_point:]
            
            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models_results = {}
            
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            lr_score = r2_score(y_test, lr_pred)
            lr_mae = mean_absolute_error(y_test, lr_pred)
            
            # Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            rf_score = r2_score(y_test, rf_pred)
            rf_mae = mean_absolute_error(y_test, rf_pred)
            
            # Choose best model
            best_model = rf_model if rf_score > lr_score else lr_model
            best_model_name = 'Random Forest' if rf_score > lr_score else 'Linear Regression'
            best_score = max(rf_score, lr_score)
            
            # Make future prediction using only available features
            current_value = self.data[target_asset].iloc[-1]
            
            # Create prediction confidence
            prediction_confidence = 'High' if best_score > 0.7 else 'Medium' if best_score > 0.4 else 'Low'
            
            return {
                'asset': target_asset,
                'current_value': round(current_value, 2),
                'prediction_model': best_model_name,
                'model_performance': {
                    'r2_score': round(best_score, 3),
                    'confidence': prediction_confidence
                },
                'linear_regression': {
                    'r2_score': round(lr_score, 3),
                    'mae': round(lr_mae, 2)
                },
                'random_forest': {
                    'r2_score': round(rf_score, 3),
                    'mae': round(rf_mae, 2)
                },
                'important_features': important_features[:5],
                'data_quality': {
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': len(important_features)
                },
                'warning': 'Predictions are based on historical patterns and should not be used as sole investment advice.'
            }
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def get_price_history(self, assets=None, days=None):
        """Return price history data with optional filtering"""
        if assets is None:
            assets = self.numeric_columns
        
        # Filter assets that exist in our data
        available_assets = [asset for asset in assets if asset in self.numeric_columns]
        
        data_to_use = self.data.copy()
        
        # Limit to recent days if specified
        if days:
            data_to_use = data_to_use.tail(days)
        
        history_data = {}
        for asset in available_assets:
            history_data[asset] = {
                'dates': data_to_use.index.strftime('%Y-%m-%d').tolist(),
                'values': data_to_use[asset].round(2).tolist(),
                'latest_value': round(data_to_use[asset].iloc[-1], 2),
                'change_30d': self._calculate_change(data_to_use[asset], 30),
                'change_7d': self._calculate_change(data_to_use[asset], 7)
            }
        
        return history_data
    
    def _calculate_change(self, series, days):
        """Calculate percentage change over specified days"""
        if len(series) < days:
            return None
        
        try:
            old_value = series.iloc[-days]
            new_value = series.iloc[-1]
            change = ((new_value / old_value) - 1) * 100
            return round(change, 2)
        except:
            return None

# Global variable
api = None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': api is not None,
        'records_count': len(api.data) if api else 0
    })

@app.route('/api/assets')
def get_assets():
    """Return available assets with metadata"""
    if not api:
        return jsonify({'error': 'API not initialized'}), 500
    
    return jsonify({
        'assets': api.numeric_columns,
        'total_assets': len(api.numeric_columns),
        'total_records': len(api.data),
        'date_range': {
            'start': api.data.index.min().strftime('%Y-%m-%d'),
            'end': api.data.index.max().strftime('%Y-%m-%d')
        },
        'data_quality': {
            'missing_values': api.data.isnull().sum().sum(),
            'completeness': round((1 - api.data.isnull().sum().sum() / api.data.size) * 100, 2)
        }
    })

@app.route('/api/correlations')
def get_correlations():
    """Return correlation analysis"""
    if not api:
        return jsonify({'error': 'API not initialized'}), 500
    
    return jsonify(api.get_correlation_data())

@app.route('/api/predict/<asset>')
def predict_asset(asset):
    """Make prediction for a specific asset"""
    if not api:
        return jsonify({'error': 'API not initialized'}), 500
    
    if asset not in api.numeric_columns:
        return jsonify({
            'error': 'Asset not found',
            'available_assets': api.numeric_columns
        }), 404
    
    prediction_data = api.train_and_predict(asset)
    return jsonify(prediction_data)

@app.route('/api/history')
def get_history():
    """Return price history with optional filtering"""
    if not api:
        return jsonify({'error': 'API not initialized'}), 500
    
    assets = request.args.getlist('assets')
    days = request.args.get('days', type=int)
    
    if not assets:
        assets = api.numeric_columns
    
    history_data = api.get_price_history(assets, days)
    return jsonify(history_data)

@app.route('/api/compare')
def compare_assets():
    """Compare two assets with detailed analysis"""
    if not api:
        return jsonify({'error': 'API not initialized'}), 500
    
    asset1 = request.args.get('asset1')
    asset2 = request.args.get('asset2')
    
    if not asset1 or not asset2:
        return jsonify({'error': 'Both asset1 and asset2 parameters are required'}), 400
    
    if asset1 not in api.numeric_columns or asset2 not in api.numeric_columns:
        return jsonify({
            'error': 'One or both assets not found',
            'available_assets': api.numeric_columns
        }), 404
    
    try:
        # Calculate correlation
        correlation = api.data[asset1].corr(api.data[asset2])
        
        # Performance analysis for different periods
        performance_analysis = {}
        for period in [7, 30, 90]:
            if len(api.data) >= period:
                recent_data = api.data.tail(period)
                asset1_change = api._calculate_change(recent_data[asset1], period)
                asset2_change = api._calculate_change(recent_data[asset2], period)
                
                performance_analysis[f'{period}d'] = {
                    asset1: asset1_change,
                    asset2: asset2_change
                }
        
        return jsonify({
            'asset1': asset1,
            'asset2': asset2,
            'correlation': round(correlation, 3) if not np.isnan(correlation) else None,
            'relationship': 'Positive' if correlation > 0 else 'Negative' if correlation < 0 else 'None',
            'strength': api._get_correlation_strength(abs(correlation)) if not np.isnan(correlation) else 'Unknown',
            'performance_analysis': performance_analysis,
            'current_values': {
                asset1: round(api.data[asset1].iloc[-1], 2),
                asset2: round(api.data[asset2].iloc[-1], 2)
            },
            'analysis_summary': {
                'description': f"Correlation coefficient: {correlation:.3f}" if not np.isnan(correlation) else "Unable to calculate correlation",
                'interpretation': api._get_correlation_interpretation(correlation)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Comparison error: {str(e)}'}), 500

def _get_correlation_interpretation(correlation):
    """Get human-readable correlation interpretation"""
    if np.isnan(correlation):
        return "Unable to determine relationship"
    elif correlation > 0.7:
        return "Strong positive relationship - they tend to move in the same direction"
    elif correlation > 0.3:
        return "Moderate positive relationship - some tendency to move together"
    elif correlation > -0.3:
        return "Weak relationship - movements are largely independent"
    elif correlation > -0.7:
        return "Moderate negative relationship - tendency to move in opposite directions"
    else:
        return "Strong negative relationship - they typically move in opposite directions"

if __name__ == '__main__':
    # Configuration
    CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', '../tests/index.csv')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    try:
        print("Initializing Value Transfer API...")
        api = ValueTransferAPI(CSV_FILE_PATH)
        print("✓ API initialized successfully!")
        print(f"✓ Loaded {len(api.data)} records")
        print(f"✓ Available assets: {len(api.numeric_columns)}")
        print(f"✓ Date range: {api.data.index.min().strftime('%Y-%m-%d')} to {api.data.index.max().strftime('%Y-%m-%d')}")
        print(f"✓ Starting server on port {PORT}")
        
        app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
        
    except FileNotFoundError as e:
        print(f"❌ CSV file not found: {e}")
        print("Please check the file path or set CSV_FILE_PATH environment variable")
    except ValueError as e:
        print(f"❌ Data error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")





