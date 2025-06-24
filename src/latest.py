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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # CORS support

class ValueTransferAPI:
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path, dayfirst=True)
        self.prepare_data()
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self):
        """Prepare the data with proper date handling"""
        # Handle different date formats
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True, errors='coerce')
        elif 'date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['date'], dayfirst=True, errors='coerce')
        else:
            # If no date column, create one
            self.data['Date'] = pd.date_range(start='2020-01-01', periods=len(self.data), freq='D')
        
        self.data.set_index('Date', inplace=True)
        
        # Forward fill and backward fill for missing values
        self.data = self.data.ffill().bfill()
        
        # Ensure all columns are numeric
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Drop any remaining NaN values
        self.data = self.data.dropna()
        
        self.numeric_columns = self.data.columns.tolist()
        
    def get_correlation_data(self):
        """Return correlation data with better handling of negative correlations"""
        correlation_matrix = self.data.corr()
        
        # Find correlation pairs
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                asset1, asset2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                # Skip if correlation is NaN
                if pd.isna(corr_value):
                    continue
                    
                correlation_pairs.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'correlation': round(corr_value, 3),
                    'relationship': 'Positive' if corr_value > 0 else 'Negative',
                    'strength': self._get_correlation_strength(abs(corr_value))
                })
        
        # Sort from strongest to weakest (by absolute value)
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'matrix': correlation_matrix.to_dict(),
            'pairs': correlation_pairs[:20],  # Top 20 relationships
            'assets': self.numeric_columns,
            'statistics': self._get_correlation_statistics(correlation_pairs)
        }
    
    def _get_correlation_strength(self, abs_corr):
        """Determine correlation strength"""
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
    
    def _get_correlation_statistics(self, correlation_pairs):
        """Get correlation statistics"""
        if not correlation_pairs:
            return {}
            
        correlations = [pair['correlation'] for pair in correlation_pairs]
        return {
            'positive_count': len([c for c in correlations if c > 0]),
            'negative_count': len([c for c in correlations if c < 0]),
            'strong_positive': len([c for c in correlations if c > 0.6]),
            'strong_negative': len([c for c in correlations if c < -0.6]),
            'average_correlation': round(np.mean(correlations), 3)
        }
    
    def create_features(self, target_asset, prediction_days=1):
        """Enhanced feature engineering for time series prediction"""
        features_df = self.data.copy()
        
        # Lag features (previous values)
        for col in self.numeric_columns:
            for lag in [1, 2, 3, 5, 7, 14, 30]:
                features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
        
        # Moving averages
        for col in self.numeric_columns:
            features_df[f'{col}_ma_7'] = features_df[col].rolling(window=7).mean()
            features_df[f'{col}_ma_14'] = features_df[col].rolling(window=14).mean()
            features_df[f'{col}_ma_30'] = features_df[col].rolling(window=30).mean()
        
        # Price changes and returns
        for col in self.numeric_columns:
            features_df[f'{col}_pct_1d'] = features_df[col].pct_change(periods=1)
            features_df[f'{col}_pct_7d'] = features_df[col].pct_change(periods=7)
            features_df[f'{col}_pct_30d'] = features_df[col].pct_change(periods=30)
        
        # Volatility measures
        for col in self.numeric_columns:
            features_df[f'{col}_volatility_7d'] = features_df[col].rolling(window=7).std()
            features_df[f'{col}_volatility_30d'] = features_df[col].rolling(window=30).std()
        
        # Relative strength index (RSI) approximation
        for col in self.numeric_columns:
            delta = features_df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        # Inter-asset ratios
        for col in self.numeric_columns:
            if col != target_asset:
                features_df[f'{col}_to_{target_asset}_ratio'] = features_df[col] / features_df[target_asset]
        
        # Bollinger Bands
        for col in self.numeric_columns:
            rolling_mean = features_df[col].rolling(window=20).mean()
            rolling_std = features_df[col].rolling(window=20).std()
            features_df[f'{col}_bb_upper'] = rolling_mean + (rolling_std * 2)
            features_df[f'{col}_bb_lower'] = rolling_mean - (rolling_std * 2)
            features_df[f'{col}_bb_position'] = (features_df[col] - rolling_mean) / (rolling_std * 2)
        
        # Time-based features
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        
        # Clean data
        features_df = features_df.dropna()
        
        return features_df
    
    def train_and_predict(self, target_asset, prediction_days=1):
        """Enhanced training with proper time series validation and multiple day predictions"""
        features_df = self.create_features(target_asset, prediction_days)
        
        if len(features_df) < 100:
            raise ValueError("Insufficient data for training")
        
        # Create target variable (future value)
        y = features_df[target_asset].shift(-prediction_days)
        X = features_df.drop(columns=[target_asset])
        
        # Remove rows where target is NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("No valid data after preprocessing")
        
        # Feature selection based on correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        # Select top features, but ensure we have at least 10
        n_features = min(max(len(correlations) // 3, 10), 50)
        important_features = correlations.head(n_features).index.tolist()
        
        X_selected = X[important_features]
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Use the last split for final training
        train_idx, test_idx = list(tscv.split(X_selected))[-1]
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models with better parameters
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_score = r2_score(y_test, lr_pred)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        
        rf_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_score = r2_score(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        
        # Future prediction using most recent data
        last_features = X_selected.iloc[-1:] 
        last_features_scaled = scaler.transform(last_features)
        
        lr_future = lr_model.predict(last_features_scaled)[0]
        rf_future = rf_model.predict(last_features_scaled)[0]
        current_value = self.data[target_asset].iloc[-1]
        
        # Calculate confidence intervals (simple approach)
        lr_std = np.std(y_test - lr_pred)
        rf_std = np.std(y_test - rf_pred)
        
        # Get statistical summary
        stats = self._get_asset_statistics(target_asset)
        
        return {
            'asset': target_asset,
            'prediction_horizon': f"{prediction_days} day(s)",
            'current_value': round(current_value, 2),
            'linear_regression': {
                'prediction': round(lr_future, 2),
                'change_percent': round(((lr_future/current_value-1)*100), 2),
                'confidence_interval': [
                    round(lr_future - 1.96 * lr_std, 2),
                    round(lr_future + 1.96 * lr_std, 2)
                ],
                'r2_score': round(lr_score, 3),
                'mae': round(lr_mae, 2)
            },
            'random_forest': {
                'prediction': round(rf_future, 2),
                'change_percent': round(((rf_future/current_value-1)*100), 2),
                'confidence_interval': [
                    round(rf_future - 1.96 * rf_std, 2),
                    round(rf_future + 1.96 * rf_std, 2)
                ],
                'r2_score': round(rf_score, 3),
                'mae': round(rf_mae, 2)
            },
            'important_features': important_features[:10],
            'statistics': stats,
            'data_quality': {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(important_features)
            }
        }
    
    def _get_asset_statistics(self, asset):
        """Get comprehensive statistics for an asset including IQR, Q1, Q3"""
        data = self.data[asset]
        
        return {
            'mean': round(data.mean(), 2),
            'median': round(data.median(), 2),
            'std': round(data.std(), 2),
            'min': round(data.min(), 2),
            'max': round(data.max(), 2),
            'q1': round(data.quantile(0.25), 2),
            'q3': round(data.quantile(0.75), 2),
            'iqr': round(data.quantile(0.75) - data.quantile(0.25), 2),
            'skewness': round(data.skew(), 3),
            'kurtosis': round(data.kurtosis(), 3),
            'volatility_30d': round(data.tail(30).std(), 2),
            'return_30d': round(((data.iloc[-1] / data.iloc[-30] - 1) * 100), 2) if len(data) >= 30 else 0
        }
    
    def get_price_history(self, assets=None):
        """Return price history data with proper date formatting"""
        if assets is None:
            assets = self.numeric_columns[:4]  # First 4 assets as default
        
        history_data = {}
        for asset in assets:
            if asset in self.data.columns:
                history_data[asset] = {
                    'dates': [date.strftime('%Y-%m-%d') for date in self.data.index],
                    'values': self.data[asset].tolist(),
                    'statistics': self._get_asset_statistics(asset)
                }
        
        return history_data
    
    def predict_multiple_days(self, target_asset, days_ahead=30):
        """Predict multiple days ahead"""
        predictions = []
        
        for day in range(1, days_ahead + 1):
            try:
                result = self.train_and_predict(target_asset, prediction_days=day)
                predictions.append({
                    'day': day,
                    'date': (self.data.index[-1] + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'lr_prediction': result['linear_regression']['prediction'],
                    'rf_prediction': result['random_forest']['prediction'],
                    'lr_confidence': result['linear_regression']['confidence_interval'],
                    'rf_confidence': result['random_forest']['confidence_interval']
                })
            except Exception as e:
                print(f"Error predicting day {day}: {str(e)}")
                continue
        
        return {
            'asset': target_asset,
            'predictions': predictions,
            'current_value': round(self.data[target_asset].iloc[-1], 2)
        }

# Global variable
api = None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/assets')
def get_assets():
    """Return available assets with enhanced information"""
    return jsonify({
        'assets': api.numeric_columns,
        'total_records': len(api.data),
        'date_range': {
            'start': api.data.index.min().strftime('%Y-%m-%d'),
            'end': api.data.index.max().strftime('%Y-%m-%d')
        },
        'data_frequency': 'Daily',
        'last_updated': api.data.index.max().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/correlations')
def get_correlations():
    """Return enhanced correlation analysis"""
    return jsonify(api.get_correlation_data())

@app.route('/api/predict/<asset>')
def predict_asset(asset):
    """Make prediction for a specific asset with configurable horizon"""
    if asset not in api.numeric_columns:
        return jsonify({'error': 'Asset not found'}), 404
    
    days = request.args.get('days', 1, type=int)
    if days < 1 or days > 30:
        return jsonify({'error': 'Days must be between 1 and 30'}), 400
    
    try:
        prediction_data = api.train_and_predict(asset, prediction_days=days)
        return jsonify(prediction_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-multiple/<asset>')
def predict_multiple_days(asset):
    """Predict multiple days ahead"""
    if asset not in api.numeric_columns:
        return jsonify({'error': 'Asset not found'}), 404
    
    days = request.args.get('days', 7, type=int)
    if days < 1 or days > 30:
        return jsonify({'error': 'Days must be between 1 and 30'}), 400
    
    try:
        prediction_data = api.predict_multiple_days(asset, days_ahead=days)
        return jsonify(prediction_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics/<asset>')
def get_asset_statistics(asset):
    """Get comprehensive statistics for an asset"""
    if asset not in api.numeric_columns:
        return jsonify({'error': 'Asset not found'}), 404
    
    try:
        stats = api._get_asset_statistics(asset)
        return jsonify({
            'asset': asset,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    """Return price history with proper date formatting"""
    assets = request.args.getlist('assets')
    if not assets:
        assets = api.numeric_columns[:4]  # Default to first 4 assets
    
    history_data = api.get_price_history(assets)
    return jsonify(history_data)

@app.route('/api/compare')
def compare_assets():
    """Compare two assets with enhanced analysis"""
    asset1 = request.args.get('asset1')
    asset2 = request.args.get('asset2')
    
    if not asset1 or not asset2:
        return jsonify({'error': 'Both asset1 and asset2 parameters are required'}), 400
    
    if asset1 not in api.numeric_columns or asset2 not in api.numeric_columns:
        return jsonify({'error': 'One or both assets not found'}), 404
    
    # Calculate correlation
    correlation = api.data[asset1].corr(api.data[asset2])
    
    # Performance analysis for different periods
    performance_periods = [7, 30, 90, 365]
    performance_data = {}
    
    for period in performance_periods:
        if len(api.data) >= period:
            recent_data = api.data.tail(period)
            asset1_change = ((recent_data[asset1].iloc[-1] / recent_data[asset1].iloc[0]) - 1) * 100
            asset2_change = ((recent_data[asset2].iloc[-1] / recent_data[asset2].iloc[0]) - 1) * 100
            
            performance_data[f'{period}_days'] = {
                asset1: round(asset1_change, 2),
                asset2: round(asset2_change, 2),
                'difference': round(asset1_change - asset2_change, 2)
            }
    
    # Get statistics for both assets
    stats1 = api._get_asset_statistics(asset1)
    stats2 = api._get_asset_statistics(asset2)
    
    return jsonify({
        'asset1': asset1,
        'asset2': asset2,
        'correlation': round(correlation, 3),
        'relationship': 'Positive' if correlation > 0 else 'Negative',
        'strength': api._get_correlation_strength(abs(correlation)),
        'performance_comparison': performance_data,
        'statistics_comparison': {
            asset1: stats1,
            asset2: stats2
        },
        'value_transfer_analysis': {
            'description': f"Correlation between {asset1} and {asset2} is {correlation:.3f}",
            'interpretation': api._get_correlation_interpretation(correlation),
            'trading_signal': api._get_trading_signal(correlation, performance_data)
        }
    })

def _get_correlation_interpretation(self, correlation):
    """Get correlation interpretation"""
    if correlation > 0.7:
        return "Strong positive relationship - assets move together"
    elif correlation > 0.3:
        return "Moderate positive relationship - some tendency to move together"
    elif correlation > -0.3:
        return "Weak relationship - little predictable pattern"
    elif correlation > -0.7:
        return "Moderate negative relationship - some tendency to move oppositely"
    else:
        return "Strong negative relationship - assets tend to move in opposite directions"

def _get_trading_signal(self, correlation, performance_data):
    """Generate trading signal based on correlation and performance"""
    if not performance_data:
        return "Insufficient data for trading signal"
    
    latest_performance = performance_data.get('30_days', {})
    if not latest_performance:
        return "Insufficient recent data"
    
    if correlation < -0.5:
        return "Potential value transfer opportunity - consider contrarian strategy"
    elif correlation > 0.7:
        return "Assets move together - consider momentum strategy"
    else:
        return "Neutral signal - no clear directional bias"

if __name__ == '__main__':
    # CSV file path - UPDATE THIS PATH
    CSV_FILE_PATH = "../tests/data.csv"  # Update with your actual file path
    
    try:
        api = ValueTransferAPI(CSV_FILE_PATH)
        print("Enhanced Value Transfer API started successfully!")
        print(f"Total data points: {len(api.data)} records")
        print(f"Date range: {api.data.index.min()} to {api.data.index.max()}")
        print(f"Available assets: {', '.join(api.numeric_columns)}")
        print("\nNew Features Added:")
        print("- Multi-day predictions (1-30 days ahead)")
        print("- Statistical analysis with IQR, Q1, Q3")
        print("- Enhanced correlation analysis")
        print("- Confidence intervals for predictions")
        print("- Better date formatting")
        print("- Improved feature engineering")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        print("Please update the CSV_FILE_PATH variable with the correct path to your data file.")
    except Exception as e:
        print(f"Error starting API: {e}")