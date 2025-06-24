# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------- #
# UYUMLU VE SON SÜRÜM GELİŞTİRİLMİŞ VALUE TRANSFER PREDICTOR API
# --------------------------------------------------------------------------- #
# Açıklama: Bu script, predictor.py'deki tüm gelişmiş analiz ve modelleme
# yeteneklerini içerir. MODELLER temizlenmiş veriyi, GRAFİKLER ise
# orijinal veriyi kullanarak hem tahmin kalitesini hem de görsel doğruluğu
# en üst seviyede tutar.
# --------------------------------------------------------------------------- #

# --- Gerekli Kütüphaneler ---
import pandas as pd
import numpy as np
import warnings
import os
import traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =========================================================================== #
# BÖLÜM 1: GELİŞMİŞ ANALİZ SINIFI
# =========================================================================== #

class EnhancedValueTransferPredictor:
    """
    Enhanced Financial Asset Value Transfer Analysis and Prediction System
    """
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)
        self.original_data_with_date_index = None
        self.outliers_info = {}
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}
        self.prepare_data()

    def prepare_data(self):
        """Enhanced data preparation with outlier detection"""
        print("=== DATA PREPARATION ===")
        try:
            self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True)
        except:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Orijinal veriyi sakla (grafikler için)
        self.original_data_with_date_index = self.data.copy()
        
        print(f"Original data shape: {self.data.shape}")
        # IQR ile aykırı değerleri tespit et ve modeller için temiz bir kopya oluştur
        self.data_cleaned = self.handle_outliers_for_modeling(self.data[self.numeric_columns])
        self.data_cleaned = self.data_cleaned.fillna(method='ffill').fillna(method='bfill')
        print(f"Data for modeling (cleaned): {self.data_cleaned.shape}")

    def handle_outliers_for_modeling(self, df):
        """
        Find outliers using IQR and return a new DataFrame with replaced values for modeling.
        This method identifies outliers and replaces them with boundary values in a *copy* of the data.
        """
        df_clean = df.copy()
        print("\n--- Outlier Analysis for Modeling (IQR Method) ---")
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
            # Aykırı değer bilgilerini sakla
            self.outliers_info[column] = {'count': len(outliers), 'lower_bound': lower_bound, 'upper_bound': upper_bound}
            
            # Modellerin kullanacağı kopyada aykırı değerleri sınır değerleriyle değiştir
            df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
            df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
        return df_clean

    def analyze_correlations(self):
        """Enhanced correlation analysis with detailed insights"""
        # Korelasyon gibi istatistiksel analizler için temizlenmiş veriyi kullan
        correlation_matrix = self.data_cleaned.corr()
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                asset1, asset2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.8: strength = 'Very Strong'
                elif abs(corr_value) > 0.6: strength = 'Strong'
                elif abs(corr_value) > 0.4: strength = 'Moderate'
                elif abs(corr_value) > 0.2: strength = 'Weak'
                else: strength = 'Very Weak'
                
                correlation_pairs.append({
                    'asset1': asset1, 'asset2': asset2, 'correlation': round(corr_value, 3),
                    'relationship': 'Positive' if corr_value > 0 else 'Negative',
                    'strength': strength
                })
        
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return {
            'matrix': correlation_matrix.round(4).to_dict(),
            'pairs': correlation_pairs[:15],
            'assets': self.numeric_columns
        }

    def create_advanced_features(self, target_asset):
        """Create advanced features for better prediction from cleaned data"""
        # Model eğitimi için özellikler temizlenmiş veriden oluşturulmalı
        features_df = self.data_cleaned.copy()
        for col in self.numeric_columns:
            if col != target_asset:
                features_df[f'{col}_to_{target_asset}_ratio'] = features_df[col] / features_df[target_asset]
            for window in [5, 10, 20]:
                features_df[f'{col}_ma_{window}'] = features_df[col].rolling(window=window).mean()
            for period in [1, 7, 30]:
                features_df[f'{col}_pct_change_{period}d'] = features_df[col].pct_change(periods=period)
            features_df[f'{col}_volatility_30d'] = features_df[col].rolling(window=30).std()
        features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
        return features_df

    def train_models(self, target_asset='Bitcoin'):
        """Enhanced model training with time series validation using cleaned data"""
        print(f"\n=== ENHANCED MODEL TRAINING FOR {target_asset} ===")
        features_df = self.create_advanced_features(target_asset)
        if len(features_df) == 0: return
        
        y = features_df[target_asset]
        X = features_df.drop(columns=self.numeric_columns)
        
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(25).index.tolist()
        X_selected = X[top_features]
        
        tscv = TimeSeriesSplit(n_splits=5)
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        print("Performing time series cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_selected)):
            X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            scaler = StandardScaler().fit(X_train)
            X_train_scaled, X_val_scaled = scaler.transform(X_train), scaler.transform(X_val)
            
            rf_model.fit(X_train_scaled, y_train)
            lr_model.fit(X_train_scaled, y_train)
            
        lr_pred = lr_model.predict(X_val_scaled)
        rf_pred = rf_model.predict(X_val_scaled)
        
        self.evaluation_results[target_asset] = {
            'linear_regression': {'r2_score': r2_score(y_val, lr_pred)},
            'random_forest': {'r2_score': r2_score(y_val, rf_pred)}
        }
        
        self.models[target_asset] = {
            'linear_regression': lr_model, 'random_forest': rf_model,
            'features': top_features, 'scaler': scaler
        }
        print(f"Model training for {target_asset} completed.")

    def predict_future_values(self, target_asset='Bitcoin'):
        """Enhanced future value prediction"""
        if target_asset not in self.models:
            self.train_models(target_asset)
        
        model_info = self.models[target_asset]
        features_df = self.create_advanced_features(target_asset)
        if len(features_df) == 0: return {'error': 'Cannot create features'}
        
        latest_features_scaled = model_info['scaler'].transform(features_df[model_info['features']].iloc[-1:])
        
        lr_next = model_info['linear_regression'].predict(latest_features_scaled)[0]
        rf_next = model_info['random_forest'].predict(latest_features_scaled)[0]
        current_value = self.data_cleaned[target_asset].iloc[-1]
        
        lr_eval = self.evaluation_results.get(target_asset, {}).get('linear_regression', {})
        rf_eval = self.evaluation_results.get(target_asset, {}).get('random_forest', {})

        return {
            'current_value': round(current_value, 2),
            'linear_regression': {
                'prediction': round(lr_next, 2),
                'change_percent': round(((lr_next / current_value) - 1) * 100, 2),
                'r2_score': round(lr_eval.get('r2_score', 0), 3)
            },
            'random_forest': {
                'prediction': round(rf_next, 2),
                'change_percent': round(((rf_next / current_value) - 1) * 100, 2),
                'r2_score': round(rf_eval.get('r2_score', 0), 3)
            },
            'important_features': model_info['features'][:5]
        }

    def interpret_correlation(self, correlation):
        if correlation < -0.7: return f"Very strong negative correlation ({correlation:.3f}): Clear value transfer."
        elif correlation < -0.3: return f"Moderate negative correlation ({correlation:.3f}): Some inverse relationship exists."
        elif correlation > 0.7: return f"Very strong positive correlation ({correlation:.3f}): Almost identical movement."
        elif correlation > 0.3: return f"Moderate positive correlation ({correlation:.3f}): Assets tend to move together."
        else: return f"Weak correlation ({correlation:.3f}): Limited relationship."

    def get_asset_comparison(self, asset1, asset2):
        correlation = self.data_cleaned[asset1].corr(self.data_cleaned[asset2])
        recent_data = self.data_cleaned.tail(30)
        asset1_change = ((recent_data[asset1].iloc[-1] / recent_data[asset1].iloc[0]) - 1) * 100
        asset2_change = ((recent_data[asset2].iloc[-1] / recent_data[asset2].iloc[0]) - 1) * 100

        if abs(correlation) > 0.7: strength = 'Strong'
        elif abs(correlation) > 0.4: strength = 'Medium'
        else: strength = 'Weak'

        return {
            'asset1': asset1, 'asset2': asset2, 'correlation': round(correlation, 3),
            'relationship': 'Positive' if correlation > 0 else 'Negative', 'strength': strength,
            'recent_performance': { asset1: round(asset1_change, 2), asset2: round(asset2_change, 2) },
            'value_transfer_analysis': {'interpretation': self.interpret_correlation(correlation)}
        }

    def get_price_history(self, assets=None):
        """Get price history from original data for correct visualization."""
        if not assets: assets = self.numeric_columns
        
        # *** DEĞİŞİKLİK BURADA: Grafiklerin doğru görünmesi için orijinal veriyi kullanıyoruz. ***
        data_to_use = self.original_data_with_date_index
        
        history = {asset: {'dates': data_to_use.index.strftime('%d-%m-%Y').tolist(), 
                           'values': data_to_use[asset].round(2).tolist()}
                   for asset in assets if asset in self.numeric_columns}
        return history


# =========================================================================== #
# BÖLÜM 2: FLASK API KATMANTI
# =========================================================================== #

app = Flask(__name__)
CORS(app)
predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/assets', methods=['GET'])
def get_assets():
    if predictor is None: return jsonify({'error': 'System not initialized'}), 503
    return jsonify({
        'assets': predictor.numeric_columns,
        'total_records': len(predictor.original_data_with_date_index),
        'date_range': {
            'start': predictor.original_data_with_date_index.index.min().strftime('%d-%m-%Y'),
            'end': predictor.original_data_with_date_index.index.max().strftime('%d-%m-%Y')
        }
    })

@app.route('/api/outliers/<string:asset>', methods=['GET'])
def get_outlier_info(asset):
    if predictor is None: return jsonify({'error': 'System not initialized'}), 503
    if asset not in predictor.outliers_info:
        return jsonify({'error': 'Asset not found or no outlier info available'}), 404
    return jsonify({asset: predictor.outliers_info[asset]})

@app.route('/api/correlations', methods=['GET'])
def get_correlations():
    if predictor is None: return jsonify({'error': 'System not initialized'}), 503
    return jsonify(predictor.analyze_correlations())

@app.route('/api/predict/<string:asset>', methods=['GET'])
def predict_asset(asset):
    if predictor is None: return jsonify({'error': 'System not initialized'}), 503
    if asset not in predictor.numeric_columns:
        return jsonify({'error': f'Asset not found: {asset}'}), 404
    try:
        prediction_data = predictor.predict_future_values(asset)
        return jsonify(prediction_data)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}', 'trace': traceback.format_exc()}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    if predictor is None: return jsonify({'error': 'System not initialized'}), 503
    assets = request.args.getlist('assets')
    history_data = predictor.get_price_history(assets)
    return jsonify(history_data)

@app.route('/api/compare', methods=['GET'])
def compare_assets():
    if predictor is None: return jsonify({'error': 'System not initialized'}), 503
    asset1 = request.args.get('asset1')
    asset2 = request.args.get('asset2')
    if not asset1 or not asset2:
        return jsonify({'error': 'Both asset1 and asset2 parameters are required'}), 400
    try:
        comparison = predictor.get_asset_comparison(asset1, asset2)
        if 'error' in comparison: return jsonify(comparison), 404
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': f'Comparison error: {str(e)}', 'trace': traceback.format_exc()}), 500

# =========================================================================== #
# BÖLÜM 3: UYGULAMA BAŞLATMA
# =========================================================================== #

if __name__ == '__main__':
    # !!! ÖNEMLİ: Lütfen CSV dosyanızın yolunu buraya belirtin !!!
    CSV_FILE_PATH = "../tests/index.csv" # <--- BU SATIRI GEREKİRSE DEĞİŞTİRİN

    print("🚀 Gelişmiş Değer Transferi Tahmin API'si başlatılıyor...")
    
    try:
        predictor = EnhancedValueTransferPredictor(CSV_FILE_PATH)
        print("✅ Sistem başarıyla başlatıldı ve veriler yüklendi.")
        print(f"📊 Toplam Varlık Sayısı: {len(predictor.numeric_columns)}")
        print(f"🗓️ Veri Tarih Aralığı: {predictor.original_data_with_date_index.index.min().strftime('%Y-%m-%d')} - {predictor.original_data_with_date_index.index.max().strftime('%Y-%m-%d')}")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except FileNotFoundError:
        print(f"❌ HATA: Belirtilen yolda CSV dosyası bulunamadı: '{os.path.abspath(CSV_FILE_PATH)}'")
        print("Lütfen `app.py` içindeki `CSV_FILE_PATH` değişkenini doğru dosya yolu ile güncelleyin.")
    except Exception as e:
        print(f"❌ API başlatılırken kritik bir hata oluştu: {e}")
        print(traceback.format_exc())