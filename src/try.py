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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Uyarıları bastır
warnings.filterwarnings('ignore')

# Flask uygulamasını başlat
app = Flask(__name__)
CORS(app)  # CORS desteği ekle

class EnhancedValueTransferAPI:
    def __init__(self, csv_file_path):
        """
        Gelişmiş Finansal Varlık Değer Transferi Analizi ve Tahmin Sistemi
        """
        self.data = pd.read_csv(csv_file_path, dayfirst=True)
        self.original_data = None
        self.outliers_info = {}
        self.prepare_data()
        self.models = {}
        self.scalers = {}

    def prepare_data(self):
        """Aykırı değer tespiti ile geliştirilmiş veri hazırlama"""
        print("=== VERİ HAZIRLAMA ===")
        
        # Orijinal veriyi sakla
        self.original_data = self.data.copy()
        
        # Tarih sütununu esnek bir şekilde datetime formatına çevir
        try:
            self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True)
        except:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
        self.data.set_index('Date', inplace=True)
        
        # Sayısal sütunları al
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Orijinal veri boyutu: {self.data.shape}")
        print(f"Bulunan varlıklar: {self.numeric_columns}")
        
        # IQR yöntemiyle aykırı değerleri temizle
        self.data_cleaned = self.remove_outliers_iqr(self.data[self.numeric_columns])
        
        # Eksik değerleri doldur
        self.data_cleaned = self.data_cleaned.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Temizleme sonrası veri boyutu: {self.data_cleaned.shape}")
        print(f"Tarih aralığı: {self.data_cleaned.index.min()} - {self.data_cleaned.index.max()}")
        print(f"Kaldırılan aykırı değer sayısı: {len(self.data) - len(self.data_cleaned)} satır")
        
        return {
            'status': 'success',
            'total_rows': len(self.data_cleaned),
            'date_range': {
                'start': self.data_cleaned.index.min().strftime('%Y-%m-%d'),
                'end': self.data_cleaned.index.max().strftime('%Y-%m-%d')
            },
            'assets': self.numeric_columns,
            'outliers_removed': len(self.data) - len(self.data_cleaned)
        }

    def remove_outliers_iqr(self, df):
        """Interquartile Range (IQR) metodu ile aykırı değerleri kaldırma"""
        df_clean = df.copy()
        outliers_info = {}
        
        print("\n--- Aykırı Değer Tespiti (IQR Metodu) ---")
        
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outliers_info[column] = {
                'count': len(outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
            
            print(f"{column}: {len(outliers)} aykırı değer tespit edildi (sınırlar: {lower_bound:.2f} - {upper_bound:.2f})")
            
            # Aykırı değerleri sınır değerleri ile değiştir
            df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
            df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
            
        self.outliers_info = outliers_info
        return df_clean

    def get_correlation_data(self):
        """Detaylı içgörülerle geliştirilmiş korelasyon analizi"""
        correlation_matrix = self.data_cleaned.corr()
        
        # Detaylı analiz ile korelasyon çiftlerini bul
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                asset1, asset2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                # İlişki gücünü ve yönünü belirle
                if abs(corr_value) > 0.8:
                    strength = 'Çok Güçlü'
                elif abs(corr_value) > 0.6:
                    strength = 'Güçlü'
                elif abs(corr_value) > 0.4:
                    strength = 'Orta'
                elif abs(corr_value) > 0.2:
                    strength = 'Zayıf'
                else:
                    strength = 'Çok Zayıf'
                    
                direction = 'Pozitif' if corr_value > 0 else 'Negatif'
                
                correlation_pairs.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'correlation': round(corr_value, 4),
                    'direction': direction,
                    'strength': strength,
                    'relationship': direction,
                    'value_transfer': 'Aynı Yön' if corr_value > 0 else 'Ters Yön'
                })
        
        # Mutlak korelasyon değerine göre sırala
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'matrix': correlation_matrix.round(4).to_dict(),
            'pairs': correlation_pairs[:50],  # En güçlü 50 ilişki
            'strongest_positive': [p for p in correlation_pairs if p['direction'] == 'Pozitif'][:10],
            'strongest_negative': [p for p in correlation_pairs if p['direction'] == 'Negatif'][:10],
            'value_transfer_candidates': [p for p in correlation_pairs if abs(p['correlation']) > 0.3][:20],
            'assets': self.numeric_columns
        }

    def create_advanced_features(self, target_asset, lookback_days=30):
        """Daha iyi tahmin için gelişmiş özellikler oluşturma"""
        features_df = self.data_cleaned.copy()
        
        # 1. Fiyat oranları (değer transferi göstergeleri)
        for col in self.numeric_columns:
            if col != target_asset:
                with np.errstate(divide='ignore', invalid='ignore'):
                    features_df[f'{col}_to_{target_asset}_ratio'] = features_df[col] / features_df[target_asset]
                    features_df[f'{target_asset}_to_{col}_ratio'] = features_df[target_asset] / features_df[col]
        
        # 2. Hareketli ortalamalar (çoklu zaman dilimleri)
        for col in self.numeric_columns:
            for window in [5, 10, 20, 30]:
                features_df[f'{col}_ma_{window}'] = features_df[col].rolling(window=window).mean()
                with np.errstate(divide='ignore', invalid='ignore'):
                    features_df[f'{col}_ma_{window}_ratio'] = features_df[col] / features_df[f'{col}_ma_{window}']

        # 3. Fiyat değişiklikleri ve momentum
        for col in self.numeric_columns:
            for period in [1, 3, 7, 14]:
                features_df[f'{col}_pct_change_{period}d'] = features_df[col].pct_change(periods=period)
                features_df[f'{col}_price_change_{period}d'] = features_df[col].diff(periods=period)

        # 4. Volatilite göstergeleri
        for col in self.numeric_columns:
            for window in [7, 14, 21]:
                features_df[f'{col}_volatility_{window}d'] = features_df[col].rolling(window=window).std()
                with np.errstate(divide='ignore', invalid='ignore'):
                    features_df[f'{col}_volatility_{window}d_ratio'] = features_df[f'{col}_volatility_{window}d'] / features_df[col]

        # 5. Teknik göstergeler
        for col in self.numeric_columns:
            # RSI benzeri gösterge
            delta = features_df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
                features_df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bandı konumu
            ma_20 = features_df[col].rolling(window=20).mean()
            std_20 = features_df[col].rolling(window=20).std()
            features_df[f'{col}_bb_upper'] = ma_20 + (2 * std_20)
            features_df[f'{col}_bb_lower'] = ma_20 - (2 * std_20)
            with np.errstate(divide='ignore', invalid='ignore'):
                features_df[f'{col}_bb_position'] = (features_df[col] - ma_20) / (2 * std_20)

        # 6. Çapraz varlık momentumu
        for col in self.numeric_columns:
            if col != target_asset:
                col_momentum = features_df[col].pct_change(periods=20)
                target_momentum = features_df[target_asset].pct_change(periods=20)
                features_df[f'{col}_vs_{target_asset}_momentum'] = col_momentum - target_momentum
        
        # 7. Zaman tabanlı özellikler
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        features_df['day_of_month'] = features_df.index.day
        
        # Sonsuz ve NaN değerleri kaldır
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()
        
        return features_df

    def train_and_predict(self, target_asset):
        """Geliştirilmiş model eğitimi ve tahmini"""
        if target_asset not in self.numeric_columns:
            return {'error': f'Varlık {target_asset} bulunamadı'}
            
        # Özellikleri oluştur
        features_df = self.create_advanced_features(target_asset)
        
        if len(features_df) == 0:
            return {'error': 'Özellik mühendisliği sonrası geçerli veri yok'}
            
        # Hedef ve özellikleri ayır
        y = features_df[target_asset]
        X = features_df.drop(columns=self.numeric_columns)  # Tüm orijinal fiyat sütunlarını kaldır
        
        # Korelasyona dayalı özellik seçimi
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(20).index.tolist()
        X_selected = X[top_features]
        
        # Eğitim-test ayrımı (zaman serisine duyarlı)
        split_index = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected.iloc[:split_index], X_selected.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        # Ölçeklendirme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelleri eğit
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_score = r2_score(y_test, lr_pred)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_score = r2_score(y_test, rf_pred)
        
        # Gelecek tahmini
        latest_features = X_selected.iloc[-1:]
        latest_features_scaled = scaler.transform(latest_features)
        
        lr_future = lr_model.predict(latest_features_scaled)[0]
        rf_future = rf_model.predict(latest_features_scaled)[0]
        current_value = self.data_cleaned[target_asset].iloc[-1]
        
        # Modelleri sakla
        self.models[target_asset] = {
            'linear_regression': lr_model,
            'random_forest': rf_model,
            'features': top_features,
            'scaler': scaler
        }
        
        return {
            'current_value': round(current_value, 2),
            'linear_regression': {
                'prediction': round(lr_future, 2),
                'change_percent': round(((lr_future / current_value - 1) * 100), 2),
                'r2_score': round(lr_score, 3)
            },
            'random_forest': {
                'prediction': round(rf_future, 2),
                'change_percent': round(((rf_future / current_value - 1) * 100), 2),
                'r2_score': round(rf_score, 3)
            },
            'important_features': top_features[:5],
            'model_comparison': {
                'best_model': 'random_forest' if rf_score > lr_score else 'linear_regression',
                'performance_difference': round(abs(rf_score - lr_score), 3)
            }
        }

    def predict_future_values(self, target_asset, days_ahead=7):
        """Geliştirilmiş gelecek değer tahmini"""
        if target_asset not in self.models:
            train_result = self.train_and_predict(target_asset)
            if 'error' in train_result:
                return train_result
        
        model_info = self.models[target_asset]
        lr_model = model_info['linear_regression']
        rf_model = model_info['random_forest']
        scaler = model_info['scaler']
        features = model_info['features']
        
        # En son özellikleri al
        features_df = self.create_advanced_features(target_asset)
        if len(features_df) == 0:
            return {'error': 'Tahmin için özellik oluşturulamıyor'}
            
        latest_features = features_df[features].iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        
        # Mevcut değer ve tarih
        current_value = self.data_cleaned[target_asset].iloc[-1]
        current_date = self.data_cleaned.index[-1]
        
        # Tek adımlı tahminler
        lr_next = lr_model.predict(latest_features_scaled)[0]
        rf_next = rf_model.predict(latest_features_scaled)[0]
        
        # Çok adımlı tahminler (basitleştirilmiş yaklaşım)
        predictions = []
        
        for i in range(1, min(days_ahead + 1, 15)):  # Doğruluk için 14 gün ile sınırla
            future_date = current_date + timedelta(days=i)
            
            # Basit trend tabanlı ekstrapolasyon
            trend_factor = 1 + (i * 0.001)  # Küçük günlük trend
            
            lr_future = lr_next * trend_factor
            rf_future = rf_next * trend_factor
            
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'days_ahead': i,
                'linear_regression': round(lr_future, 2),
                'random_forest': round(rf_future, 2),
                'average': round((lr_future + rf_future) / 2, 2)
            })
            
        return {
            'target_asset': target_asset,
            'current_value': round(current_value, 2),
            'current_date': current_date.strftime('%Y-%m-%d'),
            'next_day_prediction': {
                'linear_regression': {
                    'value': round(lr_next, 2),
                    'change_percent': round(((lr_next / current_value) - 1) * 100, 2)
                },
                'random_forest': {
                    'value': round(rf_next, 2),
                    'change_percent': round(((rf_next / current_value) - 1) * 100, 2)
                }
            },
            'future_predictions': predictions,
            'prediction_confidence': 'Yüksek' if len(predictions) <= 3 else 'Orta' if len(predictions) <= 7 else 'Düşük'
        }

    def get_price_history(self, assets=None, days=None):
        """Fiyat geçmişi verilerini döndür"""
        if assets is None:
            # Varsayılan varlıklar - ilk 10 veya 10'dan azsa hepsi
            assets = self.numeric_columns[:min(10, len(self.numeric_columns))]
            
        data_to_use = self.data_cleaned if days is None else self.data_cleaned.tail(days)
        
        history_data = {}
        for asset in assets:
            if asset in self.data_cleaned.columns:
                history_data[asset] = {
                    'dates': data_to_use.index.strftime('%Y-%m-%d').tolist(),
                    'values': data_to_use[asset].round(2).tolist(),
                    'current_value': round(data_to_use[asset].iloc[-1], 2),
                    'change_30d': round(((data_to_use[asset].iloc[-1] / data_to_use[asset].iloc[-30 if len(data_to_use) > 30 else 0]) - 1) * 100, 2) if len(data_to_use) > 1 else 0
                }
                
        return {
            'history': history_data,
            'date_range': {
                'start': data_to_use.index.min().strftime('%Y-%m-%d'),
                'end': data_to_use.index.max().strftime('%Y-%m-%d')
            },
            'total_records': len(data_to_use)
        }

    def compare_assets(self, asset1, asset2, days=30):
        """Geliştirilmiş varlık karşılaştırması"""
        if asset1 not in self.numeric_columns or asset2 not in self.numeric_columns:
            return {'error': 'Varlıklardan biri veya her ikisi bulunamadı'}
            
        # Son verileri al
        recent_data = self.data_cleaned.tail(days)
        
        if len(recent_data) == 0:
            return {'error': 'Yakın tarihli veri mevcut değil'}
            
        # Korelasyonu hesapla
        correlation = self.data_cleaned[[asset1, asset2]].corr().iloc[0, 1]
        
        # Performans metriklerini hesapla
        asset1_start = recent_data[asset1].iloc[0]
        asset1_end = recent_data[asset1].iloc[-1]
        asset1_change = ((asset1_end / asset1_start) - 1) * 100
        
        asset2_start = recent_data[asset2].iloc[0]
        asset2_end = recent_data[asset2].iloc[-1]
        asset2_change = ((asset2_end / asset2_start) - 1) * 100
        
        # Volatilite
        asset1_volatility = recent_data[asset1].pct_change().std() * np.sqrt(252) * 100
        asset2_volatility = recent_data[asset2].pct_change().std() * np.sqrt(252) * 100
        
        # İlişki gücünü belirle
        if abs(correlation) > 0.7:
            strength = 'Güçlü'
        elif abs(correlation) > 0.4:
            strength = 'Orta'
        else:
            strength = 'Zayıf'
            
        return {
            'asset1': asset1,
            'asset2': asset2,
            'correlation': round(correlation, 3),
            'relationship': 'Pozitif' if correlation > 0 else 'Negatif',
            'strength': strength,
            'recent_performance': {
                asset1: {
                    'change_percent': round(asset1_change, 2),
                    'volatility': round(asset1_volatility, 2)
                },
                asset2: {
                    'change_percent': round(asset2_change, 2),
                    'volatility': round(asset2_volatility, 2)
                }
            },
            'value_transfer_analysis': {
                'description': f"{asset1} ve {asset2} arasındaki korelasyon {correlation:.3f}",
                'interpretation': self.interpret_correlation(correlation),
                'trading_signal': self.generate_trading_signal(asset1_change, asset2_change, correlation)
            }
        }

    def interpret_correlation(self, correlation):
        """Geliştirilmiş korelasyon yorumlaması"""
        if correlation < -0.7:
            return f"Çok güçlü negatif korelasyon ({correlation:.3f}) - Net değer transferi: biri önemli ölçüde yükseldiğinde diğeri düşer"
        elif correlation < -0.5:
            return f"Güçlü negatif korelasyon ({correlation:.3f}) - Güçlü değer transferi ilişkisi"
        elif correlation < -0.3:
            return f"Orta düzeyde negatif korelasyon ({correlation:.3f}) - Bir miktar ters ilişki mevcut"
        elif correlation < 0.3:
            return f"Zayıf korelasyon ({correlation:.3f}) - Varlıklar arasında sınırlı ilişki"
        elif correlation < 0.5:
            return f"Orta düzeyde pozitif korelasyon ({correlation:.3f}) - Varlıklar birlikte hareket etme eğiliminde"
        elif correlation < 0.7:
            return f"Güçlü pozitif korelasyon ({correlation:.3f}) - Varlıklar çok benzer şekilde hareket eder"
        else:
            return f"Çok güçlü pozitif korelasyon ({correlation:.3f}) - Neredeyse aynı hareket kalıpları"

    def generate_trading_signal(self, change1, change2, correlation):
        """Basit alım-satım sinyalleri üretme"""
        if abs(correlation) < 0.3:
            return "Net bir sinyal yok - zayıf korelasyon"
            
        if correlation < -0.3:  # Negatif korelasyon
            if change1 > 2 and change2 < -2:
                return "Değer transferi doğrulandı - Varlık 1, Varlık 2'nin aleyhine kazanıyor"
            elif change1 < -2 and change2 > 2:
                return "Değer transferi doğrulandı - Varlık 2, Varlık 1'in aleyhine kazanıyor"
            else:
                return "Korelasyon mevcut ancak son zamanlarda net bir transfer yok"
        else:  # Pozitif korelasyon
            if change1 > 0 and change2 > 0:
                return "Her iki varlık da birlikte yükseliyor - benzer piyasa güçleri"
            elif change1 < 0 and change2 < 0:
                return "Her iki varlık da birlikte düşüyor - benzer piyasa baskıları"
            else:
                return "Pozitif korelasyona rağmen karışık sinyaller"

    def get_outlier_analysis(self):
        """Detaylı aykırı değer analizini döndür"""
        total_outliers = sum([info['count'] for info in self.outliers_info.values()])
        
        return {
            'outliers_by_asset': self.outliers_info,
            'total_outliers': total_outliers,
            'assets_with_outliers': [asset for asset, info in self.outliers_info.items() if info['count'] > 0]
        }

# Global değişken
api = None

@app.route('/')
def index():
    """Ana sayfa"""
    try:
        # Eğer bir index.html şablonunuz varsa onu render eder.
        return render_template('index.html')
    except:
        # Şablon yoksa veya bir hata oluşursa, API endpoint'lerini listeleyen bir JSON döner.
        return jsonify({
            'message': 'Gelişmiş Değer Transferi API',
            'status': 'active',
            'endpoints': {
                '/api/assets': 'Mevcut varlıkları al',
                '/api/correlations': 'Korelasyon analizini al',
                '/api/predict/<asset>': 'Varlık değerini tahmin et',
                '/api/history': 'Fiyat geçmişini al',
                '/api/compare': 'İki varlığı karşılaştır',
                '/api/outliers': 'Aykırı değer analizini al',
                '/api/status': 'Sistem durumunu al'
            }
        })

@app.route('/api/assets')
def get_assets():
    """Gelişmiş bilgilerle mevcut varlıkları döndür"""
    return jsonify({
        'assets': api.numeric_columns,
        'total_records': len(api.data_cleaned),
        'date_range': {
            'start': api.data_cleaned.index.min().strftime('%Y-%m-%d'),
            'end': api.data_cleaned.index.max().strftime('%Y-%m-%d')
        },
        'outliers_removed': len(api.data) - len(api.data_cleaned) if hasattr(api, 'data') else 0
    })

@app.route('/api/correlations')
def get_correlations():
    """Geliştirilmiş korelasyon analizini döndür"""
    return jsonify(api.get_correlation_data())

@app.route('/api/predict/<asset>')
def predict_asset(asset):
    """Belirli bir varlık için tahmin yap"""
    if asset not in api.numeric_columns:
        return jsonify({'error': 'Varlık bulunamadı'}), 404
        
    try:
        prediction_data = api.train_and_predict(asset)
        return jsonify(prediction_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<asset>/future')
def predict_asset_future(asset):
    """Belirli bir varlık için gelecek tahminleri yap"""
    if asset not in api.numeric_columns:
        return jsonify({'error': 'Varlık bulunamadı'}), 404
        
    days = request.args.get('days', 7, type=int)
    days = min(days, 14)  # 14 gün ile sınırla
    
    try:
        prediction_data = api.predict_future_values(asset, days)
        return jsonify(prediction_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    """Fiyat geçmişini döndür"""
    assets = request.args.getlist('assets')
    days = request.args.get('days', type=int)
    
    if not assets:
        assets = None  # Varsayılan varlıkları kullanacak
        
    try:
        history_data = api.get_price_history(assets, days)
        return jsonify(history_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare')
def compare_assets_route():
    """İki varlığı karşılaştır"""
    asset1 = request.args.get('asset1', 'Bitcoin')
    asset2 = request.args.get('asset2', 'Gold')
    days = request.args.get('days', 30, type=int)
    
    try:
        comparison_data = api.compare_assets(asset1, asset2, days)
        return jsonify(comparison_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/outliers')
def get_outliers():
    """Aykırı değer analizini al"""
    return jsonify(api.get_outlier_analysis())

@app.route('/api/status')
def get_status():
    """Sistem durumunu al"""
    return jsonify({
        'status': 'active',
        'data_info': {
            'total_records': len(api.data_cleaned),
            'assets_count': len(api.numeric_columns),
            'date_range': {
                'start': api.data_cleaned.index.min().strftime('%Y-%m-%d'),
                'end': api.data_cleaned.index.max().strftime('%Y-%m-%d')
            }
        },
        'models_trained': list(api.models.keys()),
        'assets_available': api.numeric_columns,
        'features': {
            'outlier_detection': True,
            'enhanced_correlation_analysis': True,
            'multi_step_prediction': True,
            'feature_importance': True,
            'time_series_split': True,
            'advanced_features': True
        }
    })

if __name__ == '__main__':
    # CSV dosya yolu - Bu yolu kendi dosya konumunuza göre güncelleyin
    CSV_FILE_PATH = "../tests/index.csv"  # <-- BU YOLU GÜNCELLEYİN

    try:
        print("Gelişmiş Değer Transferi API'si başlatılıyor...")
        api = EnhancedValueTransferAPI(CSV_FILE_PATH)
        print("\n" + "="*60)
        print("🚀 GELİŞMİŞ DEĞER TRANSFERİ API'Sİ BAŞLATILDI! 🚀")
        print("="*60)
        print(f"📊 Toplam veri kaydı: {len(api.data_cleaned)}")
        print(f"📈 Mevcut varlık sayısı: {len(api.numeric_columns)}")
        print(f"📅 Tarih aralığı: {api.data_cleaned.index.min().strftime('%Y-%m-%d')} to {api.data_cleaned.index.max().strftime('%Y-%m-%d')}")
        print(f"🔧 Kaldırılan aykırı değer: {len(api.data) - len(api.data_cleaned)}")
        print("\n📡 Mevcut API endpointleri:")
        print("   • GET /api/assets - Tüm mevcut varlıkları listele")
        print("   • GET /api/correlations - Korelasyon analizini al")
        print("   • GET /api/predict/<asset> - Tekli tahmin al")
        print("   • GET /api/predict/<asset>/future?days=7 - Gelecek tahminlerini al")
        print("   • GET /api/history?assets=Bitcoin,Gold&days=30 - Fiyat geçmişini al")
        print("   • GET /api/compare?asset1=Bitcoin&asset2=Gold&days=30 - Varlıkları karşılaştır")
        print("   • GET /api/outliers - Aykırı değer analizini al")
        print("   • GET /api/status - Sistem durumunu al")
        print("="*60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)

    except FileNotFoundError:
        print("\n" + "❌ HATA: CSV dosyası bulunamadı! ❌")
        print(f"Lütfen dosyanın şu yolda mevcut olduğunu kontrol edin: {CSV_FILE_PATH}")
        print("Lütfen `CSV_FILE_PATH` değişkenini veri dosyanızın doğru yolu ile güncelleyin.")
    except Exception as e:
        print(f"\n❌ API başlatılırken beklenmedik bir hata oluştu: {e}")

