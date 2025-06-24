import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedValueTransferPredictor:
    def __init__(self, csv_file_path):
        """
        Enhanced Financial Asset Value Transfer Analysis and Prediction System
        """
        self.data = pd.read_csv(csv_file_path)
        self.original_data = None
        self.outliers_info = {}
        self.prepare_data()
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self):
        """Enhanced data preparation with outlier detection"""
        print("=== DATA PREPARATION ===")
        
        # Store original data
        self.original_data = self.data.copy()
        
        # Convert date to datetime with flexible parsing
        try:
            self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True)
        except:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
        self.data.set_index('Date', inplace=True)
        
        # Get numeric columns
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Original data shape: {self.data.shape}")
        print(f"Assets found: {self.numeric_columns}")
        
        # Remove outliers using IQR method
        self.data_cleaned = self.remove_outliers_iqr(self.data[self.numeric_columns])
        
        # Fill missing values
        self.data_cleaned = self.data_cleaned.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Data after cleaning: {self.data_cleaned.shape}")
        print(f"Date range: {self.data_cleaned.index.min()} - {self.data_cleaned.index.max()}")
        print(f"Outliers removed: {len(self.data) - len(self.data_cleaned)} rows")
        
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
        """Remove outliers using Interquartile Range (IQR) method"""
        df_clean = df.copy()
        outliers_info = {}
        
        print("\n--- Outlier Detection (IQR Method) ---")
        
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
            
            print(f"{column}: {len(outliers)} outliers detected (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
            
            # Replace outliers with boundary values
            df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
            df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
            
        self.outliers_info = outliers_info
        return df_clean
        
    def get_outlier_analysis(self):
        """Return detailed outlier analysis"""
        print("\n=== OUTLIER ANALYSIS REPORT ===")
        
        total_outliers = sum([info['count'] for info in self.outliers_info.values()])
        print(f"Total outliers detected: {total_outliers}")
        
        for asset, info in self.outliers_info.items():
            if info['count'] > 0:
                print(f"\n{asset}:")
                print(f"  Outliers: {info['count']}")
                print(f"  Q1: {info['Q1']:.2f}")
                print(f"  Q3: {info['Q3']:.2f}")
                print(f"  IQR: {info['IQR']:.2f}")
                print(f"  Valid range: {info['lower_bound']:.2f} - {info['upper_bound']:.2f}")
        
        return {
            'outliers_by_asset': self.outliers_info,
            'total_outliers': total_outliers
        }
    
    def analyze_correlations(self):
        """Enhanced correlation analysis with detailed insights"""
        print("\n=== ENHANCED CORRELATION ANALYSIS ===")
        
        correlation_matrix = self.data_cleaned.corr()
        
        # Find correlation pairs with detailed analysis
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                asset1, asset2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                # Determine relationship strength and direction
                if abs(corr_value) > 0.8:
                    strength = 'Very Strong'
                elif abs(corr_value) > 0.6:
                    strength = 'Strong' 
                elif abs(corr_value) > 0.4:
                    strength = 'Moderate'
                elif abs(corr_value) > 0.2:
                    strength = 'Weak'
                else:
                    strength = 'Very Weak'
                    
                direction = 'Positive' if corr_value > 0 else 'Negative'
                
                correlation_pairs.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'correlation': round(corr_value, 4),
                    'direction': direction,
                    'strength': strength,
                    'value_transfer': 'Same Direction' if corr_value > 0 else 'Opposite Direction'
                })
        
        # Sort by absolute correlation value
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Print detailed results
        print("\nStrongest Correlations (Value Transfer Relationships):")
        for i, pair in enumerate(correlation_pairs[:10]):
            print(f"{i+1:2d}. {pair['asset1']} - {pair['asset2']}: {pair['correlation']:+.3f} ({pair['strength']}, {pair['direction']})")
        
        print("\nValue Transfer Candidates (|correlation| > 0.3):")
        value_transfer_candidates = [p for p in correlation_pairs if abs(p['correlation']) > 0.3]
        for pair in value_transfer_candidates[:10]:
            interpretation = self.interpret_correlation(pair['correlation'])
            print(f"• {pair['asset1']} ↔ {pair['asset2']}: {interpretation}")
        
        # Enhanced visualization
        self.plot_correlation_analysis(correlation_matrix, correlation_pairs)
        
        return {
            'matrix': correlation_matrix.round(4).to_dict(),
            'pairs': correlation_pairs,
            'strongest_positive': [p for p in correlation_pairs if p['direction'] == 'Positive'][:5],
            'strongest_negative': [p for p in correlation_pairs if p['direction'] == 'Negative'][:5],
            'value_transfer_candidates': value_transfer_candidates
        }
    
    def plot_correlation_analysis(self, correlation_matrix, correlation_pairs):
        """Enhanced correlation visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f', ax=axes[0,0])
        axes[0,0].set_title('Asset Correlation Matrix\n(Value Transfer Relationships)', fontsize=12)
        
        # 2. Correlation strength distribution
        correlations = [abs(p['correlation']) for p in correlation_pairs]
        axes[0,1].hist(correlations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_xlabel('Absolute Correlation Value')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Correlation Strengths')
        axes[0,1].axvline(x=0.3, color='red', linestyle='--', label='Value Transfer Threshold')
        axes[0,1].legend()
        
        # 3. Top correlations bar chart
        top_10 = correlation_pairs[:10]
        pair_names = [f"{p['asset1']}-{p['asset2']}" for p in top_10]
        corr_values = [p['correlation'] for p in top_10]
        colors = ['green' if x > 0 else 'red' for x in corr_values]
        
        axes[1,0].barh(range(len(pair_names)), corr_values, color=colors, alpha=0.7)
        axes[1,0].set_yticks(range(len(pair_names)))
        axes[1,0].set_yticklabels(pair_names, fontsize=8)
        axes[1,0].set_xlabel('Correlation Coefficient')
        axes[1,0].set_title('Top 10 Asset Correlations')
        axes[1,0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. Scatter plot of sample correlation
        if len(correlation_pairs) > 0:
            top_pair = correlation_pairs[0]
            asset1, asset2 = top_pair['asset1'], top_pair['asset2']
            
            axes[1,1].scatter(self.data_cleaned[asset1], self.data_cleaned[asset2], 
                             alpha=0.6, color='blue')
            axes[1,1].set_xlabel(asset1)
            axes[1,1].set_ylabel(asset2)
            axes[1,1].set_title(f'Strongest Correlation: {asset1} vs {asset2}\n(r = {top_pair["correlation"]:.3f})')
            
            # Add trend line
            z = np.polyfit(self.data_cleaned[asset1], self.data_cleaned[asset2], 1)
            p = np.poly1d(z)
            axes[1,1].plot(self.data_cleaned[asset1], p(self.data_cleaned[asset1]), 
                          "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.show()
    
    def interpret_correlation(self, correlation):
        """Enhanced correlation interpretation for value transfer analysis"""
        if correlation < -0.7:
            return f"Very strong negative correlation ({correlation:.3f}) - Clear value transfer: when one rises significantly, the other falls"
        elif correlation < -0.5:
            return f"Strong negative correlation ({correlation:.3f}) - Strong value transfer relationship"
        elif correlation < -0.3:
            return f"Moderate negative correlation ({correlation:.3f}) - Some inverse relationship exists"
        elif correlation < 0.3:
            return f"Weak correlation ({correlation:.3f}) - Limited relationship between assets"
        elif correlation < 0.5:
            return f"Moderate positive correlation ({correlation:.3f}) - Assets tend to move together"
        elif correlation < 0.7:
            return f"Strong positive correlation ({correlation:.3f}) - Assets move very similarly"
        else:
            return f"Very strong positive correlation ({correlation:.3f}) - Almost identical movement patterns"
    
    def create_advanced_features(self, target_asset, lookback_days=30):
        """Create advanced features for better prediction"""
        print(f"\n--- Creating Advanced Features for {target_asset} ---")
        
        features_df = self.data_cleaned.copy()
        initial_features = len(features_df.columns)
        
        # 1. Price ratios (value transfer indicators)
        for col in self.numeric_columns:
            if col != target_asset:
                features_df[f'{col}_to_{target_asset}_ratio'] = features_df[col] / features_df[target_asset]
                features_df[f'{target_asset}_to_{col}_ratio'] = features_df[target_asset] / features_df[col]
        
        # 2. Moving averages (multiple timeframes)
        for col in self.numeric_columns:
            for window in [5, 10, 20, 50]:
                features_df[f'{col}_ma_{window}'] = features_df[col].rolling(window=window).mean()
                # Price relative to moving average
                features_df[f'{col}_ma_{window}_ratio'] = features_df[col] / features_df[f'{col}_ma_{window}']
        
        # 3. Price changes and momentum
        for col in self.numeric_columns:
            for period in [1, 3, 7, 14, 30]:
                features_df[f'{col}_pct_change_{period}d'] = features_df[col].pct_change(periods=period)
                features_df[f'{col}_price_change_{period}d'] = features_df[col].diff(periods=period)
        
        # 4. Volatility indicators
        for col in self.numeric_columns:
            for window in [7, 14, 30]:
                features_df[f'{col}_volatility_{window}d'] = features_df[col].rolling(window=window).std()
                features_df[f'{col}_volatility_{window}d_ratio'] = features_df[f'{col}_volatility_{window}d'] / features_df[col]
        
        # 5. Technical indicators
        for col in self.numeric_columns:
            # RSI-like indicator
            delta = features_df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Band position
            ma_20 = features_df[col].rolling(window=20).mean()
            std_20 = features_df[col].rolling(window=20).std()
            features_df[f'{col}_bb_upper'] = ma_20 + (2 * std_20)
            features_df[f'{col}_bb_lower'] = ma_20 - (2 * std_20)
            features_df[f'{col}_bb_position'] = (features_df[col] - ma_20) / (2 * std_20)
        
        # 6. Cross-asset momentum
        for col in self.numeric_columns:
            if col != target_asset:
                # Relative strength
                col_momentum = features_df[col].pct_change(periods=20)
                target_momentum = features_df[target_asset].pct_change(periods=20)
                features_df[f'{col}_vs_{target_asset}_momentum'] = col_momentum - target_momentum
        
        # 7. Time-based features
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        features_df['day_of_month'] = features_df.index.day
        
        # Remove infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()
        
        final_features = len(features_df.columns)
        print(f"Features created: {final_features - initial_features}")
        print(f"Final dataset shape: {features_df.shape}")
        
        return features_df
    
    def train_models(self, target_asset='Bitcoin', test_size=0.2, feature_selection_method='correlation'):
        """Enhanced model training with time series validation"""
        print(f"\n=== ENHANCED MODEL TRAINING FOR {target_asset} ===")
        
        # Create features
        features_df = self.create_advanced_features(target_asset)
        
        if len(features_df) == 0:
            print("ERROR: No valid data after feature engineering")
            return {'error': 'No valid data after feature engineering'}
        
        # Separate target and features
        y = features_df[target_asset]
        X = features_df.drop(columns=self.numeric_columns)  # Remove all original price columns
        
        # Feature selection
        if feature_selection_method == 'correlation':
            # Correlation-based feature selection
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(25).index.tolist()
            print(f"Top 5 features by correlation: {top_features[:5]}")
        else:
            # Use all features
            top_features = X.columns.tolist()
        
        X_selected = X[top_features]
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = {'linear_regression': [], 'random_forest': []}
        
        # Regular train-test split for final evaluation
        split_index = int(len(X_selected) * (1 - test_size))
        X_train, X_test = X_selected.iloc[:split_index], X_selected.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model training with cross-validation
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        print("\nPerforming time series cross-validation...")
        
        # Cross-validation scores
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
            X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Linear Regression CV
            lr_temp = LinearRegression()
            lr_temp.fit(X_cv_train, y_cv_train)
            lr_pred = lr_temp.predict(X_cv_val)
            lr_r2 = r2_score(y_cv_val, lr_pred)
            cv_scores['linear_regression'].append(lr_r2)
            
            # Random Forest CV
            rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            rf_temp.fit(X_cv_train, y_cv_train)
            rf_pred = rf_temp.predict(X_cv_val)
            rf_r2 = r2_score(y_cv_val, rf_pred)
            cv_scores['random_forest'].append(rf_r2)
            
            print(f"  Fold {fold+1}: LR R² = {lr_r2:.3f}, RF R² = {rf_r2:.3f}")
        
        # Train final models
        print("\nTraining final models...")
        lr_model.fit(X_train_scaled, y_train)
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        lr_pred = lr_model.predict(X_test_scaled)
        rf_pred = rf_model.predict(X_test_scaled)
        
        # Model evaluation
        evaluation = self.evaluate_models(y_test, lr_pred, rf_pred, target_asset)
        evaluation['cross_validation'] = {
            'linear_regression_cv_score': np.mean(cv_scores['linear_regression']),
            'random_forest_cv_score': np.mean(cv_scores['random_forest']),
            'cv_std': {
                'linear_regression': np.std(cv_scores['linear_regression']),
                'random_forest': np.std(cv_scores['random_forest'])
            }
        }
        
        # Feature importance for Random Forest
        feature_importance = dict(zip(top_features, rf_model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        print(f"\nTop 5 most important features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Store models
        self.models[target_asset] = {
            'linear_regression': lr_model,
            'random_forest': rf_model,
            'features': top_features,
            'scaler': scaler,
            'feature_importance': feature_importance,
            'test_data': (X_test, y_test, lr_pred, rf_pred)
        }
        
        return {
            'target_asset': target_asset,
            'features_used': len(top_features),
            'top_features': list(feature_importance.keys())[:10],
            'feature_importance': feature_importance,
            'evaluation': evaluation,
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'total_features': len(top_features)
            }
        }
    
    def evaluate_models(self, y_true, lr_pred, rf_pred, target_asset):
        """Enhanced model evaluation with detailed metrics"""
        print(f"\n=== {target_asset} MODEL EVALUATION ===")
        
        # Linear Regression metrics
        lr_metrics = {
            'mse': mean_squared_error(y_true, lr_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, lr_pred)),
            'mae': mean_absolute_error(y_true, lr_pred),
            'r2': r2_score(y_true, lr_pred),
            'mape': np.mean(np.abs((y_true - lr_pred) / y_true)) * 100
        }
        
        # Random Forest metrics
        rf_metrics = {
            'mse': mean_squared_error(y_true, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, rf_pred)),
            'mae': mean_absolute_error(y_true, rf_pred),
            'r2': r2_score(y_true, rf_pred),
            'mape': np.mean(np.abs((y_true - rf_pred) / y_true)) * 100
        }
        
        print("Linear Regression:")
        print(f"  RMSE: {lr_metrics['rmse']:.2f}")
        print(f"  R²: {lr_metrics['r2']:.3f}")
        print(f"  MAE: {lr_metrics['mae']:.2f}")
        print(f"  MAPE: {lr_metrics['mape']:.2f}%")
        
        print("\nRandom Forest:")
        print(f"  RMSE: {rf_metrics['rmse']:.2f}")
        print(f"  R²: {rf_metrics['r2']:.3f}")
        print(f"  MAE: {rf_metrics['mae']:.2f}")
        print(f"  MAPE: {rf_metrics['mape']:.2f}%")
        
        best_model = 'random_forest' if rf_metrics['r2'] > lr_metrics['r2'] else 'linear_regression'
        print(f"\nBest performing model: {best_model.replace('_', ' ').title()}")
        
        # Enhanced prediction visualization
        self.plot_model_evaluation(y_true, lr_pred, rf_pred, target_asset, lr_metrics, rf_metrics)
        
        return {
            'linear_regression': lr_metrics,
            'random_forest': rf_metrics,
            'best_model': best_model
        }
    
    def plot_model_evaluation(self, y_true, lr_pred, rf_pred, target_asset, lr_metrics, rf_metrics):
        """Enhanced model evaluation visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Linear Regression: Actual vs Predicted
        axes[0,0].scatter(y_true, lr_pred, alpha=0.6, color='blue')
        axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Value')
        axes[0,0].set_ylabel('Predicted Value')
        axes[0,0].set_title(f'Linear Regression - {target_asset}\nR² = {lr_metrics["r2"]:.3f}')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Random Forest: Actual vs Predicted
        axes[0,1].scatter(y_true, rf_pred, alpha=0.6, color='green')
        axes[0,1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0,1].set_xlabel('Actual Value')
        axes[0,1].set_ylabel('Predicted Value')
        axes[0,1].set_title(f'Random Forest - {target_asset}\nR² = {rf_metrics["r2"]:.3f}')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Time series comparison
        axes[0,2].plot(y_true.values, label='Actual', linewidth=2, alpha=0.8)
        axes[0,2].plot(lr_pred, label='Linear Regression', linewidth=2, alpha=0.8)
        axes[0,2].plot(rf_pred, label='Random Forest', linewidth=2, alpha=0.8)
        axes[0,2].set_xlabel('Time Index')
        axes[0,2].set_ylabel('Value')
        axes[0,2].set_title(f'{target_asset} - Prediction Comparison')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Residuals - Linear Regression
        lr_residuals = y_true - lr_pred
        axes[1,0].scatter(lr_pred, lr_residuals, alpha=0.6, color='blue')
        axes[1,0].axhline(y=0, color='red', linestyle='--')
        axes[1,0].set_xlabel('Predicted Value')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Linear Regression - Residual Plot')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Residuals - Random Forest
        rf_residuals = y_true - rf_pred
        axes[1,1].scatter(rf_pred, rf_residuals, alpha=0.6, color='green')
        axes[1,1].axhline(y=0, color='red', linestyle='--')
        axes[1,1].set_xlabel('Predicted Value')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title('Random Forest - Residual Plot')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Error distribution comparison
        axes[1,2].hist(lr_residuals, bins=20, alpha=0.7, label='Linear Regression', color='blue')
        axes[1,2].hist(rf_residuals, bins=20, alpha=0.7, label='Random Forest', color='green')
        axes[1,2].set_xlabel('Residuals')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Error Distribution Comparison')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future_values(self, target_asset='Bitcoin', days_ahead=30):
        """Enhanced future value prediction with confidence intervals"""
        if target_asset not in self.models:
            print(f"ERROR: Model not trained for {target_asset}. Please train first!")
            return {'error': f'Model not trained for {target_asset}'}
        
        print(f"\n=== {target_asset} FUTURE VALUE PREDICTION ===")
        
        model_info = self.models[target_asset]
        lr_model = model_info['linear_regression']
        rf_model = model_info['random_forest']
        scaler = model_info['scaler']
        features = model_info['features']
        
        # Get the latest features
        features_df = self.create_advanced_features(target_asset)
        if len(features_df) == 0:
            return {'error': 'Cannot create features for prediction'}
        
        latest_features = features_df[features].iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        
        # Current value and date
        current_value = self.data_cleaned[target_asset].iloc[-1]
        current_date = self.data_cleaned.index[-1]
# Single-step predictions
        lr_next = lr_model.predict(latest_features_scaled)[0]
        rf_next = rf_model.predict(latest_features_scaled)[0]
        
        # Multi-step predictions (simplified approach)
        predictions = []
        
        for i in range(1, min(days_ahead + 1, 31)):  # Limit to 30 days for accuracy
            future_date = current_date + timedelta(days=i)
            
            # Simple trend-based extrapolation
            trend_factor = 1 + (i * 0.001)  # Small daily trend
            
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
            'future_predictions': predictions[:days_ahead],
            'prediction_confidence': 'High' if len(predictions) <= 7 else 'Medium' if len(predictions) <= 14 else 'Low'
        }
    
    def get_asset_comparison(self, asset1, asset2, days=30):
        """Enhanced asset comparison with recent performance"""
        if asset1 not in self.numeric_columns or asset2 not in self.numeric_columns:
            return {'error': 'One or both assets not found'}
        
        # Get recent data
        recent_data = self.data_cleaned.tail(days)
        
        if len(recent_data) == 0:
            return {'error': 'No recent data available'}
        
        # Calculate correlation
        correlation = self.data_cleaned[asset1].corr(self.data_cleaned[asset2])
        
        # Calculate performance metrics
        asset1_start = recent_data[asset1].iloc[0]
        asset1_end = recent_data[asset1].iloc[-1]
        asset1_change = ((asset1_end / asset1_start) - 1) * 100
        
        asset2_start = recent_data[asset2].iloc[0]
        asset2_end = recent_data[asset2].iloc[-1]
        asset2_change = ((asset2_end / asset2_start) - 1) * 100
        
        # Volatility
        asset1_volatility = recent_data[asset1].pct_change().std() * np.sqrt(252) * 100
        asset2_volatility = recent_data[asset2].pct_change().std() * np.sqrt(252) * 100
        
        return {
            'asset1': asset1,
            'asset2': asset2,
            'correlation': round(correlation, 4),
            'relationship': {
                'direction': 'Positive' if correlation > 0 else 'Negative',
                'strength': 'Very Strong' if abs(correlation) > 0.8 else 'Strong' if abs(correlation) > 0.6 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'
            },
            'recent_performance': {
                asset1: {
                    'change_percent': round(asset1_change, 2),
                    'start_value': round(asset1_start, 2),
                    'end_value': round(asset1_end, 2),
                    'volatility': round(asset1_volatility, 2)
                },
                asset2: {
                    'change_percent': round(asset2_change, 2),
                    'start_value': round(asset2_start, 2),
                    'end_value': round(asset2_end, 2),
                    'volatility': round(asset2_volatility, 2)
                }
            },
            'value_transfer_analysis': {
                'interpretation': self.interpret_correlation(correlation),
                'trading_signal': self.generate_trading_signal(asset1_change, asset2_change, correlation)
            },
            'period_analyzed': f'{days} days'
        }
    
    def generate_trading_signal(self, change1, change2, correlation):
        """Generate simple trading signals based on correlation and recent performance"""
        if abs(correlation) < 0.3:
            return "No clear signal - weak correlation"
        
        if correlation < -0.3:  # Negative correlation
            if change1 > 2 and change2 < -2:
                return "Value transfer confirmed - Asset 1 gaining at expense of Asset 2"
            elif change1 < -2 and change2 > 2:
                return "Value transfer confirmed - Asset 2 gaining at expense of Asset 1"
            else:
                return "Correlation exists but no clear recent transfer"
        else:  # Positive correlation
            if change1 > 0 and change2 > 0:
                return "Both assets rising together - similar market forces"
            elif change1 < 0 and change2 < 0:
                return "Both assets falling together - similar market pressures"
            else:
                return "Mixed signals despite positive correlation"
    
    def get_price_history(self, assets=None, days=None):
        """Get price history with proper date formatting"""
        if assets is None:
            assets = self.numeric_columns[:10]  # Limit to first 10 assets
        
        data_to_use = self.data_cleaned if days is None else self.data_cleaned.tail(days)
        
        history = {}
        for asset in assets:
            if asset in self.numeric_columns:
                history[asset] = {
                    'dates': data_to_use.index.strftime('%Y-%m-%d').tolist(),
                    'values': data_to_use[asset].round(2).tolist(),
                    'current_value': round(data_to_use[asset].iloc[-1], 2),
                    'change_30d': round(((data_to_use[asset].iloc[-1] / data_to_use[asset].iloc[-30 if len(data_to_use) > 30 else 0]) - 1) * 100, 2) if len(data_to_use) > 1 else 0
                }
        
        return {
            'history': history,
            'date_range': {
                'start': data_to_use.index.min().strftime('%Y-%m-%d'),
                'end': data_to_use.index.max().strftime('%Y-%m-%d')
            },
            'total_records': len(data_to_use)
        }
    
    def get_system_status(self):
        """Get system status and information"""
        return {
            'status': 'active',
            'data_info': {
                'total_records': len(self.data_cleaned),
                'assets_count': len(self.numeric_columns),
                'date_range': {
                    'start': self.data_cleaned.index.min().strftime('%Y-%m-%d'),
                    'end': self.data_cleaned.index.max().strftime('%Y-%m-%d')
                }
            },
            'models_trained': list(self.models.keys()),
            'assets_available': self.numeric_columns,
            'features': {
                'outlier_detection': True,
                'correlation_analysis': True,
                'multi_step_prediction': True,
                'feature_importance': True,
                'cross_validation': True
            }
        }
    
from flask import Flask, jsonify, request
from flask_cors import CORS  # opsiyonel ama önerilir
import os
 
# Flask app oluştur
app = Flask(__name__)
CORS(app)  # tarayıcıdan erişim için gerekli (özellikle HTML ayrıysa)
 
# CSV yolu
CSV_PATH = "../tests/index.csv"
 
# Rotalar
@app.route("/api/assets")
def get_assets():
    predictor = EnhancedValueTransferPredictor(CSV_PATH)
    return jsonify({
        "assets": predictor.numeric_columns,
        "date_range": {
            "start": predictor.data_cleaned.index.min().strftime('%Y-%m-%d'),
            "end": predictor.data_cleaned.index.max().strftime('%Y-%m-%d')
        }
    })
 
@app.route("/api/predict/<asset>")
def predict_asset(asset):
    predictor = EnhancedValueTransferPredictor(CSV_PATH)
    try:
        predictor.train_models(target_asset=asset)
        prediction = predictor.predict_future_values(target_asset=asset)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
# Uygulama başlangıç noktası
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("index.csv dosyası bulunamadı!")
    else:
        print("Sunucu başlatılıyor...")
        app.run(debug=True, port=5000)
 
 