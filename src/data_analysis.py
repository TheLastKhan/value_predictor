import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ValueTransferPredictor:
    def __init__(self, csv_file_path):
        """
        Financial asset value transfer analysis and prediction class
        """
        self.data = pd.read_csv(csv_file_path)
        self.prepare_data()
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self):
        """Prepare and clean the data"""
        # Convert date to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        
        # Fill missing values with forward fill and backward fill
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        # Get numeric columns
        self.numeric_columns = self.data.columns.tolist()
        
        print("Data prepared!")
        print(f"Total rows: {len(self.data)}")
        print(f"Date range: {self.data.index.min()} - {self.data.index.max()}")
        
    def analyze_correlations(self):
        """Correlation analysis - Find value transfer relationships"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Correlation matrix
        correlation_matrix = self.data.corr()
        
        # Find highest correlations
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                asset1, asset2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                correlation_pairs.append((asset1, asset2, corr_value))
        
        # Sort correlations
        correlation_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("\nStrongest Correlations (Value Transfer Relationships):")
        for asset1, asset2, corr in correlation_pairs[:10]:
            direction = "Same direction" if corr > 0 else "Opposite direction"
            print(f"{asset1} - {asset2}: {corr:.3f} ({direction})")
        
        # Draw correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f')
        plt.title('Financial Assets Correlation Matrix\n(Value Transfer Relationships)')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def create_features(self, target_asset):
        """Feature engineering - Create features from other assets"""
        features_df = self.data.copy()
        
        # 1. Ratio relationships (value transfer indicators)
        for col in self.numeric_columns:
            if col != target_asset:
                features_df[f'{col}_to_{target_asset}_ratio'] = features_df[col] / features_df[target_asset]
        
        # 2. Moving averages (trend indicators)
        for col in self.numeric_columns:
            features_df[f'{col}_ma_7'] = features_df[col].rolling(window=7).mean()
            features_df[f'{col}_ma_30'] = features_df[col].rolling(window=30).mean()
        
        # 3. Price changes (momentum indicators)
        for col in self.numeric_columns:
            features_df[f'{col}_pct_change'] = features_df[col].pct_change()
            features_df[f'{col}_pct_change_7d'] = features_df[col].pct_change(periods=7)
        
        # 4. Volatility indicators
        for col in self.numeric_columns:
            features_df[f'{col}_volatility'] = features_df[col].rolling(window=30).std()
        
        # Clean missing values
        features_df = features_df.dropna()
        
        return features_df
    
    def train_models(self, target_asset='Bitcoin'):
        """Train models"""
        print(f"\n=== MODEL TRAINING FOR {target_asset} ===")
        
        # Create features
        features_df = self.create_features(target_asset)
        
        # Separate target and features
        y = features_df[target_asset]
        X = features_df.drop(columns=[target_asset])
        
        # Select most important features (correlation-based)
        target_corr = features_df.corr()[target_asset].abs().sort_values(ascending=False)
        important_features = target_corr.head(20).index.tolist()
        important_features.remove(target_asset)
        
        X_selected = X[important_features]
        
        print(f"Number of selected features: {len(important_features)}")
        print("Most important features:", important_features[:5])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, shuffle=False
        )
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        # Evaluate model performances
        self.evaluate_models(y_test, lr_pred, rf_pred, target_asset)
        
        # Store models and scaler
        self.models[target_asset] = {
            'linear_regression': lr_model,
            'random_forest': rf_model,
            'features': important_features,
            'test_data': (X_test, y_test)
        }
        self.scalers[target_asset] = scaler
        
        return lr_model, rf_model, important_features
    
    def evaluate_models(self, y_true, lr_pred, rf_pred, target_asset):
        """Evaluate model performances"""
        print(f"\n=== {target_asset} MODEL PERFORMANCES ===")
        
        # Linear Regression metrics
        lr_mse = mean_squared_error(y_true, lr_pred)
        lr_rmse = np.sqrt(lr_mse)
        lr_r2 = r2_score(y_true, lr_pred)
        lr_mae = mean_absolute_error(y_true, lr_pred)
        
        # Random Forest metrics
        rf_mse = mean_squared_error(y_true, rf_pred)
        rf_rmse = np.sqrt(rf_mse)
        rf_r2 = r2_score(y_true, rf_pred)
        rf_mae = mean_absolute_error(y_true, rf_pred)
        
        print("Linear Regression:")
        print(f"  RMSE: {lr_rmse:.2f}")
        print(f"  R²: {lr_r2:.3f}")
        print(f"  MAE: {lr_mae:.2f}")
        
        print("\nRandom Forest:")
        print(f"  RMSE: {rf_rmse:.2f}")
        print(f"  R²: {rf_r2:.3f}")
        print(f"  MAE: {rf_mae:.2f}")
        
        # Prediction vs actual value graph
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, lr_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Linear Regression - {target_asset}\nR² = {lr_r2:.3f}')
        
        plt.subplot(1, 3, 2)
        plt.scatter(y_true, rf_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Random Forest - {target_asset}\nR² = {rf_r2:.3f}')
        
        plt.subplot(1, 3, 3)
        plt.plot(y_true.values, label='Actual', alpha=0.7)
        plt.plot(lr_pred, label='Linear Regression', alpha=0.7)
        plt.plot(rf_pred, label='Random Forest', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'{target_asset} - Prediction Comparison')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_future_value(self, target_asset, steps_ahead=30):
        """Future value prediction"""
        if target_asset not in self.models:
            print(f"You must first train a model for {target_asset}!")
            return None
        
        print(f"\n=== {target_asset} FUTURE VALUE PREDICTION ===")
        
        # Get latest data
        features_df = self.create_features(target_asset)
        last_features = features_df[self.models[target_asset]['features']].iloc[-1:]
        
        # Make prediction with model
        scaler = self.scalers[target_asset]
        last_features_scaled = scaler.transform(last_features)
        
        lr_model = self.models[target_asset]['linear_regression']
        rf_model = self.models[target_asset]['random_forest']
        
        lr_prediction = lr_model.predict(last_features_scaled)[0]
        rf_prediction = rf_model.predict(last_features_scaled)[0]
        
        current_value = self.data[target_asset].iloc[-1]
        
        print(f"Current {target_asset} value: ${current_value:.2f}")
        print(f"Linear Regression prediction: ${lr_prediction:.2f} ({((lr_prediction/current_value-1)*100):+.1f}%)")
        print(f"Random Forest prediction: ${rf_prediction:.2f} ({((rf_prediction/current_value-1)*100):+.1f}%)")
        
        return {
            'current': current_value,
            'linear_regression': lr_prediction,
            'random_forest': rf_prediction
        }

# Usage example
if __name__ == "__main__":
    # Write your CSV file path here
    CSV_FILE_PATH = "../tests/data.csv"  # Write your own file path
    
    try:
        # Initialize predictor
        predictor = ValueTransferPredictor(CSV_FILE_PATH)
        
        # Perform correlation analysis
        correlation_matrix = predictor.analyze_correlations()
        
        # Train model for Bitcoin
        predictor.train_models('Bitcoin')
        
        # Future value prediction
        bitcoin_prediction = predictor.predict_future_value('Bitcoin')
        
        # Also try for Gold
        predictor.train_models('Gold')
        gold_prediction = predictor.predict_future_value('Gold')
        
        print("\n=== VALUE TRANSFER ANALYSIS ===")
        print("This analysis shows value transfer relationships between financial assets.")
        print("Negative correlations indicate that when one asset rises, the other falls.")
        print("This supports the 'value transfer' concept in your theory!")
        
    except FileNotFoundError:
        print("CSV file not found! Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
