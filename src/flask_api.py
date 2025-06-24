from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import traceback
import os
from predictor import EnhancedValueTransferPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global predictor instance
predictor = None

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the prediction system with CSV data"""
    global predictor
    try:
        data = request.get_json()
        csv_file_path = data.get('csv_file_path')
        
        if not csv_file_path:
            return jsonify({'error': 'CSV file path is required'}), 400
        
        if not os.path.exists(csv_file_path):
            return jsonify({'error': f'CSV file not found: {csv_file_path}'}), 404
        
        # Initialize predictor
        predictor = EnhancedValueTransferPredictor(csv_file_path)
        
        # Get system status
        status = predictor.get_system_status()
        
        return jsonify({
            'success': True,
            'message': 'System initialized successfully',
            'system_status': status
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to initialize system: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """Get current system status"""
    global predictor
    try:
        if predictor is None:
            return jsonify({
                'initialized': False,
                'message': 'System not initialized. Please call /api/initialize first.'
            })
        
        status = predictor.get_system_status()
        status['initialized'] = True
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get system status: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/correlations', methods=['GET'])
def get_correlations():
    """Get correlation analysis"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        correlations = predictor.analyze_correlations()
        
        return jsonify({
            'success': True,
            'correlations': correlations
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to analyze correlations: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/outliers', methods=['GET'])
def get_outlier_analysis():
    """Get outlier analysis"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        outliers = predictor.get_outlier_analysis()
        
        return jsonify({
            'success': True,
            'outliers': outliers
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to analyze outliers: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train prediction models for a specific asset"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        target_asset = data.get('target_asset')
        test_size = data.get('test_size', 0.2)
        feature_selection = data.get('feature_selection_method', 'correlation')
        
        if not target_asset:
            return jsonify({'error': 'target_asset is required'}), 400
        
        # Check if asset exists
        if target_asset not in predictor.numeric_columns:
            return jsonify({
                'error': f'Asset {target_asset} not found',
                'available_assets': predictor.numeric_columns
            }), 400
        
        # Train the model
        result = predictor.train_models(
            target_asset=target_asset,
            test_size=test_size,
            feature_selection_method=feature_selection
        )
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            'training_result': result
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to train model: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_future():
    """Predict future values for an asset"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        target_asset = data.get('target_asset')
        days_ahead = data.get('days_ahead', 30)
        
        if not target_asset:
            return jsonify({'error': 'target_asset is required'}), 400
        
        # Make predictions
        predictions = predictor.predict_future_values(
            target_asset=target_asset,
            days_ahead=days_ahead
        )
        
        if 'error' in predictions:
            return jsonify(predictions), 400
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to make predictions: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare_assets():
    """Compare two assets"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        asset1 = data.get('asset1')
        asset2 = data.get('asset2')
        days = data.get('days', 30)
        
        if not asset1 or not asset2:
            return jsonify({'error': 'Both asset1 and asset2 are required'}), 400
        
        # Compare assets
        comparison = predictor.get_asset_comparison(
            asset1=asset1,
            asset2=asset2,
            days=days
        )
        
        if 'error' in comparison:
            return jsonify(comparison), 400
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to compare assets: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/history', methods=['POST'])
def get_price_history():
    """Get price history for assets"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json() or {}
        assets = data.get('assets')
        days = data.get('days')
        
        # Get price history
        history = predictor.get_price_history(assets=assets, days=days)
        
        return jsonify({
            'success': True,
            'history': history
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get price history: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/assets', methods=['GET'])
def get_available_assets():
    """Get list of available assets"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        return jsonify({
            'success': True,
            'assets': predictor.numeric_columns,
            'count': len(predictor.numeric_columns)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get assets: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/models', methods=['GET'])
def get_trained_models():
    """Get information about trained models"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        models_info = {}
        for asset, model_data in predictor.models.items():
            models_info[asset] = {
                'features_count': len(model_data['features']),
                'top_features': list(model_data['feature_importance'].keys())[:5],
                'feature_importance': dict(list(model_data['feature_importance'].items())[:10])
            }
        
        return jsonify({
            'success': True,
            'trained_models': list(predictor.models.keys()),
            'models_info': models_info
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get model info: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/batch-train', methods=['POST'])
def batch_train_models():
    """Train models for multiple assets"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        data = request.get_json()
        assets = data.get('assets', [])
        test_size = data.get('test_size', 0.2)
        feature_selection = data.get('feature_selection_method', 'correlation')
        
        if not assets:
            # Train for all assets if none specified
            assets = predictor.numeric_columns[:5]  # Limit to first 5 for performance
        
        results = {}
        for asset in assets:
            try:
                result = predictor.train_models(
                    target_asset=asset,
                    test_size=test_size,
                    feature_selection_method=feature_selection
                )
                results[asset] = result
            except Exception as e:
                results[asset] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'batch_results': results,
            'assets_trained': len([r for r in results.values() if 'error' not in r])
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to batch train models: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics_dashboard():
    """Get comprehensive analytics for dashboard"""
    global predictor
    try:
        if predictor is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        # Get basic system info
        status = predictor.get_system_status()
        
        # Get top correlations
        correlations = predictor.analyze_correlations()
        top_correlations = correlations['pairs'][:10]
        
        # Get recent performance for all assets
        recent_history = predictor.get_price_history(days=30)
        
        # Calculate portfolio metrics
        portfolio_metrics = {}
        for asset in predictor.numeric_columns[:10]:  # Limit for performance
            try:
                recent_data = predictor.data_cleaned[asset].tail(30)
                if len(recent_data) > 1:
                    portfolio_metrics[asset] = {
                        'current_value': round(recent_data.iloc[-1], 2),
                        'change_30d': round(((recent_data.iloc[-1] / recent_data.iloc[0]) - 1) * 100, 2),
                        'volatility': round(recent_data.pct_change().std() * 100, 2),
                        'max_value_30d': round(recent_data.max(), 2),
                        'min_value_30d': round(recent_data.min(), 2)
                    }
            except:
                continue
        
        return jsonify({
            'success': True,
            'analytics': {
                'system_status': status,
                'top_correlations': top_correlations,
                'portfolio_metrics': portfolio_metrics,
                'trained_models_count': len(predictor.models),
                'data_quality': {
                    'total_records': len(predictor.data_cleaned),
                    'outliers_handled': True,
                    'missing_values_handled': True
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get analytics: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Enhanced Value Transfer Predictor API is running',
        'initialized': predictor is not None
    })

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    endpoints = {
        'POST /api/initialize': 'Initialize system with CSV file',
        'GET /api/status': 'Get system status',
        'GET /api/health': 'Health check endpoint',
        'GET /api/correlations': 'Get correlation analysis',
        'GET /api/outliers': 'Get outlier analysis',
        'POST /api/train': 'Train model for specific asset',
        'POST /api/batch-train': 'Train models for multiple assets',
        'POST /api/predict': 'Predict future values',
        'POST /api/compare': 'Compare two assets',
        'POST /api/history': 'Get price history',
        'GET /api/assets': 'Get available assets',
        'GET /api/models': 'Get trained models info',
        'GET /api/analytics': 'Get comprehensive analytics'
    }
    
    return jsonify({
        'name': 'Enhanced Value Transfer Predictor API',
        'version': '1.0.0',
        'description': 'Advanced financial asset prediction and analysis system',
        'endpoints': endpoints,
        'usage': {
            'initialize': {
                'method': 'POST',
                'url': '/api/initialize',
                'body': {'csv_file_path': 'path/to/your/data.csv'}
            },
            'train': {
                'method': 'POST',
                'url': '/api/train',
                'body': {
                    'target_asset': 'BTC',
                    'test_size': 0.2,
                    'feature_selection_method': 'correlation'
                }
            },
            'predict': {
                'method': 'POST',
                'url': '/api/predict',
                'body': {
                    'target_asset': 'BTC',
                    'days_ahead': 30
                }
            }
        }
    })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Value Transfer Predictor API...")
    print("📊 Available endpoints:")
    print("   - GET  /                   : API documentation")
    print("   - GET  /api/health         : Health check")
    print("   - POST /api/initialize     : Initialize system")
    print("   - GET  /api/status         : System status")
    print("   - GET  /api/correlations   : Correlation analysis")
    print("   - POST /api/train          : Train model")
    print("   - POST /api/predict        : Make predictions")
    print("   - POST /api/compare        : Compare assets")
    print("   - GET  /api/analytics      : Dashboard analytics")
    print("\n🌐 Server starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)