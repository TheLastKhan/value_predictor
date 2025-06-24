// Global variables
        let assetsData = [];
        const API_BASE = 'http://localhost:5000/api';

        // Runs when page loads
        document.addEventListener('DOMContentLoaded', function() {
            loadAssets();
        });

        // Tab switching function
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tab buttons
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        // Load assets
        async function loadAssets() {
            try {
                const response = await fetch(`${API_BASE}/assets`);
                const data = await response.json();
                assetsData = data.assets;
                
                // Populate dropdowns
                populateSelects();
                
                console.log('Assets loaded:', data);
            } catch (error) {
                console.error('Error loading assets:', error);
                showError('Error loading asset data. Please ensure the backend server is running.');
            }
        }

        // Populate select elements
        function populateSelects() {
            const selects = [
                document.getElementById('assetSelect'),
                document.getElementById('asset1Select'),
                document.getElementById('asset2Select')
            ];
            
            selects.forEach(select => {
                // Clear existing options (except first one)
                while (select.children.length > 1) {
                    select.removeChild(select.lastChild);
                }
                
                // Add new options
                assetsData.forEach(asset => {
                    const option = document.createElement('option');
                    option.value = asset;
                    option.textContent = asset;
                    select.appendChild(option);
                });
            });
        }

        // Make asset prediction
        async function predictAsset() {
            const assetSelect = document.getElementById('assetSelect');
            const selectedAsset = assetSelect.value;
            
            if (!selectedAsset) {
                alert('Please select an asset!');
                return;
            }
            
            const resultDiv = document.getElementById('predictionResult');
            resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Calculating prediction...</div>';
            
            try {
                const response = await fetch(`${API_BASE}/predict/${selectedAsset}`);
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayPredictionResult(data, selectedAsset);
                createPredictionChart(data, selectedAsset);
                
            } catch (error) {
                console.error('Prediction error:', error);
                resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        // Display prediction result
        function displayPredictionResult(data, asset) {
            const resultDiv = document.getElementById('predictionResult');
            
            const html = `
                <div class="prediction-result">
                    <h4>${asset} - Value Prediction Analysis</h4>
                    <div class="prediction-grid">
                        <div class="metric">
                            <div class="metric-value">${data.current_value}</div>
                            <div class="metric-label">Current Value</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value ${data.linear_regression.change_percent >= 0 ? 'positive' : 'negative'}">
                                ${data.linear_regression.prediction}
                            </div>
                            <div class="metric-label">Linear Regression Prediction</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value ${data.random_forest.change_percent >= 0 ? 'positive' : 'negative'}">
                                ${data.random_forest.prediction}
                            </div>
                            <div class="metric-label">Random Forest Prediction</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value ${data.linear_regression.change_percent >= 0 ? 'positive' : 'negative'}">
                                ${data.linear_regression.change_percent >= 0 ? '+' : ''}${data.linear_regression.change_percent}%
                            </div>
                            <div class="metric-label">Expected Change</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <strong>Model Performance:</strong><br>
                        Linear Regression R²: ${data.linear_regression.r2_score}<br>
                        Random Forest R²: ${data.random_forest.r2_score}
                    </div>
                    <div style="margin-top: 10px;">
                        <strong>Important Factors:</strong> ${data.important_features.join(', ')}
                    </div>
                </div>
            `;
            
            resultDiv.innerHTML = html;
        }

        // Create prediction chart
        function createPredictionChart(data, asset) {
            const trace1 = {
                x: ['Current', 'Linear Reg.', 'Random Forest'],
                y: [data.current_value, data.linear_regression.prediction, data.random_forest.prediction],
                type: 'bar',
                marker: {
                    color: ['#667eea', '#38a169', '#e53e3e']
                },
                text: [`${data.current_value}`, `${data.linear_regression.prediction}`, `${data.random_forest.prediction}`],
                textposition: 'auto'
            };
            
            const layout = {
                title: `${asset} - Value Prediction Comparison`,
                xaxis: { title: 'Prediction Model' },
                yaxis: { title: 'Value ($)' },
                margin: { t: 50, l: 50, r: 50, b: 50 },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot('predictionChart', [trace1], layout, {responsive: true});
        }

        // Load correlation analysis
        async function loadCorrelations() {
            const resultDiv = document.getElementById('correlationResults');
            resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Performing correlation analysis...</div>';
            
            try {
                const response = await fetch(`${API_BASE}/correlations`);
                const data = await response.json();
                
                displayCorrelationResults(data);
                createCorrelationHeatmap(data);
                
            } catch (error) {
                console.error('Correlation error:', error);
                resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        // Display correlation results
        function displayCorrelationResults(data) {
            const resultDiv = document.getElementById('correlationResults');
            
            let html = '<div style="margin-top: 20px;"><h4>🔗 Strong Value Transfer Relationships</h4>';
            
            data.pairs.forEach(pair => {
                const isPositive = pair.correlation > 0;
                html += `
                    <div class="correlation-item">
                        <div>
                            <strong>${pair.asset1}</strong> ↔ <strong>${pair.asset2}</strong><br>
                            <small>${pair.strength} ${pair.relationship} Relationship</small>
                        </div>
                        <div class="correlation-value ${isPositive ? 'correlation-positive' : 'correlation-negative'}">
                            ${pair.correlation}
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            resultDiv.innerHTML = html;
        }

function createCorrelationHeatmap(data) {
    const assets = data.assets;
    const matrix = data.matrix;
    
    const z = [];
    const x = assets;
    const y = assets;
    
    for (let i = 0; i < assets.length; i++) {
        const row = [];
        for (let j = 0; j < assets.length; j++) {
            row.push(matrix[assets[i]][assets[j]]);
        }
        z.push(row);
    }
    
    const trace = {
        x: x,
        y: y,
        z: z,
        type: 'heatmap',
        colorscale: [
            [0, '#e53e3e'],  // Red for negative
            [0.5, '#ffffff'], // White for zero
            [1, '#38a169']   // Green for positive
        ],
        zmid: 0
    };
    
    const layout = {
        title: {
        text: '<b>Asset Correlation Heatmap</b>', // HTML <b> tag for bold
        font: {
            size: 18, // Optional: set font size
            family: 'Arial', // Optional: set font family
            color: '#333' // Optional: set text color
        }
        },
        margin: { t: 50, l: 50, r: 50, b: 50 },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        yaxis: {
            tickmode: 'linear',
            automargin: true,
            tickfont: { size: 10 }
        },
        xaxis: {
            automargin: true,
            tickangle: -45
        }
    };
    
    Plotly.newPlot('correlationHeatmap', [trace], layout, {responsive: true});
}

        // Asset comparison
        async function compareAssets() {
            const asset1 = document.getElementById('asset1Select').value;
            const asset2 = document.getElementById('asset2Select').value;
            
            if (!asset1 || !asset2) {
                alert('Please select two assets!');
                return;
            }
            
            const resultDiv = document.getElementById('comparisonResult');
            resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Performing comparison...</div>';
            
            try {
                const response = await fetch(`${API_BASE}/compare?asset1=${asset1}&asset2=${asset2}`);
                const data = await response.json();
                
                displayComparisonResult(data);
                
            } catch (error) {
                console.error('Comparison error:', error);
                resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        // Display comparison result
        function displayComparisonResult(data) {
            const resultDiv = document.getElementById('comparisonResult');
            
            const html = `
                <div class="comparison-result">
                    <h4>⚖️ ${data.asset1} vs ${data.asset2} Comparison</h4>
                    <div style="margin: 15px 0;">
                        <strong>Correlation:</strong> ${data.correlation} (${data.strength} ${data.relationship} Relationship)
                    </div>
                    <div class="prediction-grid">
                        <div class="metric">
                            <div class="metric-value ${data.recent_performance[data.asset1] >= 0 ? 'positive' : 'negative'}">
                                ${data.recent_performance[data.asset1] >= 0 ? '+' : ''}${data.recent_performance[data.asset1]}%
                            </div>
                            <div class="metric-label">${data.asset1} (30 Days)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value ${data.recent_performance[data.asset2] >= 0 ? 'positive' : 'negative'}">
                                ${data.recent_performance[data.asset2] >= 0 ? '+' : ''}${data.recent_performance[data.asset2]}%
                            </div>
                            <div class="metric-label">${data.asset2} (30 Days)</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <strong>Value Transfer Analysis:</strong><br>
                        ${data.value_transfer_analysis.interpretation}
                    </div>
                </div>
            `;
            
            resultDiv.innerHTML = html;
        }

        // Load price history
        async function loadHistory() {
            const selectElement = document.getElementById('historyAssets');
            const selectedAssets = Array.from(selectElement.selectedOptions).map(option => option.value);
            
            if (selectedAssets.length === 0) {
                alert('Please select at least one asset!');
                return;
            }
            
            try {
                const params = selectedAssets.map(asset => `assets=${asset}`).join('&');
                const response = await fetch(`${API_BASE}/history?${params}`);
                const data = await response.json();
                
                createHistoryChart(data);
                
            } catch (error) {
                console.error('Historical data error:', error);
                showError('Error loading historical data.');
            }
        }

        // Create history price chart
        function createHistoryChart(data) {
            const traces = [];
            const colors = ['#667eea', '#38a169', '#e53e3e', '#dd6b20', '#9f7aea', '#0bc5ea'];
            
            let colorIndex = 0;
            for (const [asset, assetData] of Object.entries(data)) {
                traces.push({
                    x: assetData.dates,
                    y: assetData.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: asset,
                    line: {
                        color: colors[colorIndex % colors.length],
                        width: 2
                    }
                });
                colorIndex++;
            }
            
            const layout = {
                title: 'Financial Assets - Price History',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Value ($)' },
                margin: { t: 50, l: 50, r: 50, b: 50 },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                legend: {
                    x: 0.02,
                    y: 0.98
                }
            };
            
            Plotly.newPlot('historyChart', traces, layout, {responsive: true});
        }

        // Error display function
        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: #fed7d7;
                color: #c53030;
                padding: 15px 20px;
                border-radius: 8px;
                border-left: 4px solid #e53e3e;
                z-index: 1000;
                max-width: 400px;
            `;
            errorDiv.textContent = message;
            
            document.body.appendChild(errorDiv);
            
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        }


