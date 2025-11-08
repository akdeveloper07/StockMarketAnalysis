from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)

class StockAnalyzer:
    def __init__(self):
        self.stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ITC.NS']
    
    def fetch_stock_data(self, start_date, end_date):
        """Fetch stock data from Yahoo Finance"""
        try:
            data = yf.download(self.stocks, start=start_date, end=end_date)
            return data['Close']
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    def calculate_returns(self, data):
        """Calculate daily returns"""
        return data.pct_change().dropna()
    
    def perform_pca_analysis(self, returns):
        """Perform PCA analysis on returns"""
        cov_matrix = returns.cov()
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        return cov_matrix, eigenvalues, eigenvectors
    
    def create_trend_chart(self, eigenvectors, eigenvalues):
        """Create market trend visualization"""
        plt.figure(figsize=(10, 6))
        main_vector = eigenvectors[:, 0]  # First principal component
        
        bars = plt.bar(self.stocks, main_vector, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.title('Stock Influence on Main Market Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Stocks', fontweight='bold')
        plt.ylabel('Eigenvector Value', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def create_returns_chart(self, returns):
        """Create cumulative returns chart"""
        plt.figure(figsize=(12, 6))
        cumulative_returns = (1 + returns).cumprod()
        
        for i, stock in enumerate(self.stocks):
            plt.plot(cumulative_returns.index, cumulative_returns[stock], 
                    label=stock, linewidth=2, color=self.colors[i])
        
        plt.title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Cumulative Returns', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    @property
    def colors(self):
        return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

analyzer = StockAnalyzer()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Stock Analysis API is running'})

@app.route('/api/analyze', methods=['POST'])
def analyze_stocks():
    try:
        data = request.get_json()
        start_date = data.get('start_date', '2024-09-01')
        end_date = data.get('end_date', '2024-10-01')
        
        # Fetch and process data
        stock_data = analyzer.fetch_stock_data(start_date, end_date)
        returns = analyzer.calculate_returns(stock_data)
        cov_matrix, eigenvalues, eigenvectors = analyzer.perform_pca_analysis(returns)
        
        # Prepare response data
        response_data = {
            'stock_prices': stock_data.tail().to_dict(),
            'daily_returns': returns.tail().to_dict(),
            'covariance_matrix': cov_matrix.to_dict(),
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors.tolist(),
            'analysis': {
                'main_trend_stock': analyzer.stocks[np.argmax(eigenvectors[:, 0])],
                'variance_explained': (eigenvalues[0] / np.sum(eigenvalues)) * 100,
                'total_variance': np.sum(eigenvalues)
            }
        }
        
        # Generate charts
        response_data['trend_chart'] = analyzer.create_trend_chart(eigenvectors, eigenvalues)
        response_data['returns_chart'] = analyzer.create_returns_chart(returns)
        
        return jsonify({
            'success': True,
            'data': response_data,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error performing stock analysis'
        }), 500

@app.route('/api/stocks', methods=['GET'])
def get_available_stocks():
    return jsonify({
        'stocks': analyzer.stocks,
        'count': len(analyzer.stocks)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
