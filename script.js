class StockAnalysisApp {
    constructor() {
        this.stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ITC.NS'];
        this.initializeEventListeners();
        this.setDefaultDates();
    }

    setDefaultDates() {
        const today = new Date();
        const oneMonthAgo = new Date(today);
        oneMonthAgo.setMonth(today.getMonth() - 1);
        
        document.getElementById('startDate').value = this.formatDate(oneMonthAgo);
        document.getElementById('endDate').value = this.formatDate(today);
    }

    formatDate(date) {
        return date.toISOString().split('T')[0];
    }

    initializeEventListeners() {
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeStocks();
        });
    }

    async analyzeStocks() {
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        if (!startDate || !endDate) {
            this.showError('Please select both start and end dates');
            return;
        }

        if (new Date(startDate) >= new Date(endDate)) {
            this.showError('Start date must be before end date');
            return;
        }

        this.showLoading(true);

        try {
            // Use Yahoo Finance API directly
            const stockData = await this.fetchStockData(startDate, endDate);
            const analysis = this.performAnalysis(stockData);
            this.displayResults(analysis);
        } catch (error) {
            this.showError('Analysis failed: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async fetchStockData(startDate, endDate) {
        const symbols = this.stocks.join(',');
        const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbols}?period1=${Math.floor(new Date(startDate)/1000)}&period2=${Math.floor(new Date(endDate)/1000)}&interval=1d`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (!data.chart || !data.chart.result) {
            throw new Error('Failed to fetch stock data');
        }

        return this.processChartData(data);
    }

    processChartData(data) {
        const result = {};
        
        data.chart.result.forEach((stockData, index) => {
            const symbol = stockData.meta.symbol;
            const timestamps = stockData.timestamp;
            const closes = stockData.indicators.quote[0].close;
            
            result[symbol] = {};
            timestamps.forEach((timestamp, i) => {
                const date = new Date(timestamp * 1000).toISOString().split('T')[0];
                result[symbol][date] = closes[i];
            });
        });
        
        return result;
    }

    performAnalysis(stockData) {
        // Convert to DataFrame-like structure
        const dates = Object.keys(stockData[this.stocks[0]]);
        const prices = {};
        
        this.stocks.forEach(stock => {
            prices[stock] = dates.map(date => stockData[stock][date]);
        });

        // Calculate returns
        const returns = {};
        this.stocks.forEach(stock => {
            returns[stock] = [];
            for (let i = 1; i < prices[stock].length; i++) {
                const ret = (prices[stock][i] - prices[stock][i-1]) / prices[stock][i-1];
                returns[stock].push(ret);
            }
        });

        // Calculate covariance matrix
        const covMatrix = this.calculateCovarianceMatrix(returns);
        
        // Perform PCA
        const { eigenvalues, eigenvectors } = this.performPCA(covMatrix);
        
        // Create charts
        const trendChart = this.createTrendChart(eigenvectors[0]);
        const returnsChart = this.createReturnsChart(prices, dates);

        return {
            stockData,
            returns,
            eigenvalues,
            eigenvectors,
            trendChart,
            returnsChart,
            analysis: {
                mainTrendStock: this.stocks[this.findMaxIndex(eigenvectors[0])],
                varianceExplained: (eigenvalues[0] / eigenvalues.reduce((a, b) => a + b, 0)) * 100,
                totalVariance: eigenvalues.reduce((a, b) => a + b, 0)
            }
        };
    }

    calculateCovarianceMatrix(returns) {
        const n = this.stocks.length;
        const matrix = Array(n).fill().map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const retI = returns[this.stocks[i]];
                const retJ = returns[this.stocks[j]];
                const meanI = retI.reduce((a, b) => a + b, 0) / retI.length;
                const meanJ = retJ.reduce((a, b) => a + b, 0) / retJ.length;
                
                let covariance = 0;
                for (let k = 0; k < retI.length; k++) {
                    covariance += (retI[k] - meanI) * (retJ[k] - meanJ);
                }
                matrix[i][j] = covariance / (retI.length - 1);
            }
        }
        
        return matrix;
    }

    performPCA(covMatrix) {
        // Simple PCA implementation using Jacobi method
        const { values, vectors } = this.jacobiMethod(covMatrix);
        return { eigenvalues: values, eigenvectors: vectors };
    }

    jacobiMethod(matrix) {
        const n = matrix.length;
        let vectors = Array(n).fill().map((_, i) => 
            Array(n).fill(0).map((_, j) => i === j ? 1 : 0)
        );
        let values = matrix.map(row => [...row]);
        
        for (let iter = 0; iter < 50; iter++) {
            let max = 0;
            let p = 0, q = 0;
            
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    if (Math.abs(values[i][j]) > max) {
                        max = Math.abs(values[i][j]);
                        p = i;
                        q = j;
                    }
                }
            }
            
            if (max < 1e-10) break;
            
            const theta = 0.5 * Math.atan2(2 * values[p][q], values[q][q] - values[p][p]);
            const c = Math.cos(theta);
            const s = Math.sin(theta);
            
            // Update matrix
            for (let r = 0; r < n; r++) {
                if (r !== p && r !== q) {
                    const temp1 = values[p][r];
                    const temp2 = values[q][r];
                    values[p][r] = c * temp1 - s * temp2;
                    values[r][p] = values[p][r];
                    values[q][r] = s * temp1 + c * temp2;
                    values[r][q] = values[q][r];
                }
            }
            
            const temp1 = values[p][p];
            const temp2 = values[q][q];
            const temp3 = values[p][q];
            
            values[p][p] = c * c * temp1 + s * s * temp2 - 2 * c * s * temp3;
            values[q][q] = s * s * temp1 + c * c * temp2 + 2 * c * s * temp3;
            values[p][q] = values[q][p] = 0;
            
            // Update eigenvectors
            for (let r = 0; r < n; r++) {
                const temp1 = vectors[r][p];
                const temp2 = vectors[r][q];
                vectors[r][p] = c * temp1 - s * temp2;
                vectors[r][q] = s * temp1 + c * temp2;
            }
        }
        
        const eigenvalues = values.map((row, i) => row[i]);
        
        // Sort by eigenvalues
        const indices = eigenvalues.map((_, i) => i)
            .sort((a, b) => eigenvalues[b] - eigenvalues[a]);
        
        const sortedEigenvalues = indices.map(i => eigenvalues[i]);
        const sortedEigenvectors = indices.map(i => vectors.map(row => row[i]));
        
        return { values: sortedEigenvalues, vectors: sortedEigenvectors };
    }

    findMaxIndex(arr) {
        return arr.reduce((maxIndex, item, index) => 
            item > arr[maxIndex] ? index : maxIndex, 0
        );
    }

    createTrendChart(mainVector) {
        const canvas = document.createElement('canvas');
        canvas.width = 600;
        canvas.height = 400;
        const ctx = canvas.getContext('2d');
        
        // Draw chart
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const maxVal = Math.max(...mainVector.map(Math.abs));
        const barWidth = 60;
        const spacing = 20;
        const startX = 50;
        const baseY = 300;
        
        // Draw bars
        mainVector.forEach((value, index) => {
            const x = startX + index * (barWidth + spacing);
            const height = (value / maxVal) * 200;
            const y = baseY - height;
            
            ctx.fillStyle = value >= 0 ? '#3498db' : '#e74c3c';
            ctx.fillRect(x, y, barWidth, height);
            
            // Stock label
            ctx.fillStyle = '#2c3e50';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(this.stocks[index].split('.')[0], x + barWidth/2, baseY + 20);
            
            // Value label
            ctx.fillText(value.toFixed(3), x + barWidth/2, y - 10);
        });
        
        // Title
        ctx.font = 'bold 16px Arial';
        ctx.fillStyle = '#2c3e50';
        ctx.textAlign = 'center';
        ctx.fillText('Stock Influence on Main Market Trend', canvas.width/2, 30);
        
        return canvas.toDataURL();
    }

    createReturnsChart(prices, dates) {
        const canvas = document.createElement('canvas');
        canvas.width = 600;
        canvas.height = 400;
        const ctx = canvas.getContext('2d');
        
        // Draw chart
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
        const startX = 50;
        const endX = canvas.width - 50;
        const startY = 50;
        const endY = canvas.height - 50;
        
        // Calculate cumulative returns
        const cumulativeReturns = {};
        this.stocks.forEach((stock, stockIndex) => {
            cumulativeReturns[stock] = [1];
            for (let i = 1; i < prices[stock].length; i++) {
                const ret = (prices[stock][i] - prices[stock][0]) / prices[stock][0];
                cumulativeReturns[stock].push(1 + ret);
            }
        });
        
        // Find min and max values
        let minVal = Infinity, maxVal = -Infinity;
        this.stocks.forEach(stock => {
            cumulativeReturns[stock].forEach(val => {
                minVal = Math.min(minVal, val);
                maxVal = Math.max(maxVal, val);
            });
        });
        
        // Draw lines
        this.stocks.forEach((stock, stockIndex) => {
            ctx.strokeStyle = colors[stockIndex];
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            cumulativeReturns[stock].forEach((val, i) => {
                const x = startX + (i / (cumulativeReturns[stock].length - 1)) * (endX - startX);
                const y = endY - ((val - minVal) / (maxVal - minVal)) * (endY - startY);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
        });
        
        // Title
        ctx.font = 'bold 16px Arial';
        ctx.fillStyle = '#2c3e50';
        ctx.textAlign = 'center';
        ctx.fillText('Cumulative Returns Over Time', canvas.width/2, 30);
        
        return canvas.toDataURL();
    }

    displayResults(analysis) {
        // Update metrics
        document.getElementById('mainTrendStock').textContent = analysis.analysis.mainTrendStock;
        document.getElementById('varianceExplained').textContent = 
            analysis.analysis.varianceExplained.toFixed(2) + '%';
        document.getElementById('totalVariance').textContent = 
            analysis.analysis.totalVariance.toFixed(6);

        // Display charts
        document.getElementById('trendChart').src = analysis.trendChart;
        document.getElementById('returnsChart').src = analysis.returnsChart;

        // Display stock prices table
        this.displayStockPrices(analysis.stockData);

        // Display eigenvalues and eigenvectors table
        this.displayEigenData(analysis.eigenvalues, analysis.eigenvectors);

        // Show results section
        document.getElementById('resultsSection').classList.remove('hidden');
    }

    displayStockPrices(pricesData) {
        const tableBody = document.querySelector('#pricesTable tbody');
        tableBody.innerHTML = '';

        const dates = Object.keys(pricesData[this.stocks[0]]).slice(-5);

        dates.forEach(date => {
            const row = document.createElement('tr');
            
            // Date cell
            const dateCell = document.createElement('td');
            dateCell.textContent = new Date(date).toLocaleDateString();
            row.appendChild(dateCell);

            // Stock price cells
            this.stocks.forEach(stock => {
                const priceCell = document.createElement('td');
                priceCell.textContent = pricesData[stock][date]?.toFixed(2) || 'N/A';
                row.appendChild(priceCell);
            });

            tableBody.appendChild(row);
        });
    }

    displayEigenData(eigenvalues, eigenvectors) {
        const tableBody = document.querySelector('#eigenTable tbody');
        tableBody.innerHTML = '';

        eigenvalues.forEach((eigenvalue, index) => {
            const row = document.createElement('tr');
            
            // Component cell
            const compCell = document.createElement('td');
            compCell.textContent = `PC${index + 1}`;
            compCell.style.fontWeight = 'bold';
            row.appendChild(compCell);

            // Eigenvalue cell
            const evalCell = document.createElement('td');
            evalCell.textContent = eigenvalue.toFixed(6);
            row.appendChild(evalCell);

            // Eigenvector cells
            eigenvectors[index].forEach((value, stockIndex) => {
                const vecCell = document.createElement('td');
                vecCell.textContent = value.toFixed(4);
                
                // Highlight the main component
                if (index === 0) {
                    vecCell.style.background = 'linear-gradient(135deg, #3498db, #2980b9)';
                    vecCell.style.color = 'white';
                    vecCell.style.fontWeight = 'bold';
                }
                
                row.appendChild(vecCell);
            });

            tableBody.appendChild(row);
        });
    }

    showLoading(show) {
        const spinner = document.getElementById('loadingSpinner');
        if (show) {
            spinner.classList.remove('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
        } else {
            spinner.classList.add('hidden');
        }
    }

    showError(message) {
        alert('Error: ' + message);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new StockAnalysisApp();
});
