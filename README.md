# 📈 Stock Price Predictor

A comprehensive machine learning project that predicts stock prices using historical data and technical indicators. This project implements multiple ML algorithms including Linear Regression, Random Forest, and LSTM neural networks to forecast future stock prices.

## 🌟 Features

- **Real-time Data Fetching**: Automatically downloads stock data from Yahoo Finance
- **Technical Analysis**: Calculates 20+ technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- **Multiple ML Models**: Implements and compares different prediction algorithms
- **Interactive Visualizations**: Beautiful charts for technical analysis and model performance
- **Performance Evaluation**: Comprehensive metrics (RMSE, MAE, R²) with visual comparisons
- **Future Price Prediction**: Forecasts stock prices for specified time periods
- **Modular Design**: Easy to customize for different stocks and time periods

## 🎯 Supported Models

1. **Linear Regression**: Simple baseline model for price prediction
2. **Random Forest**: Ensemble method with high accuracy and feature importance analysis
3. **LSTM Neural Network**: Deep learning model for time series prediction (optional)

## 🛠️ Installation

### Basic Installation (Recommended)
```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn
```

### Full Installation (with LSTM support)
```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Optional Enhancements
```bash
pip install plotly  # For interactive plots
pip install xgboost lightgbm  # Alternative advanced models
```

## 🚀 Quick Start

### Basic Usage
```python
from stock_predictor import StockPredictor

# Initialize predictor for Apple stock
predictor = StockPredictor('AAPL', period='2y')

# Fetch and process data
predictor.fetch_data()
predictor.calculate_technical_indicators()
predictor.prepare_features()

# Train models and evaluate
predictor.train_models()
predictor.evaluate_models()

# Generate predictions and visualizations
predictor.predict_future(days=1)
predictor.plot_results()
```

### Command Line Usage
```bash
python stock_predictor.py
```

## 📊 Technical Indicators

The project calculates the following technical indicators for feature engineering:

### Moving Averages
- Simple Moving Averages (5, 10, 20, 50 days)
- Exponential Moving Averages (12, 26 days)

### Momentum Indicators
- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence with signal line

### Volatility Indicators
- **Bollinger Bands**: Price volatility and support/resistance levels
- **Price Volatility**: Rolling standard deviation of returns

### Volume Indicators
- Volume moving average and ratios
- Volume-price relationship analysis

### Price Action Indicators
- High-Low ratios
- Open-Close relationships
- Multi-timeframe price changes

## 📈 Example Output

```
AAPL Stock Price Prediction Summary
==================================================

Linear Regression:
  RMSE: $2.45
  MAE:  $1.87
  R²:   0.9234

Random Forest:
  RMSE: $1.89
  MAE:  $1.43
  R²:   0.9567

Best Model: Random Forest (R² = 0.9567)

AAPL Price Prediction (Random Forest Model):
Current Price: $150.25
Predicted Price (1 days): $151.80
Expected Change: $1.55 (+1.03%)
```

## 🎨 Visualizations

The project generates several types of plots:

1. **Actual vs Predicted Prices**: Scatter plot comparing model predictions with real prices
2. **Time Series Comparison**: Line chart showing prediction accuracy over time
3. **Model Performance**: Bar chart comparing R² scores across different models
4. **Error Distribution**: Histogram of prediction errors
5. **Technical Analysis Charts**: Price action with moving averages, RSI, MACD, and Bollinger Bands

## 🔧 Configuration

### Stock Symbols
You can analyze any stock available on Yahoo Finance:
```python
# Popular stocks
predictor = StockPredictor('TSLA')    # Tesla
predictor = StockPredictor('GOOGL')   # Google
predictor = StockPredictor('MSFT')    # Microsoft
predictor = StockPredictor('AMZN')    # Amazon
predictor = StockPredictor('NVDA')    # NVIDIA

# International stocks
predictor = StockPredictor('RELIANCE.NS')  # Reliance (India)
predictor = StockPredictor('ASML.AS')      # ASML (Netherlands)
```

### Time Periods
```python
predictor = StockPredictor('AAPL', period='1y')   # 1 year
predictor = StockPredictor('AAPL', period='2y')   # 2 years
predictor = StockPredictor('AAPL', period='5y')   # 5 years
predictor = StockPredictor('AAPL', period='max')  # Maximum available
```

### Prediction Horizons
```python
predictor.prepare_features(target_days=1)   # Next day prediction
predictor.prepare_features(target_days=7)   # Next week prediction
predictor.prepare_features(target_days=30)  # Next month prediction
```

## 📁 Project Structure

```
stock-price-predictor/
├── stock_predictor.py          # Main predictor class
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── examples/                   # Usage examples
│   ├── basic_usage.py
│   └── advanced_features.py
└── docs/                       # Additional documentation
    ├── technical_indicators.md
    └── model_explanations.md
```

## 🧪 Model Performance

### Typical Performance Metrics
- **Linear Regression**: R² = 0.85-0.92, good baseline performance
- **Random Forest**: R² = 0.90-0.96, excellent for most stocks
- **LSTM**: R² = 0.88-0.94, best for highly volatile stocks

### Performance Factors
- **Data Quality**: More historical data generally improves predictions
- **Market Conditions**: Models perform better in trending markets
- **Stock Volatility**: Less volatile stocks are easier to predict
- **Feature Engineering**: Technical indicators significantly improve accuracy

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This project is for educational purposes only
2. **Past Performance**: Historical data doesn't guarantee future results
3. **Market Risk**: Stock trading involves significant financial risk
4. **Model Limitations**: No model can predict stock prices with 100% accuracy
5. **Real Trading**: Always do additional research before making investment decisions

## 🐛 Troubleshooting

### Common Issues

**TensorFlow Installation Error**
```bash
# Check Python version (needs 3.9-3.12)
python --version

# Upgrade pip first
python -m pip install --upgrade pip

# Install TensorFlow
pip install tensorflow
```

**Data Fetching Issues**
- Check internet connection
- Verify stock symbol exists on Yahoo Finance
- Try different time period if data is limited

**Memory Issues with Large Datasets**
- Reduce the time period (`period='1y'` instead of `period='max'`)
- Use fewer technical indicators
- Increase system RAM or use cloud computing

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute:

1. **Bug Reports**: Open an issue with details about the problem
2. **Feature Requests**: Suggest new technical indicators or ML models
3. **Code Improvements**: Submit pull requests with enhancements
4. **Documentation**: Help improve documentation and examples

### Development Setup
```bash
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor
pip install -r requirements.txt
python stock_predictor.py
```

## 📚 Educational Value

This project demonstrates several important concepts:

### Machine Learning
- Feature engineering with financial data
- Time series prediction techniques
- Model comparison and evaluation
- Cross-validation for time series data

### Finance
- Technical analysis indicators
- Stock market data interpretation
- Risk assessment and prediction accuracy

### Programming
- Object-oriented design patterns
- Data visualization with matplotlib/seaborn
- API integration (Yahoo Finance)
- Error handling and data validation

## 🎓 Learning Resources

- **Technical Analysis**: [Investopedia Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
- **Machine Learning**: [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- **Time Series**: [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- **Financial Data**: [Yahoo Finance API](https://pypi.org/project/yfinance/)

## 📊 Data Source

**Primary**: Yahoo Finance via yfinance library
- **Coverage**: Global stocks, ETFs, indices, cryptocurrencies
- **Data Quality**: Professional-grade financial data
- **Update Frequency**: Real-time during market hours
- **Historical Data**: Up to several decades for major stocks
- **Cost**: Free (no API key required)

## 🔮 Future Enhancements

Potential improvements and features:

- [ ] **Sentiment Analysis**: Incorporate news sentiment data
- [ ] **Multi-stock Portfolio**: Predict portfolio performance
- [ ] **Real-time Alerts**: Email/SMS notifications for price targets
- [ ] **Web Interface**: Flask/Django web application
- [ ] **Backtesting**: Historical strategy performance testing
- [ ] **Risk Management**: Position sizing and stop-loss recommendations
- [ ] **Additional Models**: XGBoost, LightGBM, Prophet
- [ ] **Feature Selection**: Automated feature importance analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free financial data
- **Scikit-learn**: For machine learning algorithms
- **Pandas**: For data manipulation and analysis
- **Matplotlib/Seaborn**: For data visualization
- **TensorFlow**: For deep learning capabilities

## 📞 Contact

For questions, suggestions, or collaboration:

- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

**⭐ If you find this project helpful, please give it a star on GitHub!**