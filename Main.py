#!/usr/bin/env python3
"""
Stock Price Predictor
A comprehensive machine learning project for predicting stock prices using historical data.

Features:
- Data collection from Yahoo Finance
- Technical indicators calculation
- Multiple ML models (Linear Regression, Random Forest, LSTM)
- Interactive visualizations
- Performance evaluation
- Future price predictions

Dependencies:
pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow plotly
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# For LSTM (optional - requires tensorflow)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("TensorFlow not available. LSTM model will be skipped.")

# For interactive plots (optional)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Interactive plots will be skipped.")

class StockPredictor:
    def __init__(self, symbol, period="2y"):
        """
        Initialize the Stock Predictor
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Data period ('1y', '2y', '5y', 'max')
        """
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            print(f"Successfully fetched {len(self.data)} days of data for {self.symbol}")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators for feature engineering"""
        df = self.data.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / df['BB_width']
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price change indicators
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_5d'] = df['Close'].pct_change(periods=5)
        df['Volatility'] = df['Price_change'].rolling(window=20).std()
        
        # High-Low indicators
        df['HL_ratio'] = df['High'] / df['Low']
        df['OC_ratio'] = df['Open'] / df['Close']
        
        self.data = df
        print("Technical indicators calculated successfully")
    
    def prepare_features(self, target_days=1):
        """
        Prepare features and target for machine learning
        
        Args:
            target_days (int): Number of days ahead to predict
        """
        df = self.data.copy()
        
        # Define feature columns
        feature_cols = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_signal',
            'RSI', 'BB_width', 'BB_position',
            'Volume_ratio', 'Price_change', 'Price_change_5d',
            'Volatility', 'HL_ratio', 'OC_ratio'
        ]
        
        # Create target (future close price)
        df['Target'] = df['Close'].shift(-target_days)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("No valid data after removing NaN values")
        
        self.features = df[feature_cols]
        self.target = df['Target']
        
        print(f"Prepared {len(self.features)} samples with {len(feature_cols)} features")
        
    def train_models(self, test_size=0.2, random_state=42):
        """Train multiple machine learning models"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size, 
            random_state=random_state, shuffle=False
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # 1. Linear Regression
        print("Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        self.models['Linear Regression'] = lr_model
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf_model.fit(X_train_scaled, y_train)
        self.models['Random Forest'] = rf_model
        
        # 3. LSTM (if available)
        if LSTM_AVAILABLE:
            print("Training LSTM...")
            lstm_model = self.create_lstm_model(X_train_scaled, y_train)
            if lstm_model:
                self.models['LSTM'] = lstm_model
        
        print(f"Trained {len(self.models)} models successfully")
    
    def create_lstm_model(self, X_train, y_train, sequence_length=60):
        """Create and train LSTM model"""
        try:
            # Prepare data for LSTM
            X_lstm = []
            y_lstm = []
            
            for i in range(sequence_length, len(X_train)):
                X_lstm.append(X_train[i-sequence_length:i])
                y_lstm.append(y_train.iloc[i])
            
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train the model
            model.fit(X_lstm, y_lstm, batch_size=32, epochs=50, verbose=0)
            
            return model
            
        except Exception as e:
            print(f"Error training LSTM: {e}")
            return None
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            try:
                if name == 'LSTM':
                    # Special handling for LSTM
                    predictions = self.predict_lstm(model, self.X_test)
                else:
                    predictions = model.predict(self.X_test)
                
                # Calculate metrics
                mse = mean_squared_error(self.y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, predictions)
                r2 = r2_score(self.y_test, predictions)
                
                results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'predictions': predictions
                }
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        self.results = results
        return results
    
    def predict_lstm(self, model, X_test, sequence_length=60):
        """Make predictions with LSTM model"""
        predictions = []
        
        for i in range(sequence_length, len(X_test)):
            sequence = X_test[i-sequence_length:i].reshape(1, sequence_length, -1)
            pred = model.predict(sequence, verbose=0)[0][0]
            predictions.append(pred)
        
        # Pad with NaN for alignment
        predictions = [np.nan] * sequence_length + predictions
        return np.array(predictions)
    
    def plot_results(self):
        """Plot model performance and predictions"""
        if not hasattr(self, 'results'):
            print("No results to plot. Run evaluate_models() first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.symbol} Stock Price Prediction Results', fontsize=16)
        
        # Plot 1: Actual vs Predicted for best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['R²'])
        ax1 = axes[0, 0]
        ax1.scatter(self.y_test, self.results[best_model]['predictions'], alpha=0.6)
        ax1.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Price')
        ax1.set_ylabel('Predicted Price')
        ax1.set_title(f'Actual vs Predicted ({best_model})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time series comparison
        ax2 = axes[0, 1]
        test_dates = self.data.index[-len(self.y_test):]
        ax2.plot(test_dates, self.y_test.values, label='Actual', linewidth=2)
        ax2.plot(test_dates, self.results[best_model]['predictions'], 
                label=f'Predicted ({best_model})', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.set_title('Price Prediction Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: Model comparison
        ax3 = axes[1, 0]
        models = list(self.results.keys())
        r2_scores = [self.results[model]['R²'] for model in models]
        bars = ax3.bar(models, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
        ax3.set_ylabel('R² Score')
        ax3.set_title('Model Comparison (R² Score)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 4: Error distribution
        ax4 = axes[1, 1]
        errors = self.y_test.values - self.results[best_model]['predictions']
        ax4.hist(errors, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Error Distribution ({best_model})')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_technical_analysis(self):
        """Plot technical analysis indicators"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{self.symbol} Technical Analysis', fontsize=16)
        
        # Plot 1: Price and Moving Averages
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        ax1.plot(self.data.index, self.data['MA_20'], label='MA 20', alpha=0.8)
        ax1.plot(self.data.index, self.data['MA_50'], label='MA 50', alpha=0.8)
        ax1.fill_between(self.data.index, self.data['BB_upper'], self.data['BB_lower'], 
                        alpha=0.2, label='Bollinger Bands')
        ax1.set_ylabel('Price')
        ax1.set_title('Price and Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='orange')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.set_ylabel('RSI')
        ax2.set_title('Relative Strength Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: MACD
        ax3 = axes[2]
        ax3.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        ax3.plot(self.data.index, self.data['MACD_signal'], label='Signal', color='red')
        ax3.bar(self.data.index, self.data['MACD'] - self.data['MACD_signal'], 
               label='Histogram', alpha=0.3)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('MACD')
        ax3.set_title('MACD Indicator')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=30):
        """Predict future stock prices"""
        if not self.models:
            print("No trained models available. Train models first.")
            return
        
        # Use the best performing model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['R²'])
        best_model = self.models[best_model_name]
        
        # Get the latest features
        latest_features = self.features.iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Make prediction
        if best_model_name == 'LSTM':
            # For LSTM, we need a sequence
            sequence_length = 60
            if len(self.features) >= sequence_length:
                sequence = self.scaler.transform(self.features.iloc[-sequence_length:].values)
                sequence = sequence.reshape(1, sequence_length, -1)
                future_price = best_model.predict(sequence, verbose=0)[0][0]
            else:
                print("Not enough data for LSTM prediction")
                return
        else:
            future_price = best_model.predict(latest_features_scaled)[0]
        
        current_price = self.data['Close'].iloc[-1]
        price_change = future_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        print(f"\n{self.symbol} Price Prediction ({best_model_name} Model):")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price ({days} days): ${future_price:.2f}")
        print(f"Expected Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
        
        return {
            'current_price': current_price,
            'predicted_price': future_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'model_used': best_model_name
        }
    
    def print_summary(self):
        """Print a summary of model performance"""
        if not hasattr(self, 'results'):
            print("No results available. Run evaluate_models() first.")
            return
        
        print(f"\n{self.symbol} Stock Price Prediction Summary")
        print("=" * 50)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  RMSE: ${metrics['RMSE']:.2f}")
            print(f"  MAE:  ${metrics['MAE']:.2f}")
            print(f"  R²:   {metrics['R²']:.4f}")
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['R²'])
        print(f"\nBest Model: {best_model} (R² = {self.results[best_model]['R²']:.4f})")

def main():
    """Main function to run the stock predictor"""
    # Configuration
    STOCK_SYMBOL = "AAPL"  # Change this to any stock symbol
    DATA_PERIOD = "2y"     # Data period: 1y, 2y, 5y, max
    
    print(f"Stock Price Predictor for {STOCK_SYMBOL}")
    print("=" * 40)
    
    # Initialize predictor
    predictor = StockPredictor(STOCK_SYMBOL, DATA_PERIOD)
    
    # Fetch data
    if not predictor.fetch_data():
        return
    
    # Calculate technical indicators
    predictor.calculate_technical_indicators()
    
    # Prepare features
    predictor.prepare_features(target_days=1)
    
    # Train models
    predictor.train_models()
    
    # Evaluate models
    predictor.evaluate_models()
    
    # Print summary
    predictor.print_summary()
    
    # Make future prediction
    predictor.predict_future(days=1)
    
    # Plot results
    predictor.plot_results()
    predictor.plot_technical_analysis()
    
    return predictor

if __name__ == "__main__":
    # Run the main function
    predictor = main()
    
    # Example of using the predictor interactively
    print("\nTo use this predictor with different stocks:")
    print("predictor = StockPredictor('GOOGL', '1y')")
    print("predictor.fetch_data()")
    print("predictor.calculate_technical_indicators()")
    print("predictor.prepare_features()")
    print("predictor.train_models()")
    print("predictor.evaluate_models()")
    print("predictor.plot_results()")