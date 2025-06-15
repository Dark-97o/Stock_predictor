import yfinance as yf
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

class SimpleStockPredictor:
    def __init__(self, symbol, period="1y"):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.scaler = StandardScaler()
        self.X_test = None
        self.y_test = None
        
    def fetch_data(self):
        try:
            print(f"Fetching data for {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            print(f"✓ Successfully fetched {len(self.data)} days of data")
            return True
            
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return False
    
    def add_technical_indicators(self):
        print("Calculating technical indicators...")
        
        self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['Price_Change'] = self.data['Close'].pct_change()
        self.data['Price_Change_5d'] = self.data['Close'].pct_change(periods=5)
        self.data['Volatility'] = self.data['Price_Change'].rolling(window=20).std()
        self.data['HL_Spread'] = self.data['High'] - self.data['Low']
        self.data['HL_Ratio'] = self.data['High'] / self.data['Low']
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=10).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        print("✓ Technical indicators calculated")
    
    def prepare_features(self):
        print("Preparing features...")
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20',
            'Price_Change', 'Price_Change_5d',
            'Volatility', 'HL_Spread', 'HL_Ratio',
            'Volume_Ratio', 'RSI'
        ]
        
        self.data['Target'] = self.data['Close'].shift(-1)
        clean_data = self.data.dropna()
        if len(clean_data) == 0:
            raise ValueError("No valid data after cleaning")
        
        self.features = clean_data[feature_columns]
        self.target = clean_data['Target']
        print(f"✓ Prepared {len(self.features)} samples with {len(feature_columns)} features")
        
    def train_models(self, test_size=0.2):
        print("Training models...")
        split_index = int(len(self.features) * (1 - test_size))
        X_train = self.features.iloc[:split_index]
        X_test = self.features.iloc[split_index:]
        y_train = self.target.iloc[:split_index]
        y_test = self.target.iloc[split_index:]
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.X_test = X_test_scaled
        self.y_test = y_test
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        self.models['Linear Regression'] = lr_model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        self.models['Random Forest'] = rf_model
        print(f"✓ Trained {len(self.models)} models")
    
    def evaluate_models(self):
        print("Evaluating models...")
        results = {}
        
        for name, model in self.models.items():
            predictions = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': predictions
            }
        self.results = results
        return results
    
    def print_results(self):
        print(f"\n{self.symbol} Stock Price Prediction Results")
        print("=" * 50)
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  RMSE: ${metrics['RMSE']:.2f}")
            print(f"  MAE:  ${metrics['MAE']:.2f}")
            print(f"  R²:   {metrics['R²']:.4f}")
        
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['R²'])
        print(f"\nBest Model: {best_model} (R² = {self.results[best_model]['R²']:.4f})")
    
    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.symbol} Stock Price Prediction Analysis', fontsize=16)
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['R²'])
        best_predictions = self.results[best_model]['predictions']
        axes[0, 0].scatter(self.y_test, best_predictions, alpha=0.6)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price')
        axes[0, 0].set_ylabel('Predicted Price')
        axes[0, 0].set_title(f'Actual vs Predicted ({best_model})')
        axes[0, 0].grid(True, alpha=0.3)
        test_dates = self.data.index[-len(self.y_test):]
        axes[0, 1].plot(test_dates, self.y_test.values, label='Actual', linewidth=2)
        axes[0, 1].plot(test_dates, best_predictions, label='Predicted', linewidth=2)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].set_title('Price Prediction Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        models = list(self.results.keys())
        r2_scores = [self.results[model]['R²'] for model in models]
        axes[1, 0].bar(models, r2_scores, color=['skyblue', 'lightcoral'])
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Model Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        for i, (model, score) in enumerate(zip(models, r2_scores)):
            axes[1, 0].text(i, score + 0.01, f'{score:.3f}', ha='center')
        
        errors = self.y_test.values - best_predictions
        axes[1, 1].hist(errors, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_stock_analysis(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.symbol} Stock Analysis', fontsize=16)
        axes[0, 0].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(self.data.index, self.data['MA_5'], label='MA 5', alpha=0.8)
        axes[0, 0].plot(self.data.index, self.data['MA_20'], label='MA 20', alpha=0.8)
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].set_title('Price and Moving Averages')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(self.data.index, self.data['RSI'], color='orange', linewidth=2)
        axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7)
        axes[0, 1].set_ylabel('RSI')
        axes[0, 1].set_title('Relative Strength Index')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)
        axes[1, 0].plot(self.data.index, self.data['Volatility'], color='red', linewidth=2)
        axes[1, 0].set_ylabel('Volatility')
        axes[1, 0].set_title('Price Volatility')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].bar(self.data.index, self.data['Volume'], alpha=0.6)
        axes[1, 1].set_ylabel('Volume')
        axes[1, 1].set_title('Trading Volume')
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict_future(self):
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['R²'])
        best_model = self.models[best_model_name]
        latest_features = self.features.iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        future_price = best_model.predict(latest_features_scaled)[0]
        current_price = self.data['Close'].iloc[-1]
        price_change = future_price - current_price
        price_change_pct = (price_change / current_price) * 100
        print(f"\n{self.symbol} Next Day Price Prediction:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${future_price:.2f}")
        print(f"Expected Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
        print(f"Model Used: {best_model_name}")
        return {
            'current_price': current_price,
            'predicted_price': future_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'model_used': best_model_name
        }

def main():
    STOCK_SYMBOL = "AAPL"
    DATA_PERIOD = "1y"
    print(f"Simple Stock Price Predictor for {STOCK_SYMBOL}")
    print("=" * 50)
    predictor = SimpleStockPredictor(STOCK_SYMBOL, DATA_PERIOD)
    if not predictor.fetch_data():
        return
    predictor.add_technical_indicators()
    predictor.prepare_features()
    predictor.train_models()
    predictor.evaluate_models()
    predictor.print_results()
    predictor.predict_future()
    predictor.plot_results()
    predictor.plot_stock_analysis()
    return predictor

if __name__ == "__main__":
    predictor = main()