import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def calculate_metrics(data):
    """Calculate unique stocks and average returns."""
    # Number of unique stocks
    unique_stocks = data['Name'].nunique()
    
    # Calculate average return for each stock
    data['daily_return'] = data.groupby('Name')['close'].pct_change()
    avg_returns = data.groupby('Name')['daily_return'].mean()
    
    return unique_stocks, avg_returns

def calculate_volatility(data):
    """Calculate volatility (standard deviation) of daily returns for each stock."""
    data['daily_return'] = data.groupby('Name')['close'].pct_change()
    return data.groupby('Name')['daily_return'].std()

def plot_data(avg_returns, volatility):
    """Plot average returns vs. volatility."""
    plt.figure(figsize=(14, 7))
    sns.scatterplot(x=volatility, y=avg_returns)
    plt.title('Average Returns vs. Volatility')
    plt.xlabel('Volatility (Std. Deviation of Returns)')
    plt.ylabel('Average Daily Returns')
    plt.grid(True, which="both", ls="--")
    plt.show()

# Main execution
data = load_data('all_stocks_5yr.csv')
unique_stocks, avg_returns = calculate_metrics(data)
volatility = calculate_volatility(data)
plot_data(avg_returns, volatility)

print(unique_stocks)
print(avg_returns.sort_values(ascending=False).head(10))
print(volatility.sort_values(ascending=False).head(10))
