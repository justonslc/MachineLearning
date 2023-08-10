import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import inspect
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_metrics(data):
    unique_stocks = data['Name'].nunique()
    data['daily_return'] = data.groupby('Name')['close'].pct_change()
    avg_returns = data.groupby('Name')['daily_return'].mean()
    
    return unique_stocks, avg_returns

def calculate_volatility(data):
    """Calculate volatility (standard deviation) of daily returns for each stock."""
    data['daily_return'] = data.groupby('Name')['close'].pct_change()
    return data.groupby('Name')['daily_return'].std()

#train, predict, and split the data
def knn_regression(features, target, n_neighbors=5):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return knn, rmse

#train, predict, and split the data
def decision_tree_regression(features, target, max_depth=None):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    dt = DecisionTreeRegressor(max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return dt, rmse

def plot_data(avg_returns, volatility):
    plt.figure(figsize=(14, 7))
    sns.scatterplot(x=volatility, y=avg_returns)
    plt.title('Average Returns vs. Volatility')
    plt.xlabel('Volatility (Std. Deviation of Returns)')
    plt.ylabel('Average Daily Returns')
    plt.grid(True, which="both", ls="--")
    plt.show()
    
def plot_top_stocks(avg_returns, volatility):
    risk_reward_score = avg_returns / volatility
    top_stocks = risk_reward_score.sort_values(ascending=False).head(10)
    plt.figure(figsize=(16, 8))
    sns.scatterplot(x=volatility, y=avg_returns, alpha=0.6)
    for stock in top_stocks.index:
        plt.annotate(stock, 
                     (volatility[stock], avg_returns[stock]),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')

    plt.title('Average Returns vs. Volatility with Top Stocks Highlighted')
    plt.xlabel('Volatility (Std. Deviation of Returns)')
    plt.ylabel('Average Daily Returns')
    plt.grid(True, which="both", ls="--")
    plt.show()

    return top_stocks

# Main 
data = load_data('all_stocks_5yr.csv')
unique_stocks, avg_returns = calculate_metrics(data)
volatility = calculate_volatility(data)

# Plot average returns vs. volatility with top stocks highlighted
top_stocks = plot_top_stocks(avg_returns, volatility)

print(unique_stocks)
print(avg_returns.sort_values(ascending=False).head(10))
print(volatility.sort_values(ascending=False).head(10))
print("\nTop 10 stocks based on risk-reward score:")
print(top_stocks)

# Prepare data for machine learning models
features = volatility.values.reshape(-1, 1)  # Using volatility as the feature
target = avg_returns.values  # Predicting average returns

# print knn
knn_model, knn_rmse = knn_regression(features, target)
print(f"Root Mean Squared Error (RMSE) of kNN model: {knn_rmse:.4f}")

# print decision tree
dt_model, dt_rmse = decision_tree_regression(features, target, max_depth=3)
print(f"Root Mean Squared Error (RMSE) of Decision Tree model: {dt_rmse:.4f}")
print(dt_model)

# graph of decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=['Volatility'], rounded=True)
plt.show()

# Plot average returns vs. volatility
plot_data(avg_returns, volatility)

print(unique_stocks)
print(avg_returns.sort_values(ascending=False).head(10))
print(volatility.sort_values(ascending=False).head(10))
