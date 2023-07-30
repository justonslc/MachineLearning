from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as mtick
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    "Load the dataset."
    return pd.read_csv(file_path)

def plot_data(df):
    "Plot the relationships between variables."
    fig, axs = plt.subplots(3, figsize=(10,15))  # Data was very difficult to read if it wasnt 3
    plt.suptitle("Features vs Price")
    for i, col in enumerate(df.drop(columns=["price"]).columns[:3]): 
        axs[i].title.set_text(f"{col} vs price")
        axs[i].scatter(df[col], df["price"], alpha=0.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjusted the top spacing
    plt.show()

    
def plot_projected_prices(projected_prices):
    "Plot the projected average prices over the next five years."
    plt.figure(figsize=(8, 6))
    plt.bar(projected_prices.index, projected_prices['Projected Average Price'])
    plt.xlabel('Year')
    plt.ylabel('Projected Average Price (USD)')
    plt.title('Projected Average Computer Price Over the Next 10 Years')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))  # Format y-axis as dollars
    plt.show()


def split_data(df, dependent_variable):
    "Split the dataset into features (X) and target variable (y), and scale the variables."
    independent_variables = df.drop(columns=[dependent_variable]).columns.tolist()
    X = df[independent_variables].values
    y = df[dependent_variable].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    "Train the model."
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def evaluate_model(model, X_test, y_test):
    "Calculate and print the performance metrics."
    y_pred = model.predict(X_test)
    print(f"The root mean squared error for the test dataset is:  ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"The mean absolute error for the test dataset is: ${mean_absolute_error(y_test, y_pred):.2f}")
    print(f"The R2_score for the test dataset is:            {r2_score(y_test, y_pred):.2f}")

# Load the dataset
df = load_data('ComputerPricesData.csv')

# Plot the relationships between variables
plot_data(df)

# Identify any columns that still have 'yes'/'no' values
binary_cols = []
for col in df.columns:
    if 'yes' in df[col].values or 'no' in df[col].values:
        binary_cols.append(col)

# Convert 'yes'/'no' to 1/0
df[binary_cols] = df[binary_cols].replace({'yes': 1, 'no': 0})

# Calculate the current average price
current_avg_price = df['price'].mean()

# Define the annual growth rate
growth_rate = 0.0181

# Create a DataFrame to hold the projected prices
projected_prices = pd.DataFrame(index=range(2023, 2033), columns=['Projected Average Price'])

# Calculate the projected price for each year
for year in projected_prices.index:
    years_passed = year - 2023
    projected_prices.loc[year, 'Projected Average Price'] = current_avg_price * (1 + growth_rate) ** years_passed

# Display the DataFrame
print(projected_prices)

# Plot the projected prices
plot_projected_prices(projected_prices)

# Split the data
X_train, X_test, y_train, y_test = split_data(df, 'price')

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
evaluate_model(model, X_test, y_test)
