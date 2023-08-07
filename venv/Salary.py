import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def plot_data_with_logistic_and_r2(data):
    try:
        # Encode the revenue categories to numerical values
        unique_revenues = sorted(data['Revenue'].unique())
        revenue_to_int = {rev: i for i, rev in enumerate(unique_revenues)}
        data['Revenue_encoded'] = data['Revenue'].map(revenue_to_int)

        plt.figure(figsize=(12, 8))

        # Plot scatter points
        for label, df in data.groupby('Revenue'):
            plt.scatter(df['Revenue_encoded'], df['Avg Salary'], label=label, s=50, alpha=0.6, edgecolors='w', linewidth=0.5)

        # Fit the logistic curve
        popt, _ = curve_fit(logistic, data['Revenue_encoded'], data['Avg Salary'])
        x = np.linspace(min(data['Revenue_encoded']), max(data['Revenue_encoded']), 1000)
        y = logistic(x, *popt)
        
        # Calculate R-squared
        predicted = logistic(data['Revenue_encoded'], *popt)
        r2 = r2_score(data['Avg Salary'], predicted)
        
        plt.plot(x, y, 'r-', label=f'Fitted Logistic Curve\n$R^2$ = {r2:.2f}')

        plt.title('Correlation between Average Salary Estimate and Company Revenue with Logistic Curve')
        plt.xlabel('Company Revenue')
        plt.ylabel('Average Salary ($K)')
        plt.xticks(rotation=45, ticks=range(len(unique_revenues)), labels=unique_revenues)
        plt.legend(title='Company Revenue', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting data: {e}")
        
def binary_classification_plot_with_scaling(data):
    # Create binary target: 0 if revenue is below $1 billion, 1 otherwise
    data['Above_1B'] = data['Revenue'].apply(lambda x: 0 if 'million' in x or 'Unknown' in x or 'Less than $1 million' in x else 1)

    # Scale the 'Avg Salary' data
    scaler = StandardScaler()
    data['Scaled_Avg_Salary'] = scaler.fit_transform(data[['Avg Salary']])
    
    # Fit a logistic regression model on scaled data
    X = data[['Scaled_Avg_Salary']]
    y = data['Above_1B']

    logistic_model = LogisticRegression()
    logistic_model.fit(X, y)

    plt.figure(figsize=(12, 8))

    # Plot data points
    plt.scatter(data['Scaled_Avg_Salary'], data['Above_1B'], alpha=0.5, label='Data points')

    # Plot logistic curve
    x_vals = np.linspace(data['Scaled_Avg_Salary'].min(), data['Scaled_Avg_Salary'].max(), 1000)
    y_probs = logistic_model.predict_proba(x_vals.reshape(-1, 1))[:, 1]
    plt.plot(x_vals, y_probs, color='red', label='Logistic Curve')

    plt.title('Probability of Company Revenue being Above $1 Billion based on Scaled Average Salary')
    plt.xlabel('Scaled Average Salary')
    plt.ylabel('Probability of Revenue > $1B')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Output model coefficients and intercept to the terminal
    coef = logistic_model.coef_[0][0]
    intercept = logistic_model.intercept_[0]
    return data[['Avg Salary', 'Scaled_Avg_Salary']], f"Model Coefficient: {logistic_model.coef_[0][0]:.4f}, Intercept: {logistic_model.intercept_[0]:.4f}"
    
def filter_data(data):
    try:
        data = data.dropna(subset=['Salary Estimate', 'Job Title', 'Size', 'Revenue', 'Degree'])
        filters = ['Salary Estimate', 'Size', 'Revenue', 'Degree']
        for col in filters:
            data = data[data[col] != '-1']
            data = data[data[col] != 'na']

        # Further refine the filter for "Per Hour" in the 'Salary Estimate'
        data = data[~data['Salary Estimate'].str.contains("Per Hour")]
        return data
    except Exception as e:
        print(f"Error filtering data: {e}")
        return None
    
def binary_classification_plot(data):
    # Create binary target: 0 if revenue is below $1 billion, 1 otherwise
    data['Above_1B'] = data['Revenue'].apply(lambda x: 0 if 'million' in x or 'Unknown' in x or 'Less than $1 million' in x else 1)

    # Fit a logistic regression model
    X = data[['Avg Salary']]
    y = data['Above_1B']

    logistic_model = LogisticRegression()
    logistic_model.fit(X, y)

    plt.figure(figsize=(12, 8))

    # Plot data points
    plt.scatter(data['Avg Salary'], data['Above_1B'], alpha=0.5, label='Data points')

    # Plot logistic curve
    x_vals = np.linspace(data['Avg Salary'].min(), data['Avg Salary'].max(), 1000)
    y_probs = logistic_model.predict_proba(x_vals.reshape(-1, 1))[:, 1]
    plt.plot(x_vals, y_probs, color='red', label='Logistic Curve')

    plt.title('Probability of Company Revenue being Above $1 Billion based on Average Salary')
    plt.xlabel('Average Salary ($K)')
    plt.ylabel('Probability of Revenue > $1B')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function
    #binary_classification_plot(filtered_data)

def extract_average_salary(salary_str):
    try:
        salary_str = salary_str.replace("Employer Provided Salary:", "").replace("(Glassdoor est.)", "").replace("(Employer est.)", "").strip()
        if '-' in salary_str:
            min_salary = int(salary_str.split('-')[0].replace('$', '').replace('K', ''))
            max_salary = int(salary_str.split('-')[1].replace('$', '').replace('K', ''))
        else:  # if only one value is provided
            min_salary = max_salary = int(salary_str.replace('$', '').replace('K', ''))
        return (min_salary + max_salary) / 2
    except Exception as e:
        print(f"Error extracting average salary: {e}")
        return None
    
def plot_data(data):
    try:
        plt.figure(figsize=(12, 8))
        for label, df in data.groupby('Revenue'):
            plt.scatter(df['Revenue'], df['Avg Salary'], label=label, s=50, alpha=0.6, edgecolors='w', linewidth=0.5)

        plt.title('Correlation between Average Salary Estimate and Company Revenue')
        plt.xlabel('Company Revenue')
        plt.ylabel('Average Salary ($K)')
        plt.xticks(rotation=45)
        plt.legend(title='Company Revenue', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting data: {e}")
            
def train_predictive_model(data):
    # Creating a mapping from revenue strings to integers
    unique_revenues = data['Revenue'].unique()
    revenue_to_int = {rev: i for i, rev in enumerate(unique_revenues)}
    
    # Filtering out revenue categories 
    value_counts = data['Revenue'].value_counts()
    to_keep = value_counts[value_counts > 1].index
    data = data[data['Revenue'].isin(to_keep)]
    
    data.loc[:, 'Revenue_encoded'] = data['Revenue'].map(revenue_to_int)

    # Splitting data into training and test sets
    X = data[['Revenue_encoded']]
    y = data['Avg Salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Evaluating the model
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Data: {mse:.2f}")

    # Prediction function
    def predict(revenue_str):
        if revenue_str in revenue_to_int:
            revenue_encoded = revenue_to_int[revenue_str]
            salary_pred = lr.predict([[revenue_encoded]])[0]
            return salary_pred
        else:
            print(f"Error: Unseen revenue label '{revenue_str}'")
            return None

    return predict

def plot_with_prediction(data, predictor, revenue_label):
    # Plotting the given data
    plt.figure(figsize=(12, 8))
    for label, df in data.groupby('Revenue'):
        plt.scatter(df['Revenue'], df['Avg Salary'], label=label, s=50, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    # Getting the prediction for the revenue 
    predicted_salary = predictor(revenue_label)
    
    # Plotting the predicted value
    plt.scatter(revenue_label, predicted_salary, color='red', marker='*', s=150, label=f'Predicted for {revenue_label}', edgecolors='w', linewidth=0.5)
    
    # Setting graph atributes 
    plt.title('Correlation between Average Salary Estimate and Company Revenue with Prediction')
    plt.xlabel('Company Revenue')
    plt.ylabel('Average Salary ($K)')
    plt.xticks(rotation=45)
    plt.legend(title='Company Revenue', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Using the refactored functions
data = load_data("data_cleaned_2021.csv")
# reason for looking for everything without 10 billion. Is there is a fault in the data set. It has a problem returning this data correctly. 
data = data[data['Revenue'] != '10+ billion (USD)']
if data is not None:
    filtered_data = filter_data(data)
    if filtered_data is not None:
        filtered_data['Avg Salary'] = filtered_data['Salary Estimate'].apply(extract_average_salary)
        scaled_data_output, model_output = binary_classification_plot_with_scaling(filtered_data)
        print(model_output)
        print(scaled_data_output.head())
        plot_data(filtered_data)
        plot_data_with_logistic_and_r2(filtered_data)
        binary_classification_plot(filtered_data)
        binary_classification_plot_with_scaling(filtered_data)
        predictor = train_predictive_model(filtered_data)
        plot_with_prediction(filtered_data, predictor, "10+ billion (USD)")
        
        # Use a different revenue category for prediction
        revenue_example = "$10+ billion (USD)"  # predicted_salary=intercept+coefficient×encoded_revenue. 
        predicted_salary = predictor(revenue_example)
        
        if predicted_salary is not None:
            print(f"Predicted Average Salary for {revenue_example}: ${predicted_salary:.2f}K")
            print(predicted_salary)
        else:
            print(f"Could not predict average salary for {revenue_example}.")





