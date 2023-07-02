import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('C:/Users/justo/module1dataset/ds_salaries.csv')
# Display the first few rows of the dataset
print(dataset.head())

# Get information about the dataset
print(dataset.info())

# Generate descriptive statistics
print(dataset.describe())
column_names = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']

# Create a histogram using Matplotlib
for column in column_names:
    plt.hist(dataset[column], bins=10)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Employee Graph of {column}')
    plt.show()
