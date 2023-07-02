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

for column in column_names:
    plt.scatter(dataset[column], range(len(dataset)), label=column)

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Index')
plt.title('Scatter Plot of Employee Data')

# Add legend
plt.legend()

# Show the plot
plt.show()
