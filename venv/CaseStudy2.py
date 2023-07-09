import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from scipy import stats

# Read the first 5 rows from CSV for easier reading
dataset = pd.read_csv('C:/Users/justo/module1dataset/ds_salaries.csv', nrows=5)

# Specific columns
salary_column = dataset['salary']
location_column = dataset['company_location']

# Convert columns to different tuples, list, dicrionary yadayada ya
salary_list = salary_column.values.tolist()  #  to List
location_set = set(location_column)  #  to Set
salary_dict = salary_column.to_dict()  #  to Dictionary
location_tuple = tuple(location_column)  #  to Tuple

# Print the data structures
print("Salary column as a List:")
print(salary_list[:5])

print("\nCompany location column as a Set:")
print(location_set)

print("\nSalary column as a Dictionary:")
print(salary_dict)

print("\nCompany location column as a Tuple:")
print(location_tuple)

# maximum value in each row
dataset_max = dataset.max(axis=1)
print("\nMaximum value in each row:")
print(dataset_max)

# mean for each row
dataset_mean = dataset.mean(axis=1)
print("\nMean value for each row:")
print(dataset_mean)

# histogram to each column
column_names = ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency',
                'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']
for column in column_names:
    plt.hist(dataset[column], bins=10)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Employee Graph of {column}')
    plt.show()
