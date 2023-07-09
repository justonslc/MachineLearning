import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Create salary dataset
dataset = pd.read_csv('C:/Users/justo/module1dataset/ds_salaries.csv', nrows=5)
column_names = ['work_year','salary', 
                'salary_in_usd', 'remote_ratio']
selected_data = dataset[column_names]

# perform descriptive statistical analytics using the below values
statistics = {}

for column in column_names:
    try:
        # convert column to numeric values, replacing non-numeric values with NaN ---> values that were missing were removed from the columns
        selected_data.loc[:, column] = pd.to_numeric(selected_data.loc[:, column], errors='coerce')
        
        # calculate descriptive statistics only for numeric columns
        if pd.api.types.is_numeric_dtype(selected_data[column]):
            column_stats = {
                # is to count all the values
                'Count': selected_data[column].count(),
                # is to check for all missing values in the column
                'Missing Values': selected_data[column].isnull().sum(),
                # unique values in each column that is not coorelated
                'Unique Values': selected_data[column].nunique(),
                #add the mode which is most frequent data
                'Mode': selected_data[column].mode().values,
                #find miniumum of the data
                #this is also were dispersion for range, variance standard deviation
                'Minimum': selected_data[column].min(),
                #find max of the data
                #this is also were dispersion for range, variance standard deviation
                'Maximum': selected_data[column].max(),
                #find the average of the data
                #this is also were dispersion for range, variance standard deviation
                'Mean': selected_data[column].mean(),
                #find the median of the data
                #this is also were dispersion for range, variance standard deviation
                'Median': selected_data[column].median(),
                #measure the spread of the values around the mean
                #this is also were dispersion for range, variance standard deviation
                'Standard Deviation': selected_data[column].std(),
                #measures of asymmetry
                'Skewness': selected_data[column].skew(),
                #measure of the peakness
                'Kurtosis': selected_data[column].kurtosis(),
                #top values
                'Top 5 Values': selected_data[column].head(5).values
            }
            statistics[column] = column_stats
    
            # Generate histogram for the column ----> the graph is just one that includes these values. The plt.show has a tight layout
            plt.hist(selected_data[column].dropna(), bins=10)
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {column}')
      

    except (ValueError, TypeError) as e:
        print(f"Error processing column '{column}': {str(e)}")

# Adjust subplot spacing
plt.tight_layout()

# Show the figure with all the histograms
plt.show()

# Print the descriptive statistics
for column, stats in statistics.items():
    print(f"\n{column} Statistics:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")

    # Calculate and print central tendencies
    mean = selected_data[column].mean()
    median = selected_data[column].median()
    mode = selected_data[column].mode().values
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    
    # Perform causation analysis (example)
causation_results = selected_data[['work_year', 'salary']].dropna()

try:
    # stats for causation
    # sm.add_constant adds work_year to the causationresults dataframe. 
    X = sm.add_constant(causation_results['work_year'])
    #causation results adds salary to the dataframe
    y = causation_results['salary']

    model = sm.OLS(y, X).fit()
    print(model.summary())

except Exception as e:
    print(f"Error performing causation analysis: {str(e)}")