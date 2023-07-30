import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error occurred while loading the dataset: {str(e)}")
        return None

def perform_linear_regression(data, x_column, y_column):
    try:
        model = LinearRegression()
        model.fit(data[[x_column]], data[y_column])
        y_pred = model.predict(data[[x_column]])
        mse = mean_squared_error(data[y_column], y_pred)
        r2_score = model.score(data[[x_column]], data[y_column])
        return mse, r2_score
    except Exception as e:
        print(f"Error occurred in linear regression: {str(e)}")
        return None, None

def plot_bar_chart(data, x_column, y_column, x_label, y_label, title):
    plt.figure(figsize=(10, 6))
    grouped_data = data.groupby(x_column)[y_column].mean().reset_index()
    plt.bar(grouped_data[x_column], grouped_data[y_column])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def calculate_mean_job_satisfaction(data):
    try:
        mean_job_satisfaction_all = data['JobSatisfaction'].mean()
        above_average_tenure_education = data[(data['YearsAtCompany'] > data['YearsAtCompany'].mean()) & (data['Education'] > data['Education'].mean())]
        mean_job_satisfaction_filtered = above_average_tenure_education['JobSatisfaction'].mean()
        return mean_job_satisfaction_all, mean_job_satisfaction_filtered
    except Exception as e:
        print(f"Error occurred while calculating mean job satisfaction: {str(e)}")
        return None, None

def identify_top_jobs(data, num_top_jobs=10):
    try:
        top_jobs = data.groupby('JobRole')['JobSatisfaction'].mean().nlargest(num_top_jobs)
        return top_jobs
    except Exception as e:
        print(f"Error occurred while identifying top jobs: {str(e)}")
        return None

def identify_top_employees(data):
    try:
        highest_job_satisfaction = data[data['JobSatisfaction'] == data['JobSatisfaction'].max()]
        sorted_by_education = highest_job_satisfaction.sort_values(by='Education', ascending=False)
        sorted_by_education['EducationDegree'] = sorted_by_education['Education'].map(map_education_level).apply(lambda x: x[1])
        top_employees = sorted_by_education.head(20)
        return top_employees
    except Exception as e:
        print(f"Error occurred while identifying top employees: {str(e)}")
        return None

def map_education_level(education_str):
    education_map = {
        1:"Below College",
        2:"College",
        3:"Bachelor's",
        4:"Master's",
        5:"Doctor",
    }
    return education_map.get(education_str, "Unkown")  

def map_job_satisfaction(satisfaction_level):
    job_satisfaction_map = {
        1: "Low Job Satisfaction",
        2: "Medium Job Satisfaction",
        3: "High Job Satisfaction",
        4: "Very High Job Satisfaction",
    }
    return job_satisfaction_map.get(satisfaction_level, "Unknown")


data = load_dataset('WA_Fn-UseC_-HR-Employee-Attrition.csv')
if data is not None:
    data['EducationLevel'] = data['Education'].map(map_education_level)
    data['EducationDegree'] = data['Education'].map(map_education_level)

    mse_simple, r2_score_simple = perform_linear_regression(data, 'Education', 'EnvironmentSatisfaction')
    mse_multi, r2_score_multi = perform_linear_regression(data, 'Education', 'EnvironmentSatisfaction')
    mean_job_satisfaction_all, mean_job_satisfaction_filtered = calculate_mean_job_satisfaction(data)
    top_employees = identify_top_employees(data)
    top_jobs = identify_top_jobs(data)

    print(f"Simple Linear Regression:\nMSE: {mse_simple}\nR^2 Score: {r2_score_simple}")
    print(f"Multiple Linear Regression:\nMSE: {mse_multi}\nR^2 Score: {r2_score_multi}")
    print(f"Mean Job Satisfaction (All Employees): {mean_job_satisfaction_all}")
    print(f"Mean Job Satisfaction (Filtered Employees): {mean_job_satisfaction_filtered}")

    if top_jobs is not None:
        print("\nJobs with the Highest Job Satisfaction:")
        print(top_jobs)

    # Output top employees with the highest job satisfaction
    if top_employees is not None:
        print("\nTop Employees with Highest Job Satisfaction:")
        # Create a new column for string representation of job satisfaction
        top_employees['JobSatisfactionValue'] = top_employees['JobSatisfaction'].map(map_job_satisfaction)
        print(top_employees[['JobRole', 'Education', 'JobSatisfaction', 'JobSatisfactionValue']])

    # chart for education levels
    # average JobSatisfaction grouped by EducationDegree
    plot_bar_chart(data, 'EducationDegree', 'JobSatisfaction', 'Education Degree', 'Average Job Satisfaction', 'Average Job Satisfaction vs. Education Degree')

    # chart for average JobSatisfaction grouped by JobRole
    plot_bar_chart(data, 'JobRole', 'JobSatisfaction', 'Job Role', 'Average Job Satisfaction', 'Average Job Satisfaction vs. Job Role')
