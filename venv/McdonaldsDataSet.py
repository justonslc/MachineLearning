import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(file_path):
    # Read the file with ISO-8859-1 encoding
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    return data

def check_data(data):
    # Check the data types and missing values
    print(data.shape)
    print(data.dtypes)
    print(data.isnull().sum())

def get_numerical_stats(data):
    # Get stats for numerical columns
    print(data.describe(include=[np.number]))

def check_unique_values(data, categorical_columns):
    # Checking for unique values
    for column in categorical_columns:
        print(f"Unique values for {column}:")
        print(data[column].unique())
        print()

def clean_data(data):
    # Standardize value name
    data['store_name'] = data['store_name'].str.replace("ý", "")

    # Remove the 'category' column
    data.drop('category', axis=1, inplace=True)

    # Clean and convert 'rating_count' column to int
    data['rating_count'] = data['rating_count'].str.replace(",", "").astype(int)

    # Clean 'review_time' column and separate time and value
    data[['time_value', 'time_unit']] = data['review_time'].str.split(' ', n=1, expand=True)
    data['time_value'] = data['time_value'].replace('a', 1).astype(int)

    # Drop the original 'review_time' column
    data.drop('review_time', axis=1, inplace=True)

    # Clean and convert 'rating' column to float
    data['rating'] = data['rating'].str.replace(" stars", "").str.replace(" star", "").str.replace("s", "").astype(float)

    # Clean 'review' column to remove non-alphanumeric characters
    data['review'] = data['review'].str.replace('[^a-zA-Z0-9 \n\.]', '', regex=True)

    return data

def create_bar_plot(data):
    # Calculate average rating and number of ratings
    total_rating = data['review'].count()
    average_rating = data['rating'].mean()
    num_ratings = data['rating'].count()

    # Create a bar plot for Average Rating and Number of Ratings
    plt.figure(figsize=(10,6))
    bars = plt.bar(['Average Rating', 'Number of Ratings'], [average_rating, num_ratings])
    plt.ylabel('Value')
    plt.title('Average Rating and Number of Ratings')

    # Annotate the actual values on the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

    plt.show()

def display_cleaned_data(data):
    # Display a few rows of cleaned data
    print(data.head())

def save_cleaned_data(data, file_path):
    # Save cleaned data to a CSV file
    data.to_csv(file_path, index=False)
file_path = r'C:\Users\justo\module1dataset\McDonald_s_Reviews.csv'
categorical_columns = ['store_name', 'category', 'rating_count', 'review_time', 'rating']
cleaned_file_path = r'C:\Users\justo\module1dataset\Cleaned_McDonald_s_Reviews.csv'

data = read_data(file_path)
check_data(data)
get_numerical_stats(data)
check_unique_values(data, categorical_columns)
data = clean_data(data)
create_bar_plot(data)
display_cleaned_data(data)
save_cleaned_data(data, cleaned_file_path)
