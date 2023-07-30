# Import necessary libraries for data manipulation, ML models, and visualization
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to load and preprocess data
def load_and_preprocess_data(filepath, rows=None):
    # Load the data from CSV file
    data = pd.read_csv(filepath, nrows=rows)
    
    # Drop columns from the data
    data = data.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1)
    
    # Initialize LabelEncoder
    le = LabelEncoder()
    
    # Convert class labels to numerical values
    data['class'] = le.fit_transform(data['class'])
    
    # Return preprocessed data
    return data

# Function to split data into training and testing sets
def split_data(data):
    # Define features (X) and target (y)
    X = data.drop('class', axis=1)
    y = data['class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Return split data
    return X_train, X_test, y_train, y_test

# Function to normalize data
def normalize_data(X_train, X_test):
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to the training data
    scaler.fit(X_train)
    
    # Normalize the training data
    X_train = scaler.transform(X_train)
    
    # Normalize the testing data
    X_test = scaler.transform(X_test)
    
    # Return normalized data
    return X_train, X_test

# Function to train and evaluate KNN classifier
def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors):
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    # Use the trained classifier to predict labels for the testing data
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate  predictions
    y_proba = knn.predict_proba(X_test)
    
    # Return accuracy, report, confusion matrix, and predictions
    return accuracy, report, cm, y_proba

# Function to select features
def feature_selection(X_train_df, X_train, y_train, X_test, k):
    # Initialize SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit the selector to the training data
    selector.fit(X_train, y_train)
    
    # Get the selected feature names
    selected_features = X_train_df.columns[selector.get_support()]
    
    # Transform the training and testing data
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Return transformed data and selected feature names
    return X_train_selected, X_test_selected, selected_features

# Function to demonstrate overfitting and underfitting
def demonstrate_overfitting_and_underfitting(X_train, y_train, X_test, y_test):
    # Define list of neighbors
    neighbors_list = [1, 3, 5, 10, 15, 20]
    
    # For each neighbor, train and evaluate the classifier
    for n_neighbors in neighbors_list:
        accuracy, report, cm, _ = train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors)
        print("Neighbors:", n_neighbors)
        print("Accuracy: {:.2f}".format(accuracy))
        print(report)
        print("Confusion Matrix:")
        print(cm)
        print("="*60)

# Function to demonstrate correctness
def demonstrate_correctness(X_train, y_train, X_test, y_test):
    # Train and evaluate the classifier with 5 neighbors
    accuracy, report, cm, y_proba = train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=5)
    print("Correctness - KNN with 5 neighbors")
    print("Accuracy: {:.2f}".format(accuracy))
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("Probabilistic Predictions:")
    print(y_proba)

# Function to demonstrate bias-variance tradeoff
def demonstrate_bias_variance_tradeoff(X_train, y_train, X_test, y_test):
    # Define list of neighbors
    neighbors_list = [1, 3, 5, 10, 15, 20]
    
    # For each neighbor, calculate cross validation score
    for n_neighbors in neighbors_list:
        # Initialize KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # Fit the classifier to the training data
        knn.fit(X_train, y_train)
        
        # Calculate cross validation score
        cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
        print("Neighbors:", n_neighbors)
        print("Cross Validation Score: {:.2f}".format(np.mean(cv_scores)))

    # Print Bias-Variance Tradeoff
    print("Bias-Variance Tradeoff")
    for n_neighbors in neighbors_list:
        # Initialize KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # Fit the classifier to the training data
        knn.fit(X_train, y_train)
        print("Neighbors:", n_neighbors)
        print("Training Accuracy: {:.2f}".format(knn.score(X_train, y_train)))
        print("Testing Accuracy: {:.2f}".format(knn.score(X_test, y_test)))

# Function to demonstrate feature extraction and selection
def demonstrate_feature_extraction_and_selection(X_train, y_train, X_test, y_test):
    # Define k_values
    k_values = [1, 2, 3]
    
    # For each k_value, select features and train and evaluate the classifier
    for k in k_values:
        X_train_selected, X_test_selected, selected_features = feature_selection(data.drop('class', axis=1), X_train, y_train, X_test, k=k)
        accuracy, report, cm, _ = train_and_evaluate_knn(X_train_selected, y_train, X_test_selected, y_test, n_neighbors=5)
        print("Number of Features Selected:", k)
        print("Selected Features:", selected_features.values)
        print("Accuracy: {:.2f}".format(accuracy))
        print(report)
        print("Confusion Matrix:")
        print(cm)
        print("="*60)

# Function to create 3D graph
def demonstrate_three_dimensional_graph(X_train, y_train, X_test, y_test):
    # Initialize figure and 
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add scatter plots to the subplot
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis', label='Training Data', s=30, alpha=0.7)
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='inferno', marker='^', label='Test Data', s=30, alpha=0.7)
    
    # Set labels for the subplot
    ax.set_xlabel('Right Ascension (ra)')
    ax.set_ylabel('Declination (dec)')
    ax.set_zlabel('u')
    
    # Set viewing angle and title for the subplot
    ax.view_init(30, 30)
    ax.set_title('3D Scatter Plot of Iris Dataset')
    
    # Add legend to the subplot
    ax.legend()
    
    # Display the plot
    plt.show()

# Function to tune parameters
def hyperparameter_tuning(X_train, y_train):
    # Initialize KNN classifier
    knn = KNeighborsClassifier()
    
    # Define parameter grid
    param_grid = {'n_neighbors': [1, 3, 5, 10, 15, 20]}
    
    # Initialize git search
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    
    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print("Best Parameters:")
    print(grid_search.best_params_)

# Load and preprocess the data
data = load_and_preprocess_data("C:/Users/justo/module1dataset/Skyserver_SQL2_27_2018 6_51_39 PM.csv", rows=70)
X_train, X_test, y_train, y_test = split_data(data)
X_train, X_test = normalize_data(X_train, X_test)

# Tune hyperparameters
hyperparameter_tuning(X_train, y_train)

# Demonstrate Overfitting and Underfitting
demonstrate_overfitting_and_underfitting(X_train, y_train, X_test, y_test)

# Demonstrate Correctness
demonstrate_correctness(X_train, y_train, X_test, y_test)

# Demonstrate Bias-Variance Tradeoff
demonstrate_bias_variance_tradeoff(X_train, y_train, X_test, y_test)

# Demonstrate Feature Extraction and Selection
demonstrate_feature_extraction_and_selection(X_train, y_train, X_test, y_test)

# Demonstrate Three-dimensional Graph
demonstrate_three_dimensional_graph(X_train, y_train, X_test, y_test)
