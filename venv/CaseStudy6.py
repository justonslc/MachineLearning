import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_split_data(filename, features, target):
    try:
        data = pd.read_csv(filename)
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, data
    except Exception as e:
        print(f"Error loading and splitting data: {e}")
        return None, None, None, None, None

def visualize_results(data, feature, target, conf_matrix):
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['No Heart Disease', 'Heart Disease'], 
                yticklabels=['No Heart Disease', 'Heart Disease'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data, x=feature, y=target, alpha=0.5)
    plt.title(f'Scatter Plot of {feature} vs. {target}')
    plt.ylabel('Heart Disease')
    plt.show()

    plt.figure(figsize=(10,6))
    sns.boxplot(data=data, x=target, y=feature, palette='coolwarm')
    plt.title(f'Box Plot of {feature} for People With and Without {target}')
    plt.xlabel('Heart Disease')
    plt.ylabel(feature)
    plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'])
    plt.show()

def make_predictions_for_older_age(model, scaler, data, features, age_threshold=40):
    """
    Predict heart disease for individuals over a specified age using the provided model.
    """
    older_data = data[data['age'] > age_threshold]
    X_older = older_data[features]
    X_older_scaled = scaler.transform(X_older)

    # Check for 'sex' column to differentiate between men = 1 and women = 0
    if 'sex' in data.columns:
        men_data = older_data[older_data['sex'] == 1]
        women_data = older_data[older_data['sex'] == 0]

        men_X = men_data[features]
        men_scaled_X = scaler.transform(men_X)
        men_predictions = model.predict(men_scaled_X)
        men_probabilities = model.predict_proba(men_scaled_X)[:, 1]

        women_X = women_data[features]
        women_scaled_X = scaler.transform(women_X)
        women_predictions = model.predict(women_scaled_X)
        women_probabilities = model.predict_proba(women_scaled_X)[:, 1]

        men_results = list(zip(men_data['age'], men_predictions, men_probabilities))
        women_results = list(zip(women_data['age'], women_predictions, women_probabilities))
        
        return men_results, women_results
    else:
        return [], []


# Re-run the pipeline with the corrected file path
def main_pipeline_with_scaling():
    X_train, X_test, y_train, y_test, data = load_and_split_data('heart.csv', ['age'], 'target')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy with scaling: {accuracy:.2f}")
    print("\nClassification Report with scaling:\n")
    print(class_report)
    visualize_results(data, 'age', 'target', conf_matrix)
    men_results, women_results = make_predictions_for_older_age(log_reg, scaler, data, ['age'])
    return X_train_scaled, X_test_scaled, men_results, women_results

X_train_scaled_output, X_test_scaled_output, men_results_output, women_results_output = main_pipeline_with_scaling()

# Display the first 5 rows of the scaled training and test data for the 'age' feature
X_train_scaled_output[:10], X_test_scaled_output[:10]

