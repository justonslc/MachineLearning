import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class PlanetData:
    def __init__(self):
        self.planet = {"density": [], "gravity": [], "target": []}
    # Checking whether density values is a list and its length 
    # If this condition is true that means the range is represented by a listwith two elements. 
    def generate_data(self, density_values, gravity_values, target_values):
        # This generates 10 random numbers 
        if isinstance(density_values, list) and len(density_values) == 2:
            density_data = [random.uniform(*density_values) for _ in range(10)]
        #Density values as either single list of value
        else:
            density_data = density_values
        #Gravity values is a range represented by the list with two elements like density values.
        if isinstance(gravity_values, list) and len(gravity_values) == 2:
            # This generates 10 random numbers 
            gravity_data = [random.uniform(*gravity_values) for _ in range(10)]
        else:
            #Gravity values as either single list of valu
            gravity_data = gravity_values

        # Ensure that all three arrays are the same length
        min_length = min(len(density_data), len(gravity_data), len(target_values))
        # Append the first length of elements of density, gravity and target into their arrays.
        self.planet["density"].extend(density_data[:min_length])
        self.planet["gravity"].extend(gravity_data[:min_length])
        self.planet["target"].extend(target_values[:min_length])

    def get_data(self):
        return self.planet


class PlanetClassifier:
    #This takes the parameter planet_data to create the classifier
    def __init__(self, planet_data):
        #Planet data is used to create a pandas dataframe. 
        self.planet_df = pd.DataFrame(planet_data)
        #This method splits the data into training and testing sets. 
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def split_data(self):
        # Create the input features x by dropping the target column from the dataframe
        X = self.planet_df.drop("target", axis="columns")
        # Create the target by extracting the values from the target column
        Y = self.planet_df["target"]
        #Split the data into training and testing sets using the train_test_split
        return train_test_split(X, Y, test_size=0.25)

    def train_and_evaluate_model(self):
        # Convert the data to numpy arrays
        X_train_np = np.array(self.X_train)
        y_train_np = np.array(self.y_train)
        X_test_np = np.array(self.X_test)
        y_test_np = np.array(self.y_test)

        # Create the knn classifier
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train_np, y_train_np)

        # Evaluate the model
        model_accuracy = knn.score(X_test_np, y_test_np)
        print("Model accuracy =", model_accuracy)

        # Create a DataFrame with test data and corresponding density and gravity values
        test_data_df = pd.DataFrame(self.X_test, columns=["gravity", "density"])

        # Print test data and corresponding density and gravity values
        print("\nTest data:")
        print(test_data_df)

        # Predict target values
        y_prediction = knn.predict(X_test_np)

        # Print predicted target values
        print("\nPredicted target =", y_prediction)

        # Calculate and print confusion matrix
        confusion_matrix_result = confusion_matrix(y_test_np, y_prediction)
        print("\nConfusion Matrix:")
        print(confusion_matrix_result)

    def plot_data(self):
        #Plot scatter using gravity, density and venus. Colors on the legend are purple blue and yellow
        plt.scatter(self.planet_df["gravity"], self.planet_df["density"], c=self.planet_df["target"], cmap='viridis')
        plt.legend(["Jupiter", "Saturn", "Venus", "Test"])
        plt.title("Planets density and gravity")
        plt.xlabel("Gravity")
        plt.ylabel("Density")
        plt.show()

# Generate and plot data
planet_data = PlanetData()
#This is data that I gathered from Planetary Fact Sheet https://nssdc.gsfc.nasa.gov/planetary/factsheet/planet_table_ratio.html
planet_data.generate_data((1.2, 1.4), (22, 26), [0]*10)  # Jupiter
planet_data.generate_data((0.6, 0.8), (9, 12), [1]*10)   # Saturn
planet_data.generate_data((5.0, 5.5), (8, 10), [2]*10)   # Venus
#The test data is just random numbers. I didnt know which numbers to use, so I just guessed. 
planet_data.generate_data([0.74, 0.85, 0.63, 0.35, 0.22, 0.15, 0.31, 0.59, 0.42, 0.88],  # Test
                          [2.5, 2.7, 4.8, 7.6, 18.5, 15.6, 21.5, 3.7, 8.5, 1.8], 
                          [0, 0, 1, 1, 2, 2, 2, 1, 1, 0])

planet_classifier = PlanetClassifier(planet_data.get_data())

# Plot data
planet_classifier.plot_data()

# Train and evaluate model
planet_classifier.train_and_evaluate_model()
