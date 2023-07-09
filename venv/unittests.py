import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class TestDescriptiveStats(unittest.TestCase):

    def setUp(self):
        # Load sample salary dataset
        dataset = pd.read_csv('C:/Users/justo/module1dataset/ds_salaries.csv', nrows=5)
        self.column_names = ['work_year', 'salary', 'salary_in_usd', 'remote_ratio']
        self.selected_data = dataset[self.column_names]

    def test_column_properties(self):
        # Test column properties such as numeric type, missing values, and unique values
        expected_counts = [5, 5, 5, 5]
        expected_missing_values = [0, 0, 0, 0]
        expected_unique_values = [1, 5, 5, 1]

        for i, column in enumerate(self.column_names):
            selected_column = self.selected_data[column]
            with self.subTest(column=column):
                self.assertTrue(pd.api.types.is_numeric_dtype(selected_column))
                self.assertEqual(selected_column.count(), expected_counts[i])
                self.assertEqual(selected_column.isnull().sum(), expected_missing_values[i])
                self.assertEqual(selected_column.nunique(), expected_unique_values[i])

    def test_causation_analysis(self):
        # Test the causation analysis for different combinations of independent and dependent variables
        combinations = [('work_year', 'salary'), ('salary_in_usd', 'remote_ratio')]

        for independent_var, dependent_var in combinations:
            causation_results = self.selected_data[[independent_var, dependent_var]].dropna()
            X = sm.add_constant(causation_results[independent_var])
            y = causation_results[dependent_var]

            try:
                model = sm.OLS(y, X).fit()
                self.assertIsNotNone(model.summary())
            except Exception as e:
                self.fail(f"Error performing causation analysis for {independent_var} -> {dependent_var}: {str(e)}")

if __name__ == '__main__':
    unittest.main()
