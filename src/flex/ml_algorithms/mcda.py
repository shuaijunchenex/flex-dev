from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#TODO: unit test

class MultiCriteriaDecisionAnalysis:
    def __init__(self) -> None:
        pass

    def reverse_min_max_normalizer(self, x):
        x_max = np.max(x)
        x_min = np.min(x)
        return (x_max - x) / (x_max  - x_min)

    @staticmethod
    def topsis(data, weights=None, criteria=None, normalize='max'):
        """
        Implementation of the TOPSIS method, consistent with the input and output of grey_relational_analysis.

        Parameters:
        - data: pandas DataFrame, rows represent samples, columns represent criteria.
        - weights: list of weights for the criteria (default is equal weights).
        - criteria: list of criteria types, 'max' for benefit criteria, 'min' for cost criteria (default is all benefit criteria).
        - normalize: str, method of data normalization ('max', 'min-max', or 'mean').

        Returns:
        - scores: pandas Series, TOPSIS scores for each sample.
        """

        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Number of criteria
        n_criteria = data.shape[1]

        # Process weights
        if weights is None:
            weights = np.ones(n_criteria) / n_criteria
        else:
            weights = np.array(weights, dtype=float)
            if len(weights) != n_criteria:
                raise ValueError("The length of weights must be equal to the number of criteria.")
            weights = weights / weights.sum()

        # Process criteria types
        if criteria is None:
            criteria = ['max'] * n_criteria
        else:
            if len(criteria) != n_criteria:
                raise ValueError("The length of criteria must be equal to the number of criteria.")

        # Data normalization
        if normalize == 'max':
            norm_data = data / np.sqrt((data ** 2).sum())
        elif normalize == 'min-max':
            norm_data = (data - data.min()) / (data.max() - data.min())
        elif normalize == 'mean':
            norm_data = data / data.mean()
        else:
            raise ValueError("The 'normalize' parameter should be 'max', 'min-max', or 'mean'.")

        # Weighted normalized matrix
        weighted_data = norm_data * weights

        # Determine ideal and negative-ideal solutions
        ideal_solution = pd.Series(index=data.columns)
        negative_ideal_solution = pd.Series(index=data.columns)

        for i, crit in enumerate(criteria):
            if crit == 'max':
                ideal_solution[i] = weighted_data.iloc[:, i].max()
                negative_ideal_solution[i] = weighted_data.iloc[:, i].min()
            elif crit == 'min':
                ideal_solution[i] = weighted_data.iloc[:, i].min()
                negative_ideal_solution[i] = weighted_data.iloc[:, i].max()
            else:
                raise ValueError("Elements in criteria list should be 'max' or 'min'.")

        # Calculate distances to the ideal and negative-ideal solutions
        distance_to_ideal = np.sqrt(((weighted_data - ideal_solution) ** 2).sum(axis=1))
        distance_to_negative_ideal = np.sqrt(((weighted_data - negative_ideal_solution) ** 2).sum(axis=1))

        # Calculate the relative closeness
        scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

        # Convert scores to pandas Series
        scores = pd.Series(scores, index=data.index)

        print("TOPSIS Scores:", scores)

        return scores
            

    def grey_relational_analysis(self, data, reference_series=None, normalize='max', rho=0.5, weights=None):
        """
        Grey Relational Analysis (GRA) function with metric weights

        Parameters:
        - data: pandas DataFrame, where rows represent samples and columns represent indicators
        - reference_series: pandas Series, the reference sequence (default is the maximum value of each indicator)
        - normalize: str, method for data normalization ('max', 'min-max', or 'mean')
        - rho: float, distinguishing coefficient in the range [0, 1], default is 0.5
        - weights: list or array-like, weights for each indicator (default is equal weights)

        Returns:
        - relational_grade: pandas Series, grey relational grades for each sample
        """
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Step 1: Data normalization
        if normalize == 'max':
            # Max-value normalization
            data_normalized = data / data.max()
        elif normalize == 'min-max':
            # Min-max normalization
            data_normalized = (data - data.min()) / (data.max() - data.min())
        elif normalize == 'mean':
            # Mean normalization
            data_normalized = data / data.mean()
        else:
            raise ValueError("Parameter 'normalize' should be 'max', 'min-max', or 'mean'.")

        # Step 2: Determine the reference sequence
        if reference_series is None:
            # Use the maximum value of each indicator as the reference sequence by default
            reference_series = data_normalized.max()
        else:
            # Normalize the reference sequence using the same method
            if normalize == 'max':
                reference_series = reference_series / data.max()
            elif normalize == 'min-max':
                reference_series = (reference_series - data.min()) / (data.max() - data.min())
            elif normalize == 'mean':
                reference_series = reference_series / data.mean()

        # Step 3: Calculate the grey relational coefficient
        # Compute the absolute difference matrix
        diff_matrix = abs(data_normalized - reference_series)

        # Compute the minimum and maximum differences
        delta_min = diff_matrix.min().min()
        delta_max = diff_matrix.max().max()

        # Compute the grey relational coefficient matrix
        relation_coefficient = (delta_min + rho * delta_max) / (diff_matrix + rho * delta_max)

        # Step 4: Calculate the grey relational grade with weights
        if weights is None:
            # Use equal weights if none are provided
            weights = np.ones(data.shape[1]) / data.shape[1]
        else:
            # Normalize the weights to sum to 1
            weights = np.array(weights)
            if len(weights) != data.shape[1]:
                raise ValueError("Length of weights must match the number of indicators.")
            weights = weights / weights.sum()

        # Convert weights to a DataFrame for multiplication
        weights_series = pd.Series(weights, index=data.columns)

        # Multiply the grey relational coefficients by the weights
        weighted_coefficients = relation_coefficient.mul(weights_series, axis=1)

        # Sum the weighted coefficients to get the grey relational grade
        relational_grade = weighted_coefficients.sum(axis=1)

        print("Grey Relational Grade:", relational_grade)

        return relational_grade

    def grey_relational_analysis_old(self, input, eps = 0.5, weight = None, normalization = 'default'): # eps is the ratio in range (1,0)
        input = pd.DataFrame(input)

        if len(input) == 1:
            return 
        
        # normalization
        if normalization == 'default':
            input = input/input.mean(axis = 0) # xi/mean
        elif normalization == 'zscore':
            input = StandardScaler().fit_transform(input)
        elif normalization == 'minmax':
            input = MinMaxScaler().fit_transform(input)
        elif normalization == 'reverse_minmax':
            '''fit for metric smaller the better'''
            input = self.reverse_min_max_normalizer(input)
        elif normalization == 'none':
            pass

        # GRA procedure
        input = pd.DataFrame(input)
        #print(input)
        X = input.iloc[:, 1:]
        Y = input.iloc[:, 0]
        absX0_X1 = abs(X.sub(Y, axis = 0))
        a = absX0_X1.min().min()
        b = absX0_X1.max().max()

        GRC = (a+eps*b)/(absX0_X1+eps*b)
        #print(GRC)
        if weight is None:
            #print(pd.DataFrame(GRC.mean(axis = 0)))
            return pd.DataFrame(GRC.mean(axis = 0)) # .sort_values(ascending=False)
        else:
            dict = {}
            for item in GRC:
                grey_relational_grade = 0
                for index, value in enumerate(GRC[item]):
                    grey_relational_grade += value*weight[index]/sum(weight)
                    dict[item] = grey_relational_grade
            #df = pd.DataFrame.from_dict(dict, orient='index', columns=['Value'])
            #print(dict)
            return dict #.sort_values(ascending=False, by = 'Value')