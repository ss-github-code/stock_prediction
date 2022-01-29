import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and 
# uses averaging to improve the predictive accuracy and control over-fitting.
class AlgoRandomForest:
    def __init__(self, data_handler):
        self.data_handler = data_handler

        X_train, y_train = data_handler.X_train, data_handler.y_train
        X_val, y_val = data_handler.X_val, data_handler.y_val

        # grid search for random forest hyper parameters
        grid = {'n_estimators' : [200, 300, 500], 'max_depth' : [3],
                'max_features' : [4,8], 'random_state' : [0]
               }

        val_scores = []
        rf_model = RandomForestRegressor()
        for g in ParameterGrid(grid):
            rf_model.set_params(**g)
            rf_model.fit(X_train, y_train)
            val_scores.append(rf_model.score(X_val, y_val))
        
        best_index = np.argmax(val_scores) # If .score() changed to a metric, check this for change to argmin
        best_params = ParameterGrid(grid)[best_index]

        rf_model.set_params(**best_params) # set the best params using unpack operator
        rf_model.fit(X_train, y_train)     # fit the training data
        self.rf_model = rf_model

    def get_forecasts(self):
        rf_model = self.rf_model
        data_handler = self.data_handler

        X_test = data_handler.X_test
        X_val = data_handler.X_val

        # create validation dataframe for 'val_pred' and 'val_actual'
        val_forecasts = rf_model.predict(X_val)
        val_forecasts  = pd.DataFrame({'predicted_mean': val_forecasts})
        test_forecasts = rf_model.predict(X_test)
        test_forecasts  = pd.DataFrame({'predicted_mean': test_forecasts})
        forecasts = pd.concat([val_forecasts, test_forecasts])
        return forecasts['predicted_mean']