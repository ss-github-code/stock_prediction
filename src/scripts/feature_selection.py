import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import shap
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
'''
Feature selection is done using XGBoost and SHAP.
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. 
It implements machine learning algorithms under the Gradient Boosting framework. 
XGBoost provides a parallel tree boosting (also known as GBDT) that solve many data science problems in a fast and accurate way.

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. 
It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.
'''
class FeatureSelector:
    def __init__(self, data_handler, max_num_features, show_plot):
        self.data_handler = data_handler
        self.select_features_using_xgboost(max_num_features, show_plot)

    def select_features_using_xgboost(self, max_num_features, show_plot):
        data_handler = self.data_handler
        scaler = MinMaxScaler()

        X_train, y_train = data_handler.X_train, data_handler.y_train
        X_val, y_val = data_handler.X_val, data_handler.y_val

        y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).reshape(-1)
        y_val = scaler.transform(y_val.values.reshape(-1, 1)).reshape(-1)

        gsearch_params={
             # colsample_bytree: the fraction of features (randomly selected) that will be used to train each tree.
             # gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree.
            'max_depth': [1,2,3,4], 'learning_rate': [0.001, 0.005, 0.01], 'colsample_bytree': [0.5, 0.75],
            'n_estimators': [250, 500, 750], 'objective': ['reg:squarederror'], 'gamma':[0, 0.1, 0.2]
        }

        param_grid = ParameterGrid(gsearch_params)
        best_score, best_score_train = (-10000, -10000)
        best_params, best_params_train = (0, 0)       
        
        print('Start: gridsearch using xgboost')

        for params in param_grid:  # iterate params until best score is found
            # print(f"Testing {params}...")

            # init xgboost regressor on each param set
            xgb_regressor = xgb.XGBRegressor(**params)
            trained_model = xgb_regressor.fit(X_train, y_train)
            
            val_score = trained_model.score(X_val, y_val)
            train_score = trained_model.score(X_train, y_train)
            #print(f"Test Score: {test_score}")
            #print(f"Train Score: {train_score}")
            
            if val_score > best_score:
                best_score = val_score
                best_params = params
                best_train = train_score
                best_model = trained_model

        print(f"Best VALIDATION R^2 is {best_score} with params: {best_params}")
        print(f"TRAIN R^2 for best test params is {best_train}")
        xgb_best = best_model

        #plot_res(xgb_best.predict(X_val), y_val)
        feature_importance_df = get_shap_importances(xgb_best, X_val, data_handler.features, show_plot)
        feature_importance_dict = feature_importance_df.abs().sum().sort_values(ascending=False).to_dict() # sum over columns
        
        self.important_features = [] # choose the most important features based on the sum of absolute SHAP values
        max_feature_length = max_num_features
        for key, val in feature_importance_dict.items():
            if val <= 0 or len(self.important_features) > (max_feature_length-1):
                break  
            self.important_features.append(key)
        return

# Tree SHAP feature importance generation
def get_shap_importances(model, X_val, features, show_plot=False):
    """Generates feature importances based on tree shap"""
    
    # intialize xgb regressor with best params
    # initialize treeshap explainer with fitted model
    explainer = shap.TreeExplainer(model)
    
    # predict test data with  the model's explainer
    shap_values = explainer.shap_values(X_val)
    # create summary feature importance chart
    shap.summary_plot(shap_values, X_val, plot_type="bar", max_display=20, show=show_plot) 
    
    feature_importance_df = pd.DataFrame(shap_values, columns=features)
    return feature_importance_df

def plot_res(y_hat, y_val):
    # Helper function used for plotting residuals during training and testing
    rolling_window = 7

    y_hat_rolling = pd.DataFrame(y_hat).rolling(rolling_window).mean()
    y_val_rolling = pd.DataFrame(y_val).rolling(rolling_window).mean()
    
    n = range(len(y_hat_rolling))
    
    plt.figure(figsize=(15, 6))
    plt.plot(n, y_val_rolling.iloc[:,0], 
             color='red', label='y_test', alpha=0.5)
    plt.plot(y_hat_rolling.iloc[:,0],
             color='black', label='y_pred', alpha=0.8)

    plt.legend()
    plt.show()