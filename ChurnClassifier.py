import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Logger import CustomLogger
import warnings

warnings.filterwarnings("ignore")

# Example usage:
logger = CustomLogger()


class MLClassifier:
    """
    Classifier model for ML tasks uses Logistic regression, random forest and XGBoost
    """

    def __init__(self, learning_rate=0.01, max_iter=100, n_estimators=100, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state

        # # Initialize models with class_weight or scale_pos_weight to handle imbalanced datasets
        self.logistic_model = LogisticRegression(C=1 / self.learning_rate, max_iter=self.max_iter,
                                                 class_weight='balanced', random_state=self.random_state)
        self.random_forest_model = RandomForestClassifier(n_estimators=self.n_estimators,
                                                          random_state=self.random_state, class_weight='balanced')
        self.xgboost_model = xgb.XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                               random_state=self.random_state, scale_pos_weight=1)
        self.catboost_model = CatBoostClassifier(iterations=self.max_iter, learning_rate=self.learning_rate,
                                                 depth=6, loss_function='Logloss', eval_metric='AUC', verbose=10,
                                                 auto_class_weights='Balanced')
        self.svc_model = SVC(C=1 / self.learning_rate, kernel='rbf', probability=True, cache_size=300,
                             class_weight='balanced', verbose=True, max_iter=self.max_iter,
                             random_state=self.random_state)
        self.scaler = MinMaxScaler()

        # self.scaler = StandardScaler()
        self.best_model = None

    def scale_features(self, X):
        """
        Applies MinMax Scaler normalization

        Parameters
        ----------
        X : features

        Returns
        -------
        normalized features

        """
        try:
            return self.scaler.fit_transform(X)
            logger.log('Successfully performed standardization of dataset')
        except Exception as e:
            raise ValueError(f"Error in normailizing the data: {e}")
            logger.log("An error was raised while attempting to perform standardization ", level='ERROR')

    def train(self, X, y):
        """
        Trains logistic regression model, random forrest, xgboost and catboost classifier

        Parameters
        ----------
        X : features
        y : target

        Returns
        -------
        trained model instances

        """
        try:
            X_scaled = self.scale_features(X)
            self.logistic_model.fit(X_scaled, y)
            logger.log("Successfully trained the Logistic Regression Classifier Model!")
            self.random_forest_model.fit(X, y)
            logger.log("\nSuccessfully trained the Random Forrest Classifier Model!")
            self.xgboost_model.fit(X, y)
            logger.log("\nSuccessfully trained the XGBoost Classifier Model!")
            self.svc_model.fit(X, y)
            logger.log("\nSuccessfully trained the Support Vector Classifier Model!")
            # Create Pool object for CatBoost
            train_pool = Pool(data=X, label=y)
            self.catboost_model.fit(train_pool)
            # self.catboost_model.fit(X,y,  eval_set=(X_valid, y_valid),early_stopping_rounds=50,use_best_model=True)
            logger.log('Successfully the CatBoost Classifier Model!')
        except Exception as e:
            raise ValueError(f"Error in training data: {e}")
            logger.log("An error was raised while attempting to train one or ML models ", level='ERROR')

    def predict(self, X):
        """

        :param X:
        :return:
        """
        try:
            X_scaled = self.scaler.transform(X)
            if self.best_model == 'logistic':
                logger.log('Selected Logistic Regression Classifier Model as best model!')
                return self.logistic_model.predict(X_scaled)

            elif self.best_model == 'random_forest':
                logger.log('Selected Random Forrest Classifier Model as best model!')
                return self.random_forest_model.predict(X)
            elif self.best_model == 'catboost':
                logger.log('Selected CatBoost Classifier Model as best model!')
                return self.catboost_model.predict(X)
            elif self.best_model == 'support_vector':
                logger.log('Selected Support Vector Classifier Model as best model!')
                return self.svc_model.predict(X)
            else:
                logger.log('Selected XGBoost Classifier Model as best model!')
                return self.xgboost_model.predict(X)
        except Exception as e:
            raise ValueError(f"Error while performing the prediction step: {e}")
            logger.log("An error was raised during the prediction step", level='ERROR')

    def evaluate(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        try:
            # Evaluate logistic regression
            X_scaled = self.scaler.transform(X)
            y_pred_logistic = self.logistic_model.predict(X_scaled)
            y_pred_proba_logistic = self.logistic_model.predict_proba(X_scaled)[:, 1]
            auc_logistic = roc_auc_score(y, y_pred_proba_logistic)

            # Evaluate random forest
            y_pred_rf = self.random_forest_model.predict(X)
            y_pred_proba_rf = self.random_forest_model.predict_proba(X)[:, 1]
            auc_rf = roc_auc_score(y, y_pred_proba_rf)

            # Evaluate XGBoost
            y_pred_xgb = self.xgboost_model.predict(X)
            y_pred_proba_xgb = self.xgboost_model.predict_proba(X)[:, 1]
            auc_xgb = roc_auc_score(y, y_pred_proba_xgb)

            # Evaluate SVC
            y_pred_svc = self.svc_model.predict(X)
            y_pred_proba_svc = self.svc_model.predict_proba(X)[:, 1]
            auc_svc = roc_auc_score(y, y_pred_proba_svc)

            # Evaluate CatBoost
            y_pred_cb = self.catboost_model.predict(X)
            y_pred_proba_cb = self.catboost_model.predict_proba(X)[:, 1]
            auc_cb = roc_auc_score(y, y_pred_proba_cb)

            # Choose the best model based on ROC AUC score
            auc_scores = {'logistic': auc_logistic, 'random_forest': auc_rf, 'xgboost': auc_xgb, 'catboost': auc_cb,
                          'support_vector': auc_svc}
            self.best_model = max(auc_scores, key=auc_scores.get)

            if self.best_model == 'logistic':
                y_pred = y_pred_logistic
                y_pred_proba = y_pred_proba_logistic
                auc = auc_logistic
            elif self.best_model == 'random_forest':
                y_pred = y_pred_rf
                y_pred_proba = y_pred_proba_rf
                auc = auc_rf
            elif self.best_model == 'catboost':
                y_pred = y_pred_cb
                y_pred_proba = y_pred_proba_cb
                auc = auc_cb

            elif self.best_model == 'support_vector':
                y_pred = y_pred_svc
                y_pred_proba = y_pred_proba_svc
                auc = auc_svc

            else:
                y_pred = y_pred_xgb
                y_pred_proba = y_pred_proba_xgb
                auc = auc_xgb

            accuracy = accuracy_score(y, y_pred)
            conf_matrix = confusion_matrix(y, y_pred)
            class_report = classification_report(y, y_pred)
            fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

            evaluation_results = {
                "best_model": self.best_model,
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix,
                "Classification_report": class_report,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": auc
            }
            logger.log('Successfully saved the evaluation result as a dictionary object')
            logger.log(evaluation_results)

            return evaluation_results

        except Exception as e:
            raise ValueError(f"Error while performing the model evaluation step: {e}")
            logger.log("An error was raised during the evaluation step", level='ERROR')

    def find_optimal_clusters(self, X, max_k=5):
        """

        :param X:
        :param max_k:
        :return:
        """
        try:
            X_scaled = self.scale_features(X)
            self.wcss = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, init="k-means++", n_init=12, random_state=self.random_state)
                kmeans.fit(X_scaled)
                self.wcss.append(kmeans.inertia_)

            diffs = np.diff(self.wcss)
            diffs_ratio = diffs[:-1] / diffs[1:]
            optimal_k = np.argmin(diffs_ratio) + 2  # +2 because of zero-based indexing and the diff shifts results by 1
            logger.log('Successfully found the optimal count of clusters')
            logger.log(optimal_k)
            return optimal_k
        except Exception as e:
            raise ValueError(f"Error while finding the optimal number of clusters : {e}")
            logger.log("An error was raised while finding the optimal number of clusters", level='ERROR')

    def plot_elbow_curve(self, max_k=5):
        """

        :param max_k:
        :return:
        """
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, max_k + 1), self.wcss, marker='o')
            plt.title('Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
            plt.xticks(range(1, max_k + 1))
            plt.grid(True)
            plt.show(block=False)
            logger.log('Successfully plotted the elbow curve  the optimal count of clusters')
        except Exception as e:
            raise ValueError(f"Error while rendering the Elbow plot : {e}")
            logger.log("An error was raised while rendering the Elbow plot", level='ERROR')

    def find_clusters(self, X, n_clusters=4):
        """

        :param X:
        :param n_clusters:
        :return:
        """
        try:
            X_scaled = self.scale_features(X)
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=12, random_state=self.random_state)
            logger.log('Successfully trained cluster algorthim with the optimal count of clusters')
            return kmeans.fit_predict(X_scaled)
        except Exception as e:
            raise ValueError(f"Error while finding the number of clusters : {e}")
            logger.log("An error was raised while finding the number of clusters", level='ERROR')

    def split_train_test(self, X, y, test_size=0.2):
        """

        :param X:
        :param y:
        :param test_size:
        :return:
        """
        try:
            logger.log('Successfully splitted the dataset into train-test sets')
            return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        except Exception as e:
            raise ValueError(f"Error while splitting the dataset to train-test sets : {e}")
            logger.log("An error was raised while splitting the dataset to train-test sets", level='ERROR')

    def plot_confusion_matrix(self, conf_matrix, cmap='viridis'):
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show(block=False)
        except Exception as e:
            raise ValueError(f"Error while plotting the confusion matrix : {e}")
            logger.log("An error was raised while plotting the confusion matrix", level='ERROR')


    def plot_roc_curve(self, fpr, tpr):
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', label='ROC Curve')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            plt.show(block=False)
        except Exception as e:
            raise ValueError(f"Error while plotting the Receiver Operating Characteristic (ROC) Curve : {e}")
            logger.log("An error was raised while plotting the Receiver Operating Characteristic (ROC) Curve", level= 'ERROR')


    def predict_churn_probability(self, X):
        """
        :param X:
        :return:
        """
        try:
            X_scaled = self.scaler.transform(X)
            if self.best_model == 'logistic':
                y_pred_proba = self.logistic_model.predict_proba(X_scaled)[:, 1]
            elif self.best_model == 'random_forest':
                y_pred_proba = self.random_forest_model.predict_proba(X)[:, 1]

            elif self.best_model == 'catboost':
                y_pred_proba = self.catboost_model.predict_proba(X)[:, 1]

            elif self.best_model == 'xgboost':
                y_pred_proba = self.xgboost_model.predict_proba(X)[:, 1]

            elif self.best_model == 'support_vector':
                y_pred_proba = self.svc_model.predict_proba(X)[:, 1]
            else:
                pass
            logger.log('Successfully splitted the dataset into train-test sets')
            return y_pred_proba
        except Exception as e:
            raise ValueError(f"Error while predicting churn probability : {e}")
            logger.log("An error was raised while predicting churn probability", level='ERROR')


def plot_class_distribution(df, target):
    """
    Plots the distribution of the target variable.

    Parameters:
    - df: DataFrame, the input dataframe containing the target variable.
    - target: str, the name of the target variable column.

    Returns:
    - None
    """
    # Plot count distribution
    try:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.countplot(x=target, data=df)
        plt.title('Count Plot: Class Distribution of Target Variable')
        plt.xlabel('Class')
        plt.ylabel('Count')

        # Plot pie chart
        plt.subplot(1, 2, 2)
        df[target].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
        plt.title('Pie Plot : Class Distribution of Target Variable')
        plt.ylabel('')

        plt.tight_layout()
        plt.show(block=False)
        logger.log("Successfully rendered distribution plots of target variable")
    except Exception as e:
        #raise ValueError(f"Error while rendering distribution plots of target variable : {e}")
        #logger.log("An error was raised while rendering distribution plots of target variable", level='ERROR')

 def tune_parameters(self, X, y, model_name, param_grid, search_type='grid', cv=5):
        """
        Tune parameters of the specified model using GridSearchCV or RandomizedSearchCV.

        Parameters:
        - X: Features
        - y: Target variable
        - model_name: Name of the model to tune ('logistic', 'random_forest', 'xgboost')
        - param_grid: Dictionary of parameters to search
        - search_type: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        - cv: Number of cross-validation folds

        Returns:
        - Best parameters found
        """
        if model_name == 'logistic':
            model = self.logistic_model
        elif model_name == 'random_forest':
            model = self.random_forest_model
        elif model_name == 'xgboost':
            model = self.xgboost_model
        elif model_name == 'support_vector':
            model = self.svc_model
        elif model_name == 'catboost':
            model = self.catboost_model
        else:
            raise ValueError("Invalid model_name. Choose from 'logistic', 'random_forest', 'xgboost', 'support_vector', 'catboost'.")

        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        elif search_type == 'random':
            search = RandomizedSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=self.random_state)
        else:
            raise ValueError("Invalid search_type. Choose 'grid' or 'random'.")

        X_scaled = self.scale_features(X)
        search.fit(X_scaled, y)

        best_params = search.best_params_
        best_score = search.best_score_

        if model_name == 'logistic':
            self.logistic_model = search.best_estimator_
        elif model_name == 'random_forest':
            self.random_forest_model = search.best_estimator_
        elif model_name == 'xgboost':
            self.xgboost_model = search.best_estimator_

        return best_params, best_score


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hi PyCharm')
