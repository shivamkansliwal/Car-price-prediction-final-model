from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import numpy as np


class Model_Finder:
    def __init__(self, file_object, logger_object):

        self.file_object = file_object
        self.logger_object = logger_object

        self.random_forest = RandomForestRegressor()
        self.prediction_random_forest = None
        self.random_forest_score = None
        self.max_features = None
        self.min_samples_leaf = None
        self.min_samples_split = None
        self.max_depth = None
        self.criterion = None
        self.n_estimators = None
        self.random_forest_grid = None
        self.param_grid = None

        self.knn = KNeighborsRegressor()
        self.param_grid_knn = None
        self.p = None
        self.n_neighbors = None
        self.leaf_size = None
        self.algorithm = None
        self.knn_grid = None
        self.knn_score = None
        self.prediction_knn = None

        self.decision_tree = DecisionTreeRegressor()
        self.decision_tree_grid = None
        self.decision_tree_score = None
        self.prediction_decision_tree = None

    def get_best_params_for_random_forest(self, train_data, test_data):
        """
            Method Name: get_best_params_for_random_forest
            Description: The method is used for obtaining the parameters for random forest which gives the best accuracy

        :param train_data: Dataset to be used for model training
        :param test_data: Dataset to be used for model testing
        :return: The model with best parameter
        """
        self.logger_object.log(self.file_object, "Entered the get_best_params_for_random_forest method in "
                                                 "Model_Finder class.")
        try:
            self.param_grid = {"n_estimators": [int(x) for x in np.linspace(start=100, stop=120, num=12)],
                               "criterion": ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
                               "max_depth": [int(x) for x in np.linspace(5, 10, num=4)],
                               "min_samples_split": [2, 5, 10],
                               "min_samples_leaf": [1, 2],
                               "max_features": ["sqrt", "log2"]}

            self.random_forest_grid = RandomizedSearchCV(estimator=self.random_forest,
                                                         param_distributions=self.param_grid, cv=5,
                                                         verbose=3)
            self.random_forest_grid.fit(train_data, test_data)

            self.n_estimators = self.random_forest_grid.best_params_["n_estimators"]
            self.criterion = self.random_forest_grid.best_params_["criterion"]
            self.max_depth = self.random_forest_grid.best_params_["max_depth"]
            self.min_samples_split = self.random_forest_grid.best_params_["min_samples_split"]
            self.min_samples_leaf = self.random_forest_grid.best_params_["min_samples_leaf"]
            self.max_features = self.random_forest_grid.best_params_["max_features"]

            self.random_forest = RandomForestRegressor(n_estimators=self.n_estimators, criterion=self.criterion,
                                                       max_depth=self.max_depth,
                                                       min_samples_split=self.min_samples_split,
                                                       min_samples_leaf=self.min_samples_leaf,
                                                       max_features=self.max_features)
            self.random_forest.fit(train_data, test_data)
            self.logger_object.log(self.file_object, "Random Forest best params: " + str(
                self.random_forest_grid.best_params_) + ". Exited the get_best_params_for_random_forest method")

            return self.random_forest

        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_best_params_for_random_forest method "
                                                     "in Model_Finder class. "
                                                     "Exception "
                                                     "message: " + str(e))
            self.logger_object.log(self.file_object, "Random Forest model tuning failed. Exited the "
                                                     "get_best_params_for_random_forest method.")
            raise Exception()

    def get_best_params_for_KNN(self, train_data, test_data):
        """
            Method Name: get_best_params_for_KNN
            Description: The method is used for obtaining the parameters for knn which gives the best accuracy

        :param train_data: Dataset to be used for model training
        :param test_data: Dataset to be used for model testing
        :return: The model with best parameter
        """
        self.logger_object.log(self.file_object, "Entered the get_best_params_for_KNN method in Model_Finder class")
        try:
            self.param_grid_knn = {
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 17, 24, 28, 30, 35],
                'n_neighbors': [4, 5, 8, 10, 11],
                'p': [1, 2]
            }
            self.knn_grid = RandomizedSearchCV(estimator=self.knn, param_distributions=self.param_grid_knn, cv=5,
                                               verbose=3)
            self.knn_grid.fit(train_data, test_data)

            self.algorithm = self.knn_grid.best_params_['algorithm']
            self.leaf_size = self.knn_grid.best_params_['leaf_size']
            self.n_neighbors = self.knn_grid.best_params_['n_neighbors']
            self.p = self.knn_grid.best_params_['p']

            self.knn = KNeighborsRegressor(algorithm=self.algorithm, leaf_size=self.leaf_size,
                                           n_neighbors=self.n_neighbors, p=self.p)

            self.knn.fit(train_data, test_data)
            self.logger_object.log(self.file_object, "KNNRegressor best params: "
                                   + str(self.knn_grid.best_params_) + ". Exited the get_best_params_for_KNN method.")

            return self.knn

        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_best_params_for_KNN method "
                                                     "in Model_Finder class. "
                                                     "Exception "
                                                     "message: " + str(e))
            self.logger_object.log(self.file_object, "KNN model tuning failed. Exited the "
                                                     "get_best_params_for_KNN method.")
            raise Exception()

    def get_best_params_for_decision_tree(self, train_data, test_data):
        """
            Method Name: get_best_params_for_decision_tree
            Description: The method is used for obtaining the parameters for decision tree which gives the best accuracy

        :param train_data: Dataset to be used for model training
        :param test_data: Dataset to be used for model testing
        :return: The model with best parameter
        """
        self.logger_object.log(self.file_object, "Entered the get_best_params_for_decision_tree method in "
                                                 "Model_Finder class")
        try:
            self.param_grid = {"criterion": ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
                               "max_depth": [int(x) for x in np.linspace(5, 10, num=4)],
                               "min_samples_split": [2, 5, 10],
                               "min_samples_leaf": [1, 2],
                               "max_features": ["sqrt", "log2"]}

            self.decision_tree_grid = RandomizedSearchCV(estimator=self.decision_tree,
                                                         param_distributions=self.param_grid, cv=5,
                                                         verbose=3)
            self.decision_tree_grid.fit(train_data, test_data)

            self.criterion = self.decision_tree_grid.best_params_["criterion"]
            self.max_depth = self.decision_tree_grid.best_params_["max_depth"]
            self.min_samples_split = self.decision_tree_grid.best_params_["min_samples_split"]
            self.min_samples_leaf = self.decision_tree_grid.best_params_["min_samples_leaf"]
            self.max_features = self.decision_tree_grid.best_params_["max_features"]

            self.decision_tree = DecisionTreeRegressor(criterion=self.criterion,
                                                       max_depth=self.max_depth,
                                                       min_samples_split=self.min_samples_split,
                                                       min_samples_leaf=self.min_samples_leaf,
                                                       max_features=self.max_features)
            self.decision_tree.fit(train_data, test_data)
            self.logger_object.log(self.file_object, "Decision Tree best params: " + str(
                self.decision_tree_grid.best_params_) + ". Exited the get_best_params_for_decision_tree method")

            return self.decision_tree

        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_best_params_for_decision_tree method "
                                                     "in Model_Finder class. "
                                                     "Exception "
                                                     "message: " + str(e))
            self.logger_object.log(self.file_object, "Decision Tree model tuning failed. Exited the "
                                                     "get_best_params_for_decision_tree method.")
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
            Method Name: get_best_model
            Description: The method is used to find out the model with best r2_score

        :param train_x: Training dataset containing independent features
        :param train_y: Training dataset containing dependent feature
        :param test_x:  Test dataset containing independent features
        :param test_y:  Test dataset containing dependent feature
        :return: The best model name and model object
        """
        self.logger_object.log(self.file_object, "Entered the get_best_model method in Model_Finder class")
        try:
            self.random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            self.prediction_random_forest = self.random_forest.predict(test_x)
            self.random_forest_score = r2_score(test_y, self.prediction_random_forest)
            self.logger_object.log(self.file_object, "Random Forest r2_score: " + str(self.random_forest_score))

            self.knn = self.get_best_params_for_KNN(train_x, train_y)
            self.prediction_knn = self.knn.predict(test_x)
            self.knn_score = r2_score(test_y, self.prediction_knn)
            self.logger_object.log(self.file_object, "KNN r2_score: " + str(self.knn_score))

            self.decision_tree = self.get_best_params_for_decision_tree(train_x, train_y)
            self.prediction_decision_tree = self.decision_tree.predict(test_x)
            self.decision_tree_score = r2_score(test_y, self.prediction_decision_tree)
            self.logger_object.log(self.file_object, "Decision Tree r2_score: " + str(self.decision_tree_score))

            if (self.random_forest_score > self.decision_tree_score) and (self.random_forest_score > self.knn_score):
                return "Random Forest", self.random_forest
            elif (self.decision_tree_score > self.random_forest_score) and (self.decision_tree_score > self.knn_score):
                return "Decision Tree", self.decision_tree
            else:
                return "KNN", self.knn

        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_best_model method in Model_Finder "
                                                     "class. Exception "
                                                     "message: " + str(e))
            self.logger_object.log(self.file_object, "Model Selections Failed! Exited the "
                                                     "get_best_model method ")
            raise Exception()
