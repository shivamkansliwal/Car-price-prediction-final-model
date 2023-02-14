import pandas as pd
import numpy as np
import pickle

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


class Preprocessor:

    def __init__(self, file_object, logger_object):
        self.label_encoder = None
        self.data = None
        self.imputed_data = None
        self.imputed_array = None
        self.null_counts = None
        self.null_present = None
        self.label = None
        self.features = None
        self.file_object = file_object
        self.logger_object = logger_object

    def drop_unnecessary_columns(self, data, column_names):
        """
            Method Name: drop_unnecessary_columns
            Description: This method drops the unwanted columns

            :param data: A pandas dataframe
            :param column_names: List of columns that is required to drop
            :return: pandas dataframe with the specified list of columns removed
        """
        self.logger_object.log(self.file_object, "Entered the drop_unnecessary_columns method in Preprocessor class.")

        car_data = data.drop(column_names, axis=1)
        return car_data

    def is_null_present(self, data):
        """
            Method Name: is_null_present
            Description: This method checks whether there are null values present in the dataframe or not

        :param data: A pandas dataframe
        :return: A boolean value. True if null values are present in the dataframe, False otherwise

        On Failure: Raise Exception.
        """
        self.logger_object.log(self.file_object, "Entered the is_null_present method in Preprocessor class.")
        self.null_present = False

        try:
            self.null_counts = data.isna().sum()
            for null in self.null_counts:
                if null > 0:
                    self.null_present = True
                    break

            if self.null_present:
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null["columns"] = data.columns
                dataframe_with_null["missing value counts"] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv("null_values.csv")

            self.logger_object.log(self.file_object, "Finding null value is success!Exited the is_null_present method "
                                                     "in Preprocessor class")
            return self.null_present

        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in is_null_present method. Exception "
                                                     "message: " + str(e))
            self.logger_object.log(self.file_object, "Finding null value is un-successfully! Exited the "
                                                     "is_null_present method in Preprocessor class")
            raise Exception()

    def replace_invalid_values_with_null(self, data):
        """
            Method Name: replace_invalid_values_with_null
            Description: This method replaces the null values in each feature with np.NaN for further imputation

        :param data: A pandas dataframe
        :return: Pandas dataframe with the null values in each feature replaced with np.NaN
        """
        self.logger_object.log(self.file_object, "Entered the replace_invalid_values_with_null method in Preprocessor "
                                                 "class.")

        for column in data.columns:
            count = data[column][data[column] == 'null'].count()
            if count != 0:
                data[column] = data[column].replace('null', np.nan)
        return data

    def impute_missing_values(self, data):
        """
            Method Name: impute_missing_values
            Description: This method replaces all the missing values in the dataframe using KNN Imputer

        :param data: A pandas dataframe
        :return: A dataframe which has all the missing values imputed
        """
        self.logger_object.log(self.file_object, "Entered the impute_missing_values method in Preprocessor class.")
        try:
            imputer = KNNImputer(n_neighbors=3, weights="uniform", missing_values=np.nan)
            imputed_array = imputer.fit_transform(data)

            imputed_data = pd.DataFrame(data=np.round(imputed_array), columns=data.columns)
            self.logger_object.log(self.file_object, "Imputing missing values successful. Exited the "
                                                     "impute_missing_values method in Preprocessor class")

            return imputed_data

        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in impute_missing_values method. Exception "
                                                     "message: " + str(e))
            self.logger_object.log(self.file_object, "Imputing missing values failed! Exited the "
                                                     "impute_missing_values method in Preprocessor class")
            raise Exception()

    def encode_categorical_features(self, data):
        """
            Method Name: encode_categorical_features
            Description: This method encodes categorical features using LabelEncoder

        :param data: A Pandas dataframe
        :return: Pandas dataframe with categorical features encoded
        """
        self.logger_object.log(self.file_object, "Entered the encode_categorical_features method in Preprocessor class.")

        self.label_encoder = LabelEncoder()
        categorical_features = ["fuel", "seller_type", "transmission", "owner"]

        for column in categorical_features:
            self.label_encoder.fit(data[column])
            data[column] = self.label_encoder.transform(data[column])

        with open('D://Car_Price_Prediction//EncoderPickle//enc.pickle', 'wb') as file:
            pickle.dump(self.label_encoder, file)

        return data

    def separate_label_feature(self, data, label_column_name):
        """
            Method Name: separate_label_feature
            Description: This method separates the features and label column

        :param data: A pandas dataframe
        :param label_column_name: Name of the label column
        :return: Two separate dataframes, one containing  features and other containing labels

        On Failure: Raise Exception.
        """

        self.logger_object.log(self.file_object, "Entered the separate_label_feature method in Preprocessor class .")
        try:
            self.features = data.drop(labels=label_column_name, axis=1)
            self.label = data[label_column_name]
            self.logger_object.log(self.file_object, "Label Separation Successful. Exited the separate_label_feature "
                                                     "method in Preprocessor class.")

        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in separate_label_feature method. Exception "
                                                     "message: " + str(e))
            self.logger_object.log(self.file_object, "Label Separation UnSuccessful. Exited the "
                                                     "separate_label_feature method in Preprocessor class.")
            raise Exception()
