import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocessing
from best_model_finder import model_tuner
from application_logging import logger
from file_operations import file_methods
from datetime import datetime


class Train_Model:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open('Training_Logs/ModelTrainingLog.txt', "a+")

    def train_model(self):
        start_time = datetime.now()
        self.log_writer.log(self.file_object, "Start of Training!! Start time: " + str(start_time))
        car_data = pd.read_csv("Car details.csv")

        try:
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            car_data = preprocessor.drop_unnecessary_columns(car_data, ["name", "year", "torque"])

            car_data = preprocessor.replace_invalid_values_with_null(car_data)

            # extract value of engine and mileage features
            car_data['engine'] = car_data['engine'].str.extract('([^\s]+)').astype(float)
            car_data['mileage'] = car_data['mileage'].str.extract('([^\s]+)').astype(float)

            # extract value of 'max_power' features
            car_data['max_power'] = car_data['max_power'].str.extract('([^\s]+)')
            car_data['max_power'] = car_data['max_power'][~(car_data['max_power'] == 'bhp')]
            car_data['max_power'] = car_data['max_power'].astype(float)

            is_null_present = preprocessor.is_null_present(car_data)

            car_data = preprocessor.encode_categorical_features(car_data)

            if is_null_present:
                car_data = preprocessor.impute_missing_values(car_data)

            x_data = car_data.drop(columns=['selling_price'])
            y_data = car_data["selling_price"]

            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 / 3, random_state=355)

            model_finder = model_tuner.Model_Finder(self.file_object, self.log_writer)
            best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)

            # saving the best model to the directory.
            file_op = file_methods.File_Operation(self.file_object, self.log_writer)
            save_model = file_op.save_model(best_model, best_model_name)

            end_time = datetime.now()
            self.log_writer.log(self.file_object, "Successful end of training!! End time: " + str(end_time))
            self.log_writer.log(self.file_object, "Training Duration: " + str(end_time - start_time))

        except Exception as e:
            self.log_writer.log(self.file_object, "Unsuccessful end of training")
            raise Exception()
