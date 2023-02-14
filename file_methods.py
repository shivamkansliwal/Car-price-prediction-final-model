import pickle
import os


class File_Operation:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory = 'D:\\Car_Price_Prediction\\models\\'

    def save_model(self, model, filename):
        """
            Method Name: save_model
            Description: Save the model file to the directory specified.

        :param model: Machine learning model object
        :param filename: Machine learning model name
        :return: File gets saved to the directory specified
        """
        self.logger_object.log(self.file_object, "Entered the save_model of File_Operation class.")
        try:
            path = os.path.join(self.model_directory)

            with open(path + '\\' + filename + '.sav',
                      'wb') as f:
                # save the model to file
                pickle.dump(model, f)

            self.logger_object.log(self.file_object, 'Model File ' + filename + ' saved. Exited the save_model method '
                                                                                'of the File_Operation class')

            return 'success'

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in save_model method of the File_Operation class. Exception '
                                   'message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' could not be saved. Exited the save_model method of '
                                                              'the File_Operation class')
            raise Exception()

    def load_model(self, filename):
        """

        :param filename:
        :return:
        """
        self.logger_object.log(self.file_object, "Entered the save_model of File_Operation class.")
        try:
            with open(self.model_directory + filename + '.sav', 'rb') as f:
                self.logger_object.log(self.file_object,
                                       'Model File ' + filename + 'loaded. Exited the load_model method of the '
                                                                  'Model_Finder class')
                return pickle.load(f)
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in load_model method of the Model_Finder class. Exception '
                                   'message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + 'could not be saved. Exited the load_model method of '
                                                              'the Model_Finder class')
            raise Exception()
