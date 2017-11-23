'''
Tools to transform data frames for machine learning.
'''
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class PercentageColumn(BaseEstimator, TransformerMixin):
    '''
    Turn a string percentage column in a Pandas DataFrame like 12.5% into numerical values
    like 0.125.
    '''

    def __init__(self, column_names=[]):
        '''
        Parameters
        ----------
        column_names: list
            Columns (series) that will be replaced.
        '''
        self._column_names = column_names

    def fit(self, data_frame):
        '''
        No specific need to fit.
        '''
        return self

    def transform(self, data_frame):
        '''
        Transform each target columns/series and return a copied DataFrame
        with percentage values.
        '''
        # modifying in place, so get a fresh copy to allow idempotent runs
        data_frame = data_frame.copy()

        def scrub(x):
            '''
            A percentage as a number, or a NaN
            '''
            if x:
                return float(x.replace('%', '')) / 100.0
            else:
                return np.nan

        for name in self._column_names:
            data_frame[name] = data_frame[name].map(scrub)

        return data_frame


class ClassifierColumnSet(BaseEstimator, TransformerMixin):
    '''
    Given a Pandas DataFrame, convert it into a dense numpy array for machine learning.

    This classifier can then be pickled or stored for subsequent use in a prediction service.
    '''

    def __init__(self, input_categorical=[], input_numerical=[], output_categorical=None):
        '''
        Parameters
        ----------
        input_categorical: list
            Column names with categorical data will be one-hot encoded
        input_numerical: list
            Column names with numerical data that will be scaled
        output_categorical: str
            Single column name for the output that will be one-hot encoded

        '''
        self._input_label_encoders = {
            name: LabelEncoder() for name in input_categorical
        }
        self._input_one_hot_encoders = {
            name: OneHotEncoder() for name in input_categorical
        }
        self._input_scalers = {
            name: StandardScaler() for name in input_numerical
        }
        self._output_categorical = output_categorical
        self._output_label_encoder = LabelEncoder()
        self._output_one_hot_encoder = OneHotEncoder()

    def fit(self, data_frame):
        '''
        Given a data frame, fit all encoders and scalers.
        '''
        for name, label_encoder in self._input_label_encoders.items():
            handle_none = list(map(str, data_frame[name]))
            encoded = label_encoder.fit_transform(handle_none)
            self._input_one_hot_encoders[name].fit(encoded.reshape(-1, 1))
        for name, scaler in self._input_scalers.items():
            as_feature_array = np.array(data_frame[name]).reshape(-1, 1)
            zeroed = np.nan_to_num(as_feature_array)
            scaler.fit(zeroed)
        if self._output_categorical:
            handle_none = list(map(str, data_frame[self._output_categorical]))
            encoded = self._output_label_encoder.fit_transform(handle_none)
            self._output_one_hot_encoder.fit(encoded.reshape(-1, 1))
        return self

    def transform(self, data_frame):
        '''
        Given a data frame, transform it to an encoded state.

        Returns
        -------
        A tuple (X, Y) of inputs and outputs each packaged into a 2d array.
        '''
        components = []
        for name, label_encoder in self._input_label_encoders.items():
            handle_none = list(map(str, data_frame[name]))
            encoded = label_encoder.transform(handle_none)
            components.append(self._input_one_hot_encoders[name].transform(encoded.reshape(-1, 1)).todense())
        for name, scaler in self._input_scalers.items():
            as_feature_array = np.array(data_frame[name]).reshape(-1, 1)
            zeroed = np.nan_to_num(as_feature_array)
            components.append(scaler.transform(zeroed))
        X = np.concatenate(components, 1)
        if self._output_categorical:
            handle_none = list(map(str, data_frame[self._output_categorical]))
            encoded = self._output_label_encoder.transform(handle_none)
            Y = self._output_one_hot_encoder.transform(encoded.reshape(-1, 1))
        return (X, Y)
