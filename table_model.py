'''
Prepare Pandas DataFrame format for Keras.
'''
import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class ExtractColumn(BaseEstimator, TransformerMixin):
    '''
    Pick a DataFrame apart by column.
    '''

    def __init__(self, column_or_series_name):
        '''
        Parameters
        ----------
        column_or_series_name : str
            Key into the data frame to extact a column or series.
        '''
        self._column_or_series_name = column_or_series_name

    def fit(self, X):
        '''
        No specific need to fit.
        '''
        return self

    def transform(self, data_frame):
        '''
        Parameters
        ----------
        data_frame : Pandas DataFrame
        '''
        return data_frame[self._column_or_series_name]

    def __repr__(self):
        '''
        Repr shows the column name for easier debugging.
        '''
        return "ExtractColumn('{0}')".format(self._column_or_series_name)


class PercentageColumn(BaseEstimator, TransformerMixin):
    '''
    Turn a string percentage column in a Pandas DataFrame like 12.5% into numerical values
    like 0.125.
    '''

    def fit(self, data_frame):
        '''
        No specific need to fit.
        '''
        return self

    def transform(self, X):
        '''
        Transform a column of data into numerical percentage values.

        Parameters
        ----------
        X : pandas series or numpy array
        '''

        def percent(x):
            '''
            A percentage as a number, or a NaN.
            '''
            if x:
                return float(x.replace('%', '')) / 100.0
            else:
                return np.nan

        return np.array(X.map(percent)).reshape(-1, 1)


class NumericColumn(BaseEstimator, TransformerMixin):
    '''
    Take a numeric value column and standardize it.
    '''

    def __init__(self):
        '''
        Set up the internal transformation.
        '''
        self._transformer = StandardScaler()

    def fit(self, X):
        '''
        Fit the standardization.
        '''
        as_feature_array = np.array(X).reshape(-1, 1)
        zeroed = np.nan_to_num(as_feature_array)
        self._transformer.fit(zeroed)
        return self

    def transform(self, X):
        '''
        Transform a column of data into numerical percentage values.

        Parameters
        ----------
        X : pandas series or numpy array
        '''
        as_feature_array = np.array(X).reshape(-1, 1)
        zeroed = np.nan_to_num(as_feature_array)
        return self._transformer.transform(zeroed)


class CategoricalColumn(BaseEstimator, TransformerMixin):
    '''
    Take a string or key categorical column and transform it
    to one hot encodings.
    '''

    def __init__(self):
        '''
        Set up the internal transformation.
        '''
        self._labeler = LabelEncoder()
        self._encoder = OneHotEncoder()

    def fit(self, X):
        '''
        Fit the label and encoding
        '''
        handle_none = list(map(str, X))
        encoded = self._labeler.fit_transform(handle_none)
        self._encoder.fit(encoded.reshape(-1, 1))
        return self

    def transform(self, X):
        '''
        Transform a column of data into one hot encodings.

        Parameters
        ----------
        X : pandas series or numpy array
        '''
        handle_none = list(map(str, X))
        encoded = self._labeler.transform(handle_none)
        return self._encoder.transform(encoded.reshape(-1, 1)).todense()


class TableModel(BaseEstimator):
    '''
    Base TableModel, this deals with:
    * Specifying transformers at the column/series level for input
        (X) and output (Y) features
    * Combining transformed columns into a feature tensor
    * An override point to build a Keras model
    * Loading and Saving trained weights

    This is an in memory model, and will preserve the original DataFrame.

    >>> import table_model
    >>> import pandas as pd
    >>> data = pd.DataFrame({'a': ['1%', '99%'], 'b': [1, 2], 'c': ['aa', 'bb'], 'd': [True, False]})
    >>> model = table_model.TableModel( \
            transformers={ \
                'a': table_model.PercentageColumn(), \
                'b': table_model.NumericColumn(), \
                'c': table_model.CategoricalColumn(), \
                'd': table_model.CategoricalColumn(), \
            }, \
            output_name='d' \
        )
    >>> X, Y = model.transform_data(data)
    >>> X
    matrix([[ 0.01, -1.  ,  1.  ,  0.  ],
            [ 0.99,  1.  ,  0.  ,  1.  ]])
    >>> Y
    matrix([[ 0.,  1.],
            [ 1.,  0.]])
    '''

    def __init__(self, transformers={}, output_name=None):
        '''

        Initialize the model with column/series mappings, only those columns
        specified will be used in the model.

        Parameters
        ----------
        transformers : dict
            Mapping from column/series name to scikit learn style transformer.
        output_name : str
            Name of a column/series that will be the output.
        '''
        try:
            self._output = make_pipeline(
                ExtractColumn(output_name),
                transformers[output_name]
            )
            transformers = transformers.copy()
            del transformers[output_name]
        except KeyError:
            raise KeyError('{0} has no transformer'.format(output_name))
        # each column is an extraction and then a transformation
        pipelines = [(name, make_pipeline(ExtractColumn(name), transformer))
                     for name, transformer in transformers.items()]
        self._transformers = FeatureUnion(pipelines, n_jobs=os.cpu_count())

    def transform_data(self, data_frame, verbose=True):
        '''
        Fit and model based on the passed Pandas DataFrame.
        This will:
        * Fit all column/series
        * Transform all column / series into a tensor
        * Transform the output column / series into a tensor

        Subclasses that classify or regress call this to get (X, Y) data
        and then pass along to a keras model.

        '''
        return (
            self._transformers.fit_transform(data_frame),
            self._output.fit_transform(data_frame)
        )
