import datetime
import random
from unittest import TestCase

import numpy as np
import pandas as pd
from mlops_hw1.custom_transformer.transformer import MeanSubtractionTransformer


class TestTransformer(TestCase):
    SIZE_OF_DF = 100

    def return_test_dataframe(self):
        df_dict = {'continuous_col': [random.random() for _ in range(self.SIZE_OF_DF)],
                   'string_col': [random.choice(["male", "female"]) for _ in range(self.SIZE_OF_DF)],
                   'datetime_col': [datetime.datetime(2022, 11, 2) for _ in range(self.SIZE_OF_DF)]}

        df_dict['continuous_col'][35] = np.nan

        return pd.DataFrame.from_dict(df_dict)

    def test_have_incorrect_columns(self):
        df = self.return_test_dataframe()

        transformer = MeanSubtractionTransformer()

        transformed_df = transformer.fit_transform(df)

        self.assertTrue(abs(transformed_df['continuous_col'].mean() - 0) < 1e-9)
        self.assertTrue(all(df['string_col'] == transformed_df['string_col']))
        self.assertTrue(all(df['datetime_col'] == transformed_df['datetime_col']))
        self.assertTrue(all(df.columns == transformed_df.columns))



