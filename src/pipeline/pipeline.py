from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def create_preprocessing_pipeline(X):

    categorical_cols = X.select_dtypes(
        include=['object']
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                StandardScaler(),
                numerical_cols
            ),
            (
                'cat',
                OneHotEncoder(
                    drop='first',
                    handle_unknown='ignore'
                ),
                categorical_cols
            )
        ]
    )

    return preprocessor