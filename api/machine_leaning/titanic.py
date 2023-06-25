import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm
from lightgbm import LGBMClassifier
import pickle
import warnings

warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)


class GenerateModel:
    base_dir = Path(__file__).resolve().parent

    def __init__(self):
        pass

    @classmethod
    def __read_data(cls) -> pd.DataFrame:
        path_to_data = cls.base_dir / "data" / "train.csv"
        return pd.read_csv(path_to_data)

    @staticmethod
    def __extract_requierd_columns(df: pd.DataFrame):
        required_columns = [
            "Survived",
            "Sex",
            "Pclass",
            "Age",
            "Parch",
            "SibSp",
        ]
        return df[required_columns]

    @staticmethod
    def encode_sex(x: str):
        if x in ("male", "男性"):
            return 1
        elif x in ("female", "女性"):
            return 0
        else:
            return np.nan

    @classmethod
    def __preprocess_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        tmp_df = cls.__extract_requierd_columns(df)
        tmp_df["Sex"] = tmp_df["Sex"].apply(lambda x: cls.encode_sex(x))
        return tmp_df

    @staticmethod
    def __train_model(df: pd.DataFrame) -> lightgbm.sklearn.LGBMModel:
        """目的変数をSurviveとし、LightGBMによる学習を行います"""
        y = df["Survived"]
        X = df.drop(["Survived"], axis=1)

        model = LGBMClassifier()
        model.fit(X.values, y.values)
        return model

    @classmethod
    def __save_model(cls, model: lightgbm.sklearn.LGBMModel):
        """pickle形式で作成したlightGBMモデルを保存します"""
        path_to_model = cls.base_dir / "model" / "model.pkl"
        with open(path_to_model, "wb") as f:
            pickle.dump(model, f)

    @classmethod
    def generate_model(cls):
        """モデルの学習から保存まで一貫して行う"""
        df = cls.__read_data()
        preprocessed_df = cls.__preprocess_df(df)
        lgbm_model = cls.__train_model(preprocessed_df)
        cls.__save_model(lgbm_model)
