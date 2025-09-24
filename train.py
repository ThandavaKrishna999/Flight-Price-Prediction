import pandas as pd
import numpy as np
import cloudpickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler,
    PowerTransformer, FunctionTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
    RareLabelEncoder, MeanEncoder, CountFrequencyEncoder
)

# =========================================
# Custom functions / transformers
# =========================================

def is_north(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=["source", "destination"])
    columns = X.columns.to_list()
    north_cities = ["Delhi", "Kolkata", "Mumbai", "New Delhi"]
    return (
        X.assign(**{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        }).drop(columns=columns)
    )

def part_of_day(X, morning=4, noon=12, eve=16, night=20):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    columns = X.columns.to_list()
    X_temp = X.assign(**{col: pd.to_datetime(X.loc[:, col]).dt.hour for col in columns})
    return (
        X_temp.assign(**{
            f"{col}_part_of_day": np.select(
                [X_temp.loc[:, col].between(morning, noon, inclusive="left"),
                 X_temp.loc[:, col].between(noon, eve, inclusive="left"),
                 X_temp.loc[:, col].between(eve, night, inclusive="left")],
                ["morning", "afternoon", "evening"],
                default="night"
            )
            for col in columns
        }).drop(columns=columns)
    )

def duration_category(X, short=180, med=400):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=["duration"])
    return X.assign(duration_cat=np.select(
        [X.duration.lt(short), X.duration.between(short, med, inclusive="left")],
        ["short", "medium"], default="long"
    )).drop(columns="duration")

def is_over(X, value=1000):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=["duration"])
    return X.assign(**{f"duration_over_{value}": X.duration.ge(value).astype(int)}).drop(columns="duration")

def is_direct(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=["total_stops"])
    return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))

def have_info(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=["additional_info"])
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))


class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1):
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma

    def fit(self, X, y=None):
        # Ensure DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        if not self.variables:
            self.variables = X.select_dtypes(include="number").columns.to_list()

        self.reference_values_ = {
            col: (
                X.loc[:, col]
                .quantile(self.percentiles)
                .values
                .reshape(-1, 1)
            )
            for col in self.variables
        }
        return self

    def transform(self, X):
        # Ensure DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.variables)

        objects = []
        for col in self.variables:
            columns = [f"{col}_rbf_{int(p * 100)}" for p in self.percentiles]
            obj = pd.DataFrame(
                data=rbf_kernel(X.loc[:, [col]], Y=self.reference_values_[col], gamma=self.gamma),
                columns=columns
            )
            objects.append(obj)
        return pd.concat(objects, axis=1)


# =========================================
# Pipelines
# =========================================

# airline
air_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

# doj
feature_to_extract = ["month", "week", "day_of_week", "day_of_year"]
doj_transformer = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True, format="mixed")),
    ("scaler", MinMaxScaler())
])

# source & destination
location_pipe1 = Pipeline(steps=[
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", MeanEncoder(variables=["source", "destination"])),
    ("scaler", PowerTransformer())
])
location_transformer = FeatureUnion(transformer_list=[
    ("part1", location_pipe1),
    ("part2", FunctionTransformer(func=is_north))
])

# dep_time & arrival_time
time_pipe1 = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
    ("scaler", MinMaxScaler())
])
time_pipe2 = Pipeline(steps=[
    ("part", FunctionTransformer(func=part_of_day)),
    ("encoder", CountFrequencyEncoder()),
    ("scaler", MinMaxScaler())
])
time_transformer = FeatureUnion(transformer_list=[
    ("part1", time_pipe1),
    ("part2", time_pipe2)
])

# duration
duration_pipe1 = Pipeline(steps=[
    ("rbf", RBFPercentileSimilarity()),
    ("scaler", PowerTransformer())
])
duration_pipe2 = Pipeline(steps=[
    ("cat", FunctionTransformer(func=duration_category)),
    ("encoder", OrdinalEncoder(categories=[["short", "medium", "long"]]))
])
duration_union = FeatureUnion(transformer_list=[
    ("part1", duration_pipe1),
    ("part2", duration_pipe2),
    ("part3", FunctionTransformer(func=is_over)),
    ("part4", StandardScaler())
])
duration_transformer = Pipeline(steps=[
    ("outliers", Winsorizer(capping_method="iqr", fold=1.5)),
    ("imputer", SimpleImputer(strategy="median")),
    ("union", duration_union)
])

# total stops
total_stops_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("", FunctionTransformer(func=is_direct))
])

# additional_info
info_pipe1 = Pipeline(steps=[
    ("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
info_union = FeatureUnion(transformer_list=[
    ("part1", info_pipe1),
    ("part2", FunctionTransformer(func=have_info))
])
info_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("union", info_union)
])

# Column Transformer
column_transformer = ColumnTransformer(transformers=[
    ("air", air_transformer, ["airline"]),
    ("doj", doj_transformer, ["date_of_journey"]),
    ("location", location_transformer, ["source", "destination"]),
    ("time", time_transformer, ["dep_time", "arrival_time"]),
    ("dur", duration_transformer, ["duration"]),
    ("stops", total_stops_transformer, ["total_stops"]),
    ("info", info_transformer, ["additional_info"])
], remainder="passthrough")

# Feature Selector
estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
selector = SelectBySingleFeaturePerformance(
    estimator=estimator,
    scoring="r2",
    threshold=0.095
)

# Preprocessor Pipeline
preprocessor = Pipeline(steps=[
    ("ct", column_transformer),
    ("selector", selector)
])

# =========================================
# Train & Save
# =========================================
print("📂 Loading dataset...")
train = pd.read_csv("train.csv")
X_train = train.drop(columns="price")
y_train = train.price.copy()

print("⚙️ Fitting preprocessor...")
preprocessor.fit(X_train, y_train)

print("💾 Saving preprocessor with cloudpickle...")
with open("preprocessor.joblib", "wb") as f:
    cloudpickle.dump(preprocessor, f)

print("✅ Preprocessor saved successfully as preprocessor.joblib")


import xgboost as xgb
import pickle

print("⚙️ Training XGBoost model...")

# Transform training data with fitted preprocessor
X_train_pre = preprocessor.transform(X_train)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train_pre, label=y_train)

# Set model parameters (you can tweak these)
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

# Train model
xgb_model = xgb.train(params, dtrain, num_boost_round=200)

# Save model
with open("xgboost-model", "wb") as f:
    pickle.dump(xgb_model, f)

print("✅ XGBoost model saved successfully as xgboost-model")
