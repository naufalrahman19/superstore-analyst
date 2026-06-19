import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import (
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

from data_loader import load_data

from preprocessing import (
    clean_columns,
    convert_data_types,
    create_features,
    prepare_features_target
)

from evaluate import evaluate_model


# =====================================================
# LOAD DATA
# =====================================================

df = load_data(
    "data/raw/Sample - Superstore.csv"
)

# =====================================================
# PREPROCESSING
# =====================================================

df = clean_columns(df)

df = convert_data_types(df)

df = create_features(df)

X, y = prepare_features_target(df)

# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )
)

# =====================================================
# SCALING
# =====================================================

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =====================================================
# MODELS
# =====================================================

models = {
    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42
        )
}

for name, model in models.items():

    model.fit(
        X_train_scaled,
        y_train
    )

    evaluate_model(
        model,
        X_test_scaled,
        y_test,
        name
    )

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0]
}

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

search.fit(
    X_train,
    y_train
)

best_xgb = search.best_estimator_

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    best_xgb,
    "models/best_xgb.pkl"
)

print(
    "\nBest Parameters:"
)

print(
    search.best_params_
)