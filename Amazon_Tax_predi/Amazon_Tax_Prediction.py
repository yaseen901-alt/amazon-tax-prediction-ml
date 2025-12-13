import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Amazon Tax Prediction - Portfolio Ready
# -----------------------------

# 1. Load dataset
df = pd.read_csv("Amazon.csv")
print("Dataset loaded. Shape:", df.shape)

# 2. Drop irrelevant columns
df = df.drop(["OrderID","CustomerID","CustomerName",
              "ProductID","ProductName","SellerID"], axis=1)

# 3. Define features and target
X = df.drop(columns=["Tax"])
y = df["Tax"]

# 4. Identify numeric and categorical columns
num_col = X.select_dtypes(include=["int64","float64"]).columns
cat_col = X.select_dtypes(include=["object","category"]).columns
print("Numeric columns:", list(num_col))
print("Categorical columns:", list(cat_col))

# 5. Preprocessing pipelines
num_transform = Pipeline(steps=[("scaler", StandardScaler())])
cat_transform = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocess = ColumnTransformer(transformers=[
    ("num", num_transform, num_col),
    ("cat", cat_transform, cat_col)
])

# 6. Complete pipeline with RandomForestRegressor
model = Pipeline(steps=[
    ("preprocessing", preprocess),
    ("RandomForest", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ))
])

# 7. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train the model
print("Training started...")
model.fit(X_train, y_train)
print("Training completed!")

# 9. Predict and evaluate
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"R² Score on Test Set: {score:.4f}")
