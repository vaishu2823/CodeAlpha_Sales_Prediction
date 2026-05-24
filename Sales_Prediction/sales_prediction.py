# ==========================================
# Professional Sales Prediction Project
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("advertising.csv")

# ==========================================
# Display Dataset
# ==========================================

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Information:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nStatistical Summary:\n")
print(df.describe())

# ==========================================
# Sales Distribution
# ==========================================

plt.figure(figsize=(8,6))

sns.histplot(df['Sales'], bins=20)

plt.title("Sales Distribution")

plt.savefig("images/sales_distribution.png")

plt.show()

# ==========================================
# Correlation Heatmap
# ==========================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Advertising Correlation Heatmap")

plt.savefig("images/heatmap.png")

plt.show()

# ==========================================
# Features and Target
# ==========================================

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Create Model
# ==========================================
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# ==========================================
# Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")


# Actual vs Predicted Graph


plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.savefig("images/prediction_graph.png")

plt.show()

# Feature Importance

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_
})

print("\nFeature Importance:\n")
print(importance)

plt.figure(figsize=(8,6))

sns.barplot(
    x='Feature',
    y='Importance',
    data=importance
)

plt.title("Advertising Feature Importance")

plt.savefig("images/feature_importance.png")

plt.show()
# Custom Prediction

print("\n=== Sales Prediction System ===")

tv = float(input("Enter TV Advertising Budget: "))
radio = float(input("Enter Radio Advertising Budget: "))
newspaper = float(input("Enter Newspaper Advertising Budget: "))

custom_data = [[tv, radio, newspaper]]

prediction = model.predict(custom_data)

print(f"\nPredicted Sales: {prediction[0]:.2f}")

print("\nProject Completed Successfully!")