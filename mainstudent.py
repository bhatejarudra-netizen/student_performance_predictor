import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
data = pd.read_csv("data.csv")

# Step 3: Basic Info
print("\nFirst 5 rows:\n", data.head())
print("\nColumns:\n", data.columns)

# Step 4: Handle Missing Values
print("\nMissing Values:\n", data.isnull().sum())
data = data.dropna()

# Step 5: Features and Target
X = data[['StudyTimeWeekly', 'Absences', 'Tutoring']]
y = data['GPA']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)

# Step 7: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Prediction
y_pred = model.predict(X_test)

# Step 9: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MSE:", mse)
print("R2 Score:", r2)
print(f"Accuracy: {r2*100:.2f}%")

# Step 10: Feature Importance
print("\nFeature Importance:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Step 11: Visualization - Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Actual vs Predicted GPA")
plt.savefig("output.png")
plt.show()

# Step 12: Correlation Heatmap
plt.figure()
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# Step 13: User Input Prediction
print("\n--- Predict Student GPA ---")

study_time = float(input("Enter weekly study hours: "))
absences = int(input("Enter number of absences: "))
tutoring = int(input("Tutoring? (1 = Yes, 0 = No): "))

prediction = model.predict([[study_time, absences, tutoring]])

print(f"\nPredicted GPA: {prediction[0]:.2f}")