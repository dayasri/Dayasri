import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Data Cleaning: Drop irrelevant columns
df_cleaned = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# EDA: Check missing values and churn distribution
missing = df_cleaned.isnull().sum()
churn_rate = df_cleaned['Exited'].mean()

# Visualization: Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=df_cleaned)
plt.title("Churn Distribution")
plt.xlabel("Exited")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.close()
plt.show() 

# Feature Engineering: One-hot encoding
df_encoded = pd.get_dummies(df_cleaned, columns=['Geography', 'Gender'], drop_first=True)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(df_encoded.drop('Exited', axis=1))
y = df_encoded['Exited'].values

# Model Development
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Visualization:
importances = model.feature_importances_
features = df_encoded.drop('Exited', axis=1).columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()

# Save model
joblib.dump(model, "churn_model.pkl")

# Documentation
with open("model_report.txt", "w") as f:
    f.write("Missing Values:\n")
    f.write(str(missing) + "\n\n")
    f.write(f"Churn Rate: {churn_rate:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
