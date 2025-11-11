import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load processed datasets
test_df = pd.read_csv("data/processed/test.csv")
X_test = test_df['text']
y_test = test_df['label']

# Load the trained model + vectorizer
model = joblib.load("models/baseline_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Transform text into TF-IDF vectors
X_test_tfidf = vectorizer.transform(X_test)

# Predict on test dataset
y_pred = model.predict(X_test_tfidf)

# Classification metrics
print("\n✅ Detailed Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualize confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal (0)", "Depressed (1)"],
            yticklabels=["Normal (0)", "Depressed (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Baseline Model")
plt.show()

# Show some correct & incorrect predictions
test_df['predicted'] = y_pred
print("\nSome Correct Predictions:")
print(test_df[test_df['label'] == test_df['predicted']].head())

print("\nSome Incorrect Predictions:")
print(test_df[test_df['label'] != test_df['predicted']].head())
