import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# We now import BOTH Forest models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("Step 1: Loading data...")
dataset_url = "https://raw.githubusercontent.com/AREEG94FAHAD/TaskComplexityEval-24/refs/heads/main/problems_data.jsonl"
df = pd.read_json(dataset_url, lines=True)

# cleanup and renaming
df = df.rename(columns={
    'task_title': 'title',
    'problem_description': 'description', 
    'complexity_score': 'problem_score',
    'problem_difficulty': 'problem_class'
})
df.fillna('', inplace=True)
df['combined_text'] = df['title'] + " " + df['description'] 
if 'input_description' in df.columns:
    df['combined_text'] += " " + df['input_description']
if 'output_description' in df.columns:
    df['combined_text'] += " " + df['output_description']

print("Step 2: Vectorizing (With N-grams)...")
# TRICK 1: ngram_range=(1,2) -> Reads "binary search" as one concept, not just "binary" and "search"
# TRICK 2: max_features=10000 -> Looks at more words
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['combined_text'])

X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, df['problem_class'], df['problem_score'], test_size=0.2, random_state=42
)

print("Step 3: Training Random Forest Models...")

# --- CLASSIFIER (Replaced Logistic Regression with Random Forest) ---
# n_estimators=200 means we use 200 "students" to vote on the answer.
classifier = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced',n_jobs=-1, random_state=42)
classifier.fit(X_train, y_class_train)

# --- REGRESSOR (Same as before, just tuned) ---
regressor = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
regressor.fit(X_train, y_score_train)

# Calculate accuracy
class_acc = accuracy_score(y_class_test, classifier.predict(X_test))
score_error = mean_absolute_error(y_score_test, regressor.predict(X_test))

# print(f"   -> Classification Accuracy: {class_acc*100:.1f}%")
# print(f"   -> Score Error (MAE): {score_error:.2f}")

print("Step 4: Saving models...")
joblib.dump(classifier, 'model_class.pkl')
joblib.dump(regressor, 'model_score.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# ===== Evaluation =====

# Predictions
y_class_pred = classifier.predict(X_test)
y_score_pred = regressor.predict(X_test)

# Metrics
accuracy = accuracy_score(y_class_test, y_class_pred)

labels = sorted(y_class_test.unique())

conf_matrix = confusion_matrix(
    y_class_test,
    y_class_pred,
    labels=labels
)

mae = mean_absolute_error(y_score_test, y_score_pred)
rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))

# Display results
print("\n=== Evaluation Metrics ===")
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
print(f"Regression MAE: {mae:.3f}")
print(f"Regression RMSE: {rmse:.3f}")

print("\nConfusion Matrix (Rows = Actual, Columns = Predicted):")
print(conf_matrix)

conf_df = pd.DataFrame(
    conf_matrix,
    index=[f"Actual {lbl}" for lbl in labels],
    columns=[f"Predicted {lbl}" for lbl in labels]
)

print("\nConfusion Matrix Table:")
print(conf_df)
print("Done!")
