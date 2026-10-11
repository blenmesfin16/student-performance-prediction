import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_student_pipeline():
    print("--- [START] Running Student Performance End-to-End Workflow ---")
    
    # 1. PREPARATION
    csv_path = os.path.join("data", "student_performance.csv")
    print(f"Phase 1: Loading student dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Define features (attendance, study hours, assignment score, midterm score) and target (pass_target)
    features = ["attendance", "study_hours", "assignment_score", "midterm_score"]
    X = df[features]
    y = df["pass_target"]
    
    # 2. SPLIT
    print("Phase 2: Splitting student dataset into training and testing sets...")
    # Using a small test_size or setting stratify because we have a small sample dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # 3. TRAINING
    print("Phase 3: Training the classifier to learn pass/fail student patterns...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. TESTING
    print("Phase 4: Running model predictions on unseen test student profiles...")
    y_pred = model.predict(X_test)
    
    # 5. EVALUATION
    print("Phase 5: Calculating core evaluation metrics...")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n" + "="*40)
    print("   STUDENT PREDICTION EVALUATION REPORT")
    print("="*40)
    print(f"Accuracy  Score: {acc:.4f}")
    print(f"Precision Score: {prec:.4f}")
    print(f"Recall    Score: {rec:.4f}")
    print(f"F1-Score  Value: {f1:.4f}")
    print("="*40 + "\n")
    print("--- [SUCCESS] Student Workflow Pipeline Complete ---")

if __name__ == "__main__":
    run_student_pipeline()
