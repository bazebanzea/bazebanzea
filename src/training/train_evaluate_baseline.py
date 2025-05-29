import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import sys
import os

# Adjust the Python path to include the 'src' directory
# This allows importing modules from src.models and src.preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_model import create_decision_tree_model
# from preprocessing.basic_preprocessor import encode_categorical # Example if we needed preprocessing

def main():
    """
    Main function to demonstrate the training and evaluation pipeline for the baseline model.
    """
    print("--- Baseline Model Training and Evaluation Demonstration ---")

    # 1. Create Dummy Data
    # This data simulates a preprocessed dataset where features are numerical
    # and the target is categorical (binary in this case).
    print("\nStep 1: Creating dummy data...")
    data = {
        'feature1': [1.0, 2.5, 3.1, 4.7, 5.2, 6.8, 7.3, 8.0, 9.5, 10.2],
        'feature2': [0.5, 1.8, 2.2, 3.9, 4.5, 5.1, 6.7, 7.2, 8.8, 9.1],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Binary target variable
    }
    df = pd.DataFrame(data)
    print("Dummy DataFrame created:")
    print(df.head())

    # Separate features (X) and target (y)
    X = df[['feature1', 'feature2']]
    y = df['label']
    print("\nFeatures (X) head:")
    print(X.head())
    print("\nTarget (y) head:")
    print(y.head())

    # 2. Split Data
    print("\nStep 2: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # 3. Load Model
    print("\nStep 3: Loading the baseline Decision Tree model...")
    # Using the function from our models module
    baseline_model = create_decision_tree_model(max_depth=3, random_state=42) # Using a shallow depth for tiny dataset
    print(f"Model loaded: {baseline_model}")

    # 4. Train Model
    print("\nStep 4: Training the model...")
    baseline_model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Make Predictions
    print("\nStep 5: Making predictions on the test set...")
    y_pred = baseline_model.predict(X_test)
    print(f"Predictions made for {len(y_pred)} samples.")
    print(f"Actual test labels:    {y_test.tolist()}")
    print(f"Predicted test labels: {y_pred.tolist()}")


    # 6. Evaluate Model
    print("\nStep 6: Evaluating the model...")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    # For binary classification, precision, recall, f1 can be calculated for the positive class (label 1)
    # For multi-class, specify average parameter if needed (e.g., 'macro', 'weighted')
    precision = precision_score(y_test, y_pred, zero_division=0) # zero_division=0 to avoid warning if a class is not predicted
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (for class 1, or average if specified)")
    print(f"Recall:    {recall:.4f} (for class 1, or average if specified)")
    print(f"F1-score:  {f1:.4f} (for class 1, or average if specified)")
    
    print("\nClassification Report:")
    # The classification report provides a more detailed breakdown
    try:
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)
    except ValueError as e:
        print(f"Could not generate classification report: {e}")
        print("This can happen if the test set is too small or only contains one class after split.")

    print("\n--- Demonstration Finished ---")

if __name__ == '__main__':
    main()
