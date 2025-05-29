from sklearn.tree import DecisionTreeClassifier

def create_decision_tree_model(max_depth: int = 5, random_state: int = 42) -> DecisionTreeClassifier:
    """
    Creates an instance of a Decision Tree Classifier.

    Parameters:
    max_depth (int): The maximum depth of the tree. Default is 5.
    random_state (int): Controls the randomness of the estimator. Default is 42.

    Returns:
    DecisionTreeClassifier: An untrained instance of the Decision Tree Classifier.
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    return model

if __name__ == '__main__':
    print("--- Baseline Model Creation Demonstration ---")
    
    # Create a decision tree model instance using the function
    dt_model = create_decision_tree_model()
    
    # Print the model instance to verify its creation
    print("\nCreated Decision Tree Model instance:")
    print(dt_model)
    
    # Example with different parameters
    dt_model_custom = create_decision_tree_model(max_depth=10, random_state=123)
    print("\nCreated Decision Tree Model instance with custom parameters:")
    print(dt_model_custom)
    
    print("\n--- Demonstration Finished ---")
