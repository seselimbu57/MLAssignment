import numpy as np

# Sample dataset
X = np.array([
    [1, 'A'],
    [2, 'B'],
    [3, 'A'],
    [4, 'C'],
    [5, 'B'],
    [6, 'C'],
    [7, 'A']
])

y = np.array([0, 1, 0, 1, 1, 0, 1])

columns = ['feature1', 'feature2']  # Example column names

def calculate_information_gain(X, y, columns, attribute):
   
    # Find the index of the attribute column
    idx = columns.index(attribute)
    
    # Calculate entropy of the entire dataset
    total_entropy = calculate_entropy(X, y)
    
    # Calculate weighted entropy after splitting by the attribute
    weighted_entropy = 0.0
    unique_values = np.unique(X[:, idx])
    
    for value in unique_values:
        # Split dataset based on the attribute value
        X_split = X[X[:, idx] == value]
        y_split = y[X[:, idx] == value]
        
        # Calculate weight (probability) of this split
        prob = len(X_split) / len(X)
        
        # Calculate entropy of this split
        entropy_split = calculate_entropy(X_split, y_split)
        
        # Weighted entropy
        weighted_entropy += prob * entropy_split
    
    # Calculate information gain
    gain = total_entropy - weighted_entropy
    
    return gain

def calculate_entropy(X, y):
   
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    # Total number of samples
    total_samples = len(y)
    
    # Calculate entropy
    entropy = 0.0
    for count in class_counts:
        prob = count / total_samples
        entropy -= prob * np.log2(prob)
    
    return entropy

# Calculate information gain for a specific attribute
attribute_to_calculate = 'feature2'
gain = calculate_information_gain(X, y, columns, attribute_to_calculate)

print(f"Information Gain for '{attribute_to_calculate}': {gain}")
