import numpy as np

# Example dataset
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10]
])

y = np.array([0, 1, 0, 1, 1])

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

# Calculate entropy of the example dataset
entropy = calculate_entropy(X, y)
print(f"Entropy of the dataset: {entropy}")
