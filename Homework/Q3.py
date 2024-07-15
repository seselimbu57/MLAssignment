import numpy as np

class TreeNode:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute    
        self.label = label            
        self.children = {}            

def calculate_entropy(y):

    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_information_gain(X, y, columns, attribute):

    attr_index = columns.index(attribute)
    total_entropy = calculate_entropy(y)
    values, counts = np.unique(X[:, attr_index], return_counts=True)
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset_indices = np.where(X[:, attr_index] == value)
        subset_y = y[subset_indices]
        subset_entropy = calculate_entropy(subset_y)
        weighted_entropy += (count / len(y)) * subset_entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

def id3(X, y, columns):
    
    # Convert y to integer labels if they are strings
    if isinstance(y[0], str):
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])

    # If all target values are the same, return a leaf node with that value
    if len(np.unique(y)) == 1:
        return TreeNode(label=np.unique(y)[0])

    # If there are no more columns to split on, return the most common target value
    if len(columns) == 0:
        return TreeNode(label=np.bincount(y).argmax())

    # Find the attribute with the highest information gain
    gains = [calculate_information_gain(X, y, columns, col) for col in columns]
    best_attr_index = np.argmax(gains)
    best_attr = columns[best_attr_index]

    # Create the root node with the best attribute
    root = TreeNode(attribute=best_attr)

    # Remove the best attribute from the list of columns
    remaining_columns = [col for col in columns if col != best_attr]

    # Split the dataset based on the best attribute and recursively create branches
    attr_index = columns.index(best_attr)
    values = np.unique(X[:, attr_index])
    for value in values:
        subset_indices = np.where(X[:, attr_index] == value)
        subset_X = X[subset_indices]
        subset_y = y[subset_indices]
        child_node = id3(subset_X, subset_y, remaining_columns)
        root.children[value] = child_node

    return root

def print_tree(node, depth=0):
    """
    Print the decision tree in a readable format.
    
    Parameters:
    node : TreeNode object
        The current node of the decision tree.
    depth : int
        Current depth of the node in the tree (used for indentation).
    """
    if node.label is not None:
        print('  ' * depth, f"Label: {node.label}")
    else:
        print('  ' * depth, f"Attribute: {node.attribute}")
        for value, child_node in node.children.items():
            print('  ' * (depth + 1), f"Value '{value}':")
            print_tree(child_node, depth + 2)

if __name__ == "__main__":
    # Example usage with fabricated climate data
    X = np.array([
        ['Sunny', 'Hot', 'High', 'Weak'],
        ['Sunny', 'Hot', 'High', 'Strong'],
        ['Overcast', 'Hot', 'High', 'Weak'],
        ['Rain', 'Mild', 'High', 'Weak'],
        ['Rain', 'Cool', 'Normal', 'Weak'],
        ['Rain', 'Cool', 'Normal', 'Strong'],
        ['Overcast', 'Cool', 'Normal', 'Strong'],
        ['Sunny', 'Mild', 'High', 'Weak'],
        ['Sunny', 'Cool', 'Normal', 'Weak'],
        ['Rain', 'Mild', 'Normal', 'Weak'],
        ['Sunny', 'Mild', 'Normal', 'Strong'],
        ['Overcast', 'Mild', 'High', 'Strong'],
        ['Overcast', 'Hot', 'Normal', 'Weak'],
        ['Rain', 'Mild', 'High', 'Strong']
    ])

    y = np.array(['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'])

    columns = ['Outlook', 'Temperature', 'Humidity', 'Wind']  # Example column names

    # Build the decision tree
    decision_tree = id3(X, y, columns)

    # Print the decision tree
    print("Decision Tree:")
    print_tree(decision_tree)
