# Created by: Haris Rasul
# Date: December 22th 2023
# Python script to build the complete min depth tree and create literals
# This has a modeified dceoded threhold vs orginal paper - should test on test cacuracy later!!

from pysat.formula import CNF
from pysat.solvers import Solver
from graphviz import Digraph
import numpy as np

# Define the function to build a complete tree of a given depth
def build_complete_tree(depth):
    
    num_nodes = (2 ** (depth + 1)) - 1
    tree_structure = [None] * num_nodes
    TB, TL = [], []

    for node in range(num_nodes):
        if node < ((2 ** depth) - 1):
            TB.append(node)
            # Include feature and threshold keys for branching nodes
            tree_structure[node] = {
                'type': 'branching', 
                'children': [2 * node + 1, 2 * node + 2], 
                'feature': None, 
                'threshold': None
            }
        else:
            TL.append(node)
            tree_structure[node] = {'type': 'leaf', 'label': None}
    
    return tree_structure, TB, TL



# Define the function to create literals based on the tree structure
def create_literals(TB, TL, F, C, dataset_size):

    literals = {}
    current_index = 1

    # Create 'a' literals for feature splits at branching nodes
    for t in TB:
        for j in F:
            literals[f'a_{t}_{j}'] = current_index
            current_index += 1

    # Create 's' literals for data points directed left or right at branching nodes
    for i in range(dataset_size):
        for t in TB:
            literals[f's_{i}_{t}'] = current_index
            current_index += 1

    # Create 'z' literals for data points ending up at leaf nodes
    for i in range(dataset_size):
        for t in TL:
            literals[f'z_{i}_{t}'] = current_index
            current_index += 1

    # Create 'g' literals for labels at leaf nodes
    for t in TL:
        for c in C:
            literals[f'g_{t}_{c}'] = current_index
            current_index += 1

    return literals

def get_ancestors(node_index, side):
 
    ancestors = []
    current_index = node_index
    while True:
        parent_index = (current_index - 1) // 2
        if parent_index < 0:
            break
        # Check if current node is a left or right child
        if (side == 'left' and current_index % 2 == 1) or (side == 'right' and current_index % 2 == 0):
            ancestors.append(parent_index)
        current_index = parent_index
    return ancestors

# Helper function to sort data points by feature and create O_j
def compute_ordering(X, feature_index):
    sorted_indices = sorted(range(len(X)), key=lambda i: X[i][feature_index])
    return [(sorted_indices[i], sorted_indices[i + 1]) for i in range(len(sorted_indices) - 1)]

def build_clauses(literals, X, TB, TL, num_features, labels,true_labels):
   
    cnf = CNF()
    
    # Clause (1) and (2): Feature selection at branching nodes
    for t in TB:
        # At least one feature is chosen (Clause 2)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        cnf.append(clause)
        
        # No two features are chosen (Clause 1)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                cnf.append(clause)

    # Clause (3) and (4): Data point direction based on feature values
    for j in range(num_features):
        Oj = compute_ordering(X, j)
        for (i, ip) in Oj:
            if X[i][j] < X[ip][j]:  # Different feature values (Clause 3)
                for t in TB:
                    cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
            if X[i][j] == X[ip][j]:  # Equal feature values (Clause 4)
                for t in TB:
                    cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
                    cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i}_{t}'], literals[f's_{ip}_{t}']])

    # Clause (5 and 6): Path valididty form right traversla and left traversal 
    for t in TL:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(X)):
            # Data point i ends at leaf node t (Clause 5 and 6) - assumption made!!!
            if left_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                cnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # Clause (7): Each data point that does not end up in leaf node t has at least one deviation from the path
    for xi in range(len(X)):
        for t in TL:
            deviations = []
            left_ancestors = get_ancestors(t, 'left')  # Get left ancestors using TB indices
            right_ancestors = get_ancestors(t, 'right')  # Get right ancestors using TB indices
            # Only append deviations if there are ancestors on the corresponding side
            if left_ancestors:
                deviations.extend([-literals[f's_{xi}_{ancestor}'] for ancestor in left_ancestors])
            if right_ancestors:
                deviations.extend([literals[f's_{xi}_{ancestor}'] for ancestor in right_ancestors])
            # Only append the clause if there are any deviations
            if deviations:
                cnf.append([literals[f'z_{xi}_{t}']] + deviations)    

    # Clause (8): Each leaf node is assigned at most one label
    for t in TL:
        for c in range(len(labels)):
            for cp in range(c + 1, len(labels)):
                cnf.append([-literals[f'g_{t}_{labels[c]}'], -literals[f'g_{t}_{labels[cp]}']])

    # Clause (9) and (10): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in TB:
        # Find the data point with the lowest and highest feature value for each feature
        for j in range(num_features):
            sorted_by_feature = sorted(range(len(X)), key=lambda k: X[k][j])
            lowest_value_index = sorted_by_feature[0]
            highest_value_index = sorted_by_feature[-1]

            # Clause (9): The data point with the lowest feature value is directed left
            cnf.append([-literals[f'a_{t}_{j}'], literals[f's_{lowest_value_index}_{t}']])

            # Clause (10): The data point with the highest feature value is directed right
            cnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{highest_value_index}_{t}']])

    # Clause (11): Correct class labels for leaf nodes
    for t in TL:
        for i, xi in enumerate(X):
            label = true_labels[i]
            cnf.append([-literals[f'z_{i}_{t}'], literals[f'g_{t}_{label}']])
    
    return cnf

def set_branch_node_features(model, literals, tree_structure,features,datasetX):
   
    # For each branching node, determine the chosen feature and threshold
    for node_index in range(len(tree_structure)):
        #print(node_index)
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            # Find which feature is used for splitting at the current node
            chosen_feature = None
            for feature in features:
                if literals[f'a_{node_index}_{feature}'] in model:
                    chosen_feature = feature
                    break
            
            # If a feature is chosen, set the feature and find the threshold
            if chosen_feature is not None:
                # Set the chosen feature and computed threshold in the tree structure
                node['feature'] = chosen_feature


def solve_cnf(cnf, literals, TL, tree_structure, labels,features,datasetX):
 
    solver = Solver()
    solver.append_formula(cnf)
    if solver.solve():
        model = solver.get_model()
        # Update the tree structure with the correct labels for leaf nodes
        for t in TL:
            for label in labels:
                if literals[f'g_{t}_{label}'] in model:
                    tree_structure[t]['label'] = label
                    break
         # Set details for branching nodes
        set_branch_node_features(model, literals, tree_structure,features,datasetX)
        return model
    else:
        #print("no solution!")
        return "No solution exists"

#adjusted Logic to compute threshold on the entire dataset at each feature node branch 
def add_thresholds(tree_structure, literals, model_solution, dataset):
  
    def get_literal_value(literal):
        # Helper function to get the value of a literal from the model solution.
        return literals[literal] if literals[literal] in model_solution else -literals[literal]

    def set_thresholds(node_index, dataset):
        node = tree_structure[node_index]
        if node['type'] == 'branching':
            feature_index = int(node['feature'])

            # Instead of using the dataset_indices, we will compute the threshold based on the entire dataset.
            feature_values = dataset[:, feature_index]
            sorted_indices = np.argsort(feature_values)

            # Initialize the threshold
            threshold = None

            # Find the first instance where the direction changes and set the threshold.
            for i in range(1, len(sorted_indices)):
                left_index = sorted_indices[i - 1]
                right_index = sorted_indices[i]
                if get_literal_value(f's_{left_index}_{node_index}') > 0 and get_literal_value(f's_{right_index}_{node_index}') < 0:
                    threshold = (feature_values[left_index] + feature_values[right_index]) / 2
                    break

            # If no change in direction is found, threshold remains None.
            node['threshold'] = threshold
            
            # Continue for children nodes
            left_child_index, right_child_index = node['children'][0], node['children'][1]
            if left_child_index < len(tree_structure): # Check index is within bounds
                set_thresholds(left_child_index, dataset)
            if right_child_index < len(tree_structure): # Check index is within bounds
                set_thresholds(right_child_index, dataset)

    # Apply the threshold setting function starting from the root node
    set_thresholds(0, dataset)

    return tree_structure

# Create a matrix for each type of variable
def create_solution_matrix(literals, solution, var_type):
    # Find the maximum index for this var_type
    max_index = max(int(key.split('_')[1]) for key, value in literals.items() if key.startswith(var_type)) + 1
    max_sub_index = max(int(key.split('_')[2]) for key, value in literals.items() if key.startswith(var_type)) + 1
    
    # Initialize the matrix with zeros
    matrix = [[0 for _ in range(max_sub_index)] for _ in range(max_index)]
    
    # Fill in the matrix with 1 where the literals are true according to the solution
    for key, value in literals.items():
        if key.startswith(var_type):
            index, sub_index = map(int, key.split('_')[1:])
            matrix[index][sub_index] = 1 if value in solution else 0

    return matrix

# visualization code
def add_nodes(dot, tree, node_index=0):
    node = tree[node_index]
    if node['type'] == 'branching':
        dot.node(str(node_index), label=f"BranchNode:\n{node_index}\nFeature:{node['feature']}\nThreshold:{node['threshold']}")
        for child_index in node['children']:
            add_nodes(dot, tree, child_index)
            dot.edge(str(node_index), str(child_index))
    elif node['type'] == 'leaf':
        dot.node(str(node_index), label=f"LeafNode:\n{node_index}\nLabel: {node['label']}")

#visualization code
def visualize_tree(tree_structure):
    dot = Digraph()
    add_nodes(dot, tree_structure)
    return dot


def find_min_depth_tree(features, labels, true_labels_for_points, dataset):
    depth = 1  # Start with a depth of 1
    solution = "No solution exists"
    tree_with_thresholds = None
    tree = None
    literals = None

    while solution == "No solution exists":
        tree, TB, TL = build_complete_tree(depth)
        literals = create_literals(TB, TL, features, labels, len(dataset))
        cnf = build_clauses(literals, dataset, TB, TL, len(features), labels, true_labels_for_points)
        solution = solve_cnf(cnf, literals, TL, tree, labels, features, dataset)
        
        if solution != "No solution exists":
            tree_with_thresholds = add_thresholds(tree, literals, solution, dataset)
            dot = visualize_tree(tree_with_thresholds)
            dot.render(f'images/min_height/binary_decision_tree_min_depth_{depth}', format='png', cleanup=True)
        else:
            print('no solution at depth', depth)
            depth += 1  # Increase the depth and try again
    
    return tree_with_thresholds, literals, depth, solution

