# Created by: Haris Rasul
# Date: March 21 2024
# Python script to build the complete tree and create literals
# for a given depth and dataset. Will attempt to maximize the number of correct labels given to training dataset 
# adding soft clauses for maximizing corrcet solution for a given depth and hard clauses 

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from graphviz import Digraph
import numpy as np
from clustering_problems.min_height_tree_module import *
import math
from itertools import combinations
from collections import defaultdict, OrderedDict
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from clustering_problems.clustering_advanced import solve_wcnf_clustering, create_distance_classes, build_complete_tree_clustering, assign_clusters 

# Define the function to create literals based on the tree structure
def create_literals_cluster_tree_bicriteria(TB, TL, F, k_clusters, dataset_size,distance_classes):
   
    C = list(range(k_clusters))  # List of cluster IDs
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

    # Create 'g' literals for clusters at leaf nodes
    for t in TL:
        for c in C:
            literals[f'g_{t}_{c}'] = current_index
            current_index += 1

    # Create 'x' The cluster assigned to point 𝑖 is or comes after 𝑐
    for i in range(dataset_size):
        for c in C:
            literals[f'x_{i}_{c}'] = current_index
            current_index += 1
    
    # Create 'bw_m' literals (points in class w should NOT be clustered together)
    for w, pairs in enumerate(distance_classes):  # data_classes is a list of numpy arrays
        literals[f'bw_m_{w}'] = current_index
        current_index += 1

    # Create 'bw_p' literals (points in class w should be clustered together)
    for w, pairs in enumerate(distance_classes):
        literals[f'bw_p_{w}'] = current_index
        current_index += 1

    return literals
    
def create_literals_cluster_tree_bicriteria_CC(TB, TL, F, k_clusters, dataset_size,distance_classes):
 
    C = list(range(k_clusters))  # List of cluster IDs
    literals = {}
    current_index = 1

    # Create 'a' literals for feature splits at branching nodes


    # Create 's' literals for data points directed left or right at branching nodes


    # Create 'z' literals for data points ending up at leaf nodes


    # Create 'g' literals for clusters at leaf nodes


    # Create 'x' The cluster assigned to point 𝑖 is or comes after 𝑐
    for i in range(dataset_size):
        for c in C:
            literals[f'x_{i}_{c}'] = current_index
            current_index += 1
    
    # Create 'bw_m' literals (points in class w should NOT be clustered together)
    for w, pairs in enumerate(distance_classes):  # data_classes is a list of numpy arrays
        literals[f'bw_m_{w}'] = current_index
        current_index += 1

    # Create 'bw_p' literals (points in class w should be clustered together)
    for w, pairs in enumerate(distance_classes):
        literals[f'bw_p_{w}'] = current_index
        current_index += 1

    return literals

def build_clauses_cluster_tree_MD_MS(literals, X, TB, TL, num_features, k_clusters,
                                  CL_pairs, ML_pairs, distance_classes):
    
    ##################################################  BASE TREE ENCODINGS ################################################

    # Now the problem has become Partial MaxSAT - we will assign weights to the soft clauses Eq. (13). Eq(1-10,12) HARD clauses 
    wcnf = WCNF()
    
    # Clause (7) and (8): Feature selection at branching nodes
    for t in TB:
        # At least one feature is chosen (Clause 7)
        clause = [literals[f'a_{t}_{j}'] for j in range(num_features)]
        wcnf.append(clause)
        
        # No two features are chosen (Clause 8)
        for j in range(num_features):
            for jp in range(j + 1, num_features):
                clause = [-literals[f'a_{t}_{j}'], -literals[f'a_{t}_{jp}']]
                wcnf.append(clause)

    # Clause (9) and (10): Data point direction based on feature values
    for j in range(num_features):
        Oj = compute_ordering(X, j)
        for (i, ip) in Oj:
            if X[i][j] < X[ip][j]:  # Different feature values (Clause 9)
                for t in TB:
                    wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
            if X[i][j] == X[ip][j]:  # Equal feature values (Clause 10)
                for t in TB:
                    wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{i}_{t}'], -literals[f's_{ip}_{t}']])
                    wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{i}_{t}'], literals[f's_{ip}_{t}']])

    # Clause (11 and 12): Path valididty form right traversla and left traversal 
    for t in TL:
        left_ancestors = get_ancestors(t, 'left')
        right_ancestors = get_ancestors(t, 'right')
        for i in range(len(X)):
            # Data point i ends at leaf node t (Clause 11 and 12) - assumption made!!!
            if left_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [literals[f's_{i}_{a}'] for a in left_ancestors])
            if right_ancestors:
                wcnf.append([-literals[f'z_{i}_{t}']] + [-literals[f's_{i}_{a}'] for a in right_ancestors])

    # Clause (13): Each data point that does not end up in leaf node t has at least one deviation from the path
    for i in range(len(X)):
        for t in TL:
            deviations = []
            left_ancestors = get_ancestors(t, 'left')  # Get left ancestors using TB indices
            right_ancestors = get_ancestors(t, 'right')  # Get right ancestors using TB indices
            # Only append deviations if there are ancestors on the corresponding side
            if left_ancestors:
                deviations.extend([-literals[f's_{i}_{ancestor}'] for ancestor in left_ancestors])
            if right_ancestors:
                deviations.extend([literals[f's_{i}_{ancestor}'] for ancestor in right_ancestors])
            # Only append the clause if there are any deviations
            if deviations:
                wcnf.append([literals[f'z_{i}_{t}']] + deviations)    

    # Clause (14) and (15): Redundant constraints to prune the search space
    # These clauses are optimizations
    for t in TB:
        # Find the data point with the lowest and highest feature value for each feature
        for j in range(num_features):
            sorted_by_feature = sorted(range(len(X)), key=lambda k: X[k][j])
            lowest_value_index = sorted_by_feature[0]
            highest_value_index = sorted_by_feature[-1]

            # Clause (14): The data point with the lowest feature value is directed left
            wcnf.append([-literals[f'a_{t}_{j}'], literals[f's_{lowest_value_index}_{t}']])

            # Clause (15): The data point with the highest feature value is directed right
            wcnf.append([-literals[f'a_{t}_{j}'], -literals[f's_{highest_value_index}_{t}']])

    
    # Add remaining clauses without CONSIDEIRNG THE MS PORBLEM - ONLY MD; recall they have ids 1,...k, in application ist say 0,...,k-1 for any k
    
    # Clause 16: Unary encoding of cluster labels in each leaf
    for t in TL:  # For each leaf node
        for c in range(k_clusters - 2):  # Remember, k_clusters already 1 less than total count
            clause = [literals[f'g_{t}_{c}'], -literals[f'g_{t}_{c+1}']]
            wcnf.append(clause)
    
    # Clause 17: Data points ending at leaf node t are assigned to cluster c if g_t,c is true
    for t in TL:  # For each leaf node
        for i in range(len(X)):  # For each data point
            for c in range(k_clusters - 1):  # Cluster indexes 0 to k-2, as the last index is not needed here
                clause = [-literals[f'z_{i}_{t}'], -literals[f'g_{t}_{c}'], literals[f'x_{i}_{c}']]
                wcnf.append(clause)

    # Clause 18: Data points ending at leaf node t are NOT assigned to cluster c if g_t,c is false
    for t in TL:  # For each leaf node
        for i in range(len(X)):  # For each data point
            for c in range(k_clusters - 1):  # Cluster indexes 0 to k-2, as the last index is not needed here
                clause = [-literals[f'z_{i}_{t}'], literals[f'g_{t}_{c}'], -literals[f'x_{i}_{c}']]
                wcnf.append(clause)
    
    # Clause 19: Ensure no cluster is empty by ensuring there's at least one data point in each cluster
    for c in range(k_clusters - 1):  # Iterate over all clusters except the last one
        wcnf.append([-literals[f'x_{c}_{c}']])
    
    # Clause 20: If xi is not in cluster c, then there must be some xi' in cluster c-1, for all c < i
    for i in range(1,len(X)):  # We start from 1 to ensure there's at least one i' < i
        for c in range(1, k_clusters - 1):  # We can safely start from 1 since we need c to be at least 1 to have a c-1
            clause = [-literals[f'x_{i}_{c}']]
            for i_prime in range(i):  # For each i' less than i
                clause.append(literals[f'x_{i_prime}_{c-1}'])
            wcnf.append(clause)
    
    # Clause 21: Ensure all clusters are non-empty by requiring at least one point is assigned to each cluster
    # In this case, we focus on the second-to-last cluster k-2.
    clauseTW = [literals[f'x_{i}_{k_clusters - 2}'] for i in range(len(X))]
    wcnf.append(clauseTW)

    # Clause 22: Ensure that pairs in CL are not clustered in the first cluster (0-indexed)
    for i, i_prime in CL_pairs:  # For each pair (i, i') in the cannot-link set
        wcnf.append([literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])

    # Clause 23: Ensure that pairs in CL are not clustered in the last cluster (k-2 in 0-indexed system)
    for i, i_prime in CL_pairs:  # For each pair (i, i') in the cannot-link set
        wcnf.append([-literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])

    # Clause 24: Unconditional separating clauses for cannot-link pairs, applied to clusters from 0 to k-3
    for (i, i_prime) in CL_pairs:  # For each cannot-link pair
        for c in range(k_clusters - 2):  # Up to k-3, because we're considering c and c+1
            wcnf.append([
                -literals[f'x_{i}_{c}'], 
                -literals[f'x_{i_prime}_{c}'], 
                literals[f'x_{i}_{c+1}'], 
                literals[f'x_{i_prime}_{c+1}']
            ])

    # Clause 25 and 26: Ensure that pairs in ML are clustered together for each cluster
    for i, i_prime in ML_pairs:  # For each must-link pair
        for c in range(k_clusters - 1):  # Iterate over all clusters except the last one
            wcnf.append([-literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']]) # clause 25
            wcnf.append([literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])
    
    # Clause 27: Conditional separating clauses using distance classes and bw_m literals
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            wcnf.append([literals[f'bw_m_{w}'], literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])
    
    # Clause 28: Ensure that if bw^-_w is true, then the pair (i, i') from Dw cannot be in the second to last cluster k-2
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])
    
    # Clause 29: Conditional co-separation for non-adjacent clusters
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            # Apply the constraint for all clusters except the last (since we start from zero)
            for c in range(k_clusters - 2):                
                wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{c}'], 
                             -literals[f'x_{i_prime}_{c}'], literals[f'x_{i}_{c+1}'], literals[f'x_{i_prime}_{c+1}']])

    # Clause 30: If b^+_w is true, then pairs (i, i') in distance class w must be in the same cluster
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            for c in range(k_clusters - 1):  # Iterate through clusters 0 to k-1
                wcnf.append([-literals[f'bw_p_{w}'], -literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])

    # Clause 31: If b^+_w is true, then points (i, i') in distance class w must be in the same cluster c
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            # The points i and i' should be in the same cluster if b^+_w is true.
            # We check across all clusters from 0 to k-2 because the clusters are 0-indexed
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'bw_p_{w}'], literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    # Clause 32: Ensures that if bw^-_w is true, then the points in distance class w 
    # cannot be clustered with the points in distance class w-1 if bw^-_(w-1) is false.
    for w in range(1, len(distance_classes)):  # Starting from 1 since we're checking w against w-1
        wcnf.append([-literals[f'bw_m_{w}'], literals[f'bw_m_{w-1}']])
    
    # Clause 33: If bw^+_w is true, then distance class w must be clustered together with distance class w-1
    for w in range(1, len(distance_classes)):  # Starting from 1 since we're checking w against w-1
        wcnf.append([-literals[f'bw_p_{w}'], literals[f'bw_p_{w-1}']])
    
    # Clause 34: If bw^+_w is true, then distance class w cannot be clustered separately within itself
    for w in range(len(distance_classes)):
        wcnf.append([-literals[f'bw_p_{w}'], literals[f'bw_m_{w}']])

    # Clause 37: For each distance class w, we add a soft clause for the corresponding b^-_w literal
    # to encourage points within that class to be clustered separately
    # Max diameter and Min split Criteria solver
    for w in range(len(distance_classes)):
        wcnf.append([-literals[f'bw_m_{w}']], weight=1)

    # Clause 38: For each distance class w, we add a soft clause for the corresponding b^+_w literal
    # to encourage points within that class to be clustered together
    # This is for minimizing the simple combination of Lambda^- - Lambda^+ and is part of the objective function
    for w in range(len(distance_classes)):
        wcnf.append([literals[f'bw_p_{w}']], weight=1)   
    
    return wcnf

def build_clauses_cluster_tree_MD_MS_CC(literals, X, TB, TL, num_features, k_clusters,
                                  CL_pairs, ML_pairs, distance_classes):
 
    wcnf = WCNF()
    
    # Clause (7) and (8): Feature selection at branching nodes
    

    # Clause (9) and (10): Data point direction based on feature values
    

    # Clause (11 and 12): Path valididty form right traversla and left traversal 
     
    # Clause (13): Each data point that does not end up in leaf node t has at least one deviation from the path
    

    # Clause (14) and (15): Redundant constraints to prune the search space
    # These clauses are optimizations
     

    ################################################## CLUSTERING PROBLEM ENDCODINGS ################################################
    
    # Add remaining clauses without CONSIDEIRNG THE MS PORBLEM - ONLY MD; recall they have ids 1,...k, in application ist say 0,...,k-1 for any k
    
    # Clause 16: Unary encoding of cluster labels in each leaf
    
    
    # Clause 17: Data points ending at leaf node t are assigned to cluster c if g_t,c is true
     
    # Clause 18: Data points ending at leaf node t are NOT assigned to cluster c if g_t,c is false
     
    
    # Clause 19: Ensure no cluster is empty by ensuring there's at least one data point in each cluster
    for c in range(k_clusters - 1):  # Iterate over all clusters except the last one
        wcnf.append([-literals[f'x_{c}_{c}']])
    
    # Clause 20: If xi is not in cluster c, then there must be some xi' in cluster c-1, for all c < i
    for i in range(1,len(X)):  # We start from 1 to ensure there's at least one i' < i
        for c in range(1, k_clusters - 1):  # We can safely start from 1 since we need c to be at least 1 to have a c-1
            clause = [-literals[f'x_{i}_{c}']]
            for i_prime in range(i):  # For each i' less than i
                clause.append(literals[f'x_{i_prime}_{c-1}'])
            wcnf.append(clause)
    
    # Clause 21: Ensure all clusters are non-empty by requiring at least one point is assigned to each cluster
    # In this case, we focus on the second-to-last cluster k-2.
    clauseTW = [literals[f'x_{i}_{k_clusters - 2}'] for i in range(len(X))]
    wcnf.append(clauseTW)

    # Clause 22: Ensure that pairs in CL are not clustered in the first cluster (0-indexed)
    for i, i_prime in CL_pairs:  # For each pair (i, i') in the cannot-link set
        wcnf.append([literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])

    # Clause 23: Ensure that pairs in CL are not clustered in the last cluster (k-2 in 0-indexed system)
    for i, i_prime in CL_pairs:  # For each pair (i, i') in the cannot-link set
        wcnf.append([-literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])

    # Clause 24: Unconditional separating clauses for cannot-link pairs, applied to clusters from 0 to k-3
    for (i, i_prime) in CL_pairs:  # For each cannot-link pair
        for c in range(k_clusters - 2):  # Up to k-3, because we're considering c and c+1
            wcnf.append([
                -literals[f'x_{i}_{c}'], 
                -literals[f'x_{i_prime}_{c}'], 
                literals[f'x_{i}_{c+1}'], 
                literals[f'x_{i_prime}_{c+1}']
            ])

    # Clause 25 and 26: Ensure that pairs in ML are clustered together for each cluster
    for i, i_prime in ML_pairs:  # For each must-link pair
        for c in range(k_clusters - 1):  # Iterate over all clusters except the last one
            wcnf.append([-literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']]) # clause 25
            wcnf.append([literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])
    
    # Clause 27: Conditional separating clauses using distance classes and bw_m literals
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            wcnf.append([literals[f'bw_m_{w}'], literals[f'x_{i}_0'], literals[f'x_{i_prime}_0']])
    
    # Clause 28: Ensure that if bw^-_w is true, then the pair (i, i') from Dw cannot be in the second to last cluster k-2
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{k_clusters - 2}'], -literals[f'x_{i_prime}_{k_clusters - 2}']])
    
    # Clause 29: Conditional co-separation for non-adjacent clusters
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            # Apply the constraint for all clusters except the last (since we start from zero)
            for c in range(k_clusters - 2):                
                wcnf.append([literals[f'bw_m_{w}'], -literals[f'x_{i}_{c}'], 
                             -literals[f'x_{i_prime}_{c}'], literals[f'x_{i}_{c+1}'], literals[f'x_{i_prime}_{c+1}']])

    # Clause 30: If b^+_w is true, then pairs (i, i') in distance class w must be in the same cluster
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            for c in range(k_clusters - 1):  # Iterate through clusters 0 to k-1
                wcnf.append([-literals[f'bw_p_{w}'], -literals[f'x_{i}_{c}'], literals[f'x_{i_prime}_{c}']])

    # Clause 31: If b^+_w is true, then points (i, i') in distance class w must be in the same cluster c
    for w, pairs_array in enumerate(distance_classes):
        for pair in pairs_array:
            i, i_prime = pair
            # The points i and i' should be in the same cluster if b^+_w is true.
            # We check across all clusters from 0 to k-2 because the clusters are 0-indexed
            for c in range(k_clusters - 1):
                wcnf.append([-literals[f'bw_p_{w}'], literals[f'x_{i}_{c}'], -literals[f'x_{i_prime}_{c}']])

    # Clause 32: Ensures that if bw^-_w is true, then the points in distance class w 
    # cannot be clustered with the points in distance class w-1 if bw^-_(w-1) is false.
    for w in range(1, len(distance_classes)):  # Starting from 1 since we're checking w against w-1
        wcnf.append([-literals[f'bw_m_{w}'], literals[f'bw_m_{w-1}']])
    
    # Clause 33: If bw^+_w is true, then distance class w must be clustered together with distance class w-1
    for w in range(1, len(distance_classes)):  # Starting from 1 since we're checking w against w-1
        wcnf.append([-literals[f'bw_p_{w}'], literals[f'bw_p_{w-1}']])
    
    # Clause 34: If bw^+_w is true, then distance class w cannot be clustered separately within itself
    for w in range(len(distance_classes)):
        wcnf.append([-literals[f'bw_p_{w}'], literals[f'bw_m_{w}']])

    # Clause 37: For each distance class w, we add a soft clause for the corresponding b^-_w literal
    # to encourage points within that class to be clustered separately
    # Max diameter and Min split Criteria solver
    for w in range(len(distance_classes)):
        wcnf.append([-literals[f'bw_m_{w}']], weight=1)

    # Clause 38: For each distance class w, we add a soft clause for the corresponding b^+_w literal
    # to encourage points within that class to be clustered together
    # This is for minimizing the simple combination of Lambda^- - Lambda^+ and is part of the objective function
    for w in range(len(distance_classes)):
        wcnf.append([literals[f'bw_p_{w}']], weight=1)   
    
    
    #clause 39:
    
    for i in range(len(X)):
        for c in range(k_clusters - 2):
            wcnf.append([literals[f'x_{i}_{c}'], -literals[f'x_{i}_{c+1}']])
            
    
    return wcnf

def create_literal_matrices_bicriteria(literals, solution, dataset_size, k_clusters, TB, TL, num_features, distance_classes):
    # Initialize matrices with zeros
    a_matrix = np.zeros((len(TB), num_features), dtype=int)
    s_matrix = np.zeros((dataset_size, len(TB)), dtype=int)
    z_matrix = np.zeros((dataset_size, len(TL)), dtype=int)
    g_matrix = np.zeros((len(TL), k_clusters), dtype=int)
    x_i_c_matrix = np.zeros((dataset_size, k_clusters), dtype=int)  # Added x_i_c matrix back
    bw_m_vector = np.zeros(len(distance_classes), dtype=int)  # Corrected length based on the number of distance classes
    bw_p_vector = np.zeros(len(distance_classes), dtype=int)  # Initialize the bw_p_vector with zeros
    
    # Helper to update the matrix based on the literal and its presence in the solution
    def update_matrix(matrix, i, j, literal_index):
        if literal_index in solution:
            matrix[i, j] = 1
        elif -literal_index in solution:
            matrix[i, j] = 0

    # Iterate over all literals to fill in the matrices
    for literal, index in literals.items():
        parts = literal.split('_')
        if literal.startswith('a_'):
            t = TB.index(int(parts[1]))
            j = int(parts[2])
            update_matrix(a_matrix, t, j, index)
        elif literal.startswith('s_'):
            i = int(parts[1])
            t = TB.index(int(parts[2]))
            update_matrix(s_matrix, i, t, index)
        elif literal.startswith('z_'):
            i = int(parts[1])
            t = TL.index(int(parts[2]))
            update_matrix(z_matrix, i, t, index)
        elif literal.startswith('g_'):
            t = TL.index(int(parts[1]))
            c = int(parts[2])
            update_matrix(g_matrix, t, c, index)
        elif literal.startswith('x_'):
            i = int(parts[1])
            c = int(parts[2])
            update_matrix(x_i_c_matrix, i, c, index)

    for literal, index in literals.items():
        if literal.startswith('bw_m_'):
            w = int(literal.split('_')[2])  # Extract the class index directly from the literal
            # If the index is positive in the solution, it's true, else false.
            bw_m_vector[w] = 1 if index in solution else 0

    for literal, index in literals.items():
        if literal.startswith('bw_p_'):
            w = int(literal.split('_')[2])  # Extract the class index directly from the literal
            # If the index is positive in the solution, it's true, else false.
            bw_p_vector[w] = 1 if index in solution else 0

    # Return the matrices
    return a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector

def create_literal_matrices_bicriteria_CC(literals, solution, dataset_size, k_clusters, TB, TL, num_features, distance_classes):
    # Initialize matrices with zeros
    
    x_i_c_matrix = np.zeros((dataset_size, k_clusters), dtype=int)  # Added x_i_c matrix back
    bw_m_vector = np.zeros(len(distance_classes), dtype=int)  # Corrected length based on the number of distance classes
    bw_p_vector = np.zeros(len(distance_classes), dtype=int)  # Initialize the bw_p_vector with zeros
    
    # Helper to update the matrix based on the literal and its presence in the solution
    def update_matrix(matrix, i, j, literal_index):
        if literal_index in solution:
            matrix[i, j] = 1
        elif -literal_index in solution:
            matrix[i, j] = 0

    # Iterate over all literals to fill in the matrices
    for literal, index in literals.items():
        parts = literal.split('_')
        
        
        if literal.startswith('x_'):
            i = int(parts[1])
            c = int(parts[2])
            update_matrix(x_i_c_matrix, i, c, index)

    for literal, index in literals.items():
        if literal.startswith('bw_m_'):
            w = int(literal.split('_')[2])  # Extract the class index directly from the literal
            # If the index is positive in the solution, it's true, else false.
            bw_m_vector[w] = 1 if index in solution else 0

    for literal, index in literals.items():
        if literal.startswith('bw_p_'):
            w = int(literal.split('_')[2])  # Extract the class index directly from the literal
            # If the index is positive in the solution, it's true, else false.
            bw_p_vector[w] = 1 if index in solution else 0

    # Return the matrices
    return x_i_c_matrix, bw_m_vector, bw_p_vector

def min_split_clustering_problem(dataset,features,k_clusters, depth, epsilon = 0, CL_pairs = np.array([]), ML_pairs = np.array([])):
   
    dataset_size = len(dataset)
    num_features = len(features)
    dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
    tree_structure, TB, TL = build_complete_tree_clustering(depth)
    literals = create_literals_cluster_tree_bicriteria(TB, TL, features, k_clusters, dataset_size,distance_classes)
    wcnf = build_clauses_cluster_tree_MD_MS(literals, dataset, TB, TL, num_features, k_clusters,
                                  CL_pairs, ML_pairs, distance_classes)
    
    solution = solve_wcnf_clustering(wcnf)

    a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector = create_literal_matrices_bicriteria(
        literals=literals,
        solution=solution,
        dataset_size=len(dataset),
        k_clusters=k_clusters,
        TB=TB,
        TL=TL,
        num_features=len(features),
        distance_classes= distance_classes
    )

    cluster_assignments, cluster_diameters = assign_clusters(
        x_i_c_matrix, dataset, k_clusters
    )

    return cluster_assignments, cluster_diameters

def min_split_clustering_problem_CC(dataset,features,k_clusters, depth, epsilon = 0, CL_pairs = np.array([]), ML_pairs = np.array([])):
    
    dataset_size = len(dataset)
    num_features = len(features)
    dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
    tree_structure, TB, TL = build_complete_tree_clustering(depth)
    literals = create_literals_cluster_tree_bicriteria_CC(TB, TL, features, k_clusters, dataset_size,distance_classes)
    wcnf = build_clauses_cluster_tree_MD_MS_CC(literals, dataset, TB, TL, num_features, k_clusters,
                                  CL_pairs, ML_pairs, distance_classes)
    
    solution = solve_wcnf_clustering(wcnf)

    a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector, bw_p_vector = create_literal_matrices_bicriteria_CC(
        literals=literals,
        solution=solution,
        dataset_size=len(dataset),
        k_clusters=k_clusters,
        TB=TB,
        TL=TL,
        num_features=len(features),
        distance_classes= distance_classes
    )

    cluster_assignments, cluster_diameters = assign_clusters(
        x_i_c_matrix, dataset, k_clusters
    )

    return cluster_assignments, cluster_diameters

