from clustering_problems.clustering_minsplit import *
from clustering_problems.clustering_advanced import *
import numpy as np
from loandra_support import loandra
from sklearn.metrics import adjusted_rand_score
from utils import *
import importlib
import random
import time
import json
importlib.reload(loandra)

def solve_clustering_problem_bicriteria_loandra(dataset,features,k_clusters, depth, epsilon, CL_pairs, ML_pairs,
                                                      loandra_path,execution_path):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree_clustering(depth)
        
        literals = create_literals_cluster_tree_bicriteria(TB, TL, features, k_clusters, dataset_size,distance_classes)
        wcnf = build_clauses_cluster_tree_MD_MS(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
    
        wcnf.to_file(execution_path)
       
        solution,cost = loandra.run_loandra_and_parse_results(loandra_path, execution_path)
        
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
        cluster_assignments, cluster_diameters = assign_clusters(x_i_c_matrix, dataset, k_clusters)
        return cluster_assignments, cluster_diameters

def solve_clustering_problem_bicriteria_loandra_CC(dataset,features,k_clusters, depth, epsilon, CL_pairs, ML_pairs,
                                                      loandra_path,execution_path):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree_clustering(depth)
        
        literals = create_literals_cluster_tree_bicriteria_CC(TB, TL, features, k_clusters, dataset_size,distance_classes)
        wcnf = build_clauses_cluster_tree_MD_MS_CC(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
    
        wcnf.to_file(execution_path)
       
        solution,cost = loandra.run_loandra_and_parse_results(loandra_path, execution_path)
        
        x_i_c_matrix, bw_m_vector, bw_p_vector = create_literal_matrices_bicriteria_CC(
                                                                                        literals=literals,
                                                                                        solution=solution,
                                                                                        dataset_size=len(dataset),
                                                                                        k_clusters=k_clusters,
                                                                                        TB=TB,
                                                                                        TL=TL,
                                                                                        num_features=len(features),
                                                                                        distance_classes= distance_classes
                                                                                        )
        cluster_assignments, cluster_diameters = assign_clusters(x_i_c_matrix, dataset, k_clusters)
        return cluster_assignments, cluster_diameters

def solve_clustering_problem_max_diameter_loandra(dataset,features,k_clusters, depth, epsilon, CL_pairs, ML_pairs,
                                                      loandra_path,execution_path):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree_clustering(depth)
        
        literals = create_literals_cluster_tree(TB, TL, features, k_clusters, dataset_size,distance_classes)
        wcnf = build_clauses_cluster_tree_MD(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
    
        wcnf.to_file(execution_path)
       
        solution,cost = loandra.run_loandra_and_parse_results(loandra_path, execution_path)
        
        a_matrix, s_matrix, z_matrix, g_matrix, x_i_c_matrix, bw_m_vector = create_literal_matrices(literals=literals,
                                                                                                    solution=solution,
                                                                                                    dataset_size=len(dataset),
                                                                                                    k_clusters=k_clusters,
                                                                                                    TB=TB,
                                                                                                    TL=TL,
                                                                                                    num_features=len(features),
                                                                                                    distance_classes= distance_classes
                                                                                                    )
        cluster_assignments, cluster_diameters = assign_clusters(x_i_c_matrix, dataset, k_clusters)
        
        return cluster_assignments, cluster_diameters

def solve_clustering_problem_max_diameter_loandra_CC(dataset,features,k_clusters, depth, epsilon, CL_pairs, ML_pairs,
                                                      loandra_path,execution_path):
        dataset_size = len(dataset)
        num_features = len(features)
        dist1, dist2, distance_classes = create_distance_classes(dataset, epsilon)
        tree_structure, TB, TL = build_complete_tree_clustering(depth)
        
        literals = create_literals_cluster_tree_CC(TB, TL, features, k_clusters, dataset_size,distance_classes)
        wcnf = build_clauses_cluster_tree_MD_CC(literals, dataset, TB, TL, num_features, k_clusters,
                                    CL_pairs, ML_pairs, distance_classes)
    
        wcnf.to_file(execution_path)
       
        solution,cost = loandra.run_loandra_and_parse_results(loandra_path, execution_path)
        
        x_i_c_matrix, bw_m_vector = create_literal_matrices_CC(literals=literals,
                                                                solution=solution,
                                                                dataset_size=len(dataset),
                                                                k_clusters=k_clusters,
                                                                TB=TB,
                                                                TL=TL,
                                                                num_features=len(features),
                                                                distance_classes= distance_classes
                                                                )
        cluster_assignments, cluster_diameters = assign_clusters(x_i_c_matrix, dataset, k_clusters)
        
        return cluster_assignments, cluster_diameters
    
def generate_constraints(true_labels, k, dataset_size, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    num_constraints = int(k * dataset_size)
    pairs = random.sample([(i, j) for i in range(dataset_size) for j in range(i + 1, dataset_size)], num_constraints)
    ML_pairs = []
    CL_pairs = []

    for (i, j) in pairs:
        if true_labels[i] == true_labels[j]:
            ML_pairs.append((i, j))
        else:
            CL_pairs.append((i, j))
    
    return np.array(ML_pairs), np.array(CL_pairs)

def run_experiment(func,k_list,true_labels_for_points,dataset,features,k_clusters,depth,epsilon,loandra_path,execution_path):
    dataset_size = len(dataset)
    result_list = {k: [] for k in k_list}
    time_list = {k: [] for k in k_list}
    for k in k_list:
        #print(f'Running with k = {k}')
        for random_state in range(11, 31):
            ML_pairs, CL_pairs = generate_constraints(true_labels_for_points, k, dataset_size,random_state=random_state)
            if k == 1.0 and random_state >= 22:
                result_list[k].append(0)
                time_list[k].append(1900)
                continue
            start_time = time.time()
            cluster_assignments, cluster_diameters = func(dataset=dataset,
                                                        features=features,
                                                        k_clusters=k_clusters,
                                                        depth = depth,
                                                        epsilon= epsilon,
                                                        CL_pairs=CL_pairs,
                                                        ML_pairs= ML_pairs,
                                                        loandra_path = loandra_path,
                                                        execution_path=execution_path)
            end_time = time.time()
            predicted_labels = [-1] * dataset_size
            for i, cluster in cluster_assignments.items():
                for point in cluster:
                    predicted_labels[point] = i
            ari = adjusted_rand_score(true_labels_for_points, predicted_labels)
            print(f"Running completed at k = {k},random state = {random_state}, depth = {depth}")
            print(f"ari = {ari}, time = {end_time - start_time}\n")
            result_list[k].append(ari)
            time_list[k].append(end_time - start_time)
    # result_list = {k: np.mean(result_list[k]) for k in k_list}
    # time_list = {k: np.mean(time_list[k]) for k in k_list}
    return result_list, time_list
def save_dict_to_json_file(dictionary, filename):
    """
    Saves a dictionary to a file in JSON format.

    :param dictionary: The dictionary to save.
    :param filename: The name of the file to save the dictionary in.
    """
    try:
        with open(filename, 'w') as file:
            json.dump(dictionary, file, indent=4)
        print(f"Dictionary successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary: {e}")
def run_experiment_with_dataset(true_labels_for_points,dataset,features,loandra_path,execution_path,folder_to_load):
    k_list = [0.0,0.1,0.25,0.5,1.0]
    # #MD
    # print('running MD')
    # MD_result_list,MD_time_list = run_experiment(func = solve_clustering_problem_max_diameter_loandra,
    #                                         k_list = k_list,
    #                                         true_labels_for_points = true_labels_for_points,
    #                                         dataset = dataset,
    #                                         features = features,
    #                                         k_clusters = 3,
    #                                         depth = 3,
    #                                         epsilon = 0.1,
    #                                         loandra_path = loandra_path,
    #                                         execution_path = execution_path
    #                                         )
    # save_dict_to_json_file(MD_result_list,f'{folder_to_load}/MD_result_list.json')
    # save_dict_to_json_file(MD_time_list,f'{folder_to_load}/MD_time_list.json')

    # #MD(CC)
    # print('running MD(CC)')
    # MD_CC_result_list,MD_CC_time_list = run_experiment(func = solve_clustering_problem_max_diameter_loandra_CC,
    #                                     k_list = k_list,           
    #                                     true_labels_for_points = true_labels_for_points,
    #                                     dataset = dataset,
    #                                     features = features,
    #                                     k_clusters = 3,
    #                                     depth = 3,
    #                                     epsilon = 0.1,
    #                                     loandra_path = loandra_path,
    #                                     execution_path = execution_path
    #                                     )
    # save_dict_to_json_file(MD_CC_result_list,f'{folder_to_load}/MD_CC_result_list.json')
    # save_dict_to_json_file(MD_CC_time_list,f'{folder_to_load}/MD_CC_time_list.json')
    #MS_MD
    print('running MS_MD')
    MS_MD_result_list,MS_MD_time_list = run_experiment(func = solve_clustering_problem_bicriteria_loandra,
                                        k_list = k_list,            
                                        true_labels_for_points = true_labels_for_points,
                                        dataset = dataset,
                                        features = features,
                                        k_clusters = 3,
                                        depth = 3,
                                        epsilon = 0.1,
                                        loandra_path = loandra_path,
                                        execution_path = execution_path
                                        )
    save_dict_to_json_file(MS_MD_result_list,f'{folder_to_load}/MS_MD_result_list.json')
    save_dict_to_json_file(MS_MD_time_list,f'{folder_to_load}/MS_MD_time_list.json')

    #MS_MD(CC)
    print('running MS_MD(CC)')
    MS_MD_CC_result_list,MS_MD_CC_time_list = run_experiment(func = solve_clustering_problem_bicriteria_loandra_CC,
                                        k_list = k_list,                  
                                        true_labels_for_points = true_labels_for_points,
                                        dataset = dataset,
                                        features = features,
                                        k_clusters = 3,
                                        depth = 3,
                                        epsilon = 0.1,
                                        loandra_path = loandra_path,
                                        execution_path = execution_path
                                        )
    save_dict_to_json_file(MS_MD_CC_result_list,f'{folder_to_load}/MS_MD_CC_result_list.json')
    save_dict_to_json_file(MS_MD_CC_time_list,f'{folder_to_load}/MS_MD_CC_time_list.json')

    #table 3
    print('running MS_MD in table 3')
    MS_MD_result_list_depth3,MS_MD_time_list_depth3 = run_experiment(func = solve_clustering_problem_bicriteria_loandra,
                                        k_list = [0.5],            
                                        true_labels_for_points = true_labels_for_points,
                                        dataset = dataset,
                                        features = features,
                                        k_clusters = 3,
                                        depth = 3,
                                        epsilon = 0.1,
                                        loandra_path = loandra_path,
                                        execution_path = execution_path
                                        )
    save_dict_to_json_file(MS_MD_result_list_depth3,f'{folder_to_load}/MS_MD_result_list_depth3.json')
    save_dict_to_json_file(MS_MD_time_list_depth3,f'{folder_to_load}/MS_MD_time_list_depth3.json')


    MS_MD_result_list_depth4,MS_MD_time_list_depth4 = run_experiment(func = solve_clustering_problem_bicriteria_loandra,
                                        k_list = [0.5],            
                                        true_labels_for_points = true_labels_for_points,
                                        dataset = dataset,
                                        features = features,
                                        k_clusters = 3,
                                        depth = 4,
                                        epsilon = 0.1,
                                        loandra_path = loandra_path,
                                        execution_path = execution_path
                                        )
    save_dict_to_json_file(MS_MD_result_list_depth4,f'{folder_to_load}/MS_MD_result_list_depth4.json')
    save_dict_to_json_file(MS_MD_time_list_depth4,f'{folder_to_load}/MS_MD_time_list_depth4.json')

    MS_MD_result_list_depth2,MS_MD_time_list_depth2 = run_experiment(func = solve_clustering_problem_bicriteria_loandra,
                                        k_list = [0.5],            
                                        true_labels_for_points = true_labels_for_points,
                                        dataset = dataset,
                                        features = features,
                                        k_clusters = 3,
                                        depth = 2,
                                        epsilon = 0.1,
                                        loandra_path = loandra_path,
                                        execution_path = execution_path
                                        )
    save_dict_to_json_file(MS_MD_result_list_depth2,f'{folder_to_load}/MS_MD_result_list_depth2.json')
    save_dict_to_json_file(MS_MD_time_list_depth2,f'{folder_to_load}/MS_MD_time_list_depth2.json')






