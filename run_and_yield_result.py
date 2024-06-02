from clustering_solver import *
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import json


def load_dataset(file_path_to_test_,delimiter_,label_position_):
        
    #file_path_to_test = 'Datasets/iris/iris.data'
    data_loader = TreeDataLoaderNumerical(file_path=file_path_to_test_, delimiter=delimiter_, label_position= label_position_)
    print("Features:", data_loader.features, data_loader.features.shape)
    print("Labels:", data_loader.labels, data_loader.labels.shape)
    print("True Labels for Points:", data_loader.true_labels_for_points, data_loader.true_labels_for_points.shape)
    #print("Dataset:\n", data_loader.dataset,data_loader.dataset.shape)
    features = data_loader.features
    labels = data_loader.labels
    true_labels_for_points = data_loader.true_labels_for_points
    dataset = data_loader.dataset
    dataset_size = len(dataset)
    return features,labels,true_labels_for_points,dataset

def main():
    loandra_path = fr"/mnt/c/Users/FPT/Desktop/ds107/ESC499-Thesis-SAT-Trees/SATreeCraft/loandra"
    execution_path = fr"dimacs/export_to_solver.cnf"

    # features,labels,true_labels_for_points,dataset = load_dataset(file_path_to_test_ = 'Datasets/wine/wine.data',
    #                                                               delimiter_ = ',', label_position_ = 0)
    # run_experiment_with_dataset(true_labels_for_points = true_labels_for_points,
    #                             dataset = dataset,
    #                             features = features,
    #                             loandra_path = loandra_path,
    #                             execution_path = execution_path,
    #                             folder_to_load = 'results/wine')
    
    features,labels,true_labels_for_points,dataset = load_dataset(file_path_to_test_ = 'Datasets/seeds/seeds.csv',
                                                                  delimiter_ = '\s+', label_position_ = -1)
    run_experiment_with_dataset(true_labels_for_points = true_labels_for_points,
                                dataset = dataset,
                                features = features,
                                loandra_path = loandra_path,
                                execution_path = execution_path,
                                folder_to_load = 'results/seeds')
    

if __name__ == "__main__":
    main()
