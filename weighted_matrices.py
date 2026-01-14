import numpy as np
import pandas as pd

# Radiosensitivity rankings
SENSITIVITY_RANKINGS = {
    'Background': 4, 
    'Bone': 4, 
    'Obtur': 4,
    'TZ': 0,       
    'CG': 0,        
    'Bladder': 6,
    'SV': 2,        
    'Rectum': 9,   
    'NVB': 10,      
}

# Sami calculated these using mean voxel distances from Prostate centres
AVG_DISTANCES = {
    'Background': 81.75,
    'TZ': 20.27,      
    'CG': 14.44, 
    'Bladder': 45.27,    
    'Obtur': 49.67,      
    'Bone': 81.98,         
    'Rectum': 47.8,    
    'SV': 38.29,           
    'NVB': 29.50,  
}

def create_sens_matrix(sensitivity_rankings):
    labels = list(sensitivity_rankings.keys())
    values = list(sensitivity_rankings.values())
    n_classes = len(labels)

    # Create grids 
    sens_of_predicted = np.array([values] * n_classes)
    sens_of_actual    = np.array([values] * n_classes).T

    # Calculate error gap
    sens_matrix = sens_of_predicted - sens_of_actual

    # Apply 2x penalty for false negatives, dangerous errors (predicted < actual)
    FN_weighting = 2
    sens_matrix[sens_matrix < 0] = sens_matrix[sens_matrix < 0] * FN_weighting
    sens_matrix = np.abs(sens_matrix)

    return sens_matrix

def create_dist_matrix(avg_distances):
    labels = list(avg_distances.keys())
    n_labels = len(labels)
    distances = list(avg_distances.values())

    distance_matrix = np.ones((n_labels, n_labels))
    for i in range(n_labels):
        distance_matrix[i, :] = distances[i]

    distance_weighting = 1 / distance_matrix

    distance_matrix = (distance_weighting * 81 / distance_weighting.sum())
    return distance_matrix

def create_weighted_matrix(sens_matrix, dist_matrix, sens_importance=0.9):
    if sens_importance < 0 or sens_importance > 1:
        raise ValueError(f'The sensitivity importance should be between 0 and 1 but got sens_importance={sens_importance}')
    if sens_matrix is None and dist_matrix is None:
        raise ValueError(f'should specify at least one of sens_matrix or dist_matrix if using a weighted matrix')
    
    # if one of the matrices is None, see the other to ones
    if sens_matrix is None:
        sens_matrix = np.ones(dist_matrix.shape)
    if dist_matrix is None:
        dist_matrix = np.ones(sens_matrix.shape)
    
    if sens_matrix.shape != dist_matrix.shape:
        raise ValueError(f'the distance and sensitivity matrices must be of the same dimensions but got sens_matrix.shape={sens_matrix.shape},dist_matrix.shape={dist_matrix.shape}')
    if sens_matrix.shape[0] != sens_matrix.shape[1]:
        raise ValueError(f'the sensitivity matrix must be square but got sens_matrix.shape={sens_matrix.shape}')
    if dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError(f'the distasnce matrix must be square but got dist_matrix.shape={dist_matrix.shape}')

    n_labels = sens_matrix.shape[0]
    accuracy_matrix = 1 - np.eye(n_labels) 
    # Normalize
    sens_matrix = sens_matrix * 72 / sens_matrix.sum() # Now both sens and acc matrices have a total of 72

    # Calculate overall importance
    importance_matrix = (sens_importance * sens_matrix * dist_matrix + (1 - sens_importance) * accuracy_matrix)

    # Apply distances
    # overall_matrix = importance_matrix * distance_matrix # We multiply both the accuracy and sensitivity matrices by distance, since both concerns are more important closer to the region of interest i.e. prostate

    # Normalize a final time
    final_matrix = importance_matrix * 72 / importance_matrix.sum()
    final_matrix = np.round(final_matrix, 2)

    return list(final_matrix)