from collections import defaultdict
from itertools import chain
import numpy as np


def get_specialist_masks(tree_node, X_data, current_mask=None):

    """
    Function used to get the masks for each specialist to be used for specialist tunning

    Parameters
    ----------
    tree_node : tuple
        The tree node representing the current condition or specialist.
    X_data : np.ndarray
        The input data for which the masks are to be generated.
    current_mask : np.ndarray, optional
        The current mask to be used for filtering the data. If None, a mask of all True values is created.

    Returns
    -------
    dict
        A dictionary where keys are specialist indices and values are boolean masks indicating the data points assigned to each specialist.
        The masks are boolean arrays of the same length as X_data, indicating which data points are assigned to each specialist.
    """

    def recursion(tree_node, X_data, current_mask=None):
        if current_mask is None:
            current_mask = np.ones(X_data.shape[0], dtype=bool)
        
        if isinstance(tree_node, tuple):  # Condition node
            condition = tree_node[0]
            left_child = tree_node[1]
            right_child = tree_node[2]
            
            # Evaluate condition
            cond_results = condition.predict(X_data) > 0
            true_mask = current_mask & cond_results
            false_mask = current_mask & ~cond_results
            
            # Process both branches
            left_masks = recursion(left_child, X_data, true_mask)
            right_masks = recursion(right_child, X_data, false_mask)
            
            # Merge results while preserving all masks
            merged = defaultdict(list)
            for sp, masks in chain(left_masks.items(), right_masks.items()):
                merged[sp].extend(masks)
                
            return merged
            
        else:  # Specialist node (leaf)
            return {tree_node: [current_mask]}
    
    result = recursion(tree_node, X_data)
    merged = defaultdict(list)

    for ind, mask in result.items(): 
        merged[ind] = np.sum(mask, axis=0).astype(bool)

    return merged    

def calculate_specialist_rmse(ensemble_tree, X_train, y_train):
    """
    Calculates RMSE for each specialist on samples where their decision paths activate
    
    Parameters:
    ensemble_tree: The trained ensemble tree structure
    X_train (array-like): Training features
    y_train (array-like): True target values
    specialists (dict): Dictionary of specialist objects
    
    Returns:
    dict: Specialist IDs mapped to their path-specific RMSE scores
    """
    # Get all activation masks
    masks = get_specialist_masks(ensemble_tree.collection, X_train)
    specialists = ensemble_tree.SPECIALISTS
    
    # Calculate RMSE for each specialist
    rmse_scores = {}
    
    for sp_id, mask_list in masks.items():
        if sp_id not in specialists:
            continue
            
        # Combine all paths for this specialist
        combined_mask = mask_list
        n_samples = np.sum(combined_mask)
        
        if n_samples == 0:
            rmse_scores[sp_id] = (0, np.nan)
            continue
            
        # Get predictions for active samples
        X_active = X_train[combined_mask]
        y_true = y_train[combined_mask]
        y_pred = specialists[sp_id].predict(X_active)
        
        # Calculate RMSE
        error = y_true - y_pred
        rmse_value = np.sqrt(np.mean(error**2))
        rmse_scores[sp_id] = (n_samples, rmse_value)
    
    return rmse_scores

