
from sklearn.metrics import silhouette_score
import numpy as np
from collections import defaultdict
import itertools
from scipy.optimize import linear_sum_assignment

from slim_gsgp_lib_np.algorithms.MULTI_SLIM.utils.mask_handling import get_specialist_masks
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.utils.utils_node import calculate_cluster_idx



# ---------------------------- SILHOUETTE SCORE ----------------------------
def silhouette_score_ensemble(ensemble, X):
    """
    Calculate the silhouette score for an ensemble of specialist models. 

    Parameters
    ----------
    
    ensemble : Ensemble
        The ensemble of specialist models.
    X : np.ndarray
        Input data for calculating the silhouette score.

    Returns
    --------
    float
        The silhouette score for the ensemble.
        
    """
    ensemble_masks = get_specialist_masks(ensemble.collection, X)
    if len(ensemble_masks) <= 1:
        return 0.0
    
    ensemble_masks = [(key,np.where(submask)[0]) for key, submask in ensemble_masks.items()]
    labels = np.zeros(len(X), dtype=int)
    for i, (key, mask) in enumerate(ensemble_masks):
        labels[mask] = i 
        
    if len(np.unique(labels)) == 1:
        return 0.0

    return silhouette_score(X, labels)


def silhouette_score_node(node): 
    """
    Calculate the silhouette score for a piecewise tree node in the ensemble tree.
        
    Parameters
    ----------
    node : Node
        The node for which the silhouette score is calculated.
        
    Returns
    -------
    float
        The silhouette score for the specified node.
    """

    labels = np.zeros(len(node.X), dtype=int)
    cluster_indices = calculate_cluster_idx(node)
    
    for i, cluster in enumerate(cluster_indices):
        labels[cluster] = i
    
    # compute silhouette score
    return silhouette_score(node.X, labels)



# ---------------------------- SPLIT DISTANCES ----------------------------
def normalized_hamming_distance(mask1, mask2):
    """
    Compute the normalized Hamming distance between two boolean masks.
    The masks must be 1D numpy arrays of the same length.
    """
    return np.sum(mask1 != mask2) / mask1.size

def split_distance(gt_masks, pred_masks):
    """
    Compute the minimum distance between a predicted binary split (2 masks)
    and a multi-case ground truth by considering all ways to merge the ground
    truth masks into 2 clusters.
    """
    if type(pred_masks) != defaultdict:
        print('Warning: pred_masks is not a defaultdict')
        return None
    
    pred_masks = [value for key, value in pred_masks.items()]

    k = len(gt_masks)
    best_distance = np.inf
    best_combo = None
    
    # Enumerate all non-empty, non-full subsets up to symmetry
    for r in range(1, (k // 2) + 1):
        for combo in itertools.combinations(range(k), r):
            other = tuple(set(range(k)) - set(combo))
            
            # Avoid duplicate partitions when k is even and r == k/2
            if k % 2 == 0 and r == k // 2 and combo > other:
                continue
            
            # Aggregate masks by OR-ing within each cluster
            agg1 = np.any([gt_masks[i] for i in combo], axis=0)
            agg2 = np.any([gt_masks[i] for i in other], axis=0)
            
            # Compute distance for this partition
            dist = composite_mask_distance([agg1, agg2], pred_masks, lambda_penalty=0)
            best_distance = min(best_distance, dist)
            if best_distance == dist:
                best_combo = combo
    
    return best_distance, best_combo

def composite_mask_distance(ground_truth_masks, predicted_masks, lambda_penalty=0.5):
    """
    Compute a composite distance between two sets of masks.
    
    ground_truth_masks and predicted_masks are lists (or arrays) of boolean masks.
    The distance is computed as the average normalized Hamming distance for the optimal 
    matching (using the Hungarian algorithm) plus a penalty for the difference in the number 
    of masks.
    
    Parameters:
      ground_truth_masks: list of 1D numpy boolean arrays (ground truth)
      predicted_masks: list of 1D numpy boolean arrays (predicted from the model)
      lambda_penalty: penalty weight for each extra/missing mask.
      
    Returns:
      composite_distance: the computed composite distance.
    """
    n_gt = len(ground_truth_masks)
    n_pred = len(predicted_masks)
    
    # Create cost matrix for the pairwise distances
    cost_matrix = np.zeros((n_gt, n_pred))
    for i, gt_mask in enumerate(ground_truth_masks):
        for j, pred_mask in enumerate(predicted_masks):
            cost_matrix[i, j] = normalized_hamming_distance(gt_mask, pred_mask)
    
    # Solve the assignment problem for the best matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    total_cost = cost_matrix[row_ind, col_ind].sum()
    avg_cost = total_cost / len(row_ind)
    
    # Add penalty for unmatched masks (if the numbers differ)
    unmatched = abs(n_gt - n_pred)
    composite_distance = avg_cost + lambda_penalty * unmatched
        
    return composite_distance

