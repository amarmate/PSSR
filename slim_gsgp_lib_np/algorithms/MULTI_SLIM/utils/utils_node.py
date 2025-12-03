
def calculate_cluster_idx(node): 
    """
    Calculate the cluster indices for a piecewise tree node in the ensemble tree.
    
    Parameters
    ----------
    node : Node
        The node for which the cluster indices are calculated.  
        
    Returns
    -------
    list
        A list of cluster indices for the specified node.
    """

    cluster_indices = []
    def recursion(node): 
        if node.children: 
            recursion(node.children[0]) 
            recursion(node.children[1])
        else: 
            cluster_indices.append(node.indices)
    recursion(node)
    return cluster_indices


def children_ids(node): 
    """
    Get the children ids of a piecewise tree node in the ensemble tree.
    
    Parameters
    ----------
    node : Node
        The node for which the children ids are calculated.
    
    Returns
    -------
    list
        A list of tuples representing the paths to the children nodes.
    """

    child_ids = []
    def recursion(node, current_path=()):
        if node.children: 
            # There are children still 
            recursion(node.children[0] , current_path + (1,))
            recursion(node.children[1], current_path + (2,))
        else: 
            child_ids.append(current_path)
    recursion(node)
    return child_ids
