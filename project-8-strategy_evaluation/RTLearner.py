# RTLearner.py

import numpy as np


class RTLearner(object):
    """Implements a Random Tree Learning algorithm"""
    
    def __init__(self, leaf_size=1, verbose=False):
        self.min_samples = leaf_size
        self.verbose = verbose
        self.model_structure = None
  
    def author(self):
        return "dcarbono3"

    def study_group(self):
        return "dcarbono3"
  
    def add_evidence(self, data_x, data_y):
        """Train the learner with provided dataset"""
        self.model_structure = self._construct_tree(data_x, data_y)
  
    def _construct_tree(self, data_x, data_y):
        """Build tree structure recursively using random feature selection"""
        if data_x.shape[0] <= self.min_samples:
            return np.array([[-1, np.mean(data_y), -1, -1]])
        
        if np.all(data_y == data_y[0]):
            return np.array([[-1, data_y[0], -1, -1]])
        
        selected_feature = np.random.randint(0, data_x.shape[1])
        threshold = np.median(data_x[:, selected_feature])
        
        if np.all(data_x[:, selected_feature] == threshold):
            return np.array([[-1, np.mean(data_y), -1, -1]])
        
        left_indices = data_x[:, selected_feature] <= threshold
        right_indices = data_x[:, selected_feature] > threshold
        
        if not np.any(left_indices) or not np.any(right_indices):
            return np.array([[-1, np.mean(data_y), -1, -1]])
        
        left_subtree = self._construct_tree(data_x[left_indices], data_y[left_indices])
        right_subtree = self._construct_tree(data_x[right_indices], data_y[right_indices])
        
        root = np.array([[selected_feature, threshold, 1, left_subtree.shape[0] + 1]])
        return np.vstack([root, left_subtree, right_subtree])

    def query(self, points):
        """Generate predictions for given test data"""
        if self.model_structure is None:
            raise ValueError("Tree has not been trained yet")
        
        results = np.zeros(points.shape[0])
        for i, point in enumerate(points):
            results[i] = self._traverse_tree(point, 0)
        return results
    
    def _traverse_tree(self, point, current_node):
        """Navigate through tree to find prediction for single data point"""
        node = self.model_structure[current_node]
        feature = int(node[0])
        
        if feature == -1:
            return node[1]
        
        threshold = node[1]
        left_start = int(node[2])
        right_start = int(node[3])
        
        if point[feature] <= threshold:
            return self._traverse_tree(point, current_node + left_start)
        else:
            return self._traverse_tree(point, current_node + right_start)