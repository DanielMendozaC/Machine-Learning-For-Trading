# DTLearner.py

""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
class DTLearner(object):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This is a Decision Tree Learner. It builds a decision tree for regression.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param leaf_size: Maximum number of samples in a leaf  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type leaf_size: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param verbose: If "verbose" is True, your code can print out information for debugging.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return "dcarbono3"  # replace tb34 with your Georgia Tech username  

    def study_group(self):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Returns
            A comma separated string of GT_Name of each member of your study group
            # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return "dcarbono3"  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.tree = self._build_tree(data_x, data_y)
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def _build_tree(self, data_x, data_y):
        """
        Recursively build the decision tree
        Returns a numpy array representing the tree
        """
        # Base cases for leaf creation
        
        # Case 1: If we have leaf_size or fewer samples, create a leaf
        if data_x.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), -1, -1]])
        
        # Case 2: If all Y values are the same, create a leaf
        if np.all(data_y == data_y[0]):
            return np.array([[-1, data_y[0], -1, -1]])
        
        # Find the best feature to split on (highest absolute correlation with Y)
        best_feature = -1
        best_corr = 0
        
        for feature in range(data_x.shape[1]):
            # Calculate correlation between this feature and Y
            if np.std(data_x[:, feature]) != 0:  # Avoid division by zero
                corr = abs(np.corrcoef(data_x[:, feature], data_y)[0, 1])
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_feature = feature
        
        # If no valid feature found, create a leaf
        if best_feature == -1:
            return np.array([[-1, np.mean(data_y), -1, -1]])
        
        # Split value is the median of the best feature
        split_val = np.median(data_x[:, best_feature])
        
        # Handle case where all values are the same (median equals all values)
        if np.all(data_x[:, best_feature] == split_val):
            return np.array([[-1, np.mean(data_y), -1, -1]])
        
        # Split the data
        left_mask = data_x[:, best_feature] <= split_val
        right_mask = data_x[:, best_feature] > split_val
        
        # If either split is empty, create a leaf
        if not np.any(left_mask) or not np.any(right_mask):
            return np.array([[-1, np.mean(data_y), -1, -1]])
        
        # Recursively build left and right subtrees
        left_tree = self._build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self._build_tree(data_x[right_mask], data_y[right_mask])
        
        # Build the current node
        # Format: [feature, split_val, left_start, right_start]
        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
        
        # Combine root, left subtree, and right subtree
        return np.vstack([root, left_tree, right_tree])

    def query(self, points):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  	
        if self.tree is None:
            raise ValueError("Tree has not been trained yet. Call add_evidence first.")
        
        predictions = np.zeros(points.shape[0])
        
        for i, point in enumerate(points):
            predictions[i] = self._query_single(point, 0)
        
        return predictions
    
    def _query_single(self, point, node_index):
        """
        Query a single point through the tree starting at node_index
        """
        node = self.tree[node_index]
        feature = int(node[0])
        
        # If this is a leaf node (feature == -1), return the value
        if feature == -1:
            return node[1]
        
        split_val = node[1]
        left_start = int(node[2])
        right_start = int(node[3])
        
        # Navigate to left or right child based on the split
        if point[feature] <= split_val:
            return self._query_single(point, node_index + left_start)
        else:
            return self._query_single(point, node_index + right_start)
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")