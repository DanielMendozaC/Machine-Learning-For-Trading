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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
class BagLearner(object):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This is a Bootstrap Aggregating Learner (Bagging). It creates an ensemble of learners
    using bootstrap sampling and averages their predictions.
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param learner: The learner class to use for each bag  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type learner: class
    :param kwargs: Keyword arguments to pass to each learner's constructor
    :type kwargs: dict
    :param bags: Number of learners to create in the ensemble
    :type bags: int
    :param boost: Whether to use boosting (optional, not implemented)
    :type boost: bool
    :param verbose: If "verbose" is True, your code can print out information for debugging.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 	 		  		  		    	 		 		   		 		  
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
        # Clear any existing learners
        self.learners = []
        
        # Get the number of samples
        n_samples = data_x.shape[0]
        
        # Create and train each bag
        for i in range(self.bags):
            # Create a new learner instance
            learner_instance = self.learner(**self.kwargs)
            
            # Bootstrap sampling: sample with replacement
            # Each bag should have the same number of samples as the original dataset
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Get the bootstrap sample
            bag_x = data_x[bootstrap_indices]
            bag_y = data_y[bootstrap_indices]
            
            # Train the learner on the bootstrap sample
            learner_instance.add_evidence(bag_x, bag_y)
            
            # Add the trained learner to our ensemble
            self.learners.append(learner_instance)
            
            if self.verbose:
                print(f"Trained bag {i+1}/{self.bags}")

    def query(self, points):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  	
        if len(self.learners) == 0:
            raise ValueError("No learners have been trained. Call add_evidence first.")
        
        # Get predictions from all learners
        all_predictions = np.zeros((len(self.learners), points.shape[0]))
        
        for i, learner in enumerate(self.learners):
            all_predictions[i] = learner.query(points)
        
        # Average the predictions across all learners
        return np.mean(all_predictions, axis=0)
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")