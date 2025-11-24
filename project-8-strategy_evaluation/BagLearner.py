# BagLearner.py
import numpy as np


class BagLearner(object):
    """Ensemble learner implementing bootstrap aggregation"""
    
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.base_model = learner
        self.model_params = kwargs
        self.num_bags = bags
        self.boost = boost
        self.verbose = verbose
        self.ensemble = []
  
    def author(self):
        return "dcarbono3"

    def study_group(self):
        return "dcarbono3"
  
    def add_evidence(self, data_x, data_y):
        """Train ensemble with provided dataset"""
        self.ensemble = []
        sample_count = data_x.shape[0]
        
        bag_idx = 0
        while bag_idx < self.num_bags:
            model = self.base_model(**self.model_params)
            
            sampled_indices = np.random.choice(sample_count, size=sample_count, replace=True)
            x_subset = data_x[sampled_indices]
            y_subset = data_y[sampled_indices]
            
            model.add_evidence(x_subset, y_subset)
            self.ensemble.append(model)
            
            if self.verbose:
                print(f"Trained bag {bag_idx+1}/{self.num_bags}")
            
            bag_idx += 1

    def query(self, points):
        """Generate predictions by aggregating ensemble outputs"""
        if len(self.ensemble) == 0:
            raise ValueError("No learners trained")
        
        num_models = len(self.ensemble)
        num_points = points.shape[0]
        prediction_matrix = np.zeros((num_models, num_points))
        
        model_idx = 0
        for model in self.ensemble:
            prediction_matrix[model_idx] = model.query(points)
            model_idx += 1
        
        aggregated_predictions = np.mean(prediction_matrix, axis=0)
        return aggregated_predictions