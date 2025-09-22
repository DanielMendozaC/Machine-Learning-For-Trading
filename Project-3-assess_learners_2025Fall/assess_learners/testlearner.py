""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import math  		  	   		 	 	 		  		  		    	 		 		   		 		  
import sys  		  	   		 	 	 		  		  		    	 		 		   		 		  
import time
import matplotlib.pyplot as plt
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it

def load_data(filename):
    """Load data from CSV file, skipping header row and handling date column"""
    inf = open(filename)
    lines = inf.readlines()
    inf.close()
    
    # Skip the first line (header) and process data
    data_lines = []
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(",")
        # Skip the first column (date) and convert rest to float
        if len(parts) > 1:
            try:
                data_lines.append([float(x) for x in parts[1:]])
            except ValueError:
                continue  # Skip any problematic lines
    
    return np.array(data_lines)

def calculate_rmse(actual, predicted):
    """Calculate Root Mean Square Error"""
    return math.sqrt(((actual - predicted) ** 2).sum() / actual.shape[0])

def experiment_1_overfitting():
    """
    Experiment 1: Research overfitting with DTLearner
    Vary leaf_size and observe overfitting
    """
    print("Running Experiment 1: DTLearner Overfitting Analysis")
    
    # Set random seed for reproducibility
    np.random.seed(903123456)  # Using GT ID as suggested
    
    # Load data
    data = load_data(sys.argv[1])
    
    # Randomly shuffle and split data (60% train, 40% test)
    np.random.shuffle(data)
    train_rows = int(0.6 * data.shape[0])
    
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]
    
    # Range of leaf sizes to test
    leaf_sizes = range(1, 51)
    in_sample_rmse = []
    out_sample_rmse = []
    
    for leaf_size in leaf_sizes:
        # Create and train DTLearner
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        
        # In-sample predictions and RMSE
        pred_y_train = learner.query(train_x)
        rmse_train = calculate_rmse(train_y, pred_y_train)
        in_sample_rmse.append(rmse_train)
        
        # Out-of-sample predictions and RMSE
        pred_y_test = learner.query(test_x)
        rmse_test = calculate_rmse(test_y, pred_y_test)
        out_sample_rmse.append(rmse_test)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, in_sample_rmse, 'b-', label='In-sample RMSE', linewidth=2)
    plt.plot(leaf_sizes, out_sample_rmse, 'r-', label='Out-of-sample RMSE', linewidth=2)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('DTLearner: Overfitting Analysis (RMSE vs Leaf Size)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiment1_overfitting.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Experiment 1 complete. Chart saved as 'experiment1_overfitting.png'")
    return leaf_sizes, in_sample_rmse, out_sample_rmse

def experiment_2_bagging():
    """
    Experiment 2: Effect of bagging on overfitting
    Compare DTLearner with and without bagging
    """
    print("Running Experiment 2: Bagging Effect on Overfitting")
    
    # Set random seed for reproducibility
    np.random.seed(903123456)
    
    # Load data
    data = load_data(sys.argv[1])
    
    # Randomly shuffle and split data
    np.random.shuffle(data)
    train_rows = int(0.6 * data.shape[0])
    
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]
    
    # Range of leaf sizes to test
    leaf_sizes = range(1, 51)
    
    # DTLearner without bagging
    dt_in_sample_rmse = []
    dt_out_sample_rmse = []
    
    # BagLearner with DTLearner (20 bags)
    bag_in_sample_rmse = []
    bag_out_sample_rmse = []
    
    for leaf_size in leaf_sizes:
        # Regular DTLearner
        dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        dt_learner.add_evidence(train_x, train_y)
        
        pred_y_train = dt_learner.query(train_x)
        dt_in_sample_rmse.append(calculate_rmse(train_y, pred_y_train))
        
        pred_y_test = dt_learner.query(test_x)
        dt_out_sample_rmse.append(calculate_rmse(test_y, pred_y_test))
        
        # Bagged DTLearner
        bag_learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, verbose=False)
        bag_learner.add_evidence(train_x, train_y)
        
        pred_y_train = bag_learner.query(train_x)
        bag_in_sample_rmse.append(calculate_rmse(train_y, pred_y_train))
        
        pred_y_test = bag_learner.query(test_x)
        bag_out_sample_rmse.append(calculate_rmse(test_y, pred_y_test))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(leaf_sizes, dt_in_sample_rmse, 'b-', label='DTLearner In-sample', linewidth=2)
    plt.plot(leaf_sizes, dt_out_sample_rmse, 'r-', label='DTLearner Out-of-sample', linewidth=2)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('DTLearner (No Bagging): RMSE vs Leaf Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(leaf_sizes, bag_in_sample_rmse, 'b--', label='BagLearner In-sample', linewidth=2)
    plt.plot(leaf_sizes, bag_out_sample_rmse, 'r--', label='BagLearner Out-of-sample', linewidth=2)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('BagLearner (20 bags): RMSE vs Leaf Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment2_bagging.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Experiment 2 complete. Chart saved as 'experiment2_bagging.png'")
    return leaf_sizes, dt_out_sample_rmse, bag_out_sample_rmse

def experiment_3_dt_vs_rt():
    """
    Experiment 3: Compare DTLearner vs RTLearner
    Use metrics other than RMSE, correlation, or time
    """
    print("Running Experiment 3: DTLearner vs RTLearner Comparison")
    
    # Set random seed for reproducibility
    np.random.seed(903123456)
    
    # Load data
    data = load_data(sys.argv[1])
    
    # Multiple random splits for more robust comparison
    num_trials = 10
    leaf_sizes = range(1, 21, 2)
    
    dt_mae_scores = []
    rt_mae_scores = []
    dt_max_error_scores = []
    rt_max_error_scores = []
    
    for leaf_size in leaf_sizes:
        dt_mae_trial = []
        rt_mae_trial = []
        dt_max_trial = []
        rt_max_trial = []
        
        for trial in range(num_trials):
            # Random shuffle for each trial
            data_copy = data.copy()
            np.random.shuffle(data_copy)
            train_rows = int(0.6 * data_copy.shape[0])
            
            train_x = data_copy[:train_rows, 0:-1]
            train_y = data_copy[:train_rows, -1]
            test_x = data_copy[train_rows:, 0:-1]
            test_y = data_copy[train_rows:, -1]
            
            # DTLearner
            dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
            dt_learner.add_evidence(train_x, train_y)
            dt_pred = dt_learner.query(test_x)
            
            # RTLearner
            rt_learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)
            rt_learner.add_evidence(train_x, train_y)
            rt_pred = rt_learner.query(test_x)
            
            # Calculate Mean Absolute Error (MAE)
            dt_mae = np.mean(np.abs(test_y - dt_pred))
            rt_mae = np.mean(np.abs(test_y - rt_pred))
            dt_mae_trial.append(dt_mae)
            rt_mae_trial.append(rt_mae)
            
            # Calculate Maximum Error
            dt_max_error = np.max(np.abs(test_y - dt_pred))
            rt_max_error = np.max(np.abs(test_y - rt_pred))
            dt_max_trial.append(dt_max_error)
            rt_max_trial.append(rt_max_error)
        
        # Average across trials
        dt_mae_scores.append(np.mean(dt_mae_trial))
        rt_mae_scores.append(np.mean(rt_mae_trial))
        dt_max_error_scores.append(np.mean(dt_max_trial))
        rt_max_error_scores.append(np.mean(rt_max_trial))
    
    # Create comparison plots
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(leaf_sizes, dt_mae_scores, 'bo-', label='DTLearner MAE', linewidth=2)
    plt.plot(leaf_sizes, rt_mae_scores, 'ro-', label='RTLearner MAE', linewidth=2)
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error')
    plt.title('DTLearner vs RTLearner: Mean Absolute Error Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(leaf_sizes, dt_max_error_scores, 'bs-', label='DTLearner Max Error', linewidth=2)
    plt.plot(leaf_sizes, rt_max_error_scores, 'rs-', label='RTLearner Max Error', linewidth=2)
    plt.xlabel('Leaf Size')
    plt.ylabel('Maximum Error')
    plt.title('DTLearner vs RTLearner: Maximum Error Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment3_dt_vs_rt.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Experiment 3 complete. Chart saved as 'experiment3_dt_vs_rt.png'")
    return leaf_sizes, dt_mae_scores, rt_mae_scores, dt_max_error_scores, rt_max_error_scores

if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    
    print("Starting experiments...")
    start_time = time.time()
    
    # Run all three experiments
    exp1_data = experiment_1_overfitting()
    exp2_data = experiment_2_bagging()
    exp3_data = experiment_3_dt_vs_rt()
    
    end_time = time.time()
    print(f"\nAll experiments completed in {end_time - start_time:.2f} seconds")
    print("Charts saved as PNG files in current directory")
    
    # Save numerical results to text file for report
    with open('p3_results.txt', 'w') as f:
        f.write("Project 3 Experimental Results\n")
        f.write("================================\n\n")
        
        f.write("Experiment 1 - Overfitting Analysis:\n")
        f.write("Leaf sizes tested: 1 to 50\n")
        f.write("Minimum out-of-sample RMSE achieved at leaf_size = " + 
                str(exp1_data[0][np.argmin(exp1_data[2])]) + "\n")
        f.write("Overfitting appears to occur when leaf_size < " + 
                str(exp1_data[0][np.argmin(exp1_data[2])]) + "\n\n")
        
        f.write("Experiment 2 - Bagging Effect:\n")
        f.write("Compared DTLearner vs BagLearner (20 bags)\n")
        f.write("Bagging generally reduces overfitting and improves out-of-sample performance\n\n")
        
        f.write("Experiment 3 - DTLearner vs RTLearner:\n")
        f.write("Metrics used: Mean Absolute Error (MAE) and Maximum Error\n")
        f.write("Results show trade-offs between deterministic and random tree approaches\n")