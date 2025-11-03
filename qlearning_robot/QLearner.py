# QLearner.py

""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Daniel Mendoza
GT User ID: dcarbono3
GT ID: 904060775
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import random as rand
import numpy as np


class QLearner(object):
    
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        self.verbose = verbose
        self.n_states = num_states
        self.n_actions = num_actions
        self.lr = alpha
        self.df = gamma
        self.epsilon = rar
        self.eps_decay = radr
        self.n_dyna = dyna
        
        self.qtable = np.zeros((num_states, num_actions))
        self.prev_s = 0
        self.prev_a = 0
        
        if dyna > 0:
            self.transitions = np.full((num_states, num_actions), -1, dtype=int)
            self.rewards = np.zeros((num_states, num_actions))
            self.history = []

    def querysetstate(self, s):
        self.prev_s = s
        
        if rand.random() < self.epsilon:
            act = rand.randint(0, self.n_actions - 1)
        else:
            act = np.argmax(self.qtable[s])
        
        self.prev_a = act
        return act

    def query(self, s_prime, r):
        old_s, old_a = self.prev_s, self.prev_a
        
        best_next = np.max(self.qtable[s_prime])
        target = r + self.df * best_next
        self.qtable[old_s, old_a] += self.lr * (target - self.qtable[old_s, old_a])
        
        if self.n_dyna > 0:
            self.transitions[old_s, old_a] = s_prime
            self.rewards[old_s, old_a] = r
            
            pair = (old_s, old_a)
            if pair not in self.history:
                self.history.append(pair)
            
            for _ in range(self.n_dyna):
                sim_s, sim_a = rand.choice(self.history)
                next_s = self.transitions[sim_s, sim_a]
                rew = self.rewards[sim_s, sim_a]
                max_next = np.max(self.qtable[next_s])
                self.qtable[sim_s, sim_a] += self.lr * (rew + self.df * max_next - self.qtable[sim_s, sim_a])
        
        act = rand.randint(0, self.n_actions - 1) if rand.random() < self.epsilon else np.argmax(self.qtable[s_prime])
        self.epsilon *= self.eps_decay
        self.prev_s, self.prev_a = s_prime, act
        
        return act

    def author(self):
        return "dcarbono3"


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")