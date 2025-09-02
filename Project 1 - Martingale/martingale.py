""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  
import matplotlib.pyplot as plt	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def author():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return "dcarbono3"  # replace tb34 with your Georgia Tech username.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def gtid():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return 904060775  # replace with your GT ID number  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    result = False  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        result = True  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return result  	

def episode(win_prob):	  	   		 	 	 		  		  		    	 		 		   		 		  
    episode_winnings = 0
    # bet_number = 0
    money_history = []
    money_history.append(episode_winnings)
    while episode_winnings < 80:
        won = False
        bet_amount = 1
        while not won:
            # wager bet_amount on black
            won = get_spin_result(win_prob)
            # bet_number = bet_number + 1
            # print(f"Episode winnings: {episode_winnings}")

            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2
            money_history.append(episode_winnings)
    # print(f"Episode winnings: {episode_winnings}") 
    return money_history

def realistic_episode(win_prob):
    """
    Episode with $256 bankroll limit
    """
    episode_winnings = 0
    bankroll = 256
    money_history = [0]  # Start at 0
    
    while episode_winnings < 80 and episode_winnings > -256:
        won = False
        bet_amount = 1
        
        while not won and episode_winnings > -256:
            # Corner case: if next bet > remaining bankroll, bet what you have
            available_money = bankroll + episode_winnings  # Current money available
            if bet_amount > available_money:
                bet_amount = available_money
            
            # If no money left, break
            if bet_amount <= 0:
                break
                
            # Make the bet
            won = get_spin_result(win_prob)
            
            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2  # Double for next bet
            
            money_history.append(episode_winnings)
            
            # Safety: stop if bankrupt
            if episode_winnings <= -256:
                episode_winnings = -256
                break
    
    return money_history


  		  	   		 	 	 		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    win_prob = 18/38  # set appropriately to the probability of a win  		  	   		 	 	 		  		  		    	 		 		   		 		  
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # add your code here to implement the experiments  	
    # for i in range(2):
    #     print(f"Spin {i+1}: {get_spin_result(win_prob)}")

# FIGURE 1
    for i in range(10):
        money_history = episode(win_prob)
        plt.plot(money_history)
    plt.title("Martingale Strategy")
    plt.xlabel("Number of Bets")
    plt.ylabel("Money")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.savefig('Figure1.png')  
    plt.close()



# FIGURE 2
    # Create matrix to store all episodes
    episodes_list = []

    for i in range(1000):
        money_history = episode(win_prob) 
        episodes_list.append(money_history)

    # Find the maximum length of any episode
    max_length = max(len(ep_data) for ep_data in episodes_list)  # Also renamed here

    # Create a matrix with NaN values for missing data
    matrix = np.full((len(episodes_list), max_length), np.nan)

    # Fill the matrix with episode data
    for i, ep_data in enumerate(episodes_list):  # Changed 'episode' to 'ep_data'
        matrix[i, :len(ep_data)] = ep_data

    # Calculate mean and std across all episodes for each bet number (ignoring NaN)
    mean_money_history = np.nanmean(matrix, axis=0)
    std_money_history = np.nanstd(matrix, axis=0)

    plt.figure(figsize=(12, 8))
    plt.plot(mean_money_history, 'b-', linewidth=2, label='Mean')
    plt.plot(range(len(mean_money_history)), mean_money_history + std_money_history, 'r--', alpha=0.7, label='Mean + Std Dev')
    plt.plot(range(len(mean_money_history)), mean_money_history - std_money_history, 'r--', alpha=0.7, label='Mean - Std Dev')
    plt.title("Figure 2: Mean Martingale Strategy (1000 Episodes)")
    plt.xlabel("Number of Bets")
    plt.ylabel("Money")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Figure2.png')  
    plt.close()

# FIGURE 3
    # Calculate median and std across all episodes for each bet number (ignoring NaN)
    median_money_history = np.nanmedian(matrix, axis=0)
    std_money_history = np.nanstd(matrix, axis=0)  # Same std dev calculation

    # Create Figure 3
    plt.figure(figsize=(12, 8))

    # Plot median line
    plt.plot(range(len(median_money_history)), median_money_history, 'b-', linewidth=2, label='Median')

    # Add the two required lines
    plt.plot(range(len(median_money_history)), median_money_history + std_money_history, 'r--', alpha=0.7, label='Median + Std Dev')
    plt.plot(range(len(median_money_history)), median_money_history - std_money_history, 'r--', alpha=0.7, label='Median - Std Dev')

    plt.title("Figure 3: Median Martingale Strategy (1000 Episodes)")
    plt.xlabel("Number of Bets")
    plt.ylabel("Money")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Figure3.png')  
    plt.close()

    win_prob = 18/38
    
    # Run 1000 realistic episodes
    episodes_list = []
    print("Running 1000 realistic episodes...")
    
    for i in range(1000):
        if i % 100 == 0:
            print(f"Episode {i}/1000")
        money_history = realistic_episode(win_prob)
        episodes_list.append(money_history)
    
    # Create matrix with NaN for missing data
    max_length = max(len(ep) for ep in episodes_list)
    matrix = np.full((len(episodes_list), max_length), np.nan)
    
    for i, ep_data in enumerate(episodes_list):
        matrix[i, :len(ep_data)] = ep_data


# FIGURE 4: Mean
    mean_money_history = np.nanmean(matrix, axis=0)
    std_money_history = np.nanstd(matrix, axis=0)
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(mean_money_history)), mean_money_history, 'b-', linewidth=2, label='Mean')
    plt.plot(range(len(mean_money_history)), mean_money_history + std_money_history, 'r--', alpha=0.7, label='Mean + Std Dev')
    plt.plot(range(len(mean_money_history)), mean_money_history - std_money_history, 'r--', alpha=0.7, label='Mean - Std Dev')
    plt.fill_between(range(len(mean_money_history)), 
                     mean_money_history - std_money_history, 
                     mean_money_history + std_money_history, 
                     alpha=0.2, color='red')
    plt.title("Figure 4: Realistic Mean Martingale Strategy (1000 Episodes)")
    plt.xlabel("Number of Bets")
    plt.ylabel("Money")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Figure4.png')  
    plt.close()
    
# FIGURE 5: Median  
    median_money_history = np.nanmedian(matrix, axis=0)
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(median_money_history)), median_money_history, 'b-', linewidth=2, label='Median')
    plt.plot(range(len(median_money_history)), median_money_history + std_money_history, 'r--', alpha=0.7, label='Median + Std Dev')
    plt.plot(range(len(median_money_history)), median_money_history - std_money_history, 'r--', alpha=0.7, label='Median - Std Dev')
    plt.fill_between(range(len(median_money_history)), 
                     median_money_history - std_money_history, 
                     median_money_history + std_money_history, 
                     alpha=0.2, color='red')
    plt.title("Figure 5: Realistic Median Martingale Strategy (1000 Episodes)")
    plt.xlabel("Number of Bets")
    plt.ylabel("Money")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Figure5.png')  
    plt.close()

    return None




	  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	 	 		  		  		    	 		 		   		 		  
