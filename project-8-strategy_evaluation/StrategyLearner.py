# StrategyLearner.py

""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
GT User ID: dcarbono3 (replace with your User ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 904060775 (replace with your GT ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 	 	 		  		  		    	 		 		   		 		  
import random  		  	
import numpy as np

  		  	   		 	 	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	 	 		  		  		    	 		 		   		 		  
import util as ut  		  	
from indicators import bollinger_band_percent, rsi, macd_histogram
from RTLearner import RTLearner
from BagLearner import BagLearner   		

def author():
    return "dcarbono3"  

def study_group():
    return "dcarbono3"  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # constructor  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.model = BagLearner(learner=RTLearner, kwargs={"leaf_size": 5}, 
                                 bags=50, boost=False, verbose=False)
        self.is_trained = False
        self.lookback_period = 5

    def _compute_technical_indicators(self, price_data):
        """
        Helper method to compute technical indicators from price data
        
        :param price_data: Series of price data
        :return: DataFrame with indicator features
        """
        bb_percent = bollinger_band_percent(price_data, lookback=20)
        rsi_indicator = rsi(price_data, period=14)
        macd_hist = macd_histogram(price_data)
        
        indicator_df = pd.DataFrame(index=price_data.index)
        indicator_df['bbp'] = bb_percent
        indicator_df['rsi'] = rsi_indicator / 100.0
        indicator_df['macd'] = macd_hist
        indicator_df.fillna(0, inplace=True)
        
        return indicator_df
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		 	 	 		  		  		    	 		 		   		 		  
    ):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # add your code to do learning here  		
        date_range = pd.date_range(sd, ed)
        price_data = ut.get_data([symbol], date_range)
        price_data = price_data[[symbol]]
        price_data.ffill(inplace=True)
        price_data.bfill(inplace=True)
        
        if self.verbose:
            print(f"Training on {symbol} from {sd} to {ed}")
        
        # Compute technical indicators
        stock_prices = price_data[symbol]
        feature_data = self._compute_technical_indicators(stock_prices)
        
        # Calculate forward-looking returns for labeling
        forward_price_change = (price_data.shift(-self.lookback_period) - price_data) / price_data
        training_labels = np.where(forward_price_change > 0.02 + self.impact, 1,
                                   np.where(forward_price_change < -0.02 - self.impact, -1,
                                           0))
        
        # Prepare training data (remove last lookback_period rows)
        feature_matrix = feature_data.values[:-self.lookback_period]
        label_vector = training_labels[:-self.lookback_period].flatten()
        
        # Train the model
        self.model.add_evidence(feature_matrix, label_vector)
        self.is_trained = True
        
        if self.verbose:
            print("Training complete")	    		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  	  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		 	 	 		  		  		    	 		 		   		 		  
    ):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	 	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	 	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	 	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	
        if not self.is_trained:
            raise Exception("Must train learner first")
        
        date_range = pd.date_range(sd, ed)
        price_data = ut.get_data([symbol], date_range)
        price_data = price_data[[symbol]]
        price_data.ffill(inplace=True)
        price_data.bfill(inplace=True)
        
        # Compute technical indicators
        stock_prices = price_data[symbol]
        feature_data = self._compute_technical_indicators(stock_prices)
        
        # Get model predictions
        feature_matrix = feature_data.values
        model_predictions = self.model.query(feature_matrix)
        
        # Initialize trades DataFrame
        trades_df = pd.DataFrame(0, index=price_data.index, columns=[symbol])
        current_position = 0
        
        # Generate trading signals
        for day_idx in range(len(price_data)):
            if day_idx < 26:
                continue
            
            predicted_signal = int(np.round(model_predictions[day_idx]))
            
            if predicted_signal == 1 and current_position <= 0:
                if current_position == 0:
                    trades_df.iloc[day_idx] = 1000
                elif current_position == -1000:
                    trades_df.iloc[day_idx] = 2000
                current_position = 1000
            elif predicted_signal == -1 and current_position >= 0:
                if current_position == 0:
                    trades_df.iloc[day_idx] = -1000
                elif current_position == 1000:
                    trades_df.iloc[day_idx] = -2000
                current_position = -1000
        
        if self.verbose:
            print(f"Generated {(trades_df != 0).sum()[0]} trades")
        
        return trades_df		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 			  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		  	   		 	 	 		  		  		    	 		 		   		 		  
