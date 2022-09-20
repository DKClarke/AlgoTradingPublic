from sklearn import svm

import backtrader as bt

import pandas as pd

from modelFuncs import getFeaturesAndTargets, getListOfAllCombos, getFeatures

import backtrader.analyzers as btanalyzers
from itertools import product
from tqdm.auto import tqdm
import time

import concurrent.futures

from backtrader.feeds import PandasData

class SignalData(PandasData):
    """
    Define pandas DataFrame structure
    """
    OHLCP = ['open', 'high', 'low', 'close', 'predicted']
    cols = OHLCP
    
    # create lines
    lines = tuple(cols)
    
    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())

# Create a Stratey
class SVRStrategy(bt.Strategy):
    
    # Strategy parameters here where we can sub in different values to optimise our strategy
    params = (

        # The model we use to predict whether to buy or sell
        #('model', svm.SVR(cache_size= 1000)),

        # The hyperparameters for the model
        #('C', 32),
        #('epsilon', 0.05),
        #('gamma', 0.1),
        ('Thigh', 0.5),
        ('Tlow', 0.5),

        # How many previous prices are we basing our prediction on?
        #('lookbackWindow', 8),

        # Are we printing our log?
        ('printlog', False)
    )

    # Define the log method
    def log(self, txt, dt=None, doprint = False):

        ''' Logging function for this strategy'''
        if self.params.printlog or doprint == True:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):

        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.data_predicted = self.datas[0].predicted

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Keep track of the most recent prices
        #self.lastprices = list()

        # Create a dictionary of our parameters
        #paramsDict = {'C': self.params.C, 'epsilon': self.params.epsilon, 'gamma': self.params.gamma}
        
        # Set the parameters of the model
        #self.params.model.set_params(**paramsDict)
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm), doprint = False)

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm), doprint = False)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None
        
    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm), doprint = False)

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        
        # Store the most recent close price
        #self.lastprices.append(self.dataclose[0])
        
        # If we have as many prices stored as we need
        #if len(self) > self.params.lookbackWindow:
            
            # Turn the list of most recent prices into a pandas series
            #featuresRaw = pd.Series(self.lastprices)
            
            # Generate normalised prices and FFT amplitudes and phases in our lookback window
            #features, targets = generateNormalizedLookback(featuresRaw, pd.Series(np.ones(len(featuresRaw))), self.params.lookbackWindow)
            #amps, phases = generateFFTLookback(featuresRaw, self.params.lookbackWindow)
            
            # Drop the oldest price in our list
            #self.lastprices.pop(0)
            
            # Put the features together in a single np array
            #combinedFeatures = np.concatenate((features,amps,phases), axis=1)
            
            # Get what we predict the next price will be (in a normalised turning point oscillator)
            #predictions = self.params.model.predict(combinedFeatures)[0]
            
            # Print the prediction
            #self.log(predictions, doprint = False)
        
            # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
            
        # Check if we are in the market
        if not self.position:

            #if predictions < self.params.Tlow:
            if self.data_predicted < self.params.Tlow:   
                    
                    # BUY, BUY, BUY!!! (with default parameters)
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])

                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.buy()
                    
        else:
        
            #if predictions > self.params.Thigh:
            if self.data_predicted > self.params.Thigh:
                
                    # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

def runBacktraderMult(runDf, parametersDict, strategy = 'SVR', commission = 0.001, cash = 20000, srtimeframe = bt.TimeFrame.Days, C=32, epsilon=0.05, gamma=0.01):

    data = SignalData(dataname=runDf)

    # Generate a cerebro instance, add cash, add validation data
    #cerebro = bt.Cerebro(exactbars = True, stdstats = False)
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    #btData = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio_A, _name='mysharpe', timeframe = srtimeframe)
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name = 'trades')
    cerebro.addanalyzer(btanalyzers.Transactions, _name = 'transactions')

    # Compute the number of parameter combinations we're trying
    noCombos = len(list(product(*(parametersDict[Name] for Name in parametersDict))))

    if strategy == 'SVR':
        strat = SVRStrategy
    
    # If we're trying multiple
    if noCombos > 1:

        # Add a strategy
        cerebro.optstrategy(
            strat,
            **parametersDict
        )

        cerebro.optcallback(cb=bt_opt_callback)

        valResults = cerebro.run(maxcpus = 1)
        
        return {
            'results': valResults
            ,'C': C
            , 'epsilon': epsilon
            , 'gamma': gamma
        }

    else:

        raise ValueError('List of parameters is not longer than 1, why are you using multiprocessing?')

# Define a callback function to update the progress bar
def bt_opt_callback(cb):
    pbar.update()

def runBacktrader(runDf, parametersDict, strategy = 'SVR', cpus = 1, commission = 0.001, cash = 20000, srtimeframe = bt.TimeFrame.Days, C=32, epsilon=0.05, gamma=0.01):

    data = SignalData(dataname=runDf)

    # Generate a cerebro instance, add cash, add validation data
    #cerebro = bt.Cerebro(exactbars = True, stdstats = False)
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    #btData = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio_A, _name='mysharpe', timeframe = srtimeframe)
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name = 'trades')
    cerebro.addanalyzer(btanalyzers.Transactions, _name = 'transactions')

    # Compute the number of parameter combinations we're trying
    noCombos = len(list(product(*(parametersDict[Name] for Name in parametersDict))))

    if strategy == 'SVR':
        strat = SVRStrategy
    
    # If we're trying multiple
    if noCombos > 1:

        # Add a strategy
        cerebro.optstrategy(
            strat,
            **parametersDict
        )

        # Create a progress bar
        global pbar
        pbar = tqdm(smoothing=0.05, desc='Optimization Runs', total=noCombos)

        cerebro.optcallback(cb=bt_opt_callback)

        # If we've specified to use multiprocessing
        # Need to wrap this code in multiprocessing yourself as other it is broken w/ custom class
        if cpus == 'None':

            valResults = cerebro.run()
             
        else:

            valResults = cerebro.run(maxcpus = int(cpus))
        
        return {
            'results': valResults
            ,'C': C
            , 'epsilon': epsilon
            , 'gamma': gamma
        } 


    # If we're just running for one combo of parameters  
    else:

        # Format the input and run
        parametersSingle = {k: parametersDict[k][0] for k,v in parametersDict.items()}
        cerebro.addstrategy(strat, **parametersSingle)

        valResults = cerebro.run()

        # Return both the output and the cerebro so we can use cerebro to plot
        return {
            'strat_results': valResults
            , 'cerebro': cerebro
        }