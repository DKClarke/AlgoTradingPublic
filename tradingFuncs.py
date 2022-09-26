from psutil import cpu_count
from sklearn import svm

import pandas as pd

from modelFuncs import getListOfAllCombos, getFeaturesAndTargets, getFeatures

from tqdm.auto import tqdm
import time
import numpy as np

import concurrent.futures
from collections import defaultdict

from multiprocessing import Pool, Process

def getModelDataMultAlt(argsList):

    # Create a progress bar
    pbar = tqdm(total=len(argsList))

    results = list()

    # Run our tradeStrategyOld class for all combinations of arguments
    with Pool() as pool:  
        
        future_parameters = [(pool.apply_async(getModelData, kwds=parameters), parameters) for parameters in argsList]
        for future, parameters in future_parameters:
            results.append(future.get())
            pbar.update()

        pool.close()
        pool.join()

    pbar.close()

    return results


def getModelDataMult(argsList):
    '''
    Function executes getModelData() as a multiprocessing task based on a list of arguments
    as provided by getDataArgs()
    '''

    # Create a progress bar
    pbar = tqdm(total=len(argsList))

    # Run our tradeStrategyOld class for all combinations of arguments
    with concurrent.futures.ProcessPoolExecutor() as executor:

        #processes = list()
        results = list()

        for chunk in (argsList[pos:pos + 32] for pos in range(0, len(argsList), 32)):

            d = defaultdict(list)
            for adict in chunk:
                for key, value in adict.items():
                    d[key].append(value)

            for result in executor.map(
                getModelData
                , d['model']
                , d['X_val']
                , d['valDates']
                , d['dataPred']
                , d['ThighRaw']
                , d['TlowRaw']
                , d['metaData']
            ):

                pbar.update()
                results.append(result)
                
    pbar.close()

    return results

def getDataArgs(trainData, valData, extremaFilter, lookbackWindow, intraHyperParameters, metaData):
    '''
    Function takes in train, validation. Takes a dictionary of how to filter our turning point
    preidction targets, a value for our lookback window, and a dictionary of all possible SVR hyperparameters
    we want to test. Returns a list of arguments to pass to getModelData. Also takes a dictionary (metaData)
    that can be used to pass metaData to the returned list

    Args:
    trainData -- pandas dataframe of data to train model with
    valData -- pandas dataframe of data to find the best hyperparameters with
    extremaFilter -- dictionary of values to filter our prediction target for
    lookbackWindow -- value of how far to extend our lookback window for feature generation
    intrahyperparameters -- dictionary of the different values we want optimise of our models hyperparameters
    metaData -- dictionary of metaData
    '''
    
    # Get our features and targets for training, and our features for validating
    X_train, y_train, trainDates = getFeaturesAndTargets(trainData.reset_index(), extremaFilter, lookbackWindow)
    
    # If we generate features for training our data
    if X_train is not None:

        # Generate our Thigh and Tlow values in 0.01 steps from 0 to 1
        ThighRaw = [x/100 for x in list(range(0,101))]
        TlowRaw = [x/100 for x in list(range(0,101))]
        
        X_val, valDates = getFeatures(valData, lookbackWindow)
            
        modelParams = getListOfAllCombos(intraHyperParameters)

        # Create a dictionary to save metadata for this run
        metaDict = dict()
        metaDict['feature_values'] = {'lookbackWindow': lookbackWindow}
        metaDict['feature_values'].update(extremaFilter)
        metaDict.update(metaData)
        
        # Compute the 'next' values for each relevant column
        valData['next_open'] = valData['Open'].shift(-1)
        valData['next_close'] = valData['Close'].shift(-1)
        valData['temp_index'] = valData.index
        valData['next_index'] = valData['temp_index'].shift(-1)
        valData.drop(columns=['temp_index'], inplace = True)

        X_val, valDates = getFeatures(valData, lookbackWindow)

        firstOpenPrice = valData['Open'].iloc[0]
        lastClosePrice = valData['Close'].iloc[-1]

        metaDict['tradingEnd'] = (firstOpenPrice, lastClosePrice)

        # Generate the instance of our SVR
        svr = svm.SVR(kernel = 'rbf')
        svr.fit(X_train, y_train)
        
        argsList = [
            {
            #'modelParams':{'C':x[0], 'epsilon':x[1]}
          'model': svr.set_params(C = x[0], epsilon = x[1])
          #, 'X_train': X_train
          #, 'y_train': y_train
          , 'X_val': X_val
          , 'valDates': valDates
          , 'dataPred': valData
          , 'ThighRaw': ThighRaw
          , 'TlowRaw': TlowRaw
          , 'metaData': {**metaDict,**{'C':x[0], 'epsilon':x[1]}}
          }
         
        for x in modelParams]
        
        return argsList
    
    else:
        
        return None

#def getModelData(modelParams, X_train, y_train, dataPred, ThighRaw, TlowRaw, metaData):
#def getModelData(modelParams, X_train, y_train, X_val, valDates, dataPred, ThighRaw, TlowRaw, metaData):
def getModelData(model, X_val, valDates, dataPred, ThighRaw, TlowRaw, metaData):

    '''
    Function takes parameters for an SVR, training data, validation data, and the Thigh and Tlow thresholds
    we want to attempt. Returns a tradeStrategyOld object for each threshold combination possible out of
    the lists provided

    Args:
    modelParams -- dictionary with k,v pairs of the hyperparameters to set for our SVR
    X_train -- training features
    y_train -- training targets
    dataPred -- raw validation data
    ThighRaw -- list of Thigh values to try
    TlowRaw -- list of Tlow values to try
    metaData -- dictionary of meta data 
    '''

    # Generate the instance of our SVR
    svr = model
    #svr.set_params(**modelParams)
    #svr.fit(X_train, y_train)

    predictions = svr.predict(X_val)

    # Create a dataset of our validation features and predictions
    predicted = pd.Series(data = predictions, index = valDates)

    # Filter out threshold values so they're restricted by what we've predicted
    threshParameters = {
    'Thigh': [x for x in ThighRaw if x >= (min(predictions)-0.01) and x <= (max(predictions)+0.01)]
    , 'Tlow': [x for x in TlowRaw if x >= (min(predictions)-0.01) and x <= (max(predictions)+0.01)]
    }

    # Get a list of all combinations of Thigh and Tlow
    paramListThresh = getListOfAllCombos(threshParameters)

    # Remove any combinations that make no sense i.e. they're the same or we sell instead of buy
    paramListThreshClean = [x for x in paramListThresh if (x[0] != x[1]) and (x[0] > x[1])]

    # Create a dictionary to save metadata for this run
    # metaDict = dict()
    #metaData['modelParams'] = modelParams
    # metaDict.update(metaData)

    threshResults = [getStratDataOld(y[1], y[0], predicted, dataPred.rename(columns = {'Open':'open', 'Close':'close'}), metaData) for y in paramListThreshClean]

    return threshResults

def getStratData(buyPoint, sellPoint, predicted):
    '''
    Function takes thresholds for when we want to buy and sell, our predictions, and our share data
    and returns a tradeStrategyOld object
    
    Args:
    buyPoint -- float
    sellPoint -- double
    predicted -- np.array
    '''

    if not isinstance(buyPoint, float) or not isinstance(sellPoint, float):
        raise TypeError('One of buyPoint or sellPoint is not a float')

    elif buyPoint > 1 or buyPoint < 0 or sellPoint > 1 or sellPoint < 0:
        raise ValueError('One of buyPoint or sellPoint is bigger than 1 or less than 0')

    # Filter for points where we would buy - if we don't have any, exit
    if len(predicted[predicted < buyPoint]) == 0:

        return None

    else:
        
        # Get the dates where we would sell
        sellPointsRaw = np.where(predicted > sellPoint)[0]

        # Create lists to store filtered dates of buying and selling
        buyPointsIdx = list()
        sellPointsIdx = list()
        
        # Set the index we previously sold on to None for now
        prevSellRow = None

        # For each buy point
        for idx in np.where(predicted < buyPoint)[0]:

            # If we have previously sold
            if not prevSellRow is None:

                # If our current buy point index is earlier than our last sell index
                if idx < prevSellRow:

                    # Skip this loop iteration
                    continue

            #print('Buy at ', valDates[idx])

            # Add this buy point date our list of dates
            buyPointsIdx.append(idx)
                
            # If we have a sell point after this buy point
            if np.any(sellPointsRaw > idx):

                #print('Sell point is ', sellRow)
                #print('Sell at ', valDates[sellPointsRaw[sellPointsRaw > idx]][0])

                sellPointsIdx.append(sellPointsRaw[sellPointsRaw > idx][0])

            # If we don't have a sell point after this buy point
            else:

                # Add exit the loop
                break

            # Set our previous sell point index to the one we've just added
            prevSellRow = sellPointsRaw[sellPointsRaw > idx][0]
        
        # Return our buy dates, sell dates, and the buy point and sell point values used
        return {
            'buyDates' : buyPointsIdx
            , 'sellDates' : sellPointsIdx
            , 'metaData' : {'buyPoint': buyPoint, 'sellPoint':sellPoint}
        }


def getStratDataOld(buyPoint, sellPoint, predicted, data, metaDictRaw):
    '''
    Function takes thresholds for when we want to buy and sell, our predictions, and our share data
    and returns a tradeStrategyOld object
    
    Args:
    buyPoint -- float
    sellPoint -- double
    predicted -- pd.Series
    data -- pd.DataFrame
    metaDictRaw -- dictionary with the keys 'tradingEnd' and 'commission'
    '''

    if not isinstance(buyPoint, float) or not isinstance(sellPoint, float):
        raise TypeError('One of buyPoint or sellPoint is not a float')

    elif buyPoint > 1 or buyPoint < 0 or sellPoint > 1 or sellPoint < 0:
        raise ValueError('One of buyPoint or sellPoint is bigger than 1 or less than 0')

    if not isinstance(predicted, pd.Series):
        raise TypeError('predicted is not a pandas series')

    if not isinstance(data, pd.DataFrame):
        raise TypeError('data is not a pd DataFrame')

    if not isinstance(metaDictRaw, dict):
        raise TypeError('metaDict is not a dictionary')
    elif 'tradingEnd' not in metaDictRaw or 'commission' not in metaDictRaw:
        raise ValueError('metaDict doesnt contain the necessary keys')

    # Filter for points where we would buy, and points where we would sell
    buyPointsRaw = predicted[predicted < buyPoint]

    # If we have no buypoints
    if buyPointsRaw.empty:
        
        return None
        
    else:

        sellPointsRaw = predicted[predicted > sellPoint]
    
        # Create lists to store filtered indices of these
        buyPointsIdx = list()
        sellPointsIdx = list()

        prevSellRow = None

        # For each buy point
        for idx, *rest in buyPointsRaw.iteritems():
            
            # If we have a previous point to sell at
            if not prevSellRow is None:

                # If our current buy point index is earlier than our last sell point
                if idx < prevSellRow:

                    # Skip this loop iteration
                    continue
            
            # Add this buy point to our list of indices
            buyPointsIdx.append(idx)

            # If we have a sell point after this buy point
            if not sellPointsRaw.loc[idx:].empty:

                # Get its row and store it
                sellRow = sellPointsRaw.loc[idx:].index[0]
                sellPointsIdx.append(sellRow)

            # If we don't have a sell point after this buy point
            else:

                # Add 'no sell' to the list and then end the looping logic
                break
            
            # Set our previous sell point to this one
            prevSellRow = sellRow
    
    # Create a dataframe of only the points where we might issue a buy order
    buyPoints = data.loc[buyPointsIdx]

    sellPointsSeries = None
    # If we have selling points
    if len(sellPointsIdx) > 0:

        # Create a pd.Series of our sell ponts to join to our buyPoints df
        sellPointsSeries = pd.Series(data = sellPointsIdx, dtype = 'datetime64[ns, UTC]', index = [x for x in buyPointsIdx if x < sellPointsIdx[-1]])
    
    buyPoints['sellPoints'] = sellPointsSeries

    # Create a dictionary to save metadata for this run
    #metaDict = metaDictRaw.copy()
    metaDictRaw['buyTrigger'] = buyPoint
    metaDictRaw['sellTrigger'] = sellPoint
        
    return tradeStrategyOld(
        buyPoints
        , data.loc[sellPointsIdx]
        , metaDictRaw['tradingEnd']
        , metaDictRaw['commission']
        , metaData = metaDictRaw
    )

def getStratResults(currArgs):
    
    data = currArgs['dataPred']

    # Generate our Thigh and Tlow values in 0.01 steps from 0 to 1
    ThighRaw = currArgs['ThighRaw']
    TlowRaw = currArgs['TlowRaw']

    # Generate the instance of our SVR
    svr = currArgs['model']

    predicted = svr.predict(currArgs['X_val'])

    # Create a dataset of our validation features and predictions

    # Filter out threshold values so they're restricted by what we've predicted
    threshParameters = {
    'Thigh': [x for x in ThighRaw if x >= (min(predicted)-0.01) and x <= (max(predicted)+0.01)]
    , 'Tlow': [x for x in TlowRaw if x >= (min(predicted)-0.01) and x <= (max(predicted)+0.01)]
    }

    # Get a list of all combinations of Thigh and Tlow
    paramListThresh = getListOfAllCombos(threshParameters)

    # Remove any combinations that make no sense i.e. they're the same or we sell instead of buy
    paramListThreshClean = [x for x in paramListThresh if (x[0] != x[1]) and (x[0] > x[1])]

    valDates = currArgs['valDates']

    resultsList = [getStratData(y[1], y[0], predicted) for y in paramListThreshClean]
    resultsListClean = [x for x in resultsList if x is not None]

    comboIdxList = [x['buyDates'] + x['sellDates'] for x in resultsListClean]
    comboIdxUnique = np.unique(comboIdxList, return_index = True)[1]
    distinctIdx = [resultsListClean[x] for x in comboIdxUnique]
    
    tradeStrategyList = list()
    for combo in distinctIdx:

        dataDict = {
                'buyNextOpen' : data.loc[valDates].iloc[combo['buyDates']]['next_open'].values  
                , 'buyNextIndex': data.loc[valDates].iloc[combo['buyDates']]['next_index'].values
                , 'buyDates': data.loc[valDates].iloc[combo['buyDates']].index
                , 'sellNextOpen' : data.loc[valDates].iloc[combo['sellDates']]['next_open'].values 
                , 'sellNextIndex': data.loc[valDates].iloc[combo['sellDates']]['next_index'].values
                , 'sellDates': data.loc[valDates].iloc[combo['sellDates']].index
                , 'tradingEnd': currArgs['metaData']['tradingEnd']
                , 'commission': currArgs['metaData']['commission']
                , 'printArg': False
                , 'metaData': combo['metaData']
        }

        tradeStrategyList.append(tradeStrategy(**dataDict))

    [x.metaData.update(currArgs['metaData']) for x in tradeStrategyList]
        
    return tradeStrategyList

def getStratResultsMult(argsList):
    '''
    Function executes getModelData() as a multiprocessing task based on a list of arguments
    as provided by getDataArgs()
    '''

    # Create a progress bar
    pbar = tqdm(total=len(argsList))

    # Run our tradeStrategy class for all combinations of arguments
    with concurrent.futures.ProcessPoolExecutor() as executor:

        #processes = list()
        results = list()

        for result in executor.map(
            getStratResults
            , argsList
        ):

            pbar.update()
            results.append(result)
                
    pbar.close()

    return results

def getStratResultsMultAlt(argsList):

    results = list()

    # Run our tradeStrategyOld class for all combinations of arguments
    with Pool() as pool:  
        for x in tqdm(pool.imap(getStratResults, argsList), total=len(argsList), leave = True, position = 0):
            results.append(x)

        pool.close()
        pool.join()

    return results

class tradeStrategy:
    """
    This class is used to evaluate a pandas dataframe that is passed with columns 'open', 'close', and 'predicted'
    against thresholds to buy and sell for the predicted column as a trading strategy
    """
    
    def __init__(self
                 , buyNextOpen
                 , buyNextIndex
                 , buyDates
                 , sellNextOpen
                 , sellNextIndex
                 , sellDates
                 , tradingEnd
                 , commission
                 , printArg = False
                 , metaData = None):
        
        self.buyNextOpen = buyNextOpen
        self.buyNextIndex = buyNextIndex
        self.buyDates = buyDates
        self.sellNextOpen = sellNextOpen
        self.sellNextIndex = sellNextIndex
        self.sellDates = sellDates

        self.commission = commission
        self.print = printArg
        self.metaData = metaData
        self.tradingEnd = tradingEnd

        self.ordersList = self.getOrdersAlt()
        self.tradesList = self.getTrades()
        self.bahReturns = self.getBahValue()
        self.stratReturns = self.getStratValue()
        
    def log(self, txt, doprint = False):
        """
        Logging function so we can turn it on or off
        """

        if self.print == True or doprint == True:
            print(txt)
            
    def getOrdersAlt(self):
        """
        Function uses our data to compute when we would buy and sell based on the predicted
        column and the buy and sell triggers, using the open and close values

        Tracks orders and commissions in a list of dictionaries.

        Limited to: 
        - only entering the market if we issue a buy order i.e. no short options
        - only one share is traded
        """

        ordersList = list()

        # For each of these possible buy points
        for buyIdx, buyDate in enumerate(self.buyDates):
        #for buyIdx, buyRow in self.buyPoints.iterrows():

            # Print info about this buy order
            #self.log(f'BUY ORDER ISSUE BASED ON {buyRow.name}')
            #self.log(f"BUY AT {buyRow['next_open']}")
            
            #self.log(f'BUY ORDER ISSUE BASED ON {buyDate}')
            #self.log(f'BUY ORDER ISSUED ON {self.buyNextIndex[buyIdx]}')
            #self.log(f"BUY AT {self.buyNextOpen[buyIdx]}")
            

            # Added to our orderslist
            ordersList.append({
                    'buy': {
                        #'price': buyRow['next_open']
                        'price': self.buyNextOpen[buyIdx]
                        #, 'commission': buyRow['next_open'] * self.commission
                        , 'commission': self.buyNextOpen[buyIdx] * self.commission
                        #, 'placed': buyRow['next_index']
                        , 'placed': self.buyNextIndex[buyIdx]
                        #, 'detected': buyRow.name
                        #, 'detected': buyDate
                    }
            })

            # Get the row of data that corresponds to when we would sell this order, and if it exists
            #if not pd.isnull(buyRow['sellPoints']):
            if buyIdx < len(self.sellDates):

                #sellRow = self.sellPoints.loc[buyRow['sellPoints']]

                #if not np.isnan(sellRow['next_open']):
                if not np.isnan(self.sellNextOpen[buyIdx]):

                    # Print details about our sale
                    #self.log(f'SELL ORDER ISSUE BASED ON {sellRow.name}')
                    #self.log(f"SELL AT {sellRow['next_open']}")
                    
                    #self.log(f'SELL ORDER ISSUE BASED ON {self.sellDates[buyIdx]}')
                    #self.log(f'SELL ORDER ISSUE ON {self.sellNextIndex[buyIdx]}')
                    #self.log(f"SELL AT {self.sellNextOpen[buyIdx]}")

                    # Added to our orderslist
                    ordersList[-1]['sell'] = {
                               # 'price': sellRow['next_open']
                                'price': self.sellNextOpen[buyIdx]
                                #, 'commission': sellRow['next_open'] * self.commission
                                , 'commission': self.sellNextOpen[buyIdx] * self.commission
                                #, 'placed': sellRow['next_index']
                                , 'placed': self.sellNextIndex[buyIdx]
                                #, 'detected': sellRow.name
                                #, 'detected': self.sellDates[buyIdx]
                            }

        return ordersList
            
    def getTrades(self):
        """
        Based on our orders in self.ordersList, compute the gross and net profit for each closed trade
        """
        
        tradesList = list()
        for orders in self.ordersList:
            
            # If we issued a sell order for this buy order
            if 'sell' in orders:
            
                gross = orders['sell']['price'] - orders['buy']['price']
                net = gross - (orders['sell']['commission'] + orders['buy']['commission'])

                tradesList.append({'gross': round(gross,2), 'net': round(net, 2)})
            
        return tradesList
    
    def getBahValue(self):
        """
        Compute the money we would have made / lost if we just bought and held
        """
        
        commission = (self.tradingEnd[1] * self.commission) + (self.tradingEnd[0] * self.commission)
        
        return round(self.tradingEnd[1] - self.tradingEnd[0] - commission, 2)
    
    def getStratValue(self):
        """
        Compute the money we made / lost using our strategy
        If we have an open trade, calculate the change in value since we bought
        """
        
        total = 0
        for orderNo in range(len(self.ordersList)):
            
            if orderNo < (len(self.tradesList)):
                total = total + self.tradesList[orderNo]['net']

            else:
                total = total + (self.tradingEnd[1] - self.ordersList[orderNo]['buy']['price'])
            
        return round(total,2)

class tradeStrategyOld:
    """
    This class is used to evaluate a pandas dataframe that is passed with columns 'open', 'close', and 'predicted'
    against thresholds to buy and sell for the predicted column as a trading strategy
    """
    
    def __init__(self
                 , buyPoints
                 , sellPoints
                 , tradingEnd
                 , commission
                 , printArg = False
                 , metaData = None):
        """
        buyPoints -- pandas dataframe with 'open', 'close', 'predicted', 'next_open', 'next_close'
            and 'next_index' columns with rows indicating the timepoints at which we would predict a buy order
        sellPoints -- pandas dataframe with 'open', 'close', 'predicted', 'next_open', 'next_close'
            and 'next_index' columns with rows indicating the timepoints at which we would predict a sell order    
        commission -- float with a max value of 1.00 that serves as broker commission
        printArg -- boolean indicating whether to print the results of each order
        """
        
        if isinstance(buyPoints, pd.DataFrame):
            if not all(item in buyPoints.columns for item in ['open','close', 'next_open', 'next_close', 'next_index']):
                raise ValueError('buyPoints doesnt contain the required columns: open, close')
        else:
                raise TypeError('buyPoints is not a pandas DataFrame')

        if isinstance(sellPoints, pd.DataFrame):
            if not all(item in sellPoints.columns for item in ['open','close', 'next_open', 'next_close', 'next_index']):
                raise ValueError('sellPoints doesnt contain the required columns: open, close')
        else:
                raise TypeError('sellPoints is not a pandas DataFrame')
            
        if isinstance(commission, float):
            if commission >= 1:
                raise ValueError('commission must be less than 1.0')
        else:
            raise TypeError('commission is not a float')
            
        if not isinstance(printArg, bool):
            raise TypeError('printArg needs to be a boolean')

        #self.data = data
        self.buyPoints = buyPoints
        self.sellPoints = sellPoints
        self.commission = commission
        self.print = printArg
        self.metaData = metaData
        self.tradingEnd = tradingEnd

        self.ordersList = self.getOrdersAlt()
        self.tradesList = self.getTrades()
        self.bahReturns = self.getBahValue()
        self.stratReturns = self.getStratValue()
        
    def log(self, txt, doprint = False):
        """
        Logging function so we can turn it on or off
        """

        if self.print == True or doprint == True:
            print(txt)
            
    def getOrdersAlt(self):
        """
        Function uses our data to compute when we would buy and sell based on the predicted
        column and the buy and sell triggers, using the open and close values

        Tracks orders and commissions in a list of dictionaries.

        Limited to: 
        - only entering the market if we issue a buy order i.e. no short options
        - only one share is traded
        """

        ordersList = list()

        # For each of these possible buy points
        for buyIdx, buyRow in self.buyPoints.iterrows():

            # Print info about this buy order
            self.log(f'BUY ORDER ISSUE BASED ON {buyRow.name}')
            self.log(f"BUY AT {buyRow['next_open']}")

            # Added to our orderslist
            ordersList.append({
                    'buy': {
                        'price': buyRow['next_open']
                        , 'commission': buyRow['next_open'] * self.commission
                        , 'placed': buyRow['next_index']
                        , 'detected': buyRow.name
                    }
            })

            # Get the row of data that corresponds to when we would sell this order, and if it exists
            if not pd.isnull(buyRow['sellPoints']):

                sellRow = self.sellPoints.loc[buyRow['sellPoints']]

                if not np.isnan(sellRow['next_open']):

                    # Print details about our sale
                    self.log(f'SELL ORDER ISSUE BASED ON {sellRow.name}')
                    self.log(f"SELL AT {sellRow['next_open']}")

                    # Added to our orderslist
                    ordersList[-1]['sell'] = {
                                'price': sellRow['next_open']
                                , 'commission': sellRow['next_open'] * self.commission
                                , 'placed': sellRow['next_index']
                                , 'detected': sellRow.name
                            }

        return ordersList
            
    def getTrades(self):
        """
        Based on our orders in self.ordersList, compute the gross and net profit for each closed trade
        """
        
        tradesList = list()
        for orders in self.ordersList:
            
            # If we issued a sell order for this buy order
            if 'sell' in orders:
            
                gross = orders['sell']['price'] - orders['buy']['price']
                net = gross - (orders['sell']['commission'] + orders['buy']['commission'])

                tradesList.append({'gross': round(gross,2), 'net': round(net, 2)})
            
        return tradesList
    
    def getBahValue(self):
        """
        Compute the money we would have made / lost if we just bought and held
        """
        
        commission = (self.tradingEnd[1] * self.commission) + (self.tradingEnd[0] * self.commission)
        
        return round(self.tradingEnd[1] - self.tradingEnd[0] - commission, 2)
    
    def getStratValue(self):
        """
        Compute the money we made / lost using our strategy
        If we have an open trade, calculate the change in value since we bought
        """
        
        total = 0
        for orderNo in range(len(self.ordersList)):
            
            if orderNo < (len(self.tradesList)):
                total = total + self.tradesList[orderNo]['net']

            else:
                total = total + (self.tradingEnd[1] - self.ordersList[orderNo]['buy']['price'])
            
        return round(total,2)