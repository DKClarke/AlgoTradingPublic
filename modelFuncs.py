from operator import truediv
import pandas as pd
import numpy as np
from scipy.fft import fft

from sklearn import svm
from sklearn.model_selection import train_test_split

import math
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import Normalizer

import itertools # For getting all combinations of a dict

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Used to retrieve historical market data for the DJIA that we use to show our approach matches the preprint
import yfinance as yf
# Used to retrieve historical market data for the DJIA that we use to show our approach matches the preprint
import quandl

import http.client, urllib.parse
import requests
import os
import json

from extremaFuncs import getLabelledExtrema, getExtremaToDrop, optimiseTpOscillator, normaliseTpOsc

def getHistoricalIntradayResponse(symbol, interval, date_to):
    '''
    Function takes a ticker symbol, a time interval (minute, hour, day etc.) and a date
    and queries stockdata.org for intraday data according to that interval. It will return all the data
    prior to the date specified. This is limited to 7 days preceding (inclusive) for minute, and 180 days for
    hour https://www.stockdata.org/documentation
    
    symbol -- string, stock ticker
    interval -- string, minute | hour | day | week | month | quarter | year
    date_to -- string
    '''
    
    # Connect to our API source
    with http.client.HTTPSConnection('api.stockdata.org') as conn:

        with open('credentials.json', 'r') as j:
            api_key = json.loads(j.read())
        
            # Parse the parameters appropriately
            params = urllib.parse.urlencode({
                'api_token': api_key
                , 'symbols': symbol
                , 'interval': interval
                , 'date_to': str(date_to.strftime('%Y-%m-%d'))
                })
            
            # Get the response from the API
            response = requests.get('https://api.stockdata.org/v1/data/intraday?{}'.format(params))
            data = response.json()
        
    # Return the data formatted in a dataframe
    return pd.json_normalize(data['data'])

def getHistoricalIntraday(symbol, startDate, endDate, interval = 'minute'):
    '''
    Function takes a ticker symbol, a start and end date (as datetime objects) and a time interval 
    (minute, hour, day etc.) and queries stockdata.org for intraday data between the start and end dates.
    https://www.stockdata.org/documentation
    
    Currently only support interval of minute or hour
    
    symbol -- string, stock ticker
    startDate -- date object e.g. date(2020,6,10)
    endDate --  date object
    interval -- string, minute | hour | day | week | month | quarter | year
    '''
    
    if interval not in ['minute', 'hour']:
        
        raise ValueError('Interval not currently supported')
        
    if endDate <= startDate:
        
        raise ValueError('endDate specified is earlier or equal to startDate')
    
    # If the user wants per minute prices, each request gets 7 days of data, so create
    # a list of dates to retrieve where each is separated by 8 days
    if interval == 'minute':
        frequency = '8d'
        
    # For hour, this is 180 days, so separate by 181d
    elif interval == 'hour':
        frequency = '181d'
    
    # Create our date range
    dateRange = pd.date_range(startDate, endDate, freq=frequency)
    
    # Compute our start and end dates in a printing friendly format
    startDatePrint = min(dateRange).strftime('%Y-%m-%d')
    endDatePrint = max(dateRange).strftime('%Y-%m-%d')
    
    # Create a folder path for saving our data
    folder = f'{symbol}_{startDatePrint}_{endDatePrint}_{interval}_historical_intraday'

    # If the folder doesn't exist, make it
    if not os.path.exists(folder):
        print('Made folder')
        os.mkdir(folder)
    
    # For each date to query the API for
    for currentDate in dateRange:
        
        # Generate a filename for what we would save the data as
        printDate = currentDate.strftime('%Y-%m-%d')
        title = (f'{symbol}_{printDate}_{interval}_historical_intraday.csv')
        savePath = os.path.join(folder, title)
        
        # If we haven't already saved that data
        if not os.path.exists(savePath):
            
            # Query, retrieve the data, save it
            print(f'Fetched data for {currentDate}')
            data = getHistoricalIntradayResponse(symbol, 'minute', currentDate)

            data.to_csv(savePath)

    return symbol, startDatePrint, endDatePrint, interval
    

def getFFTAmplitudePhase(data):
    '''
    Function takes an array of data and returns the amplitude and phases of the discrete
    fourier transform

    Keyword arguments:
    data -- np.array of 1 dimension
    '''

    # Run a fourier transform
    fourier = fft(data)
    
    # Get amplitude and phase
    amplitudes = np.abs(fourier)
    phases = np.angle(fourier)

    return amplitudes, phases

def generateFFTLookback(featuresRaw, lookbackWindow):
    '''
    Function takes an array of values to use as features, values to use as targets,
    and an integer as a lookback window. Loops through the featuresRaw and looking back over the
    indicated lookbackWindow size, it gets the fourier amplitudes and phases of the featuresRaw.
    These are returned as 2D np arrays.

    Keyword arguments:
    featuresRaw -- np.array or pd.series of 1 dimension
    lookbackWindow -- integer
    '''

    if not isinstance(featuresRaw, np.ndarray):

        raise TypeError('featuresRaw is not a np array')

    if not isinstance(lookbackWindow, int):

        raise TypeError('lookbackWindow is not an int')
    
    # Create the lists we'll use to store our outputs
    amplitudes = list()
    phases = list()
    
    # Iterate through our features
    for index, value in enumerate(featuresRaw):
        
        # If we are far enough in to have a complete lookback window
        if index>lookbackWindow-1:
            
            # Get that window
            lookback = np.array(featuresRaw[index - lookbackWindow:index])
            
            # Compute its fourier amplitudes and phases
            amp, phase = getFFTAmplitudePhase(lookback)
            
            # Add these, and the target, to our output lists
            amplitudes.append(amp)
            phases.append(phase)
            
    # Convert our output lists to np arrays
    ampArray = np.array(amplitudes)
    phaseArray = np.array(phases)
    
    AmpTransformer = Normalizer().fit(ampArray)
    PhaseTransformer = Normalizer().fit(phaseArray)
    
    #AmpTransformer = MinMaxScaler().fit(ampArray)
    #PhaseTransformer = MinMaxScaler().fit(phaseArray)
    
    # Return all
    return AmpTransformer.transform(ampArray), PhaseTransformer.transform(phaseArray)

def generateNormalizedLookback(featuresRaw, lookbackWindow):
    '''
    Function takes an array of values to use as features
    and an integer as a lookback window. Loops through the featuresRaw and looking back over the
    indicated lookbackWindow size, it gets the normalized values in that lookback window using the
    sklearn Normalizer. This is returned as 2D np array.

    Keyword arguments:
    featuresRaw -- np.array or pd.series of 1 dimension
    lookbackWindow -- integer

    Returns:
    np.array
    '''

    if not isinstance(featuresRaw, np.ndarray):

        raise TypeError('featuresRaw is not a np array')

    if not isinstance(lookbackWindow, int):

        raise TypeError('lookbackWindow is not an int')
    
    # Generate our output lists
    features = list()
    
    # Iterate through our features
    for index, value in enumerate(featuresRaw):
        
        # If we have gone far enough to have a complete lookback window
        if index>lookbackWindow-1:
            
            # Get that window
            lookback = np.array(featuresRaw[index - lookbackWindow:index])
            
            # Add it, and the target, to our lists
            features.append(lookback)
    
    # Normalize our features in the lookback windows
    featuresArray = np.array(features)
    transformer = Normalizer().fit(featuresArray)
    #transformer = MinMaxScaler().fit(featuresArray)
    
    # Return it and the target
    return transformer.transform(featuresArray)

def calcClassError(target, prediction, Thigh, Tlow):
    '''
    Return error values for a target and a prediction, provided high and low threshold values based on
    classifier performance

    Keyword arguments:
    target -- a target value from a Turning Point Oscillator bounded by 1 and 0
    prediction -- a predicted value for a Turning Point Oscillator
    Thigh --  the threshold above which we consider a prediction equivalent to 1
    Tlow -- the threshold below which we consider a prediction equivalent to 0
    '''

    # Set our 'new' target and prediction values based on classifying outcomes
    newTarget = 0
    newPrediction = 0

    # If we're at a peak, set our new target to class 2
    if target == 1:

        newTarget = 2

    # If we're at a trough, set to class 0
    elif target == 0:

        newTarget = 0

    # Else set to class 1
    else:

        newTarget = 1

    # With the prediction, if it's above threshold for a peak, class 2
    if prediction > Thigh:

        newPrediction = 2

    # If below threshold for a trough, class 0
    elif prediction < Tlow:

        newPrediction = 0

    # Else class 1
    else:

        newPrediction = 1

    outDict = {
        'target': newTarget
        , 'prediction': newPrediction
    }

    return outDict


def calcTPrime(target, prediction, Thigh, Tlow):
    '''
    Return T' for a target and a prediction, provided high and low threshold values.

    Keyword arguments:
    target -- a target value from a Turning Point Oscillator bounded by 1 and 0
    prediction -- a predicted value for a Turning Point Oscillator
    Thigh --  the threshold above which we consider a prediction equivalent to 1
    Tlow -- the threshold below which we consider a prediction equivalent to 0
    '''

    # If we're at a peak but our prediction is below our threshold
    # Set our Tprime to threshold
    if target == 1 and prediction < Thigh:
        Tprime = Thigh

    # If we're at a trough but our prediction is above our threshold
    # Set our Tprime to threshold
    elif target == 0 and prediction > Tlow:
        Tprime = Tlow

    # If we're not at a peak or trough, but our prediction is above our threshold
    # Set our Tprime to the threshold
    elif target != 1 and prediction > Thigh:
        Tprime = Thigh * 0.98

    # If we're not at a peak or a trough and our prediction is below our threshold
    # Set our Tprime to the threshold
    elif target != 0 and prediction < Tlow:
        Tprime = Tlow * 1.02

    # If we satisfied none of these criteria, our T' is just our prediction
    else:

        Tprime = prediction

    return Tprime

def getError(errorFunc, targets, predictions, Thigh, Tlow):
    '''
    Return the error for an array of targets and predictions, provided high and low threshold values.

    Keyword arguments:
    errorFunc -- a string describing the function to use to calculate our error
    target -- a 1D array of target values from a Turning Point Oscillator bounded by 1 and 0
    prediction -- a 1D array of predicted values for a Turning Point Oscillator
    Thigh --  the threshold above which we consider a prediction equivalent to 1
    Tlow -- the threshold below which we consider a prediction equivalent to 0
    '''

    if errorFunc == 'TpRMSE':
    
        # For each element in our arrays, compute the T' value, and pass our Thigh and Tlow values as arrays
        # of the same length as our target / prediction arrays
        errorVals = list(
            map(
                calcTPrime
                , targets
                , predictions
                , np.repeat(Thigh, len(targets))
                , np.repeat(Tlow, len(targets))
            )
        )
        
        # Compute the TpRMSE
        errorOutput = math.sqrt(mean_squared_error(predictions, errorVals))

    elif errorFunc == 'class':

        # For each element in our arrays, compute the classifications
        errorRaw = list(
            map(
                calcClassError
                , targets
                , predictions
                , np.repeat(Thigh, len(targets))
                , np.repeat(Tlow, len(targets))
                )
            )

        # Get the targets and predictions
        errorVals = [x['target'] for x in errorRaw]
        testPredictions = [x['prediction'] for x in errorRaw]
        
        # Compute a confusion matrix and get the true positives
        cm = confusion_matrix(testPredictions, errorVals)
        tp = np.diag(cm)

        # For each class, compute the f1 score
        f1_list = list()

        for i in [0,1,2]:

            prec = list(map(truediv, tp, np.sum(cm, axis=0)))[i]
            rec = list(map(truediv, tp, np.sum(cm, axis=1)))[i]
            f1_list.append(2* (prec * rec)/(prec + rec))

        # For the middle class, compute what percent it makes up out all values
        all_classes = np.sum(cm, axis = 1)
        middle_class = np.sum(cm[1])
        pcnt_middle = middle_class/np.sum(all_classes)

        # We want our peaks and troughs to have 5X the influence that the middle class does, so here
        # we compute what to multiply them by to do this
        mlt_factor = (pcnt_middle * 5) / (1 - pcnt_middle)

        # Compute the weighted error
        errorOutput = ((f1_list[2] * mlt_factor) + (f1_list[0] * mlt_factor) + f1_list[1]) / ((mlt_factor * 2) + 1)

    else: 

        raise ValueError("passed an error function that isn't supported")
    
    # Return the TpRMSE and the Thigh and Tlow values used
    return {
        'Errors': errorOutput
        , 'Thigh': Thigh
        , 'Tlow': Tlow
    }

def getTpRMSE(targets, predictions, Thigh, Tlow):
    '''
    Return TpRMSE for an array of targets and predictions, provided high and low threshold values.

    Keyword arguments:
    target -- a 1D array of target values from a Turning Point Oscillator bounded by 1 and 0
    prediction -- a 1D array of predicted values for a Turning Point Oscillator
    Thigh --  the threshold above which we consider a prediction equivalent to 1
    Tlow -- the threshold below which we consider a prediction equivalent to 0
    '''
    
    # For each element in our arrays, compute the T' value, and pass our Thigh and Tlow values as arrays
    # of the same length as our target / prediction arrays
    Tprime = list(
        map(
            calcTPrime
            , targets
            , predictions
            , np.repeat(Thigh, len(targets))
            , np.repeat(Tlow, len(targets))
        )
    )
    
    # Compute the TpRMSE
    TpRMSE = math.sqrt(mean_squared_error(predictions, Tprime))
    
    # Return the TpRMSE and the Thigh and Tlow values used
    return {
        'TpRMSE': TpRMSE
        , 'Thigh': Thigh
        , 'Tlow': Tlow
    }

def fitModelHyperParams(C, epsilon, gamma):
    '''
    Generates an instance of an epsilon SVR model with a rbf kernel fit with the passed hyperparameters

    Keyword arguments:
    C -- the value of to fit to the SVR
    epsilon -- the value of epsilon to fit to the SVR
    gamma -- the value of gamma to fit to the SVR
    '''
    
    # Generate the instance of our SVR
    svr = svm.SVR(kernel = 'rbf', cache_size = 1000)
    svr.set_params(C = C, epsilon = epsilon, gamma = gamma)

    
    # Return it with the fit hyperparameters
    return {
        'svr': svr
        , 'C': C
        , 'epsilon': epsilon
        , 'gamma': gamma
    }

def getErrorWrapper(errorFunc, targets, Thigh, Tlow, modelInfo):
    '''
    Get the error for a model and return it along with the hyperparameters fit. This function
    wraps around getError function so it can be used in a map function.

    Keyword arguments:
    errorFunc -- a string argument specifying what error function to use to compute error
    targets -- A 1D array of prediction targets for a turning point oscillator bounded by 0 and 1
    Thigh --  the threshold above which we consider a prediction equivalent to 1
    Tlow -- the threshold below which we consider a prediction equivalent to 0
    modelInfo -- A dictionary output by fitModelHyperParams
    '''
    
    # Assign our model predictions to a new variable
    predictions = modelInfo['preds']
    
    # Get the error value for this model
    errorVals = getError(errorFunc, targets, predictions, Thigh, Tlow)
    
    # Enrich the output with hyperparameter values
    errorVals['C'] = modelInfo['C']
    errorVals['epsilon'] = modelInfo['epsilon']
    errorVals['gamma'] = modelInfo['gamma']
    
    # Return this enriched output
    return errorVals

def getTpRMSEWrapper(targets, Thigh, Tlow, modelInfo):
    '''
    Get the TpRMSE for a model and return it along with the hyperparameters fit. This function
    wraps around getTpRMSE so it can be used in a map function.

    Keyword arguments:
    targets -- A 1D array of prediction targets for a turning point oscillator bounded by 0 and 1
    Thigh --  the threshold above which we consider a prediction equivalent to 1
    Tlow -- the threshold below which we consider a prediction equivalent to 0
    modelInfo -- A dictionary output by fitModelHyperParams
    '''
    
    # Assign our model predictions to a new variable
    predictions = modelInfo['preds']
    
    # Get the TpRMSE value for this model
    TpRMSE = getTpRMSE(targets, predictions, Thigh, Tlow)
    
    # Enrich the output with hyperparameter values
    TpRMSE['C'] = modelInfo['C']
    TpRMSE['epsilon'] = modelInfo['epsilon']
    TpRMSE['gamma'] = modelInfo['gamma']
    
    # Return this enriched output
    return TpRMSE

def getListOfAllCombos(params):
    '''
    This function takes a dictionary where each key:value pair contains a list, and returns a list of
    all possible combinations of all lists in the dictionary
    
    Keyword arguments: 
    params -- dictionary where values are lists
    '''
    
    # Get all combinations possible in our params dictionary
    combos = itertools.product(*(params[Name] for Name in params))

    # Get a list of these
    combosList = list(combos)
    
    # Return the list
    return combosList
    
def TpSVRGridSearch(errorFunc, C, epsilon, gamma, Thigh, Tlow, X_train, y_train, X_val, y_val):
    '''
    This function takes a list of hyperparameter values to search through, fits models using training data,
    computes an error value using validation data, and returns a list of dictionaries of the values and
    hyperparameter values
    
    Keyword arguments
    errorFunc -- a string argument dictating what error function to use
    C -- list of C values to iterate through
    epsilon -- list of epsilon values to iterate through
    Thigh -- list of Thigh values to iterate through
    Tlow -- list of Tlow values to iterate through
    X_train -- training features
    y_train -- training targets
    X_val -- validation features
    y_val -- validation targets
    '''
    
    # Create a dictionary of our hyperparameters
    paramGrid = {
        'C': C
        , 'epsilon': epsilon
        , 'gamma': gamma
    }

    # Get a list of the unique combinations of all hyperparameters
    paramCombosList = getListOfAllCombos(paramGrid)

    # For each combination, create an epsilon SVR instance
    models = list(
        map(
            fitModelHyperParams
            , [x[0] for x in paramCombosList]
            , [x[1] for x in paramCombosList]
            , [x[2] for x in paramCombosList]
        )
    )
    
    # Now for each instance
    for model in models:

        # Fit it to our training data then predict our validation data
        model['svr'].fit(X_train, y_train)
        model['preds'] = model['svr'].predict(X_val)

    # Create a dictionary containing the values to search for our thresholds
    threshGrid = {
        'Thigh': Thigh
        , 'Tlow': Tlow
    }
    
    # Get a list of the unique combinations of our thresholds
    threshCombosList = getListOfAllCombos(threshGrid)
    
    # Get the length of this list
    noThreshCombos = len(threshCombosList)

    # Separate our combinations again into Thigh and Tlow vectors
    ThighPass = [x[0] for x in threshCombosList]
    TlowPass = [x[1] for x in threshCombosList]

    # Create a 2D np array where the first dimension provides the validation targets for each 
    # fitted SVR and Thigh X Tlow combination
    targetsPass = np.tile(y_val, (noThreshCombos*len(models),1))

    # Likewise for our predictions
    predPass = np.repeat(models, noThreshCombos)

    # For each epsilon SVR, return the lowest error values found in our Thigh X Tlow search space, and the values of
    # Thigh and Tlow that provided it
    errorOut = list(
        map(
            getErrorWrapper 
            , [errorFunc] * len(threshCombosList)
            , targetsPass
            , ThighPass
            , TlowPass
            , predPass
        )
    )
    
    return errorOut

def plotModelOutputs(C, epsilon, gamma, X_train, y_train, X_test, y_test, Thigh, Tlow, kernel = 'rbf'):
    '''
    Fits an epsilon SVR model with the specified hyperparameters, trains it on X and y_train, fits it to
    X_test, computes the TpRMSE using Thigh and Tlow, then plots the targets, predictions, T', Thigh, Tlow,
    and error.

    Keyword arguments:
    C -- the value of to fit to the SVR
    epsilon -- the value of epsilon to fit to the SVR
    gamma -- the value of gamma to fit to the SVR
    X_train -- training features
    y_train -- training targets
    X_test -- test features
    y_test -- test targets
    Thigh --  the threshold above which we consider a prediction equivalent to 1
    Tlow -- the threshold below which we consider a prediction equivalent to 0
    '''
    
    # Fit an epsilon SVR model to training data
    svr = svm.SVR(kernel = kernel, C = C, epsilon = epsilon, gamma = gamma, cache_size = 1000)
    svr.fit(X_train, y_train)

    # Predict our test data
    predictions = svr.predict(X_test)

    # For every prediction and target, compute T' by passing the target, predictions, and Thigh and Tlow
    # We must structure Thigh and Tlow as vectors of the same length as our target / predictions
    Tprime = list(
        map(
            calcTPrime
            , y_test
            , predictions
            , np.repeat(Thigh, len(y_test))
            , np.repeat(Tlow, len(y_test)))
    )

    # Create a dataframe that stores vectors and values of interest
    plotOne = pd.DataFrame({
        'Predictions': predictions
        , 'Targets': y_test
        , 'Mean Squared Error': (Tprime - predictions)**2
        , "T'": Tprime
        , 'Thigh': Thigh
        , 'Tlow': Tlow
    })
    
    # Plot our data
    fig = make_subplots(rows=2, cols=1, shared_xaxes= True)
    
    for column in ['Predictions', 'Targets', "T'", 'Tlow', 'Thigh']:
        
        fig.add_trace(
            go.Scatter(
                x=plotOne.index,
                y=plotOne[column],
                mode='lines',
                name=column,
            ),
            row = 1,
            col = 1
        )
    
    fig.add_trace(
        go.Scatter(
            x=plotOne.index,
            y=plotOne['Mean Squared Error'],
            mode='lines',
            name='Mean Squared Error'
        ),
        row = 2,
        col = 1
    )

    fig.update_layout(
        title="T' and Mean Squared Error Plotted for Validation Dataset",
        legend_title="Legend",
    )

    fig.update_xaxes(title_text="Index", row=2, col=1)
    fig.update_yaxes(title_text="Turning Point Oscillator Value", row=1, col=1)
    fig.update_yaxes(title_text="Mean Squared Error", row=2, col=1)

    return fig

def getFeatures(trainData, lookbackWindow):
    '''
    Reads in stock data and returns its close value as features in our turning point model. Also returns
    the indices these features are built on from the original trainData

    Keyword arguments:
    trainData -- dictionary where the k,v pairs are arguments to the quandl.get function
        Alternatively, a path to a .csv file or a pandas dataFrame
    lookbackWindow -- integer value that dictates how many of the previous prices to use for our features

    Returns:
    np.array
    index
    '''

    if isinstance(trainData, dict):
       
        # Load in our price data
        #data = yf.download(**trainData)

        # Load in DJIA data from October 2008 to May 2009
        data = quandl.get(**trainData).reset_index()

    elif isinstance(trainData, str):
        
        # Read in our data and set the index column correctly
        data = pd.read_csv(trainData, index_col = 0)

    elif isinstance(trainData, pd.DataFrame):

        data = trainData

    else:

        raise TypeError('data argument is not passed a supported class')

    # If the data has no close column
    if sum([True for x in data.columns if x == 'Close']) == 0:

        # If the data being used has a value column, convert that to 'Close'
        if sum([True for x in data.columns if x == 'Value']) == 1:
            data.rename(columns = {'Value' : 'Close'}, inplace = True)

        # Else raise an error
        else :
            raise ValueError('trainData to be used contains no Close or Value column')


    # Split this into features and targets - note we are generating features using the actual value of the price
    # and using these to predict the normalised value
    featuresRaw = np.array(data.reset_index()['Close'])

    # Set a lookback window size and generate features using that lookback window
    features = generateNormalizedLookback(featuresRaw,  lookbackWindow)
    amps, phases = generateFFTLookback(featuresRaw,  lookbackWindow)

    # Here what we have is a 2D np array where for each element, the first X are the normalised raw prices,
    # the next X are the normalised FFT amplitudes of those prices, and the final X are the normalised FFT
    # phases of those prices. Note here that with a larger lookback window, the size of the array (X) gets bigger
    combinedFeatures = np.concatenate((features,amps,phases), axis=1)
    
    # Get the dates of our features / targets
    datesOfFeatures = data.iloc[lookbackWindow:].index
    
    return combinedFeatures, datesOfFeatures

def getTargets(trainData, extremaFilters):

    '''
    Reads in stock data before returning our prediction target for our turning point mmodel. It get's
    the extrema for the 'Close' prices and then filtering the extrema according to the extremaFilters value.
    Also returns the indices of the targets from the original data

    Keyword arguments:
    trainData -- dictionary where the k,v pairs are arguments to the quandl.get function
        Alternatively, a path to a .csv file or a pandas dataFrame
    extremaFilters -- dictionary where the k,v pairs are the column and value to filter >= to in the output of
        the getLabelledExtrema function call

    Returns
    np.array
    index
    '''

    if isinstance(trainData, dict):
       
        # Load in our price data
        #data = yf.download(**trainData)

        # Load in DJIA data from October 2008 to May 2009
        data = quandl.get(**trainData).reset_index()

    elif isinstance(trainData, str):
        
        # Read in our data and set the index column correctly
        data = pd.read_csv(trainData, index_col = 0)

    elif isinstance(trainData, pd.DataFrame):

        data = trainData

    else:

        raise TypeError('data argument is not passed a supported class')

    # If the data has no close column
    if sum([True for x in data.columns if x == 'Close']) == 0:

        # If the data being used has a value column, convert that to 'Close'
        if sum([True for x in data.columns if x == 'Value']) == 1:
            data.rename(columns = {'Value' : 'Close'}, inplace = True)

        # Else raise an error
        else :
            raise ValueError('trainData to be used contains no Close or Value column')


    # For each local extrema in the 'Close' prices, get its type (maxima or minima),
    # its degree, and its impact
    
    # If we're going to be using both impact and degree to filter our data
    if 'impact' in [*extremaFilters] and 'degree' in [*extremaFilters]:

        extremaDf = getLabelledExtrema(np.array(data['Close']))

    # If we're only going to be using one of them
    else:

        # Pass the key in our extremaFilters as the argument to getLabelledExtrema
        extremaDf = getLabelledExtrema(np.array(data['Close']), [*extremaFilters][0])

    extremaDf.set_index(data.index, inplace=True)

    extremaDf.index.names = ['Datetime']
    
    # Copy our DF so we can filter it without influencing the underlying data
    filteredDf = extremaDf.copy()
    
    # For each filter we want to use (e.g. impact above a certain value) apply this
    for key, value in extremaFilters.items():
        
        filteredDf = filteredDf[filteredDf[key] >= value]

    # If after filtering we only have one kind of extrema
    if len(filteredDf['label'].unique()) < 2:

        print('Filtering values returned ineligible dataframe, try another')
        
        # Return none for all arguments
        return None, None, None
    
    # If we have two
    else:

        # For each type of extrema
        for flag in filteredDf['label'].unique():
            
            # If we have less than 3 of one class
            if int(filteredDf[filteredDf['label'] == flag].groupby('label').count().reset_index()['value']) < 3:
                
                # Return none for all arguments
                return None, None, None
    
    
    # Reset our index so we don't have any gaps
    filteredDf.reset_index(inplace = True)

    # Get the indices to drop from this filtered DF
    indicesToRemove = getExtremaToDrop(filteredDf)

    # Drop the indices that it tells us to drop from the tpOsc object
    tpOsc = filteredDf.drop(indicesToRemove, axis=0)
    tpOsc.set_index(extremaDf.index.name, inplace= True)
    
    # Return a Tp oscillator that now has peaks as the highest point between two troughs, and vice versa
    newTpOscDirty = optimiseTpOscillator(
        extremaDf['value'], 
        tpOsc[['value', 'label']]
    )

    secondCleanToDrop =  getExtremaToDrop(newTpOscDirty)

    # Drop the indices that it tells us to drop from the tpOsc object
    newTpOsc = newTpOscDirty.drop(secondCleanToDrop, axis=0)

    # Now we normalise the values in our Tp oscillator to be 0 at a trough, and 1 at a peak. Inbetween values are
    # normalised between peaks and troughs
    TpOscFull = normaliseTpOsc(extremaDf, newTpOsc)

    # Filter out turning point oscillator for values that are not null, i.e., values after our first turning point
    # and before our last
    dataToUse = TpOscFull[~TpOscFull['normalised'].isnull()]

    # Split this into features and targets - note we are generating features using the actual value of the price
    # and using these to predict the normalised value
    targetRaw = np.array(dataToUse['normalised'])
    
    # Get the dates of our features / targets
    datesOfTargets = dataToUse.index
    
    return targetRaw, datesOfTargets

def getFeaturesAndTargets(trainData, extremaFilters, lookbackWindow):
    '''
    Wraps around getFeatures and getTargets to return both for a trainData argument, extremaFilters dictionary,
    and lookbackWindow. Ensures the returned features and targets correspond to the same indices.

    Keyword arguments:
    trainData -- dictionary where the k,v pairs are arguments to the quandl.get function
        Alternatively, a path to a .csv file or a pandas dataFrame
    extremaFilters -- dictionary where the k,v pairs are the column and value to filter >= to in the output of
        the getLabelledExtrema function call
    lookbackWindow -- integer value that dictates how many of the previous prices to use for our features

    Returns:
    np.array of features
    np.array of targets
    index of the features
    '''

    # Get our features and targets
    featuresRaw, datesOfFeaturesRaw = getFeatures(trainData, lookbackWindow)
    targetsRaw, datesOfTargetsRaw = getTargets(trainData, extremaFilters)

    # Filter our feature dates so they only include dates where we have targets
    #datesOfTargets = datesOfTargetsRaw[datesOfTargetsRaw > datesOfFeaturesRaw[0]]
    datesOfFeatures = datesOfFeaturesRaw[datesOfFeaturesRaw < datesOfTargetsRaw[-1]]

    # Filter our targets so they only apply for where we have features and vice versa
    targets = targetsRaw[datesOfTargetsRaw > datesOfFeaturesRaw[0]]
    features = featuresRaw[datesOfFeaturesRaw < datesOfTargetsRaw[-1]]

    if len(targets) != len(features):
        raise ValueError('Features not same length as targets')
    
    return features, targets, datesOfFeatures

def TpSVROptParams(trainData, extremaFilters, lookbackWindow, testSize, valSize, gridArgs):
    '''
    Function wraps around getFeaturesAndTargets to then split the data into train/validate/test sets
    before forming a grid search using the TpSVRGridSearch function and gridArgs. The hyperparameters that
    provide the lowest value of TpRMSE are returned.

    Keyword arguments:
    trainData -- dictionary where the k,v pairs are arguments to the yfinance.download function
        Alternatively, a path to a .csv file
    extremaFilters -- dictionary where the k,v pairs are the column and value to filter >= to in the output of
        the getLabelledExtrema function call
    lookbackWindow -- integer value that dictates how many of the previous prices to use for our features
    testSize -- double that indicates the proportion of our data to use for our test set
    valSize -- double that indicates the proportion of our non-test data to use as our validation set
    gridArgs -- dictionary with k,v pairs that provide the inputs to the TpSVRGridSearch function
    '''
    
    # For some trainData, extremaFilter value, and lookbackWindow, get our features and targets
    combinedFeatures, target, datesOfFeatures = getFeaturesAndTargets(trainData, extremaFilters, lookbackWindow)

    # Split into test and train
    X_train, X_test, y_train, y_test = train_test_split(combinedFeatures, target, test_size=testSize, random_state=1, shuffle = False)

    # Split train into train and validate
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=valSize, random_state=1, shuffle = False)

    # Perform our grid search according to the strategy outlined in the manuscript
    gridOutput = TpSVRGridSearch(
        **gridArgs
        , X_train = X_train
        , y_train = y_train
        , X_val = X_val
        , y_val = y_val
    )

    # Get out a dictionary of the hyperparameters that provided the lowest TpRMSE value
    outDf = pd.DataFrame(gridOutput)
    optimalRow = outDf[outDf['Errors'] == outDf['Errors'].min()]
    bestParams = optimalRow.to_dict(orient = 'records')[0]
    
    bestParams['X_train'] = X_train
    bestParams['y_train'] = y_train
    bestParams['X_val'] = X_val
    bestParams['y_val'] = y_val
    bestParams['X_test'] = X_test
    bestParams['y_test'] = y_test
    
    return bestParams