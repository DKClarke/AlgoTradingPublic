from scipy.signal import argrelextrema # Import to find indices of extrema of different degrees
import numpy as np # For np methods
import pandas as pd # For returning data in a pd df


def getExtremaDegrees(valuesArray):
    """
    Take a np array of numbers and return a np array where each element is the degree to which that
    index is a local extrema. Negative degrees indicate minima and vice versa.

    Keyword arguments:
    valuesArray -- a np. array
    """

    if(len(valuesArray) == 0):
        raise ValueError('Input array contains no values')

    # Throw an error if the numpy array passed in isn't numeric
    _NUMERIC_KINDS = set('uif')
    if valuesArray.dtype.kind not in _NUMERIC_KINDS:

        raise TypeError('numpy array is not numeric')
    
    # Create a np.array to store the degree of extrema at each index of input
    degreesArray = np.zeros(shape = len(valuesArray))
    
    # Iterate through our possible degree values from highest to lowest so we don't have to overwrite
    # anything
    for currDegree in reversed(range(1, len(valuesArray))):
        
         # Get the indices of the current degree minima and maxima
        ilocs_min = argrelextrema(valuesArray, np.less_equal, order=currDegree)[0]
        ilocs_max = argrelextrema(valuesArray, np.greater_equal, order=currDegree)[0]
        
        # Store the degree at these indices into degreesArray
        np.put(degreesArray, ilocs_min[np.where(degreesArray[ilocs_min] == 0)[0]], [-currDegree])
        np.put(degreesArray, ilocs_max[np.where(degreesArray[ilocs_max] == 0)[0]], [currDegree])
    
    return degreesArray

def getLocalExtrema(valuesArray):
    """
    Take a np array of numbers and return a np array where each element is the degree to which that
    index is a local extrema. Negative degrees indicate minima and vice versa.

    Limited to only finding degrees of 1 i.e. returns a np array containing indices of local extrema

    Keyword arguments:
    valuesArray -- a np. array
    """

    if(len(valuesArray) == 0):
        raise ValueError('Input array contains no values')

    # Throw an error if the numpy array passed in isn't numeric
    _NUMERIC_KINDS = set('uif')
    if valuesArray.dtype.kind not in _NUMERIC_KINDS:

        raise TypeError('numpy array is not numeric')
    
    # Create a np.array to store the degree of extrema at each index of input
    degreesArray = np.zeros(shape = len(valuesArray))
        
    # Get the indices of local minima and maxima
    ilocs_min = argrelextrema(valuesArray, np.less_equal, order=1)[0]
    ilocs_max = argrelextrema(valuesArray, np.greater_equal, order=1)[0]
        
    # Store the degree at these indices into degreesArray
    np.put(degreesArray, ilocs_min[np.where(degreesArray[ilocs_min] == 0)[0]], [-1])
    np.put(degreesArray, ilocs_max[np.where(degreesArray[ilocs_max] == 0)[0]], [1])
    
    return degreesArray


def getExtremaImpact(valuesArray, degreesArray):
    """Take a np array of numbers, and a np array identifying which of these are extrema (returned from 
    getLocalExtrema) and return a np array where each element the impact of the extrema. If the point is 
    not an extrema, the impact is zero.

    Keyword arguments:
    valuesArray -- a np. array
    degressArray -- a np. array, example is returned from getLocalExtrema
    """

    if(len(valuesArray) == 0):
        raise ValueError('Input array contains no values')

    # Throw an error if the numpy array passed in isn't numeric
    _NUMERIC_KINDS = set('uif')
    if valuesArray.dtype.kind not in _NUMERIC_KINDS:

        raise TypeError('numpy array is not numeric')

    # Get the degrees of the extrema in our values
    extremaLocs = np.concatenate([argrelextrema(valuesArray, np.less_equal)[0], argrelextrema(valuesArray, np.greater_equal)[0]])

    # Create an array to store the impact values of all our extrema
    impactArray = np.zeros(shape = len(valuesArray))

    # For each extrema
    for extrema in extremaLocs:

        # Get the value at that extrema and then all the values afterwards
        Xt = valuesArray[extrema]
        afterXt = valuesArray[extrema+1:]

        # If we have values afterwards
        if(len(afterXt) >0):
            
            # Get the locations where values are below current if we're at minima, else where higher
            XnAll = np.where(valuesArray[extrema+1:] < Xt) if degreesArray[extrema] < 0 else np.where(valuesArray[extrema+1:] > Xt)

            # If we have values that meet this criteria
            if len(XnAll[0]) > 0:

                # Get the soonest value that meets this criteria
                Xn = XnAll[0][0]

                # Find the maximum value between our current and this if we're a minima, else find minimum value
                impactPoint = afterXt[:Xn].max() if degreesArray[extrema] < 0 else afterXt[:Xn].min()

            # If we have no values ahead of us that meet our criteria
            else:

                # Get the local maxima or minima after our point
                impactPoint = afterXt.max() if degreesArray[extrema] < 0 else afterXt.min()

            ## Calculate the impact of the current extrema
            impact = (impactPoint - Xt) / Xt

            # Store our impact in our output array
            impactArray[extrema] = -impact
            
    return impactArray

def getWorstAdjacentExtrema(data, extremaType, indices):
    '''
    Function takes a 1D np array, a string value for the type of local extrema this data contains,
    and the indices these data are derived from.
    Returns a dict containing the index of the extrema that are less extreme than the most extreme point.

    Keyword arguments:
    data -- 1D np array
    extremaType -- string of either 'minima' or 'maxima'
    indices -- A list of indices as long as data
    '''

    if data is None:
        return {'indices': []}

    # Find the lowest / highest value out of them
    if extremaType == 'minima':
        toFind = data.min()
    elif extremaType == 'maxima':
        toFind = data.max()
    else:
        raise ValueError('values in label are not one of minima or maxima')

    # Get the indices that match our ideal values - there's a chance there are more than one
    matches = indices.values[data == toFind]

    # Get the indices that don't match our ideal values
    dropIndices = list()
    toDrop = indices.values[data != toFind].tolist()

    # If we have indices that don't match, store them in dropIndices
    if len(toDrop) > 0:
        dropIndices = dropIndices + toDrop

    # If we have more than one matching indices, add the all except the last to the dropIndices list
    if len(matches) > 1:
        toAdd = matches[:-1].tolist()
        dropIndices = dropIndices + toAdd

    return {'indices': dropIndices}

def getSequentialExtrema(filteredDf):
    '''
    Function takes a pd DF containing only extrema from getLabelledExtrema
    Returns a list of tupples which for each sequence of identical extrema (i.e. multiple peaks or troughs)
    contains the data, extremaType, and indices of these extrema

    Keyword arguments:
    filteredDf -- A pd df containing only extrema with a 'label' and 'value' column
    '''

    # For each set of sequential extrema of the same type
    argsList = list()
    for k, v in filteredDf.groupby((filteredDf['label'].shift() != filteredDf['label']).cumsum()):

        # If they are indeed sequential
        if len(v) > 1:

            # Pass their necessary values to our arguments list
            argsList.append({'data': np.array(v['value']), 'extrema': v.iloc[0]['label'], 'indices': v.index})

    return argsList

def getExtremaToDrop(filteredDf):
    '''
    Function takes a df output by getLabelledExtrema() that has been filtered to only include
    extrema and returns a list that contains the indices to remove from that DF so that for
    any points where there are a sequence of extrema of the same type in a row e.g. minima, minima we only retain
    the one minima with the lowest value.

    Keyword arguments:
    filteredDf -- A pd DF output by getLabelledExtrema that has been filtered to contain only extrema
    '''
    
    # Get a list of the extrema that are sequential in our DF
    storeIt = getSequentialExtrema(filteredDf)
    
    # Separate these components
    data = [x['data'] for x in storeIt]
    extremaType = [x['extrema'] for x in storeIt]
    indices = [x['indices'] for x in storeIt]

    # Get the indices to remove from each of these sequences
    results = list(
        map(
            getWorstAdjacentExtrema
            , data
            , extremaType
            , indices
        )
    )

    # Compile these into a single list
    toRemove = list()
    for item in results:
        toRemove = toRemove + (item['indices'])
        
    # Return the list
    return toRemove

def getLabelledExtrema(valuesArray, extremaType = 'both'):
    """Take a np array of numbers and return a pd Df with columns for this original
    array, the degree of the extrema in the array, the impact of the extrema in the array,
    and labels for whether the extrema are minima or maxima.

    Wraps around getExtremaDegrees() and getExtremaImpact()

    Keyword arguments:
    valuesArray -- a np. array
    extremaType -- a string that indicates whether to get impact, degrees, or both
    """

    if extremaType == 'degree':

        print('Retrieving only extrema values in degrees')

        # Get an array labelled with degrees of extrema
        extremaDegrees = getExtremaDegrees(valuesArray)

        # Create a dummy impacts array
        extremaImpacts = np.zeros(len(valuesArray))

    elif extremaType == 'impact':

        print('Retrieving only extrema values in impact')

        # Get an array of local extrema
        extremaDegrees = getLocalExtrema(valuesArray)

        # Get an array labelled with the impact of the extrema
        extremaImpacts = getExtremaImpact(valuesArray, extremaDegrees)

    elif extremaType == 'both':

        print('Retrieving extrema values in degrees')

        # Get an array labelled with degrees of extrema
        extremaDegrees = getExtremaDegrees(valuesArray)

        print('Retrieving extrema values in impact')

        # Get an array labelled with the impact of the extrema
        extremaImpacts = getExtremaImpact(valuesArray, extremaDegrees)

    else:

        raise ValueError('extremaType argument not one of impact, degree, or both')
    
    # Store these in a pd df to return
    extremaDf = pd.DataFrame({'value': valuesArray,
                         'degree': extremaDegrees,
                         'impact': extremaImpacts})

    # Get the indices of our minima and maxima
    minima = np.where(extremaDegrees < 0)[0]
    maxima = np.where(extremaDegrees > 0)[0]

    # Store labels for these in the df
    extremaDf['label'] = None
    extremaDf.loc[minima, 'label'] = 'minima'
    extremaDf.loc[maxima, 'label'] = 'maxima'

    # Convert our degrees and impacts to absolute values
    extremaDf['degree'] = extremaDf.apply(lambda x: abs(x['degree']), axis = 1)
    extremaDf['impact'] = extremaDf.apply(lambda x: abs(x['impact']), axis = 1)

    return extremaDf

def optimiseCurrentExtremaPair(rawValues):
    '''
    Takes a 1D np array, find the indices of the min and max values,
    returns them in a dict

    Keyword arguments:
    rawValues -- 1D np array
    '''

    optimalLocs = {
        'minima': np.where(rawValues == rawValues.min())[0][0],
        'maxima': np.where(rawValues == rawValues.max())[0][0]
    }

    return optimalLocs

def optimiseTpOscillator(rawValues, rawTpOsc):
    """Take the raw values in pd series, as well as the alternating seqeuence of extrema (i.e. a turning point alternator) built on these
    values, and makes sure that all our extrema represent the greatest magnitude of extrema i.e. that all our maxima are the highest
    point between two minima and vice versa. We return a version of rawTpOsc that addresses this.

    Keyword arguments:
    rawValues -- pd series of our raw values with a datetime index
    rawTpOsc -- a pd.DataFrame turning point alternator; a filtered version of inputDf that only contains extrema, and where we alternate
        from one kind of extrema to the next, with a datetime index
    """
    
    # Create a dictionary to story the type and values of our optimised extrema
    returnCopyDict = {
        'label': list()
        , 'value': list()
    }
    
    # Create a list to store their indices
    returnCopyIndex = list()

    # For each pairing of extrema in our TP oscillator
    for currInd in range(0, len(rawTpOsc)-1):

        # Get the data this represents
        extremaData = rawTpOsc.iloc[currInd : currInd + 2]

        # Get the raw data these two extrema are bounding - since we're using datetime indices this is inclusive
        inbetweenData = rawValues.loc[extremaData.index[0]:extremaData.index[1]]

        # Find the locations in this raw data of the lowest and highest points
        optimalLocs = optimiseCurrentExtremaPair(np.array(inbetweenData))

        # For each of these locations, get out that info from our raw data
        # and store it in our dictionary and list
        for k,v in optimalLocs.items():
            
            # If we haven't already saved this index
            if inbetweenData.index[v] not in returnCopyIndex:
                
                # Append our values
                returnCopyIndex.append(inbetweenData.index[v])
                returnCopyDict['label'].append(k)
                returnCopyDict['value'].append(inbetweenData.iloc[v])

    # Convert dict to pd
    returnDf = pd.DataFrame(returnCopyDict, index = returnCopyIndex)
    returnDf.sort_index(inplace = True)

    # Return our dictionary and index as a dataframe
    return returnDf

def getNormalisedTpValue(rawValue, peakValue, troughValue):
    '''
    Function used to normalise values in the TP alternator. Takes a number, the value of its nearest peak,
    and the value of its nearest trough, and normalises the number between those.

    Keyword arguments:
    rawValue -- double
    peakValue -- double; derived from the value at the nearest peak to rawValue in a TP alternator
    troughValue -- double; derived from the value at the nearest trough to rawValue in a TP alternator
    '''
    if rawValue == peakValue and rawValue == troughValue:
        return None
    else:
        normedValue = (rawValue - troughValue) / (peakValue - troughValue)
        return normedValue

def normaliseTpOsc(rawValues, TpAlt):
    """Take a df output by getLabelledExtrema, and a alternating seqeuence of extrema (i.e. a turning point alternator)
    that has been optimised using previously defined functions. Returns a pd df that normalises the values in the df
    to between 0 and 1 between each pair of altnerating extrema (i.e. created a turning point oscillator)

    Keyword arguments:
    inputDf -- a pd.DataFrame output by getLabelledExtrema
    TpAlt -- a pd.DataFrame turning point alternator; a filtered version of inputDf that only contains extrema, and where we alternate
        from one kind of extrema to the next
    """

    # Generate a np array that will store the value of our turning point oscillator
    osc = rawValues.copy()
    osc['normalised'] = None

    # For our turning point alternating sequence, set the osc to 1 or 0 at
    # maxima and minima
    osc.loc[TpAlt[TpAlt['label'] == 'maxima'].index, 'normalised'] = 1
    osc.loc[TpAlt[TpAlt['label'] == 'minima'].index, 'normalised'] = 0

    # Rewrite our labels according to our cleaned turning points oscillator
    osc['label'] = None
    osc.loc[TpAlt[TpAlt['label'] == 'maxima'].index, 'label'] = 'maxima'
    osc.loc[TpAlt[TpAlt['label'] == 'minima'].index, 'label'] = 'minima'


    # For every combination of subsequent rows (alternating extrema) in our alternator
    for currCombo in range(0, len(TpAlt)-1):

        # Get the two extrema this corresponds to
        ExtremaData = TpAlt.iloc[currCombo : currCombo+2]

        if len(ExtremaData) == 2:

            # Get the value at the peak and value at the trough
            Pt = ExtremaData[ExtremaData['label'] == 'maxima']['value'].iloc[0]
            Tt = ExtremaData[ExtremaData['label'] == 'minima']['value'].iloc[0]

            # Get the data that sits between these two extrema
            inbetweenData = rawValues.loc[ExtremaData.index[0]:ExtremaData.index[1]]

            if len(inbetweenData) > 0:

                # Normalise the data to the peak and trough
                midOsc = np.array([getNormalisedTpValue(x, Pt, Tt) for x in inbetweenData['value']])

                nanLocs = np.isnan(midOsc.astype(float))
                if nanLocs.any():
                    midOsc[nanLocs] = osc.loc[ExtremaData.index[0]]['value']
                    
                # Store this normalised value in our osc array
                osc.loc[ExtremaData.index[0]:ExtremaData.index[1], 'normalised'] = midOsc.tolist()

    return osc