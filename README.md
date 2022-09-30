# Algotrading Using Support Vector Regression

The work in this repo is based on [this preprint](https://arxiv.org/abs/1209.0127). The general idea is to use machine learning to predict 
turning points (i.e. pivots; local extrema) in share prices. To do this we first have to detect these turning points, before generating a 
Turning Point Oscillator - this is what we use as our prediction target. This oscillator normalises all prices to their nearest trough (0) and 
peak (1). We then use Support Vector Regression to predict oscillator values, and if our prediction crosses certain thresholds, we execute a 
buy / sell order. 

## [Predicting Turning Points in Share Prices.ipynb](https://github.com/DKClarke/AlgoTradingPublic/blob/main/Predicting%20Turning%20Points%20in%20Share%20Prices.ipynb)

This approach is explored and backtested as a trading strategy in the 'Predicting Turning Points in Share Prices' Jupyter notebook.

The application of this backtesting to multiple timepoints to investigate whether the performance is significantly different from default strategies
will be explored in another Jupyter notebook.

## [extremaFuncs.py](https://github.com/DKClarke/AlgoTradingPublic/blob/main/extremaFuncs.py)

This python file contains functions written to simplify the process of preprocessing share data for use in this trading strategy.

## [modelFuncs.py](https://github.com/DKClarke/AlgoTradingPublic/blob/main/modelFuncs.py)

This python file contains functions written to simplify the process of generating and using Support Vector Regression in this trading strategy.

## [tradingFuncs.py](https://github.com/DKClarke/AlgoTradingPublic/blob/main/tradingFuncs.py)

This python file contains functions written to simplify the process of evaluating our approach as a trading strategy.
