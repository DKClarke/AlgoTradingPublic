# Algotrading Using Support Vector Regression

The work in this repo is based on [this preprint](https://arxiv.org/abs/1209.0127). The general idea is to use machine learning to predict 
turning points (i.e. pivots; local extrema) in share prices. To do this we first have to detect these turning points, before generating a 
Turning Point Oscillator - this is what we use as our prediction target. This oscillator normalises all prices to their nearest trough (0) and 
peak (1). We then use Support Vector Regression to predict oscillator values, and if our prediction crosses certain thresholds, we execute a 
buy / sell order. 

## [1. Predicting Turning Points in Share Prices.ipynb](https://github.com/DKClarke/AlgoTradingPublic/blob/main/1.%20Predicting%20Turning%20Points%20in%20Share%20Prices.ipynb)

This approach is explored and backtested as a trading strategy in the '1. Predicting Turning Points in Share Prices' Jupyter notebook.

## [2. Backtesting the Turning Point Strategy.ipynb](https://github.com/DKClarke/AlgoTradingPublic/blob/main/2.%20Backtesting%20the%20Turning%20Point%20Strategy.ipynb)

In this notebook we backtest our strategy against multiple batches of time over the last 2 years so we can compute the average difference in returns between our SVR strategy and a simple buy-and-hold strategy. The reason this requires a whole other notebook, is we need to iterate over 5 million combinations to perform an exhaustive grid search!

## [extremaFuncs.py](https://github.com/DKClarke/AlgoTradingPublic/blob/main/extremaFuncs.py)

This python file contains functions written to simplify the process of preprocessing share data for use in this trading strategy.

## [modelFuncs.py](https://github.com/DKClarke/AlgoTradingPublic/blob/main/modelFuncs.py)

This python file contains functions written to simplify the process of generating and using Support Vector Regression in this trading strategy.

## [tradingFuncs.py](https://github.com/DKClarke/AlgoTradingPublic/blob/main/tradingFuncs.py)

This python file contains functions written to simplify the process of evaluating our approach as a trading strategy.
