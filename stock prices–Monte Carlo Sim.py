#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
from pandas_datareader import data

import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import datetime

import yfinance as yf
from datetime import date, timedelta
from csv import reader
import seaborn as sns
import math as math

import statistics
from statistics import variance
from fractions import Fraction as fr
sns.set()
yf.pdr_override()


# In[51]:


#function features:
    # Plots Monte Carlo Simulations using Geometric Brownian Motion
    # Compares values predicted using the Monte Carlo Simulation with the actual stock prices
    # Jagged black line on Monte Carlo Simulation is actual stock price
    # Straight black line on Monte Carlo Simulation is the value predicted by the simulation
    # Uses different amounts of historical data in the equation to evaulate which amount of historical data used is most effective in predicting stock prices
n=1000
def function(ticker):
    start_date = '1-1-2012'
    end_date = '31-12-2020'
    
    start = datetime.datetime.strptime(start_date, '%d-%m-%Y')
    end = datetime.datetime.strptime(end_date, '%d-%m-%Y')
    stock = data.get_data_yahoo(ticker, data_source='yahoo', start=start, end=end)
#     stock = data.DataReader(ticker, 'yahoo',start = datetime.strptime(start_date, '%d-%m-%Y'), end = datetime.strptime(end_date, '%d-%m-%Y'))
    n=1000
    stock['Close'].plot(figsize=(15,6))
    #start & final row indexes
    start2020_row = len(stock) - 1 - 252
    final_row = len(stock) - 1
    
    #2020start
    real_start = stock['Close'][start2020_row]

    #2020end
    real_final = stock['Close'][final_row]
    
    print("start price:", round(real_start, 2))
    print("final price:", round(real_final, 2))
    
    #time intervals used
    times = [start2020_row-252*1, 
             start2020_row-252*2, 
             start2020_row-252*3, 
             start2020_row-252*4,
             start2020_row-252*5,
             start2020_row-252*6,
             start2020_row-252*7,
             1]
    
    #actual price
    actual = []
    for x in stock['Close'][start2020_row:start2020_row + 252]:
        actual.append(x)
    
    #generate random variables array
    random = np.random.normal(0, 1, 5000000)
    print("random", random[0])

    #logged return (ALL values up until 2020)
    log_close = np.log(1 + stock['Close'][:start2020_row].pct_change())
    
    plt.figure(figsize=(15,6))
    log_plot = sns.histplot(log_close.iloc[1:], binwidth=0.002)
    plt.xlabel("Daily logarithmic return")
    plt.ylabel("Frequency")
    plt.title("Log of historical closing prices from 2012-2021", fontdict=None, loc='center')
    log_plot.set(xlim=(-0.06, 0.06))

    
#start time loop
    predicted_final = ['Na']
    Zscore = ['Na']
    fZscore = ['Na']
    AllZscores = []
    mean_list = ['Na']
    var_list = ['Na']
    drift_list = ['Na']
    sdev_list1 = ['Na']
    othersdev = ['Na']
    
    for t in range(8):
        
        #calculate drift and sdev
        mean = sum(log_close[times[t]:]) / len(log_close[times[t]:])
        var = sum(((log_close[times[t]:]-mean)**2)/len(log_close[times[t]:]))
        drift = mean - var/2
        sdev1 = var**0.5
        
        mean_list.append("{:.2e}".format(mean))
        var_list.append("{:.2e}".format(var))
        drift_list.append("{:.2e}".format(drift))
        sdev_list1.append(sdev1)
    
        #generate Geometric Brownian motion values
        GBM = []
        count = 0
        for x in range(n):
            sf = real_start
            temporary = []
            for i in range(252):
                s0 = sf
                sf = s0 * math.exp(drift + var**(0.5) * random[count])
                temporary.append(sf)
                count += 1
            GBM.append(temporary)
        GBM = np.array(GBM)
        
        
#         if t==7:
#             plt.figure(figsize=(15,6))
#             plt.plot(pd.DataFrame(GBM[0]))
#             plt.xlabel("# of Trading Days after December 31, 2019")
#             plt.ylabel("Closing Price ($)")
        
        
        #print final predicted price 
        final_prices = [0.00] * n
        for x in range(n):
            final_prices[x]=GBM[x, 251]
        predicted_price = sum(final_prices)/len(final_prices)
        
        predicted_final.append(round(predicted_price, 2))
        
        
        #Create average path
        ave_path = []
        for x in range(252):
            price_sum = 0
            #find mean
            for i in range(n):
                price_sum += GBM[i][x]
            ave_path.append(price_sum/n)
        
        
        #sdev path
        sdev_list = []
        for i in range(252):
            Sum = 0
            for x in GBM[i]:
                Sum += (x-ave_path[i])**2
            sdev_list.append((Sum/n)*0.5)   
            Sum = 0
            
        othersdev.append(sdev_list[-1])
        
        
        #Z-score
        zscores = []
        Sum = 0
        Count = 0

        for m, s, v in zip(ave_path, sdev_list, actual):
            temp_zscore = (v-m)/s
            #list of all
            zscores.append(temp_zscore)
            #mean z-score  
            #if Count > 20:
            Sum += temp_zscore 
            Count += 1
        
        ave_zscore = Sum / len(zscores)
        
        Zscore.append("{:.2e}".format(ave_zscore))
        fZscore.append("{:.2e}".format(zscores[-1]))
        AllZscores.append(zscores)
        
        
        #flip GBM
        flippedGBM = np.linspace(0.0, 252.0*n, num=252*n).reshape((252, n))
        dimension_number = 0
        element_number = 0
        
        for x in range(n):
            dimension_number = 0
            for i in range(252):
                flippedGBM[dimension_number, element_number] = GBM[element_number, dimension_number]
                dimension_number += 1
            element_number += 1

        #Plot monte Carlo graph
        plt.figure(figsize=(15,6))
        ax = plt.plot(pd.DataFrame(flippedGBM).iloc[:,0:n])
        plt.xlabel("# of Trading Days after December 31, 2019")
        plt.ylabel("Closing Price ($)")
        plt.title("Monte Carlo Sim using {} years historical data".format(t+1), fontdict=None, loc='center')

        df2 = plt.plot(pd.DataFrame(ave_path).iloc[:], color='black', linewidth=2, markersize=12)
        df3 = plt.plot(pd.DataFrame(actual).iloc[:], color='black', linewidth=2, markersize=12)
        plt.grid
        
        
        if t==0:
            plt.figure(figsize=(15,6))
            sns.histplot(final_prices, binwidth=50)
            plt.xlabel("Final closing prices ($)")
            plt.ylabel("Frequency")
            plt.title("Closing prices in 2020", fontdict=None, loc='center')
            
            plt.figure(figsize=(15,6))
            sns.histplot(np.log(final_prices), binwidth=0.03)
            plt.xlabel("Natural logarithm of final closing prices")
            plt.ylabel("Frequency")
            plt.title("Log of closing prices in 2020", fontdict=None, loc='center')

        
    df = pd.DataFrame({"Predicted": predicted_final,
                       "Z-score": Zscore,
                       "F Z-score": fZscore,
                       "Mean log": mean_list,
                       "Var": var_list,
                       "Sdev": sdev_list1,
                       "Drift": drift_list,
                       "Sdev other": othersdev
                      })
    
    #plot Z-score
    plt.figure(figsize=(15,6))
    plt.title("Z-scores for each simulation", fontdict=None, loc='center')
    for z in range(8):
        z_plot = plt.plot(pd.DataFrame(AllZscores[z]).iloc[:], linewidth=2, markersize=12)

    print(df)


# In[52]:


function('GOOG')


# In[53]:


function('JHB')


# In[54]:


function('CADUSD=X')

