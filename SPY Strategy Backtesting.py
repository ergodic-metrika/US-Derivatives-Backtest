# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:05:31 2023

@author: sigma
"""
# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import norm
#ip install fredapi
from fredapi import Fred
#pip install fredpy
import fredpy as fp
#pip install QuantLib
import QuantLib as ql
import statistics
import math
import matplotlib.pyplot as plt
from tabulate import tabulate # conda install tabulate
import yfinance as yf

from BS import BS

#Fred database to get SP500 historical prices
fred=Fred(api_key='8ac7604258e93947305e988327a4f7df')

vix=fred.get_series('VIXCLS', start_date='2018-03-01', end_date='2023-04-30')
vix
plt.plot(vix)

median_vix=np.median(vix)
print(median_vix)


average_vix=np.mean(vix)
print(average_vix)

#SPY from Yahoo Finance
#spy = yf.Ticker('SPY')

#daily_spy=spy.history(start='2022-09-11', end='2023-03-11')
#print(daily_spy)

#Options of SPY
#options = spy.option_chain('2023-03-31')
#print(options)

# For ITM Call
c_spot = 340
c_strike = 330
c_rate = 0.03
c_dte =20/365
Initial_c_dte =30/365
End_c_dte=1/365
c_vol = 0.1966
c_spy_itm_opt = BS(c_spot,c_strike,c_rate,c_dte,c_vol)
print(f'Option Price with BS Model is {c_spy_itm_opt.callPrice:0.4f}')

Initial_Day_ITM_Call_price= BS(c_spot,c_strike, c_rate, Initial_c_dte, c_vol)
print(f'Option Price with BS Model is {Initial_Day_ITM_Call_price.callPrice:0.4f}')
End_Day_ITM_Call_price= BS(c_spot,c_strike,c_rate,End_c_dte,c_vol)
print(f'Option Price with BS Model is {End_Day_ITM_Call_price.callPrice:0.4f}')

Net_ITM_Short_Call=Initial_Day_ITM_Call_price.callPrice-End_Day_ITM_Call_price.callPrice
Net_ITM_Short_Call

ITM_Call_Theta=abs(c_spy_itm_opt.callTheta)*100

print(ITM_Call_Theta)

ITM_Call_Delta=-(c_spy_itm_opt.callDelta)
print(ITM_Call_Delta)
      
#For OTM Put      
p_spot = c_spot
p_strike = 295
p_rate = c_rate
p_dte = c_dte
Initial_p_dte =30/365
End_p_dte=1/365
p_vol = c_vol
p_spy_otm_opt = BS(p_spot,p_strike,p_rate,p_dte,p_vol)
print(f'Option Price with BS Model is {p_spy_otm_opt.putPrice:0.4f}')

Initial_Day_OTM_Put_price= BS(p_spot,p_strike,p_rate,Initial_p_dte,p_vol)
print(f'Option Price with BS Model is {Initial_Day_OTM_Put_price.putPrice:0.4f}')

End_Day_OTM_Put_price= BS(p_spot,p_strike,p_rate,End_p_dte,p_vol)
print(f'Option Price with BS Model is {End_Day_OTM_Put_price.putPrice:0.4f}')
       
Net_OTM_Long_Put=End_Day_OTM_Put_price.putPrice-Initial_Day_OTM_Put_price.putPrice
Net_OTM_Long_Put

#Sum up the above SC and BP
Time_premium=(Net_ITM_Short_Call+Net_OTM_Long_Put)
print(Time_premium)

OTM_Put_Theta=p_spy_otm_opt.putTheta*100
print(OTM_Put_Theta)

OTM_Put_Delta=p_spy_otm_opt.putDelta
print(OTM_Put_Delta)

SCBP_netDelta=ITM_Call_Delta+OTM_Put_Delta
print(SCBP_netDelta)


#Import SP500 data from the Fed's FRED database
sp500=fred.get_series('SP500', start_date='2013-03-01', end_date='2023-04-30')
sp500

monthly_sp500=sp500.resample('M').last()
print(monthly_sp500)
plt.plot(sp500)

#For example, my other leg could be just represent 0.7 of the futures's Delta
mirror_partial_hedge_monthly_sp500=-monthly_sp500*0.70
print(mirror_partial_hedge_monthly_sp500)
number_to_add= Time_premium

# use a loop to add the time premium
for i in range(len(mirror_partial_hedge_monthly_sp500)):
   mirror_partial_hedge_monthly_sp500[i] += number_to_add
    
# print the modified list
print(mirror_partial_hedge_monthly_sp500)

monthly_sp500_fd =monthly_sp500.diff()
monthly_sp500_fd
plt.plot(monthly_sp500_fd)

cumul_PnL_sp500=(monthly_sp500_fd.cumsum())*100
cumul_PnL_sp500=cumul_PnL_sp500.dropna()

plt.plot(cumul_PnL_sp500)


cumul_PnL_mirror_time_leg_fd=mirror_partial_hedge_monthly_sp500.cumsum()*100
cumul_PnL_mirror_time_leg_fd=cumul_PnL_mirror_time_leg_fd.dropna()

#Capital and transaction cost set up
initial_margin=13200
Mirror_leg_using_options_margin=13200
Total_capital = initial_margin+Mirror_leg_using_options_margin
PnL_Index_Base=100
#For equity curve setup
PnL_month_net=monthly_sp500_fd+mirror_partial_hedge_monthly_sp500
print(PnL_month_net)

plt.plot(PnL_month_net)
PnL_month_net_trend=PnL_month_net.cumsum()
print(PnL_month_net_trend)
plt.plot(PnL_month_net_trend)




#################Other performancee simulation################################
PnL_month_net_trend_return=PnL_month_net.pct_change()
PnL_month_net_trend_return
# Drop the first row containing NaN value
PnL_month_net_trend_return = PnL_month_net_trend_return.dropna()

# Calculate the cumulative return
cumulative_PnL_month_net_trend_return = (1 + PnL_month_net_trend_return).cumprod()

# Calculate the running maximum of the cumulative return
running_max_PnL_month_net_trend_return = cumulative_PnL_month_net_trend_return.cummax()

# Calculate the drawdown
drawdown_PnL_month_net_trend_return = (cumulative_PnL_month_net_trend_return / running_max_PnL_month_net_trend_return) - 1

# Calculate the max drawdown
max_drawdown_PnL_month_net_trend_return = drawdown_PnL_month_net_trend_return.min()

# Print the max drawdown
print("Max Drawdown of PnL_month_net_trend_return:", max_drawdown_PnL_month_net_trend_return)


# Identify the positive returns (profits) and negative returns (losses)
profits = PnL_month_net_trend_return[PnL_month_net_trend_return > 0]
losses = PnL_month_net_trend_return[PnL_month_net_trend_return < 0]

# Calculate the sum of profits and the sum of losses (in absolute value)
sum_profits = profits.sum()
sum_losses = abs(losses.sum())

# Calculate the profit factor
profit_factor_PnL_month_net_trend_return = sum_profits / sum_losses

# Print the profit factor
print("Profit Factor of PnL_month_net_trend_return:", profit_factor_PnL_month_net_trend_return)
#Profit factor is infinite due to no losses in the monthly series


########################################################################################
equity_curve=PnL_Index_Base*(1+np.cumsum(PnL_month_net_trend_return))
print(equity_curve)
PnL_Index=pd.DataFrame(equity_curve, columns=['PnL Index'])
PnL_Index.plot()

std_dev = np.std(PnL_month_net_trend_return)
print("Standard Deviation of Monthly Return:", std_dev)


#Total Transaction cost including tax and slippage cost (TTC)
#Set at 10% of capital used
TTC=0.10



#Check for correct location
equity_curve[120]
equity_curve[2]

raw_return=(equity_curve[120]-equity_curve[2])/equity_curve[2]
raw_return

#number of years
noy=2023-2013
#annualzied power
ap=1/noy

Final_value=Total_capital*(1+raw_return)
Final_value
Initial_value=Total_capital
Initial_value
annualized_return=((Final_value/Initial_value)**ap-1)*(1-TTC)
annualized_return


portfolio_return=((Final_value-Initial_value)/Initial_value)*(1-TTC)
portfolio_return

#risk free rate for Sharpe ratio
# https://fred.stlouisfed.org/series/FEDFUNDS
#My rfr is a weighted average from the last ten years
risk_free_rate=0.4*0.004+0.2*0.015+0.2*0.001+0.2*0.025
risk_free_rate

#Sharpe ratio
excess_return =portfolio_return -risk_free_rate
excess_return
Annualized_sharpe_ratio = excess_return /std_dev*math.sqrt(12)
print("Annualized Sharpe Ratio:", Annualized_sharpe_ratio)



#******Plotting setup*****#
# Generate some data
PnL_Index.index=pd.to_datetime(PnL_Index.index)
PnL_Index.index
Date = PnL_Index.index
Date
y1 =PnL_Index
y1
y2 = monthly_sp500
y2

# Create the plot and set the first y-axis (left)
fig, ax1 = plt.subplots()
plt.xticks(rotation=90)
ax1.plot(Date, y1, 'b-')
ax1.set_xlabel('Date')
ax1.set_ylabel('PnL', color='b')
ax1.tick_params('y', colors='b')

# Set the second y-axis (right)
ax2 = ax1.twinx()
ax2.plot(Date, y2, 'k.')
ax2.set_ylabel('SP500', color='k')
ax2.tick_params('y', colors='k')

# Show the plot
plt.title('PnL vs SP500')
plt.show()
