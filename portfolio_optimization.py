import pandas as pd
import quandl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


aapl = pd.read_csv(r'C:\Users\Admin\Desktop\Programming\Python Finance\Python-for-Finance-Repo-master\09-Python-Finance-Fundamentals\AAPL_CLOSE',index_col='Date',parse_dates=True)
cisco = pd.read_csv(r'C:\Users\Admin\Desktop\Programming\Python Finance\Python-for-Finance-Repo-master\09-Python-Finance-Fundamentals\CISCO_CLOSE',index_col='Date',parse_dates=True)
ibm = pd.read_csv(r'C:\Users\Admin\Desktop\Programming\Python Finance\Python-for-Finance-Repo-master\09-Python-Finance-Fundamentals\IBM_CLOSE',index_col='Date',parse_dates=True)
amzn = pd.read_csv(r'C:\Users\Admin\Desktop\Programming\Python Finance\Python-for-Finance-Repo-master\09-Python-Finance-Fundamentals\AMZN_CLOSE',index_col='Date',parse_dates=True)


stocks = pd.concat([aapl,cisco,ibm,amzn],axis=1)
stocks.columns = ['AAPL','CSCO','IBM','AMZN']
stocks.head()
stocks.pct_change(1).mean()
stocks.pct_change(1).corr()


log_ret = np.log(stocks/stocks.shift(1))
log_ret.head()

log_ret.hist(bins=100,figsize=(12,8))

log_ret.mean()

log_ret.cov() * 252

np.random.seed(101)

num_ports = 5000
all_weights = np.zeros((num_ports,len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):

    #weights
    weights = np.array(np.random.random(4))
    weights = weights/np.sum(weights)

    # Save weights
    all_weights[ind,:] = weights

    #Expected Return
    ret_arr[ind] = np.sum( (log_ret.mean() * weights) * 252)


    #expected volatility (variance)
    vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))


    #sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

sharpe_arr.max()

sharpe_arr.argmax()
all_weights[1420,:]
max_sr_ret = ret_arr[1420]
max_sr_vol = vol_arr[1420]
plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')

def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    #return 0 if sum of the weights is 1
    return np.sum(weights) - 1

cons = ({'type':'eq','fun':check_sum})
bounds = ((0,1),(0,1),(0,1),(0,1))

init_guess = [0.25,0.25,0.25,0.25]

opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
opt_results
opt_results.x
get_ret_vol_sr(opt_results.x)

frontier_y = np.linspace(0,0.3,100)

def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]

frontier_volatility = []

for possible_return in frontier_y:
    cons = ({'type':'eq','fun':check_sum},
            {'type':'eq','fun':lambda w: get_ret_vol_sr(w)[0]-possible_return})
    result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

    frontier_volatility.append(result['fun'])


plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)
