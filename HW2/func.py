import statsmodels
import scipy
from scipy import stats
from scipy.stats import rv_continuous
import numpy as np
import math
import pandas as pd
import pickle
from statsmodels.distributions.copula.api import CopulaDistribution, StudentTCopula 

def sim_1K_quarterly(org_amt, start_age, end_age, bond_weight, adj_RMD):
    rmd_quarter = pd.read_csv('rmd_quarter_rate.csv')
    (joint_dist, quarter_all, income_fit, dividend_fit, inflation_fit) = pickle.load(open("inputs.pkl", "rb"))
    
    rmd_quarter['Adj_Rate'] = rmd_quarter.Quarterly_Rate * adj_RMD
    
    '''Apply method_quarterly method'''
    
    quarters=np.arange(start_age, end_age+0.1, 0.25)       
    balance = {}
    withdrawal = {} #VaR - value at risk
    
    for i in range(1000):
        
        '''simulate 1K of autocorrelated quarterly data start with the last observed autocorrelaton data'''
        sim_spl_i =  joint_dist.rvs(len(quarters), random_state = i)
        
        stock_bond_sim_df = pd.DataFrame(sim_spl_i)                
        stock_bond_sim_df.index = quarters
        stock_bond_sim_df.columns = ['Stock', 'Income','Dividend','Inflation','Bond']
        
        stock_bond_sim_df['Income'] = getSim(np.log(quarter_all['Bond']), 
                                             income_fit, stock_bond_sim_df.Income, type = "log")
        
        stock_bond_sim_df['Dividend'] = getSim(np.log(quarter_all['Dividend']), 
                                             dividend_fit, stock_bond_sim_df.Dividend, type = "log")
        
        stock_bond_sim_df['Inflation'] = getSim(quarter_all['Inflation'], 
                                            inflation_fit, stock_bond_sim_df.Inflation, type = "non_log")
        
        balance[i], withdrawal[i] = method_quarterly(org_amt,rmd_quarter,start_age, end_age, 
                                             stock_bond_sim_df, (1- bond_weight), bond_weight, rmd_col = "Adj_Rate")
        
    return balance, withdrawal

def method_quarterly(org_amt, rmd_quarter, start_age, end_age, stock_bond_sim_df, 
                       stock_weight, bond_weight, rmd_col):
    
    balance = [] 
    withdrawal = [] 
    org_left = org_amt

    rmd_sub = rmd_quarter.loc[(rmd_quarter.Age >= start_age) & (rmd_quarter.Age <= end_age)]
    rmd_sub.index = list(np.arange(start_age, end_age+0.1, 0.25))
    
    for i in np.arange(start_age, end_age+0.1, 0.25):
        
        if i == start_age:
            '''withdraw RMD % initial balance at the retirement moment''' 
            withdrawal_ = org_left * (rmd_sub[rmd_col][i] \
                           + stock_weight * stock_bond_sim_df.Dividend[i] \
                                 + bond_weight * stock_bond_sim_df.Income[i])
            
            withdrawal.append(withdrawal_)
            
            org_left =  (org_left - withdrawal_)                   
            balance.append(org_left)              

        else:            
            '''for the following quarters, the left balance will grow with total bond and total stock 
            returns before next quarterly withdrawal'''            
            grow_left =  org_left * (1 + bond_weight * stock_bond_sim_df.Bond[i] \
                                   + stock_weight * stock_bond_sim_df.Stock[i])

            withdrawal_ = grow_left * rmd_sub[rmd_col][i] \
                          + org_left * (stock_weight * stock_bond_sim_df.Dividend[i] \
                                        + bond_weight * stock_bond_sim_df.Income[i])
              
            if grow_left <= 0: 
                withdrawal.append(0)
                org_left = 0
                balance.append(0)                
            
            elif (withdrawal_ >= grow_left):
                withdrawal.append(grow_left)
                org_left = 0
                balance.append(0)
                
            elif (withdrawal_ < 0):
                withdrawal.append(0)
                org_left = grow_left
                balance.append(grow_left)   
                
            else:                
                org_left = grow_left - withdrawal_
                balance.append(org_left)
                withdrawal.append(withdrawal_)             

    return balance, withdrawal

def getSim(y, ar_model, error, type = "non_log"):
    '''get simulation of autocorrealtion data with simulated LR error term'''    
 
    sim_init = y[y.index.max()]
    sim_out = ar_model.intercept_ + ar_model.coef_[0] * sim_init + error   
    
    if type == "log":
        sim_out = np.exp(sim_out)
        return sim_out
    else:
        return sim_out  