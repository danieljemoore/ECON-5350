# Missouri University of Science & Technology
# Data Analytics - Case Studies (ECON5350)
# Lecturer: David Guo, PhD

import pickle
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from func import * 

from scipy import stats
from statsmodels.distributions.copula.api import CopulaDistribution, StudentTCopula 
pd.options.plotting.backend = "plotly"


st.set_page_config(page_title="Smart Retirement Plan", layout="wide")

# st.balloons()

st.title("Quarterly Withdrawal & Balance")

with st.sidebar:
    st.title("Retirement Options")
    st.markdown("- Inital Retirement Balance")
    st.markdown("- Start Age and End Age")
    st.markdown("- Investment Portfolio, e.g. 0.6 for Bond and 0.4 for Stock")
    st.markdown("- Adjustment on IRS Retirement Minimal distribution Rate.")
    
    st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)

    
    tot_bal = st.number_input('Input the Retirement Account Balance: ')
    #st.write('The initial balance of retirement account is ', tot_bal)

    bond_weight = st.number_input('Weight of Bond in Investment Portfolio: ')
    #st.write('The current number is ', stock_weight)

    adj_RMD = st.slider(
        "Retirement Minimal Distribution (RMD)", 0.2, 3.0, value = 1.0, step = 0.1, help="Apply a factor to default RMD"
    )  
    
    start, end = st.slider(
        "Range of Retirement Ages", 72, 115, (72, 100), step=1, help="Pick retirement age!"
    )
    
bal, wdrl = sim_1K_quarterly(tot_bal, start, end, bond_weight, adj_RMD)

bal_median_df = pd.DataFrame.from_dict(bal, orient = "index", columns = list(np.arange(start, end+0.1, 0.25))).median(axis =0).reset_index()
bal_median_df.columns = ["Quater_Year", "Balance"]

wdrl_median_df = pd.DataFrame.from_dict(wdrl, orient = "index", columns = list(np.arange(start, end+0.1, 0.25))).median(axis =0).reset_index()
wdrl_median_df.columns = ["Quater_Year", "Withdrawal"]


if not bal_median_df.empty:
    fig1 = px.line(bal_median_df, x="Quater_Year", y="Balance",
                 title= "Retirement Account Balance by Quarter")
    fig1.update_layout(plot_bgcolor="#f0f2f6")
    st.plotly_chart(fig1, use_container_width=True)
    
if not wdrl_median_df.empty:
    fig2 = px.line(wdrl_median_df, x="Quater_Year", y="Withdrawal",
                 title= "Quarterly Withdrawal")
    fig2.update_layout(plot_bgcolor="#262730")
    st.plotly_chart(fig2, use_container_width=True)    

else:
    st.error("No Chart to display!")
