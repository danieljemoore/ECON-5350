# Missouri University of Science & Technology
# Data Analytics - Case Studies (ECON5350)
# Lecturer: David Guo, PhD

import pickle #Parquet
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from func import * 

import scipy
from scipy import stats
from statsmodels.distributions.copula.api import CopulaDistribution, StudentTCopula 
pd.options.plotting.backend = "plotly"


st.set_page_config(page_title="Smart Retirement Plan", layout="wide")

st.snow()

st.title("Quarterly Balance & Withdrawal")

with st.sidebar:
    st.title("Retirement Options")
    st.markdown("## Inital Retirement Balance")
    tot_bal = st.number_input('Input the Retirement Account Balance: ')
    st.write(f'The initial balance of retirement account is ${tot_bal}')
    
    st.markdown("## Start Age and End Age")
    start, end = st.slider(
        "Range of Retirement Ages", 72, 115, (72, 100), step=1, help="Pick retirement age!"
    )
    
    st.markdown("## Investment Portfolio")
    bond_weight = st.number_input('Weight of Bond in Investment Portfolio: ', help="e.g. 0.6 for Bond and 0.4 for Stock")
    st.write('Weight of Stocks in Investment Portfolio ', (1 - bond_weight))
    
    st.markdown("## Retirement Minimal Distribution(RMD) Rate")
    adj_RMD = st.slider(
        "Adjustment on IRS RMD Rate", 0.2, 3.0, value = 1.0, step = 0.1, help="Apply a factor to default RMD"
    )  

    st.markdown('## Confidence Intervals')
    top = st.number_input("Top CI Band (%)", 0.500, .990, value = 0.95, step = 0.005, help="Apply a %")
    st.write(f'The current upper CI is {top*100}%')

    bttm = st.number_input("Bottom CI Band (%)", 0.005, .490, value = 0.025, step = 0.005, help="Apply a %")
    st.write(f'The current lower CI is {bttm*100}%') 

            
    st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:20px;
}
</style>
''', unsafe_allow_html=True)

with st.spinner('Loading Retirement Analysis...'):     
    bal, wdrl = sim_1K_quarterly(tot_bal, start, end, bond_weight, adj_RMD)
    pickle.dump((bal, wdrl, start, end), open("bal_wdrl.pkl", "wb"))
    bal_median_df = pd.DataFrame.from_dict(bal, orient = "index", columns = list(np.arange(start, end+0.1, 0.25))).median(axis =0).reset_index()
    bal_median_df.columns = ["Quater_Year", "Balance"]

    wdrl_median_df = pd.DataFrame.from_dict(wdrl, orient = "index", columns = list(np.arange(start, end+0.1, 0.25))).median(axis =0).reset_index()
    wdrl_median_df.columns = ["Quater_Year", "Withdrawal"]
    #new code
    bal_df = pd.DataFrame.from_dict(bal, orient = "index", columns = list(np.arange(start, end+0.1, 0.25)))
    bal_percentiles = bal_df.describe([bttm, 0.5, top])

    wdrl_df = pd.DataFrame.from_dict(wdrl, orient = "index", columns = list(np.arange(start, end+0.1, 0.25)))
    wdrl_percentiles = wdrl_df.describe([bttm, 0.5, top])
    #
    

    st.write(f'The ending balance of retirement account is ${bal_percentiles.loc["mean", end]:.2f}')
  
    if len(str(bttm)) == 5:
        bttm_per = (f'{bttm:.1%}')
    else:
        bttm_per = (f'{bttm:.0%}')
    
    if len(str(top)) == 5:
        top_per = (f'{top:.1%}')
    else:
        top_per = (f'{top:.0%}')

    if not bal_df.empty:
        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(x=bal_percentiles.columns, y= bal_percentiles.loc['50%', :],
                        mode = 'lines', name = "Median of Balance", marker_color='rgb(0,128,0)', ))
        fig3.add_traces([go.Scatter(x=bal_percentiles.columns, y=bal_percentiles.loc[bttm_per, :],
                                mode = 'lines', line_color = 'rgba(255,128,128)', showlegend = False, 
                                fill='tonexty', fillcolor = 'rgb(131, 90, 241)'),
                        go.Scatter(x=bal_percentiles.columns, y=bal_percentiles.loc[top_per, :],
                                mode = 'lines', line_color = 'rgb(255,128,128)', 
                                name = f'{top_per} confidence interval',
                                fill='tonexty', fillcolor = 'rgba(255,255,255)')])

        fig3.update_layout(legend=dict(
            yanchor="top",
            y=0.01,
            xanchor="left",
            x=0.01
        ))


        fig3.update_layout(plot_bgcolor="#f0f2f6", title='Retirement Account Balance by Quarter with Confidence Intervals')
        st.plotly_chart(fig3, use_container_width=True) 

    if not wdrl_df.empty:
        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(x=wdrl_percentiles.columns, y= wdrl_percentiles.loc['50%', :],
                        mode = 'lines', name = "Median of Balance", marker_color='rgb(0,128,0)'))
        fig4.add_traces([go.Scatter(x=wdrl_percentiles.columns, y=wdrl_percentiles.loc[bttm_per, :],
                                mode = 'lines', line_color = 'rgba(255,128,128)', showlegend = False, 
                                fill='tonexty', fillcolor = 'rgb(131, 90, 241)'),
                        go.Scatter(x=wdrl_percentiles.columns, y=wdrl_percentiles.loc[top_per, :],
                                mode = 'lines', line_color = 'rgb(255,128,128)', 
                                name = f'{top_per} confidence interval',
                                fill='tonexty', fillcolor = 'rgba(255,255,255)')])

        fig4.update_layout(legend=dict(
            yanchor="top",
            y=0.01,
            xanchor="left",
            x=0.01
        ))


        fig4.update_layout(plot_bgcolor="#f0f2f6", title='Quarterly Withdrawal with Confidence Intervals')
        st.plotly_chart(fig4, use_container_width=True) 

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
st.success('Done!')