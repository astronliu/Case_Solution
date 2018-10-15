import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

data=pd.read_csv("C:/Users/astro/Documents/Bazean_Case/Data/Forecasting_Data.csv")
#Clean up data further
#We will only look at currently active wells
#We don't know why a well was abandoned, and the total number of wells does not change too much (627 to 610) 
data=data.drop(data[data.status!='A'].index)
#Require for wells drilled before 2017 we require at least 12 months of data after the maximum production date
data["spud_year"]=pd.to_numeric(data["spud_year"] , errors='coerce')
data=data.drop(data[(data.Num_Entries<12)& (data.spud_year<2017)].index)

##################################################################################################
#Format data so it can be used to fit curves
##################################################################################################
#this will contain up to X months worth of data after the initial production date
X=24
Wells=data["Well_id"].unique()
prod=pd.DataFrame(np.nan,index=range(0,len(Wells)),columns=range(0,X+1))
#cumulative production at time t=0 is 0
#get data for the given well and sort so that max prod date is at top
for i in range(0,len(Wells)):
    temp0=data.drop(data[data.Well_id!=Wells[i]].index)
    a=max(temp0.Adj_index)
    temp0=temp0.drop(temp0[temp0['index']<a].index)
    temp0=temp0.sort_values(by=['index'])
    temp0=temp0.reset_index(drop=True)
    prod[0][i]=0
    for j in range(1,X+1):
        if j<len(temp0.index):
            z=temp0['volume_oil_formation_bbls'][j-1]
        else:
            z=np.nan
        prod[j][i]=z
#Drop wells where there are fewer than 6 months of data
prod=prod[np.isfinite(prod[6])]

##################################################################################################
#Build forecast for each well
##################################################################################################
#To build the forecast we will use a logistic growth model
#Will be using the model defined in Clark et. al "Production Forecasting with Logistic Growth Models" 2011 (SPE 144790)
#The euqation will be in the form Q(t)=Kt^n/(a+t^n)
#Where Q(t) is cumulative production at time t
#K, n and a are constants that will be determined using scipy.optimize

models=pd.DataFrame(np.nan,index=range(0,len(prod[0])),columns=['Well_id','K','a','n'])
def func(t,K,a,n):
	Q=K*t**n/(a+t**n)
	return Q
for i in range (0,len(prod.index.values)):
	#use an initial guess
	#from paper K is relatively aribtrary, but it was observed that K~10^5-10^6
	#a~10-100
	#n~0-1
	x0=[250000,25,1]
	ytemp0=prod.iloc[i].values
	ytemp1=ytemp0[~np.isnan(ytemp0)]
	#convert monthly production to cumulative
	ydat=np.zeros(len(ytemp1)-1)
	ydat[0]=ytemp1[1]
	for j in range(2,len(ytemp1)):
		ydat[j-1]=ydat[j-2]+ytemp1[j]
		xdat=np.array(list(range(1,len(ydat)+1)))
	try:
		popt,pcov =curve_fit(func,xdat,ydat,x0)
	except RuntimeError:
		popt=[np.nan,np.nan,np.nan]
	models['Well_id'][i]=Wells[prod.index.values[i]]
	models['K'][i]=popt[0]
	models['a'][i]=popt[1]
	models['n'][i]=popt[2]

##################################################################################################
#Compare predicted production to actual production
##################################################################################################
col=np.insert(np.arange(1,X+1,1).astype(str),0,'Well_id')
est=pd.DataFrame(np.nan,index=range(0,len(models['Well_id'])),columns=col)
act=pd.DataFrame(np.nan,index=range(0,len(models['Well_id'])),columns=col)
est['Well_id']=models['Well_id']
act['Well_id']=models['Well_id']
for i in range(0,len(est['Well_id'])):
	#actuals
	act_temp0=prod.iloc[i].values
	act_temp1=act_temp0[~np.isnan(act_temp0)]
	act[act.columns.values[1]][i]=act_temp1[1]
	for z in range(2, len(act_temp1)):
		act[act.columns.values[z]][i]=act[act.columns.values[z-1]][i]+act_temp1[z]
	#predictions
	temp2=models.iloc[i]
	k=temp2[1]
	a=temp2[2]
	n=temp2[3]
	for j in range (1,len(est.columns.values)):
		est[est.columns.values[j]][i]=(k*((j)**n))/(a+((j)**n))
graph_data=pd.DataFrame(np.nan,index=range(0,X+1),columns=['Well_id','Month','act','est'])
graph_data['Well_id']=np.repeat(est['Well_id'][0],X)
graph_data['Month']=range(1,X+1)
graph_data['act']=act.drop(columns=['Well_id']).iloc[0].values
graph_data['est']=est.drop(columns=['Well_id']).iloc[0].values
#rearrange data so it can be more easily used in tableau
for i in range(1,len(est['Well_id'])):
	graph_temp=pd.DataFrame(np.nan,index=range(0,X),columns=['Well_id','Month','act','est'])
	graph_temp['Well_id']=np.repeat(est['Well_id'][i],X)
	graph_temp['Month']=range(1,X+1)
	graph_temp['act']=act.drop(columns=['Well_id']).iloc[i].values
	graph_temp['est']=est.drop(columns=['Well_id']).iloc[i].values
	graph_data=graph_data.append(graph_temp,ignore_index=True)
#clean up final table by removing rows where there is no actual production
graph_data=graph_data[np.isfinite(graph_data['act'])]

##################################################################################################
#Estimates for production at 2,4,6,8,10 years
##################################################################################################
year_est=pd.DataFrame(np.nan,index=range(0,len(models['Well_id'])),columns=['Well_id','2_year','4_year','6_year','8_year','10_year'])
year_est['Well_id']=models['Well_id']

for i in range (1,6):
    for j in range(0,len(year_est['Well_id'])):
        temp2=models.iloc[j]
        k=temp2[1]
        a=temp2[2]
        n=temp2[3]
        year_est[year_est.columns.values[i]][j]=(k*((24*i)**n))/(a+((24*i)**n))

##################################################################################################
#Export results to csv 
##################################################################################################
#Could do the final comparisons in python, but easier and more insights can be drawn using Tableau
year_est.to_csv('C:/Users/astro/Documents/Bazean_Case/Data/yearly_prod_estimates.csv',index=False)
graph_data.to_csv('C:/Users/astro/Documents/Bazean_Case/Data/monthly_prod_estimates.csv',index=False)
models.to_csv('C:/Users/astro/Documents/Bazean_Case/Data/model_parameters.csv',index=False)