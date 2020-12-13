#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# In[ ]:


def calculate_beta_restricted(cluster, restricted_regressors, outcome):

    restricted_regressors = sm.add_constant(restricted_regressors)

    rest_bs_model = sm.OLS(outcome,restricted_regressors)

    rest_bs_results = rest_bs_model.fit(cov_type='cluster', 
                                        cov_kwds={'groups': cluster}, 
                                        use_correction = True)

    beta_restr = np.array(rest_bs_results.params)
    
    return beta_restr


# In[ ]:


def calculate_w_star_wild(restricted_regressors, outcome, cluster, beta_restr, residual, treatment):
    data = pd.DataFrame(restricted_regressors)
    data['outcome'] = outcome
    data['cluster'] = cluster
    data['residual'] = residual
    cluster_list = list(np.unique(cluster))
    
    bootstrap_sample = pd.DataFrame(columns=data.columns)
    
    for i in range(0,len(cluster_list)):
        draw = np.random.randint(0,len(cluster_list))
        mask = data['cluster']==cluster_list[draw]
        bootstrap_sample =  bootstrap_sample.append(data[mask], ignore_index=True)
    
    restricted_regressors_bs = bootstrap_sample.drop(columns = ['outcome', 'cluster', 'residual'])

    restricted_regressors_bs = sm.add_constant(restricted_regressors_bs)
    ## this is to create the corrective addition to the residual, if the wild factor is 0, then no addition
    bootstrap_sample['neg_2_residual'] = (-2)*bootstrap_sample['residual'] 
    
    wild_factor = np.random.randint(0,2,size=len(bootstrap_sample))
    
    #print(bootstrap_sample)
    #print(wild_factor)
    
    bootstrap_sample['y_star'] = [np.matmul(np.array(restricted_regressors_bs.astype(float).iloc[i]), beta_restr) 
                                  + np.array(bootstrap_sample['residual'].astype(float).iloc[i])
                                  + np.array(bootstrap_sample['neg_2_residual'].astype(float).iloc[i]*wild_factor[i])
                                  for i in range(0, len(bootstrap_sample)) 
                                 ]

    unrestricted_regressors = restricted_regressors_bs.copy()
    
    unrestricted_regressors['treatment'] = treatment
        
    unrestricted_regressors = sm.add_constant(unrestricted_regressors)

    unrest_bs_model = sm.OLS(bootstrap_sample['y_star'].astype(float),
                             unrestricted_regressors.astype(float)
                            )

    unrest_bs_results = unrest_bs_model.fit(cov_type='cluster', 
                                            cov_kwds={'groups': bootstrap_sample['cluster']}, 
                                            use_correction = True)

    return unrest_bs_results.tvalues[list(unrestricted_regressors.columns).index('treatment')]


# In[ ]:


def calculate_w_star_list_wild(number_bootstrap,data):
    w_star_list_wild = list(np.repeat(np.nan,number_bootstrap))
    w_star_list_wild = pd.DataFrame(w_star_list_wild, columns=['w_star'])
    for i in range(0,len(w_star_list_wild)):
        #print(i)
        try:
            w_star_added = calculate_w_star_wild(data)
            #print(w_star_added)
        except:
            w_star_added = np.nan
        #print(w_star_added)
        w_star_list_wild.iloc[i] = w_star_added
    return w_star_list_wild.dropna()

