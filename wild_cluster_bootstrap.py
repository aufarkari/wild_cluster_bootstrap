#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:


def calculate_beta_restricted(data=regression_table_1):
    regression_table_b = data
    
    regression_table_b['Residual'] = list(results.resid)

    regression_table_b['Cluster + Period'] = (regression_table_b['Community Used'].map(str) 
                                              + regression_table_b['Period'].map(str))

    cluster_period = regression_table_b['Cluster + Period'].unique()

    ### Calculating unrestricted beta

    restricted_regressors = regression_table_b[rest_regressors_cols]

    restricted_regressors = sm.add_constant(restricted_regressors)

    rest_bs_model = sm.OLS(regression_table_b['Post-treatment Price Sharing'].astype(float),
                       restricted_regressors.astype(float)#,
                       #weights=bootstrap_sample['# of Total Users']
                      )

    rest_bs_results = rest_bs_model.fit(cov_type='cluster', 
                                        cov_kwds={'groups': regression_table_b['Community Used']}, 
                                    use_correction = True)

    beta_restr = np.array(rest_bs_results.params)
    
    return beta_restr


# In[ ]:


def calculate_w_star_wild(data = regression_table, beta_restr = beta_restr):
    bootstrap_sample = pd.DataFrame(columns=data.columns)
    regression_table_c = data
    
    for i in range(0,len(cluster_used)):
        draw = np.random.randint(0,len(cluster_used))
        mask = regression_table_c['Community Used']==cluster_used[draw]
        bootstrap_sample =  bootstrap_sample.append(regression_table_c[mask], ignore_index=True)
    
    restricted_regressors = bootstrap_sample[rest_regressors_cols]

    restricted_regressors = sm.add_constant(restricted_regressors)
    ## this is to create the corrective addition to the residual, if the wild factor is 0, then no addition
    bootstrap_sample['Neg 2 Residual'] = (-2)*bootstrap_sample['Residual'] 
    
    wild_factor = np.random.randint(0,2,size=len(bootstrap_sample))
    
    #print(bootstrap_sample)
    #print(wild_factor)
    
    bootstrap_sample['Y star'] = [np.matmul(np.array(restricted_regressors.astype(float).iloc[i]), beta_restr) 
                                  + np.array(bootstrap_sample['Residual'].astype(float).iloc[i])
                                  + np.array(bootstrap_sample['Neg 2 Residual'].astype(float).iloc[i]*wild_factor[i])
                                  for i in range(0, len(bootstrap_sample)) 
                                 ]

    unrestricted_regressors = bootstrap_sample[regressors_cols]
        
    unrestricted_regressors = sm.add_constant(unrestricted_regressors)

    unrest_bs_model = sm.OLS(bootstrap_sample['Y star'].astype(float),
                             unrestricted_regressors.astype(float)
                            )

    unrest_bs_results = unrest_bs_model.fit(cov_type='cluster', 
                                            cov_kwds={'groups': bootstrap_sample['Community Used']}, 
                                            use_correction = True)

    return unrest_bs_results.tvalues[list(unrestricted_regressors.columns).index('Treatment Dummy')]


# In[ ]:


def calculate_w_star_list_wild(number_bootstrap,data=regression_table_1):
    w_star_list_wild = list(np.repeat(np.nan,number_bootstrap))
    w_star_list_wild = pd.DataFrame(w_star_list_wild, columns=['W star'])
    for i in range(0,len(w_star_list_wild)):
        #print(i)
        try:
            w_star_added = calculate_w_star_wild(data=data)
            #print(w_star_added)
        except:
            w_star_added = np.nan
        #print(w_star_added)
        w_star_list_wild.iloc[i] = w_star_added
    return w_star_list_wild.dropna()

