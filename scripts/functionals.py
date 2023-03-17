#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:35:34 2023

@author: cg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.stats import pearsonr,spearmanr,pointbiserialr
from statsmodels.stats.multitest import multipletests as multitest

def importData():
    
    '''
    importData simply reads in the semi-processed data:
            radiomics : radiomics features from all contoured lesions
            mb_path   : pathway abundance
            mb_rela   : relative abundance
            mb_mapf   : shannon diversity/inverse simpson
            baseline  : baseline clinical information
            survival  : overall and progression-free survival (iRECIST and RECIST)
    '''
    
    radiomics = pd.read_csv('data/radiomics/rad-features.csv')
    mb_path = pd.read_excel('data/mu-biome/path-abundance.xlsx')
    mb_rela = pd.read_excel('data/mu-biome/rel-abundance.xlsx')
    mb_mapf = pd.read_excel('data/mu-biome/mapfile.xlsx')
    baseline = pd.read_csv('data/clinical/baseline.csv')
    survival = pd.read_csv('data/clinical/survival.csv')
    
    return radiomics, mb_path, mb_rela, mb_mapf, baseline, survival

def consolidateMB(mb_path,mb_rela,mb_mapf,thresh=40,incl_path=False):
    
    '''
    consolidateMB takes in all semi-processed µ-biome data and consolidates it 
    into a singular table
    
    The threshold of patients for which measurements were available was set at
    40%, but can be input as an additional parameter if other thresholds are
    to be explored. A plot to visualize the feature reduction will generate.
    
    Analysis did not include pathway abundance as it is not clear what these 
    pathways even mean.
    
    INPUTS:
            mb_path   : pathway abundance
            mb_rela   : relative abundance
            mb_mapf   : shannon diversity/inverse simpson
            thresh    : 
            
    OUTPUT:
            mb_all    : all µ-biome features with at least 40% measurement saturation
    
    '''

    # only screening data
    mpf_screen = mb_mapf[mb_mapf['sample_id'].str.contains("screen")][['sdi_r','insim_r']].reset_index()
    mpf_screen = mpf_screen.drop(['index'],1)

    cols_path = mb_path.columns[mb_path.columns.str.contains("screen")]
    cols_rela = mb_rela.columns[mb_rela.columns.str.contains("screen")]
    num_patients = len(cols_path)

    path_screen = mb_path[cols_path]
    rela_screen = mb_rela[cols_rela]

    path_col_remove = (num_patients - (path_screen==0).sum(axis=1)) / num_patients * 100
    rela_col_remove = (num_patients - (rela_screen==0).sum(axis=1)) / num_patients * 100

    # VISUALIZATION
    rela_plot = []
    path_plot = []

    for i in range(101):
        rela_plot.append((rela_col_remove>=i).sum())
        path_plot.append((path_col_remove>=i).sum())
    
    plt.plot(range(101),rela_plot,color='black',label='relative abundance')
    plt.plot(range(101),path_plot,color='green',label='path abundance')
    plt.axvline(x=40,linestyle='--',color='red')
    plt.ylabel('Number of Features')
    plt.xlabel('% of Patients')
    plt.legend()
    plt.show()

    # feature reduction by measurement saturation
    rela_features = rela_screen.iloc[np.where(rela_col_remove>=thresh)[0],:].transpose()
    path_features = path_screen.iloc[np.where(path_col_remove>=thresh)[0],:].transpose()

    rela_features.columns = ['Microbiome_RELA_'+mb_rela['Species'].iloc[col] for col in rela_features.columns]
    path_features.columns = ['Microbiome_PATH_'+mb_path['# Pathway'].iloc[col] for col in path_features.columns]

    rela = rela_features.dropna(axis=1)
    path = path_features.dropna(axis=1)
    
    if ~incl_path:
        frames = [mpf_screen.reset_index(),rela.reset_index()]
    else:
        frames = [mpf_screen.reset_index(),rela.reset_index(),path.reset_index()]
    
    result = pd.concat(frames,axis=1)
    result = result.rename(columns = {"sdi_r" : "Microbiome_sdi_r"})

    return result.drop(['index','insim_r'],axis=1)   # drop insim_r due to high correlation with sdi_r (pearson r = 0.935)

def consolidateBL(baseline):
    
    '''
        consolidateBL converts words in the baseline spreadsheet into 
        machine-friendly format, with:
            
            1 --> 'Female', 'White', 'Uterine Cancer', 'Cohort A', 
                  'SD' or 'PR', 'Yes' (toxicities)
    
        ...and 0 otherwise, for the respective categories of:
            - biological sex;
            - race;
            - disease;
            - trial arm;
            - response (RECIST v1);
            - Gr1Gr2/Gr3 toxicities
    '''
    
    bl = baseline.copy()
    bl['Gender'] = np.int64(bl['Gender'] == 'Female')
    bl['Race'] = np.int64(bl['Race'] == 'White')
    bl = bl.drop('Primary Diagnosis Disease Group',axis=1)
    bl['Diagnosis'] = np.int64(bl['Diagnosis'] == 'Uterine Cancer')
    bl['Assigned Treatment Arm'] = np.int64(bl['Assigned Treatment Arm'] == 'Cohort A')
    bl['Best Response RECISTv1'] = np.int64(np.logical_or((bl['Best Response RECISTv1'] == 'PR'),(bl['Best Response RECISTv1'] == 'SD')))
    bl['Gr1 Gr2 Toxicities'] = np.int64(bl['Gr1 Gr2 Toxicities'] == 'Yes')
    bl['Gr3 Toxicities'] = np.int64(bl['Gr3 Toxicities'] == 'Yes')
    
    return bl

def sumVolWeightedAvg(radiomics):


    test = radiomics.copy()
    test = test.drop('LESIONLOC',axis=1)
    features = test.columns
    shape_features = features[1:].str.contains('shape') # logical

    volume = np.array(test['original_shape_VoxelVolume'])
    mat = np.array(test[features[1:]])
    vol_weighted_features = pd.DataFrame((mat.T * volume).T,columns=features[1:])
    vol_weighted_features.insert(0,'USUBJID',test['USUBJID'])
    vol_weighted_features = vol_weighted_features.groupby('USUBJID').sum()
    
    summed_features = test.groupby('USUBJID').sum()
    vol_sum = np.array(summed_features['original_shape_VoxelVolume'])
    mat = np.array(vol_weighted_features)
    vol_weighted_features = pd.DataFrame((mat.T / vol_sum).T,columns=features[1:])
    
    frames = [summed_features.loc[:,shape_features].reset_index(drop=True),vol_weighted_features.loc[:,~shape_features].reset_index(drop=True)] 
    sum_vol_weighted = pd.concat(frames,axis=1)
    
    return vol_weighted_features

def radiomicsPCA(radiomics,n_components=10):


    num_components = range(1,31)
    explained_var = []
    x = StandardScaler().fit_transform(radiomics.iloc[:,1:])

    for num in num_components:
        pca = PCA(n_components = num)
        pca.fit(x)
        explained_var.append(sum(pca.explained_variance_ratio_))
    
    plt.scatter(num_components,explained_var)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Fraction of Explained Variance')
    plt.axhline(y=explained_var[9],linestyle='--',color='lightgrey')
    plt.axvline(x=10,linestyle='--',color='lightgrey')

    pca = PCA(n_components = n_components)
    out = pd.DataFrame(pca.fit_transform(x))
    out.columns = ['Component '+str(i) for i in range(1,n_components+1)]
    
    return out

def consolidateRF(radiomics,n_components=10):
    
    sum_vol_weighted_features = sumVolWeightedAvg(radiomics)
    principal_components = radiomicsPCA(sum_vol_weighted_features,n_components)
    
    return principal_components

def consolidateALL(bl,rf,mb):
    
    bl = bl.drop('Subject',axis=1)
    
    frames = [bl,rf,mb] 
    
    return pd.concat(frames,axis=1)

def correlateFeatures(all_features):
    
    num_features = all_features.shape[1]
    r_mat = np.empty([num_features,num_features])
    p_mat = r_mat.copy()

    for i in range(num_features):
        # is it categorical?
        check1 = categorical_check(all_features.iloc[:,i])
        
        for j in range(num_features):
            # is it categorical?
            check2 = categorical_check(all_features.iloc[:,j])
            
            if check1 & check2:
                dct = np.array(pd.crosstab(all_features.iloc[:,i],all_features.iloc[:,j]))
                r_mat[i,j],p_mat[i,j] = cramers_v(dct)
            elif check1 | check2:
                r_mat[i,j],p_mat[i,j] = pointbiserialr(all_features.iloc[:,i],all_features.iloc[:,j])
            else:
                r_mat[i,j],p_mat[i,j] = spearmanr(all_features.iloc[:,i],all_features.iloc[:,j])


    # consolidate into a dataframe and correct for multiple testing
    r_vals = np.tril(r_mat, k=-1)
    p_vals = np.tril(p_mat, k=-1)

    # eliminate duplicate entries
    row,col = np.where(r_vals!=0)

    corr_type = []
    feature1 = []
    feature2 = []
    r_table = []
    p_table = []

    for i in range(len(row)):
        f1 = all_features.columns[row[i]]
        f2 = all_features.columns[col[i]]
        type1 = getFeatureClass(all_features.columns[row[i]])
        type2 = getFeatureClass(all_features.columns[col[i]])
        corr_type.append(type1+'-'+type2)
        feature1.append(f1)
        feature2.append(f2)
        r_table.append(r_vals[row[i],col[i]])
        p_table.append(p_vals[row[i],col[i]])
     
    corrDict = {
                'Feature 1 Name'   : feature1,
                'Feature 2 Name'   : feature2,
                'Correlation Type' : corr_type,
                'r-Value'          : r_table,
                'p-Value' : p_table}

    corr_df = pd.DataFrame(corrDict)
    corr_df['Adjusted p-Value'] = multitest(corr_df['p-Value'],method='fdr_bh')[1]
    
    return corr_df

def cramers_v(data):
    #Chi-squared test statistic, sample size, and minimum of rows and columns
    X2 = stats.chi2_contingency(data, correction=False)[0]
    p = stats.chi2_contingency(data, correction=False)[1]
    n = np.sum(data)
    minDim = min(data.shape)-1

    #calculate Cramer's V 
    V = np.sqrt((X2/n) / minDim)
    
    return V,p

def categorical_check(arr):
    num_vals = len(np.unique(arr))
    check = num_vals == 2
    
    return check

def var_filter(df,thresh):
    
    var = df.var()
    cols = df.columns
    reduced_cols = cols[np.where(var>=thresh)]  

    return reduced_cols,df[reduced_cols]

def getFeatureClass(feat):
    # series of if's
    if 'Microbiome' in feat:
        c = 'MICROBIOME'
    elif 'Component' in feat:
        c = 'IMAGE'
    else:
        c = 'CLINICAL'
        
    return c