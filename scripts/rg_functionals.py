#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:21:50 2024

@author: caryngeady
"""

# standard imports
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from numpy import interp
from scipy.stats import t


def logTransformData(df):
    
    """
    Log-transforms expression data.

    Parameters:
    - expressionData (DataFrame): The dataframe with genomics data (expressed in either FPKM or TPM).
    
    Returns:
    - out (DataFrame): The log-transformed genomics dataframe.
    """
    
    return np.log2(df.copy().astype('float') + 1)

def identifyProteinCodingGenes(expressionData):
    
    """
    Removes non-protein coding genes from the genomics dataframe.

    Parameters:
    - expressionData (DataFrame): The dataframe with genomics data.
    
    Returns:
    - out (DataFrame): The filtered genomics dataframe.
    """
    
    # load the features_gene file (effectively our dictionary/encoder)
    features_gene = pd.read_csv('data/features_gene.csv')
    
    # get the names of protein-coding genes 
    pc_genes = features_gene.gene_name[features_gene.gene_type == 'protein_coding']
    out = expressionData.loc[expressionData.index.isin(pc_genes),:].T
    
    # formatting
    # out.columns = out.iloc[0,:]
    # out = out.drop('gene_name',axis=0)
    
    return out

def identifyGenesByPrevalancy(expressionData,thresh=50,nanFlag=True):
    
    """
    Filters the columns of a dataframe based on the prevalancy of expression data.

    Parameters:
    - expressionData (DataFrame): The dataframe with genomics data.
    - thresh (float, optional): The percent prevalancy threshold - any gene that is not expressed in at least (thresh)% of patients is removed.
    - nanFlag (bool, optional): For FKPM reads, set NaN to 0.
    
    Returns:
    - out (DataFrame): The filtered genomics dataframe.
    """
    
    if nanFlag:
        # replace NaNs with 0 (FKPM reads NaN --> 0)
        expressionData[expressionData.isna()] = 0
    
    num_patients = expressionData.shape[0]

    gen_col_remove = (num_patients - (expressionData == 0).sum(axis=0)) / num_patients * 100

    # VISUALIZATION
    gen_plot = []

    for i in range(101):
        gen_plot.append((gen_col_remove>=i).sum())

    plt.plot(range(101),gen_plot,color='green') # ,label='PC genes'
    plt.axvline(x=thresh,linestyle='--',color='red')
    plt.ylabel('Protein-Coding Genes')
    plt.xlabel('% of Patients')
    # plt.legend()
    plt.show()
    
    # feature reduction by measurement saturation
    return expressionData[expressionData.columns[np.where(gen_col_remove>=thresh)[0]]]

def removeCorrelatedFeatures(df, thresh=0.1, keepVolume=False):
    """
    Filters the columns of a dataframe based on the correlation to other columns in the same dataframe.

    Parameters:
    - df (DataFrame): The dataframe.
    - thresh (float, optional): The correlation threshold. Columns with absolute correlation greater than volThresh will be dropped. Default is 0.1.
    - keepVolume (bool, optiona;) : Force-include volume as a feature (radiomics-specific).
    
    Returns:
    - out (DataFrame): The filtered radiomics dataframe.
    """
    cor = df.corr()
    cols_to_keep = cor[abs(cor) < thresh].index
    
    if keepVolume and not cols_to_keep.isin(['original_shape_VoxelVolume']).any():
        
        cols_to_keep.append('original_shape_VoxelVolume')
        

    return df[cols_to_keep]

def volumeFilter(radiomics, volThresh=0.1):
    """
    Filters the columns of a radiomics dataframe based on the correlation with the 'original_shape_VoxelVolume' column.

    Parameters:
    - radiomics (DataFrame): The radiomics dataframe.
    - volThresh (float, optional): The correlation threshold. Columns with absolute correlation greater than volThresh will be dropped. Default is 0.1.

    Returns:
    - radiomics (DataFrame): The filtered radiomics dataframe.
    """
    cor = radiomics.corr(method='spearman')['original_shape_VoxelVolume']
    cols_to_drop = cor[abs(cor) > volThresh].index

    if cols_to_drop.isin(['original_shape_VoxelVolume']).any():
        cols_to_drop = cols_to_drop.drop('original_shape_VoxelVolume')

    return radiomics.drop(cols_to_drop, axis=1)

def draw_cv_roc_curve(classifier, cv, X, y, title='ROC Curve'):
    """
    Draw a Cross Validated ROC Curve.
    Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response numpy array
    Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train,:], y[train]).predict_proba(X.iloc[test,:])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3)#,
                 # label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    auc_lower,auc_upper = calc_conf_intervals(aucs)
    plt.plot(mean_fpr, mean_tpr, 
             label=r'Mean ROC (AUC = %0.2f [conf. int. %0.2f, %0.2f])' % (mean_auc, auc_lower,auc_upper),
             lw=3, alpha=.8,color='k')

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightgrey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    # for ax in plt.gcf().axes:
    #     # ax.get_lines()[0].set_color("#CC79A7")
    plt.savefig('roc.png',transparent=True,dpi=500,bbox_inches='tight')
    plt.show()
    
    return mean_auc

def calc_conf_intervals(lst_item, confidence = 0.95):
    
    m = np.mean(lst_item)
    s = np.std(lst_item)
    dof = len(lst_item) - 1
    t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    
    return m-s*t_crit/np.sqrt(len(lst_item)), m+s*t_crit/np.sqrt(len(lst_item))

# import sklearn.metrics as metrics
# # calculate the fpr and tpr for all thresholds of the classification
# probs = model.predict_proba(X_test)
# preds = probs[:,1]
# fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
# roc_auc = metrics.auc(fpr, tpr)

# # method I: plt
# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()