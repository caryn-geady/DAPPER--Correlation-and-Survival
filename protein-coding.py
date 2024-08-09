#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:35:50 2024

@author: caryngeady
"""

# change working directory to be whatever directory this script is in
import os
os.chdir(os.path.dirname(__file__))

# standard imports
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import scripts.rg_functionals as f
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from datetime import datetime
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


# %% LOAD THE DATA

pathName = 'NSCLC' # options ['NSCLC','INSPIRE']
ID_dict = {'NSCLC'   : 'Case ID',
           'INSPIRE' : 'Sample ID'}

genomics = pd.read_csv('data/' + pathName + '/genomics.csv')
genomics.index = genomics.gene_name
genomics.drop(labels='gene_name',axis=1,inplace=True)
radiomics = pd.read_csv('data/' + pathName + '/radiomics.csv')
clinical = pd.read_csv('data/' + pathName + '/clinical.csv')
clinical = clinical[clinical[ID_dict[pathName]].isin(genomics.columns)]

# %% GENOMICS DATA PROCESSING

# convert FPKM to TPM
FKPM_j = genomics.sum(axis=None,numeric_only=True)
genomics_TPM = genomics.div(FKPM_j) * 1e6


# %%
genomics_pc = f.identifyProteinCodingGenes(genomics_TPM)                 # use only protein-coding genes
genomics_sat = f.identifyGenesByPrevalancy(genomics_pc,50,True)          # remove genes not expressed by at least 50% of samples
genomics_norm = f.logTransformData(genomics_sat)                         # log-transform the data
genomics_imp2 = KNNImputer(n_neighbors=15).fit_transform(genomics_norm)  # imput missing values using 15-NN

df = pd.DataFrame(genomics_imp2)
df.index,df.columns = genomics_norm.index,genomics_norm.columns


# %% REMOVE LOW-VARIANCE GENE EXPRESSION

# visualize the variance in the data
plt.hist(df.var(skipna=True),bins=1500,color='green')
plt.xlim(0,10)
plt.ylabel('Number of Genes')
plt.xlabel('Variance in Gene Expression')
plt.show()

# %%

numComponents = 2

clinical = clinical.sort_values(by=ID_dict[pathName]).reset_index(drop=True)

varThresh = np.median(df.var(skipna=True)) # update / user-input

    
genomics_var = df[df.columns[np.where(df.var(skipna=True)>varThresh)[0]]]
df_choice = genomics_var
df_choice = df_choice.sort_index(ignore_index=True)

# pc = PCA(n_components=numComponents).fit_transform(df_choice)
pc = PCA(n_components=numComponents).fit_transform(StandardScaler().fit_transform(df_choice))

stanford_inds = np.where(pc[:,0]>0)[0]
va_inds = np.where(pc[:,0]<0)[0]

# stanford_inds = np.where(clinical['EGFR mutation status']=='Wildtype')[0]
# va_inds = np.where(clinical['EGFR mutation status']!='Wildtype')[0]

plt.scatter(pc[stanford_inds,0],pc[stanford_inds,1],color='green',label='Stanford')
plt.scatter(pc[va_inds,0],pc[va_inds,1],color='blue',label='VA')
# plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.title('Variation Threshold: {:.2f}'.format(varThresh))
plt.savefig(pathName+'_cluster.png',dpi=500,bbox_inches='tight')
plt.show()



# %%

numComponents = 2

clinical = clinical.sort_values(by=ID_dict[pathName]).reset_index(drop=True)

varThresh = 5 # update / user-input
numGenes = []

for i in np.linspace(0,5,50):
    
    genomics_var = df[df.columns[np.where(df.var(skipna=True)>i)[0]]]
    df_choice = genomics_var
    df_choice = df_choice.sort_index(ignore_index=True)
    numGenes.append(len(df_choice.columns))
    
    # pc = PCA(n_components=numComponents).fit_transform(df_choice)
    pc = PCA(n_components=numComponents).fit_transform(StandardScaler().fit_transform(df_choice))
    
    stanford_inds = np.where(clinical['Patient affiliation']=='Stanford')[0]
    va_inds = np.where(clinical['Patient affiliation']=='VA')[0]
    
    # stanford_inds = np.where(clinical['EGFR mutation status']=='Wildtype')[0]
    # va_inds = np.where(clinical['EGFR mutation status']!='Wildtype')[0]
    
    plt.scatter(pc[stanford_inds,0],pc[stanford_inds,1],color='green',label='Stanford')
    plt.scatter(pc[va_inds,0],pc[va_inds,1],color='blue',label='VA')
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Variation Threshold: {:.1f}'.format(i))
    plt.show()
    
# %% VISUALIZE

plt.plot(np.linspace(0,5,50),numGenes)
plt.axvline(x=2.2,linestyle='--',color='black')
plt.xlabel('Expression Variance Threshold')
plt.ylabel('Number of Genes')
    
# %% TRY

numComponents = 2
col = 'Patient affiliation'

clinical = clinical.sort_values(by='Case ID').reset_index(drop=True)
colors = ['green','blue','purple','orange','red','pink','lightgray']

for i in np.linspace(0,2,50):
    
    genomics_var = df[df.columns[np.where(df.var(skipna=True)>i)[0]]]
    df_choice = genomics_var
    df_choice = df_choice.sort_index(ignore_index=True)
    
    # pc = PCA(n_components=numComponents).fit_transform(df_choice)
    pc = PCA(n_components=numComponents).fit_transform(StandardScaler().fit_transform(df_choice))
    
    cats = np.unique(clinical[col])
    
    for j in range(len(cats)):
        
        inds = np.where(clinical[col]==cats[j])[0]
        plt.scatter(pc[inds,0],pc[inds,1],label=cats[j],color=colors[j])
    
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Variation Threshold: {:.1f}'.format(i))
    plt.show()

# %% TRY WITH DIFFERENT NUMBERS OF PCS

# varThresh = 0.5
# gvar = df.var(skipna=True)
# gvar_col = df.var(axis=1,skipna=True)
# genomics_var = df[df.columns[np.where(df.var(skipna=True)>varThresh)[0]]]

# numComponents = 2
# df_choice = genomics_var

# clinical = clinical.sort_values(by='Case ID')#.reset_index()
# df_choice = df_choice.sort_index(ignore_index=True)

# pc = PCA(n_components=numComponents).fit_transform(StandardScaler().fit_transform(df_choice))
# pc = PCA(n_components=numComponents).fit_transform(df_choice)

# stanford_inds = np.where(clinical['Patient affiliation']=='Stanford')[0]
# va_inds = np.where(clinical['Patient affiliation']=='VA')[0]

# for i in range(numComponents):
#     for j in range(numComponents):
#         if i == j:
#             continue
#         plt.scatter(pc[stanford_inds,i],pc[stanford_inds,j],color='green',label='Stanford')
#         plt.scatter(pc[va_inds,i],pc[va_inds,j],color='blue',label='VA')
#         plt.legend()
#         plt.xlabel('PC'+str(i+1))
#         plt.ylabel('PC'+str(j+1))
#         plt.show()

# %% K-MEDOIDS APPROACH FOR GENES

distance_thresh = 0.5
var_thresh = np.median(df.var(skipna=True))

var = df.var()
cols = df.columns
reduced_cols = cols[np.where(var>=var_thresh)]  

# obtain the linkages array
corr = df[reduced_cols].corr()  # we can consider this as affinity matrix
distances = 1 - corr.abs().values  # pairwise distnces

distArray = ssd.squareform(distances)  # scipy converts matrix to 1d array
# print(max(distArray))
hier = hierarchy.linkage(distArray, method='average')  
hier[hier<0] = 0

fig = plt.gcf()
fig.set_size_inches(18.5, 6)
fig, ax = plt.subplots(figsize=(12, 6))

hierarchy.dendrogram(hier, truncate_mode="level", p=30, color_threshold=distance_thresh,
                                no_labels=True,above_threshold_color='k')
plt.axhline(y=1.5, color='r', linestyle='--')
plt.ylabel('distance',fontsize=20)
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
   	label.set_fontsize(20)
plt.axhline(y=distance_thresh, color='r',linestyle='--')
plt.style.context('light_background')
plt.show()

cluster_labels = hierarchy.fcluster(hier, distance_thresh, criterion="distance")
num = len(np.unique(cluster_labels))
    
print('Number of clusters: {}'.format(num))
print('Distance threshold: {}'.format(distance_thresh))


num = 2
kmeds = KMedoids(n_clusters=num,init='k-medoids++',max_iter=300,random_state=0)  # method='pam'
kmeds.fit(corr)

centers = kmeds.cluster_centers_
feature_inds = np.where(centers==1)[1]
cols_cluster = reduced_cols[feature_inds]

genomics_cluster = df[cols_cluster]

# %%

scores = []
maxnum = len(corr)

for i in range(2,maxnum):
    kmeds = KMedoids(n_clusters=i,init='k-medoids++',max_iter=300,random_state=0)  # method='pam'
    scores.append(silhouette_score(corr, kmeds.fit_predict(corr)))

# %%
plt.plot(range(2,len(scores)+2),scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

# %%

kmeds = KMedoids(n_clusters=79,init='k-medoids++',max_iter=300,random_state=0)  # method='pam'
kmeds.fit(corr)

centers = kmeds.cluster_centers_
feature_inds = np.where(centers==1)[1]
cols_cluster = reduced_cols[feature_inds]

genomics_cluster = df[cols_cluster]


# %%

from dateutil import parser

def convert_to_datetime(input_str, parserinfo=None):
    """
    Convert a string representation of a date and time to a datetime object.

    Parameters:
    - input_str (str): The string representation of the date and time.
    - parserinfo (parserinfo, optional): The parserinfo object to use for parsing the input string. Defaults to None.

    Returns:
    - datetime: The datetime object representing the input string.

    """
    return parser.parse(input_str, parserinfo=parserinfo)

# print(int(convert_to_datetime(clinical['Date of Last Known Alive'][2])-convert_to_datetime(clinical['CT Date'][2]))-clinical['Days between CT and surgery'][2])

ostime = [(convert_to_datetime(clinical['Date of Last Known Alive'][i])-convert_to_datetime(clinical['CT Date'][i])).days - clinical['Days between CT and surgery'][i] for i in range(len(clinical))]

clinical['OSTIME'] = ostime
clinical['OSEVENT'] = clinical['Survival Status'] == 'Dead'

print((convert_to_datetime(clinical['Date of Last Known Alive'][2])-convert_to_datetime(clinical['CT Date'][2])).days - clinical['Days between CT and surgery'][2])

# %% SURVIVAL
from lifelines.plotting import add_at_risk_counts
import matplotlib

matplotlib.rcParams.update({'font.size': 16,'font.sans-serif': 'Geneva'})

geneName = 'PDE4DIP'    # options ['PDE4DIP','FAM172A']    # x==OS and v==PFS

T = 'OSTIME'
E = 'OSEVENT'

ix = (genomics_cluster[geneName] > np.median(genomics_cluster[geneName])).values
T_R, E_R = clinical.loc[ix, T], clinical.loc[ix, E]
T_NR, E_NR = clinical.loc[~ix, E], clinical.loc[~ix, E]
label_R = '> Median'
label_NR = '<= Median'

# test PD versus non-PD
# ix = df2['FirstAssess'] != 'PD'
# T_R, E_R = df2.loc[ix, T], df2.loc[ix, E]
# T_NR, E_NR = df2.loc[~ix, E], df2.loc[~ix, E]
# label_R = 'non-PD'
# label_NR = 'PD'

ax = plt.subplot(111)
# ax.figsize = (6,8)

kmf_R = KaplanMeierFitter()
ax = kmf_R.fit(clinical.loc[ix][T]/365, clinical.loc[ix][E], label=label_R).plot_survival_function(ax=ax,color='green')

kmf_NR = KaplanMeierFitter()
ax = kmf_NR.fit(clinical.loc[~ix][T]/365, clinical.loc[~ix][E], label=label_NR).plot_survival_function(ax=ax,color='blue')

plt.legend(loc=(1.04, 0))
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')


from lifelines.plotting import add_at_risk_counts
add_at_risk_counts(kmf_R, kmf_NR, ax=ax)
# plt.tight_layout()
plt.savefig('KMcurve_'+geneName+'.png',dpi=500,bbox_inches="tight")

results = logrank_test(T_R, T_NR, event_observed_A=E_R, event_observed_B=E_NR)
results.print_summary()


# print 
print('Log rank p-value: ',results.p_value)   

# %% RADIOMICS PROCESSING

# feature reduction
startColInd = np.where(radiomics.columns.str.find('original')==0)[0][0]
radiomics.index = radiomics.patient_ID
radiomics = radiomics.dropna(axis=1)
features = radiomics.copy().iloc[:,startColInd:]
var = features.var()
cols_to_keep = features.columns[np.where(var>=10)]
features_varred = features[cols_to_keep]
if 'original_shape_VoxelVolume' not in cols_to_keep:
    features_varred.insert(0,'original_shape_VoxelVolume',features.original_shape_VoxelVolume)

cor = features_varred.corr(method='spearman')['original_shape_VoxelVolume']
cols_to_keep = cor[abs(cor) < 0.7].index
features_volcorr = features_varred[cols_to_keep]
if 'original_shape_VoxelVolume' not in cols_to_keep:
    features_volcorr.insert(0,'original_shape_VoxelVolume',features.original_shape_VoxelVolume)

df = features_volcorr

cor = df.corr()
m = ~(cor.mask(np.eye(len(cor), dtype=bool)).abs() > 0.7).any()
test = m.index[m.values]
cormat = features_volcorr[test].corr()

features_volcorr = features_volcorr[test]
if 'original_shape_VoxelVolume' not in features_volcorr.columns:
    features_volcorr.insert(0,'original_shape_VoxelVolume',features.original_shape_VoxelVolume)
    
# features_volcorr.iloc[:,:] = StandardScaler().fit_transform(features_volcorr.iloc[:,:])

# %%
pc_rad = PCA(n_components=2).fit_transform(features_volcorr)
pc_rad = pd.DataFrame(pc_rad,columns=['PC1','PC2'])
pc_rad.index = features_volcorr.index
plt.scatter(pc_rad.iloc[:,0],pc_rad.iloc[:,1],color='green')


# %% MERGE AND CORRELATE


df_merged = features_volcorr.merge(genomics_cluster,left_index=True, right_index=True, how='inner')
# df_merged = pc_rad.merge(genomics_cluster,left_index=True, right_index=True, how='inner')

cor = df_merged.corr()


# %%

radFeatures = df_merged.copy().iloc[:,:-2]
expression = df_merged.copy().iloc[:,-2:]


# %% CLASSIFICATION

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_classif, chi2
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from statsmodels.stats.multitest import multipletests as mtc
from scipy.stats import wilcoxon
from sklearn.neural_network import MLPClassifier

# %%
# preamble
geneChoice = 1
modelChoice = 'logistic'
maxFeatures = 6
forceVolume = True

print('--------------------')
print('Gene of Interest: {}'.format(expression.columns[geneChoice]))

outcome = expression.iloc[:,geneChoice] > np.median(expression.iloc[:,geneChoice])
train_inds, test_inds = train_test_split(range(len(outcome)),test_size=0.2,random_state=1,stratify=outcome)


models = {
            'logistic'   : [LogisticRegression(random_state=1),{
                                                                'penalty'  : ['l1', 'l2', 'elasticnet', None],
                                                                'solver'   : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                                                                'tol'      : [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
                                                                'max_iter' : [50,100,150,200]
                                                                }],
            'naivebayes' : [GaussianNB(),None],
            'kNN'        : [KNeighborsClassifier(),{
                                                                'n_neighbors' : [3,5,7,9],
                                                                'weights'     : ['uniform','distance']
                                                                }],
            'RF_classif' : [RandomForestClassifier(random_state=1),{
                                                                'n_estimators' : [50,100,150],
                                                                'max_features' : ["sqrt","log2",None],
                                                                'min_samples_split' : [5,10,15],
                                                                'min_samples_leaf' : [3,6,9]  
                                                                }],
            'MLP'        : [MLPClassifier(random_state=1),{
                                                                'activation' : ["identity", "logistic", "tanh", "relu"],
                                                                'solver' : ["lbfgs", "sgd", "adam"]#,
                                                                # 'C' : [0,0.25,0.5,0.75,1],
                                                                # 'kernel' : ["linear", "poly", "rbf", "sigmoid", "precomputed"]  
                                                                }]
         }

selected_features = []
auroc = []
auprc = []
neg_log_loss = []
mcc_lst = []
wilcoxp = []
wilcoxp.append(np.nan)
fdr = []




for i in range(6,maxFeatures+1):

    # features and outcomes
    predictors = radFeatures.iloc[train_inds,:].reset_index(drop=True)
    # remove any data points that may have missing values (sometimes core too small and nans live there instead of radiomic features)
    predInds = predictors[predictors.isnull().any(axis=1)].index
    targInds = predictors.index
    harmInds = [i for i in targInds if i not in predInds]
    # consolidate
    predictors = predictors.loc[harmInds,:]
    targets = outcome[train_inds][harmInds] 

    if forceVolume:                        # if volume is force-included in the model
        predictors.pop('original_shape_VoxelVolume')
        fs = SelectKBest(score_func=f_classif, k=i-1)
        mask = fs.fit(predictors, targets).get_support()
        predictors = predictors[predictors.columns[mask]]
        predictors['volume'] = radFeatures['original_shape_VoxelVolume'].iloc[train_inds].reset_index(drop=True)
    else:
        fs = SelectKBest(score_func=f_classif, k=i)
        mask = fs.fit(predictors, targets).get_support()
        predictors = predictors[predictors.columns[mask]]
        
    if i == 1:
        print('Progressor Fraction: %.2f' % (sum(targets==1)/len(targets)))
        print('Stable Fraction: %.2f' % (sum(targets==0)/len(targets)))
        print('Total Samples: %.0f' % (len(targets)))
        print('--------------------')    
        
    selected_features.append(predictors.columns)
    print('Features selected({}): {}'.format(len(predictors.columns),list(predictors.columns)))

    # modeling
    model = models[modelChoice][0]
    params = models[modelChoice][1]
    
    if modelChoice != 'naivebayes':
        gs = GridSearchCV(model, params, cv=5, scoring='matthews_corrcoef',n_jobs=1)
        gs.fit(predictors,targets)
        print('Best Params: ',gs.best_params_)
        model = gs.best_estimator_
    
    
    scaler = preprocessing.StandardScaler()    
    clf = make_pipeline(scaler, model)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)

   
    negLL = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='neg_log_loss', cv=cv, n_jobs=1)
    neg_log_loss.append(np.mean(negLL))
    
    auc = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='roc_auc_ovo', cv=cv, n_jobs=1)    
    # auc_lower,auc_upper = f.calc_conf_intervals(auc)
    # print('Average AUROC: {:.2f} (95% conf. int. [{:.2f},{:.2f}])'.format(np.mean(auc),auc_lower,auc_upper))
    auroc.append(np.mean(auc))
    
    aps = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='average_precision', cv=cv, n_jobs=1)    
    # aps_lower,aps_upper = f.calc_conf_intervals(aps)
    # print('Average Precision: {:.2f} (95% conf. int. [{:.2f},{:.2f}])'.format(np.mean(aps),aps_lower,aps_upper))
    auprc.append(np.mean(aps))
    
    mcc = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='matthews_corrcoef', cv=cv, n_jobs=1)    
    # mcc_lower,mcc_upper = f.calc_conf_intervals(mcc)
    # print('Average MCC: {:.2f} (95% conf. int. [{:.2f},{:.2f}])'.format(np.mean(mcc),mcc_lower,mcc_upper))
    mcc_lst.append(np.mean(mcc))
    
    f.draw_cv_roc_curve(clf, cv, predictors, targets.astype('int'), outcome.values, title='ROC Curve')
    
    # print('--------------------')
    # print('Significance Testing')
    # print('--------------------')
    
    # if i == 1:
    #     avg_precision = aps
    
    # if i > 1:
    #     print('Wilcoxon p-value: ',wilcoxon(avg_precision,aps)[1])
    #     wilcoxp.append(wilcoxon(avg_precision,aps)[1])
    
    # scores_precision, perm_scores_precision, pvalue_precision = permutation_test_score(
    #     clf, predictors.values, targets.astype('int'), scoring="matthews_corrcoef", cv=cv, n_permutations=1000
    # )
    # print('p-value: {:.3f}'.format(pvalue_precision))
    # print('FDR-corrected p-value: {:.3f}'.format(mtc(np.repeat(pvalue_precision,5))[1][0]))
    # fdr.append(mtc(np.repeat(pvalue_precision,5))[1][0])

    # print('--------------------')
    
results_df = pd.DataFrame([auroc,auprc,mcc_lst,neg_log_loss,wilcoxp,fdr]).T
results_df.columns = ['AUROC','AUPRC','MCC','NegLogLoss','Wilcoxon P-Value','FDR']
results_df.index = range(1,maxFeatures+1)
print(results_df)
