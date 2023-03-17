#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:13:07 2023

@author: cg
"""

# change working directory to be whatever directory this script is in
import os
os.chdir(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt


# custom methods
import scripts.functionals as f
  

# %% PRELIMINARIES

# load data
radiomics, mb_path, mb_rela, mb_mapf, baseline, survival = f.importData()

# clean and organize data
baseline_clean = f.consolidateBL(baseline)
radiomics_clean = f.consolidateRF(radiomics,10)
mb_clean = f.consolidateMB(mb_path,mb_rela,mb_mapf)
all_clean = f.consolidateALL(baseline_clean,radiomics_clean,mb_clean)

# tidy up
del(baseline,radiomics,mb_path,mb_rela,mb_mapf)
del(baseline_clean,radiomics_clean,mb_clean)


# %% CORRELATION
corr_results = f.correlateFeatures(all_clean)
corr_results.to_csv('correlation_results_XXYYZZZZ.csv')




# %%
ucorr,ucorr_counts = np.unique(corr_type,return_counts=True)


import pandas as pd
import numpy as np
import networkx as nx

ints = ucorr_counts
a = ['IMAGE', 'IMAGE', 'IMAGE', 'MAP', 'PATH', 'PATH', 'PATH', 'RELA']
b = ['CLINICAL', 'IMAGE', 'PATH', 'PATH', 'CLINICAL', 'MAP', 'PATH', 'CLINICAL']
df = pd.DataFrame(ints, columns=['weight'])
df['a'] = a
df['b'] = b
df

g=nx.from_pandas_edgelist(df, 'a', 'b', ['weight'])
# g['E']['C']['weight']
color_map = ['purple','green','blue','blue','blue']

    
# g['E']['C']['cost']
# nx.draw(g,with_labels=True,node_color=color_map)

pos=nx.spring_layout(g) # pos = nx.nx_agraph.graphviz_layout(G)
nx.draw_networkx(g,pos,node_color=color_map)
labels = nx.get_edge_attributes(g,'weight')

nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
plt.show()
# for r in row:
#     print(getFeatureClass(all_features.columns[r]))

# %%
from pyvis.network import Network
import pandas as pd

got_net = Network(height='500px', width='100%', bgcolor='#FFFFFF', font_color='black')

# set the physics layout of the network
got_net.barnes_hut()
got_data = pd.read_csv('/Users/EL-CAPITAN-2016/OneDrive - University of Toronto/Caryn PhD/DAPPER/corr_summary5.csv')
got_data = pd.read_csv('/Users/EL-CAPITAN-2016/OneDrive - University of Toronto/Caryn PhD/SARC/Sarcoma-Radiomics-2--Multiple-Lesions/corr_adjust.csv')

sources = got_data['Source']
targets = got_data['Target']
weights = got_data['Weight']
node1_color = got_data['Node 1 Color']
node2_color = got_data['Node 2 Color']
edge_color = got_data['Edge Color']

edge_data = zip(sources, targets, weights, node1_color, node2_color, edge_color)

for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]
    n1 = e[3]
    n2 = e[4]
    edg = e[5]
    
    

    got_net.add_node(src, src, title=src,fontsize=100, color=n1)
    got_net.add_node(dst, dst, title=dst,fontsize=100, color=n2)
    got_net.add_edge(src, dst, value=w,fontsize=100,color=edg)

neighbor_map = got_net.get_adj_list()

# add neighbor data to node hover data
for node in got_net.nodes:
    node['title'] += ' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
    node['value'] = len(neighbor_map[node['id']])
got_net.show_buttons(filter_=['physics'])
# got_net.show('/Users/EL-CAPITAN-2016/OneDrive - University of Toronto/Caryn PhD/DAPPER/corr_summary.html')
got_net.show('/Users/EL-CAPITAN-2016/OneDrive - University of Toronto/Caryn PhD/SARC/Sarcoma-Radiomics-2--Multiple-Lesions/corr_summary2.html')


# %% TESTING CORRELATIONS

frames = [principalComponents.reset_index(),radiomics.reset_index()]
result = pd.concat(frames,axis=1)
# df = result.drop(['ID'],axis=1)
# mb_all = result.drop(['index'],axis=1)   # 81 x 116 matrix of all features
# mb_all = mb_all.drop(['level_0'],axis=1)   # 81 x 116 matrix of all features
test = result.drop(['index','USUBJID'],axis=1)

corr_mat = test.corr()

# corr_mat.to_csv('connections.csv')


