#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:13:07 2023

@author: cg
"""

# change working directory to be whatever directory this script is in
import os
os.chdir(os.path.dirname(__file__))

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

# consolidate correlations and corresponding signifiance into a dataframe
corr_results = f.correlateFeatures(all_clean)

# export to csv
corr_results.to_csv('data/results/correlation_results_16Mar2023.csv')
all_clean.to_csv('data/results/reduced-feature-set.csv')