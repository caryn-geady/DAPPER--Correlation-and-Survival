## **DAPPER Correlation and Survival** ##
*General Description*

Correlative and univariate survival analyses for the LMS cohort on the DAPPER Phase II clinical trial. Features of interest included:
* image-derived/radiomic;
* microbiome/metagenomic;
* clinical data

Running the correlation script (`correlation.py`) generates and saves a spreadsheet of correlations between radiomic features, microbiome features/metagenomics and clinical data. Correlations can then be sorted by strength of association, significance or correlation type (e.g., correlation between imaging feature and microbiome feature versus correlation between microbiome feature and clinical feature). A spreadsheet containing the reduced feature set for survival analysis is also generated and saved.

Running the survival notebook (`survival.Rmd`) generates two dataframes containing results from univariate analysis of radiomic, microbiome and clinical features for Overall Survival (OS) and Progression-Free Survival (PFS). Results can then be sorted by strength of association or significance.
