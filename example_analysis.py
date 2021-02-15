#%% importing modules
import statistics
import itertools
from datetime import datetime
import sklearn.linear_model

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

import analysis_func

#%% experiment parameters
########     ###    ########    ###        ######  ##       ########    ###    ##    ## #### ##    ##  ######
##     ##   ## ##      ##      ## ##      ##    ## ##       ##         ## ##   ###   ##  ##  ###   ## ##    ##
##     ##  ##   ##     ##     ##   ##     ##       ##       ##        ##   ##  ####  ##  ##  ####  ## ##
##     ## ##     ##    ##    ##     ##    ##       ##       ######   ##     ## ## ## ##  ##  ## ## ## ##   ####
##     ## #########    ##    #########    ##       ##       ##       ######### ##  ####  ##  ##  #### ##    ##
##     ## ##     ##    ##    ##     ##    ##    ## ##       ##       ##     ## ##   ###  ##  ##   ### ##    ##
########  ##     ##    ##    ##     ##     ######  ######## ######## ##     ## ##    ## #### ##    ##  ######
EXPT_NAME = 'exampleExperiment'
DATA_DATE = '2021.02.13'
TODAY_DATE = analysis_func.TODAY_DATE_STRING()

TARGET_SAMPLE_SIZE = 50

BLOCK_TYPE = ['A','B']
TRIAL_TYPE = ['T1', 'T2', 'T3', 'T4']
CONDITION_N = len(TRIAL_TYPE)
CONDITION_ITEM_N = 20
CONDITION_TRIAL_N = CONDITION_ITEM_N
BLOCK_TRIAL_N = CONDITION_N*CONDITION_TRIAL_N
BLOCK_N = len(BLOCK_TYPE)
TOTAL_TRIAL_N = BLOCK_TRIAL_N * BLOCK_N

SAVE_TABS = True
RAW_TRIAL_DATA_TAB_NAME = 'trial'
RAW_SUBJ_DATA_TAB_NAME = 'subj'
EXCLUSION_TAB_NAME = 'exclusions'
USED_TRIAL_TAB_NAME = 'usedTrial'
USED_A_RESPONSE_TAB_NAME = 'rawA'
USED_B_RESPONSE_TAB_NAME = 'rawB'
USED_ALL_RESPONSE_TAB_NAME = 'rawA&B'
USED_A_ZSCORE_TAB_NAME = 'zA'
USED_B_ZSCORE_TAB_NAME = 'zB'
USED_ALL_ZSCORE_TAB_NAME = 'zA&B'
SUBJ_COND_MEAN_TAB_NAME = 'M'
SUBJ_COND_SD_TAB_NAME = 'SD'
REGRESSION_RESULTS_TAB_NAME = 'regression'
RESIDUALS_TAB_NAME = 'residual'
RESIDUAL_MEAN_TAB_NAME = 'residualM'
RESIDUAL_SD_TAB_NAME = 'residualSD'
STATISTICS_TAB_NAME = 'stats'
SUMMARY_TAB_NAME = 'summary'
EXCEL_FILE_NAME = 'analysis_'+EXPT_NAME+'_'+TODAY_DATE+'.xlsx'

#%% import data
trial_file_name = 'Raw Data/trial_'+EXPT_NAME+'_'+DATA_DATE+'.txt'
subj_file_name = 'Raw Data/subj_'+EXPT_NAME+'_'+DATA_DATE+'.txt'
trial = pd.read_csv(trial_file_name, sep='\t', lineterminator='\n')
subj = pd.read_csv(subj_file_name, sep='\t', lineterminator='\n')

#%% delete the repeating labels
TRIAL_FIRST_LABEL = trial.columns[0]
SUBJ_FIRST_LABEL = subj.columns[0]
trial = trial.drop(trial[trial[TRIAL_FIRST_LABEL] == TRIAL_FIRST_LABEL].index, axis=0)
subj = subj.drop(subj[subj[SUBJ_FIRST_LABEL] == SUBJ_FIRST_LABEL].index, axis=0)

#%% clean up window's mess (remove "\r")
trial = analysis_func.CLEAN_CARRIAGE_RETURN(trial)
subj = analysis_func.CLEAN_CARRIAGE_RETURN(subj)

#%% save raw data
if SAVE_TABS:
    writer = pd.ExcelWriter(EXCEL_FILE_NAME, engine = 'openpyxl')
    trial.to_excel(writer, sheet_name=RAW_TRIAL_DATA_TAB_NAME, index=False)
    subj.to_excel(writer, sheet_name=RAW_SUBJ_DATA_TAB_NAME, index=False)

#%% print columns
print(trial.columns)
print(subj.columns)

#%% delete useless columns and rows
trial = trial.drop(['Column1', 'Column2', 'Column3'], axis=1)
SUBJ_USELESS = [
    'Column1',
    'Column2',
    'Column3']
subj = subj.drop(SUBJ_USELESS, axis=1)

#%% clean up the response
trial['response'] = trial['response'].str[1:]

#%% print columns
print(trial.columns)
print(subj.columns)

#%% convert data types
TRIAL_DATATYPE_DICT = {
    'num': int,
    'condition': str,
    'stimID': str,
    'response': float #,
    # ...
    }
trial = trial.astype(TRIAL_DATATYPE_DICT)

SUBJ_DATATYPE_DICT = {
    'num': int,
    'condition': str,
    'response': float,
    'gender': str,
    'age': int #,
    # ...
    }
subj = subj.astype(SUBJ_DATATYPE_DICT)

#%% sort rows
trial = trial.sort_values(by=['num'])
subj = subj.sort_values(by=['num'])

#%% exclusion criteria
######## ##     ##  ######  ##       ##     ##  ######  ####  #######  ##    ##  ######
##        ##   ##  ##    ## ##       ##     ## ##    ##  ##  ##     ## ###   ## ##    ##
##         ## ##   ##       ##       ##     ## ##        ##  ##     ## ####  ## ##
######      ###    ##       ##       ##     ##  ######   ##  ##     ## ## ## ##  ######
##         ## ##   ##       ##       ##     ##       ##  ##  ##     ## ##  ####       ##
##        ##   ##  ##    ## ##       ##     ## ##    ##  ##  ##     ## ##   ### ##    ##
######## ##     ##  ######  ########  #######   ######  ####  #######  ##    ##  ######
# offline exclusions
TESTING_LIST = [0] # data from experimenter testing
exclusion_list_dict = {}
exclusion_list_dict['techExclusion'] = [1]
exclusion_list_dict['nonsenseExclusion'] = [2]
exclusion_list_dict['notFollowExclusion'] = [3, 4]
exclusion_list_dict['regularRatingExclusion'] = [5]
QUALITATIVE_EXCLUSIONS_CRITERIA = list(exclusion_list_dict.keys())
INSTR_Q_ATTEMP_N_CRITERION = 2 # exclude subject with > 2 for either block
IN_VIEW_CRITERION = 'true' # exclude subject with viewport < [800, 600]
QUICK_READING_CRITERION = 0 # exclude any subject with > 0 page reading time < 0.5s (consider only max time between repeats)
SERIOUS_CRITERION = 1 # exclude subject with != 1
TRIAL_IN_VIEW_CRITERION = 'true' # exclude subject with any trial with stimuli not in view
REPEAT_RATING_CRITERION = 15  # exclude any subject with > 15 repeats in ratings
TAB_SWITCH_N_CRITERION = 3 # exclude any subject with > 3 tab switches
ABNORMAL_RT_N_CRITERION = 4 # exclude subj with > 4 in either block
RT_CRITERION = 120 # trial RT > 120s is considered as abnormal
EXPT_DURATION_CRITERION = 2 # exclude subject with expt duration > 2SD from M

#%% delete testing data
trial = trial.drop(trial[trial['num'].isin(TESTING_LIST)].index, axis=0)
subj = subj.drop(subj[subj['num'].isin(TESTING_LIST)].index, axis=0)

#%% adding debriefing responses to summary
DEBRIEFING_RESPONSES = [
    'num',
    'response',
    'gender',
    'age'
    ]
by_subj_summary = subj.loc[:, DEBRIEFING_RESPONSES].copy().set_index('num')

#%% check for experiment completion
SUBJ_NUM_SET = set(subj['num'])
TRIAL_SUBJ_NUM_SET = set(trial['num'])
INCOMPLETE_LIST = list(TRIAL_SUBJ_NUM_SET - SUBJ_NUM_SET)

trial = trial.drop(trial[trial['num'].isin(INCOMPLETE_LIST)].index, axis=0)

assert len(trial['num'].unique()) == len(subj['num'].unique())
assert len(trial['num'].unique()) == len(subj['num'])

#%% splitting condition names
splitted_names = trial['condition'].str.split('_')
trial['blockType'] = list(splitted_names.str[0])
trial['trialType'] = list(splitted_names.str[1])

#%% trial number checking
grouped_trial = trial.groupby(['num'])
trial_counts = grouped_trial['num'].count()
assert (trial_counts==TOTAL_TRIAL_N).all(axis=None)
by_subj_summary['trialN'] = list(trial_counts)

grouped_cond_trial = trial.groupby(['num', 'blockType', 'trialType'])
cond_trial_counts = grouped_cond_trial['num'].count().unstack(level=['blockType','trialType'])
assert (cond_trial_counts==BLOCK_TRIAL_N/CONDITION_N).all(axis=None)
cond_trial_counts.columns = [b+t+'TrialN' for b in BLOCK_TYPE for t in TRIAL_TYPE]
by_subj_summary = by_subj_summary.join(cond_trial_counts)

#%% stimulus number checking
cond_stim_counts = grouped_cond_trial['stimID'].unique().str.len().unstack(level=['blockType','trialType'])
assert (cond_stim_counts==CONDITION_ITEM_N).all(axis=None)
cond_stim_counts.columns = [b+t+'StimN' for b in BLOCK_TYPE for t in TRIAL_TYPE]
by_subj_summary = by_subj_summary.join(cond_stim_counts)

#%% subject exclusions from technical problems, non sense,
#   not following instructions, and regular ratings
for i in QUALITATIVE_EXCLUSIONS_CRITERIA:
    by_subj_summary[i] = False
    by_subj_summary[i].loc[exclusion_list_dict[i]] = True

#%% subject exclusions from too many instruction question attempts
subj = subj.set_index('num')
by_subj_summary['instrQExclusion'] = subj['instrQAttemptN'] > INSTR_Q_ATTEMP_N_CRITERION
exclusion_list_dict['instrQExclusion'] = list(by_subj_summary[by_subj_summary['instrQExclusion']].index)

#%% subject exclusions from small viewports
by_subj_summary['viewportExclusion'] = subj['inView'] != IN_VIEW_CRITERION
exclusion_list_dict['viewportExclusion'] = list(subj[by_subj_summary['viewportExclusion']].index)

#%% subject exclusions from reading instructions too fast
by_subj_summary['quickReadingExclusion'] = subj['quickReadingN'] > QUICK_READING_CRITERION
exclusion_list_dict['quickReadingExclusion'] = list(subj[by_subj_summary['quickReadingExclusion']].index)

#%% subject exclusions from not being serious
by_subj_summary['seriousExclusion'] = subj['serious'] != SERIOUS_CRITERION
exclusion_list_dict['seriousExclusion'] = list(subj[by_subj_summary['seriousExclusion']].index)

#%% subject exclusions from having any trial without stimuli fully in view
not_in_view_trial = trial[trial['inView'] != TRIAL_IN_VIEW_CRITERION]
exclusion_list_dict['stimNotInViewExclusion'] = list(not_in_view_trial['num'].unique())
by_subj_summary['stimNotInViewExclusion'] = False
for i in exclusion_list_dict['stimNotInViewExclusion']:
    by_subj_summary.loc[i,'stimNotInViewExclusion'] = True

#%% subject exclusions from repeating ratings
exclusion_list_dict['repeatRatingExclusion'] = []
for s in subj_list:
    for b in range(BLOCK_N):
        this_ratings = list(trial[(trial['num']==s) & (trial['blockNum']==b)]['response'])
        this_repeating = any([len(list(g))>REPEAT_RATING_CRITERION for k,g in itertools.groupby(this_ratings)])
        if this_repeating and s not in exclusion_list_dict['repeatRatingExclusion']:
            exclusion_list_dict['repeatRatingExclusion'].append(s)
by_subj_summary['repeatRatingExclusion'] = False
for i in exclusion_list_dict['repeatRatingExclusion']:
    by_subj_summary.loc[i,'repeatRatingExclusion'] = True

#%% subject exclusions from many tab switches
by_subj_summary['hiddenExclusion'] = subj['hiddenCount'] > TAB_SWITCH_N_CRITERION
exclusion_list_dict['hiddenExclusion'] = list(subj[by_subj_summary['hiddenExclusion']].index)

#%% abnormal RT counts
by_subj_summary['abnormalRTN'] = 0
long_rts = trial[trial['rt'] > RT_CRITERION].copy()
long_rts_dict = long_rts.groupby('num')['num'].count().to_dict()
for k, v in long_rts_dict.items():
    by_subj_summary.loc[k, 'abnormalRTN'] = v

#%% subject exclusions from too many abnormal RTs
by_subj_summary['RTSubjExclusion'] = by_subj_summary['abnormalRTN'] > ABNORMAL_RT_N_CRITERION
exclusion_list_dict['RTSubjExclusion'] = list(by_subj_summary[by_subj_summary['RTSubjExclusion']].index)

#%% subject exclusions from long experiment duration
upper_bound = subj['duration'].mean() + subj['duration'].std()*EXPT_DURATION_CRITERION

by_subj_summary['durationExclusion'] = subj['duration'] > upper_bound
exclusion_list_dict['durationExclusion'] = list(subj[by_subj_summary['durationExclusion']].index)

#%% all subject exclusions
exclusion_count_dict = {}
all_subj_exclusions = set()
for k, v in exclusion_list_dict.items():
    all_subj_exclusions = all_subj_exclusions.union(set(v))
    exclusion_count_dict[k] = len(v)
all_subj_exclusions = list(all_subj_exclusions)

trial = trial.drop(trial[trial['num'].isin(all_subj_exclusions)].index, axis=0)
subj = subj.drop(all_subj_exclusions, axis=0)

usable_subj = list(trial['num'].unique())
usable_subj.sort()
assert set(usable_subj) == set(subj.index)
by_subj_summary['included'] = 0
by_subj_summary.loc[usable_subj, 'included'] = 1

#%% drop extra subjects
if len(usable_subj) > TARGET_SAMPLE_SIZE:
    extra_subj = usable_subj[TARGET_SAMPLE_SIZE:]

    trial = trial.drop(trial[trial['num'].isin(extra_subj)].index, axis=0)
    by_subj_summary = by_subj_summary.drop(extra_subj, axis=0)
    usable_subj = list(trial['num'].unique())

    assert len(usable_subj) == TARGET_SAMPLE_SIZE

#%% save summary tab
if SAVE_TABS:
    by_subj_summary.to_excel(writer, sheet_name=EXCLUSION_TAB_NAME)

#%% drop excluded subjects from summary
by_subj_summary = by_subj_summary.drop(all_subj_exclusions, axis=0)
assert set(usable_subj) == set(by_subj_summary.index)
assert len(usable_subj) == len(by_subj_summary.index)

#%% print columns
print(by_subj_summary.columns)

#%% keep useful columns only in summary
USEFUL_COLUMNS = [
    'response',
    'gender',
    'age']
by_subj_summary = by_subj_summary[USEFUL_COLUMNS]

#%% z-transformation within subject within condition
   ###    ##    ##    ###    ##       ##    ##  ######  ####  ######
  ## ##   ###   ##   ## ##   ##        ##  ##  ##    ##  ##  ##    ##
 ##   ##  ####  ##  ##   ##  ##         ####   ##        ##  ##
##     ## ## ## ## ##     ## ##          ##     ######   ##   ######
######### ##  #### ######### ##          ##          ##  ##        ##
##     ## ##   ### ##     ## ##          ##    ##    ##  ##  ##    ##
##     ## ##    ## ##     ## ########    ##     ######  ####  ######
grouped = trial.groupby(by=['num','blockType'])
z_transformation = lambda x: (x - x.mean()) / x.std()
trial['zScore'] = grouped['response'].transform(z_transformation)

#%% print columns
print(trial.columns)
print(subj.columns)

#%% create object for used trial tab
irrelevant_columns = [
    'rt']
used_trial = trial.drop(irrelevant_columns, axis=1)

#%% organized used data
remove_columns = ['column4', 'rt']
stripped_trial = trial.drop(remove_columns, axis=1)
stripped_trial['stimName'] = stripped_trial['stimID'].astype(str) + '_' + stripped_trial['trialType']

ratings = {}
organized_resp = {}
z_scores = {}
organized_z = {}
for i, b in enumerate(BLOCK_TYPE):
    # raw ratings stimuli as columns
    organized_resp[b] = stripped_trial[stripped_trial['blockType']==i+1].drop(['zScore','blockType'], axis=1)
    organized_resp[b] = organized_resp[b].pivot(index='num', columns='stimName', values='response').reset_index()
    organized_resp[b].columns = ['SubjNum\StimName'] + list(organized_resp[b].columns[1:])
    organized_resp[b] = organized_resp[b].set_index('SubjNum\StimName').astype('float')
    organized_resp[b].columns = [b.lower()+'_'+x for x in organized_resp[b].columns]

    # raw ratings subjects as columns
    transposed_resp[b] = organized_resp[b].copy().transpose()
    transposed_resp[b].columns = ['S'+str(x) for x in transposed_resp[b].columns]

    # z-scores stimuli as columns
    organized_z[b] = stripped_trial[stripped_trial['blockNum']==i+1].drop(['response','blockNum'], axis=1)
    organized_z[b] = organized_z[b].pivot(index='num', columns='stimName', values='zScore').reset_index()
    organized_z[b].columns = ['SubjNum\StimName'] + list(organized_z[b].columns[1:])
    organized_z[b] = organized_z[b].set_index('SubjNum\StimName').astype('float')
    organized_z[b].columns = [b.lower()+'_'+x for x in organized_z[b].columns]

    # z-scores subjects as columns
    transposed_z[b] = organized_z[b].copy().transpose()
    transposed_z[b].columns = ['S'+str(x) for x in transposed_z[b].columns]

organized_resp['all'] = organized_resp['A'].join(organized_resp['B']).reset_index()
organized_z['all'] = organized_z['A'].join(organized_z['B']).reset_index()

#%% means and sds by subject and condition
temp_dict = {}
mean_sd_tab_dict = {}
mean_sd_tab_dict['M'] = pd.DataFrame()
mean_sd_tab_dict['SD'] = pd.DataFrame()
for b in BLOCK_TYPE:
    # raw ratings condition means and sds
    this_resp = transposed_resp[b].copy()
    this_resp['trialType'] = [x[-2:] for x in list(this_resp.index)]
    grouped_resp = this_resp.groupby(by=['trialType'])
    means = grouped_resp.mean().transpose()
    sds = grouped_resp.std().transpose()

    # z-scores condition means and sds
    this_z = transposed_z[b].copy()
    this_z['trialType'] = [x[-2:] for x in list(this_z.index)]
    grouped_z = this_z.groupby(by=['trialType'])
    z_means = grouped_z.mean().transpose()
    z_sds = grouped_z.std().transpose()

    # join two DVs
    temp_dict[(b,'M')] = pd.concat([means,z_means], axis=1, keys=['Raw Ratings','Z-Scores'])
    temp_dict[(b,'SD')] = pd.concat([sds,z_sds], axis=1, keys=['Raw Ratings','Z-Scores'])
mean_sd_tab_dict['M'] = pd.concat([temp_dict[('A','M')], temp_dict[('B','M')]], axis=1, keys=['A','B'])
mean_sd_tab_dict['SD'] = pd.concat([temp_dict[('A','SD')], temp_dict[('B','SD')]], axis=1, keys=['A','B'])

#%% calculate typicalities
COLUMNS_TUPLES = [
    ('Typicality','A','raw','all'),
    ('Typicality','A','raw','T1'),
    ('Typicality','A','raw','T2'),
    ('Typicality','A','raw','T3'),
    ('Typicality','A','raw','T4'),
    ('Typicality','A','z','all'),
    ('Typicality','A','z','T1'),
    ('Typicality','A','z','T2'),
    ('Typicality','A','z','T3'),
    ('Typicality','A','z','T4'),
    ('Typicality','B','raw','all'),
    ('Typicality','B','raw','T1'),
    ('Typicality','B','raw','T2'),
    ('Typicality','B','raw','T3'),
    ('Typicality','B','raw','T4'),
    ('Typicality','B','z','all'),
    ('Typicality','B','z','T1'),
    ('Typicality','B','z','T2'),
    ('Typicality','B','z','T3'),
    ('Typicality','B','z','T4')]
MULTILEVEL_COLUMNS = pd.MultiIndex.from_tuples(COLUMNS_TUPLES, names=('Measure','BlockType','DV','TrialType'))
typicality = pd.DataFrame(index=by_subj_summary.index, columns=MULTILEVEL_COLUMNS)
for b in BLOCK_TYPE:
    this_resp = organized_resp[b].copy()
    this_z = organized_z[b].copy()
    for left_out_s in by_subj_summary.index:
        # overall
        left_out_data_resp = this_resp.loc[left_out_s].copy()
        group_resp = this_resp.drop([left_out_s])
        means_resp = group_resp.mean()
        typicality.loc[left_out_s, ('Typicality',b,'raw','all')] = means_resp.corr(left_out_data_resp)

        left_out_data_z = this_z.loc[left_out_s].copy()
        group_z = this_z.drop([left_out_s])
        means_z = group_z.mean()
        typicality.loc[left_out_s, ('Typicality',b,'z','all')] = means_z.corr(left_out_data_z)

        # each trial type
        left_out_data_resp = left_out_data_resp.to_frame()
        left_out_data_resp.columns = ['leftout']
        means_resp = means_resp.to_frame()
        means_resp.columns = ['mean']
        correlation_resp = left_out_data_resp.join(means_resp)
        correlation_resp['trialType'] = [x[-2:] for x in correlation_resp.index]

        correlation_resp = correlation_resp.groupby(by=['trialType'])
        typicality_trial_type_resp = correlation_resp.corr(method='pearson')['leftout'].unstack()['mean']

        left_out_data_z = left_out_data_z.to_frame()
        left_out_data_z.columns = ['leftout']
        means_z = means_z.to_frame()
        means_z.columns = ['mean']
        correlation_z = left_out_data_z.join(means_z)
        correlation_z['trialType'] = [x[-2:] for x in correlation_z.index]

        correlation_z = correlation_z.groupby(by=['trialType'])
        typicality_trial_type_z = correlation_z.corr(method='pearson')['leftout'].unstack()['mean']

        for t in TRIAL_TYPE:
            typicality.loc[left_out_s, ('Typicality',b,'raw',t)] = typicality_trial_type_resp[t]
            typicality.loc[left_out_s, ('Typicality',b,'z',t)] = typicality_trial_type_z[t]
by_subj_summary.columns = pd.MultiIndex.from_product([['Questions'], [''], [''], by_subj_summary.columns])
by_subj_summary = by_subj_summary.join(typicality)

#%% save tabs
if SAVE_TABS:
    used_trial.to_excel(writer, sheet_name=USED_TRIAL_TAB_NAME, index=False)
    organized_resp['A'].to_excel(writer, sheet_name=USED_A_RESPONSE_TAB_NAME)
    organized_resp['B'].to_excel(writer, sheet_name=USED_B_RESPONSE_TAB_NAME)
    organized_resp['all'].to_excel(writer, sheet_name=USED_ALL_RESPONSE_TAB_NAME)
    organized_z['A'].to_excel(writer, sheet_name=USED_A_ZSCORE_TAB_NAME)
    organized_z['B'].to_excel(writer, sheet_name=USED_B_ZSCORE_TAB_NAME)
    organized_z['all'].to_excel(writer, sheet_name=USED_ALL_ZSCORE_TAB_NAME)
    mean_sd_tab_dict['M'].to_excel(writer, sheet_name=SUBJ_COND_MEAN_TAB_NAME)
    mean_sd_tab_dict['SD'].to_excel(writer, sheet_name=SUBJ_COND_SD_TAB_NAME)

#%% split-half reliability of typicality
stats = pd.DataFrame(columns=['name', 'value'])

ITERATION_N = 10000

for b in BLOCK_TYPE:
    print('Start iterations...')
    reliabilities_resp = []
    reliabilities_z = []
    this_resp = organized_resp[b].copy()
    this_z = organized_z[b].copy()
    sums_resp = this_resp.sum()
    sums_z = this_z.sum()
    subject_indices = organized_resp[b].index
    n_leave_one_out = len(subject_indices) - 1
    for i in range(ITERATION_N):
        if i % 100 == 0:
            print('Iteration '+str(i)+'...')
        split_half_resp = pd.DataFrame(columns=['set1','set2'], index=subject_indices)
        split_half_z = pd.DataFrame(columns=['set1','set2'], index=subject_indices)
        for left_out_s in subject_indices:
            # raw ratings
            left_out_resp = this_resp.loc[left_out_s].copy()
            means_resp = (sums_resp - left_out_resp) / n_leave_one_out

            left_out_resp_1 = left_out_resp.sample(n=int(len(left_out_resp)/2))
            left_out_resp_2 = left_out_resp.drop(left_out_resp_1.index)
            means_resp_1 = means_resp.drop(left_out_resp_2.index)
            means_resp_2 = means_resp.drop(left_out_resp_1.index)
            split_half_resp.loc[left_out_s, 'set1'] = means_resp_1.corr(left_out_resp_1)
            split_half_resp.loc[left_out_s, 'set2'] = means_resp_2.corr(left_out_resp_2)

            # z-scores
            left_out_z = this_z.loc[left_out_s].copy()
            means_z = (sums_z - left_out_z) / n_leave_one_out

            left_out_z_1 = left_out_z.sample(n=int(len(left_out_z)/2))
            left_out_z_2 = left_out_z.drop(left_out_z_1.index)
            means_z_1 = means_z.drop(left_out_z_2.index)
            means_z_2 = means_z.drop(left_out_z_1.index)
            split_half_z.loc[left_out_s, 'set1'] = means_z_1.corr(left_out_z_1)
            split_half_z.loc[left_out_s, 'set2'] = means_z_2.corr(left_out_z_2)
        # listwise deletion for zero variance
        split_half_resp = split_half_resp.dropna().astype(float)
        split_half_z = split_half_z.dropna().astype(float)
        r_resp = split_half_resp['set1'].corr(split_half_resp['set2'])
        r_z = split_half_z['set1'].corr(split_half_z['set2'])
        reliabilities_resp.append(analysis_func.SPLIT_HALF_RELIABILITY_CORRECTION(r_resp))
        reliabilities_z.append(analysis_func.SPLIT_HALF_RELIABILITY_CORRECTION(r_z))
    print('Done with iterations...')
    m_resp = round(statistics.mean(reliabilities_resp), 3)
    sd_resp = round(statistics.stdev(reliabilities_resp), 3)
    m_z = round(statistics.mean(reliabilities_z), 3)
    sd_z = round(statistics.stdev(reliabilities_z), 3)
    stats.loc[len(stats)] = [b+' Raw Typicality Reliability M', m_resp]
    stats.loc[len(stats)] = [b+' Raw Typicality Reliability SD', sd_resp]
    stats.loc[len(stats)] = [b+' Z Typicality Reliability M', m_z]
    stats.loc[len(stats)] = [b+' Z Typicality Reliability SD', sd_z]

#%% correlation between A and B typicalities
stats.loc[len(stats)] = ['Correlation between typicalities', '']
r, p = analysis_func.CALCULATE_PEARSON_R(by_subj_summary[('Typicality','A','z','all')], by_subj_summary[('Typicality','B','z','all')])
stats.loc[len(stats)] = ['A & B r', r]
stats.loc[len(stats)] = ['A & B p', p]

#%% correlation with self reports
stats.loc[len(stats)] = ['Correlation with self reports', '']
SELF_REPORTS = ['response']

for b in BLOCK_TYPE:
    for k in SELF_REPORTS:
        r, p = analysis_func.CALCULATE_PEARSON_R(by_subj_summary[('Typicality',b,'z','all')], by_subj_summary[('Questions','','',k)])
        stats.loc[len(stats)] = [b+' '+k+' r', r]
        stats.loc[len(stats)] = [b+' '+k+' p', p]

#%% gender differences
stats.loc[len(stats)] = ['Gender differences in typicality', '']
males = by_subj_summary[by_subj_summary[('Questions','','','gender')] == 'M'].copy()
females = by_subj_summary[by_subj_summary[('Questions','','','gender')] == 'F'].copy()

for b in BLOCK_TYPE:
    male_mean = males[('Typicality',b,'z','all')].mean()
    male_SD = males[('Typicality',b,'z','all')].std()
    female_mean = females[('Typicality',b,'z','all')].mean()
    female_SD = females[('Typicality',b,'z','all')].std()
    t, p = ttest_ind(males[('Typicality',b,'z','all')], females[('Typicality',b,'z','all')])

    stats.loc[len(stats)] = [b+' male M', male_mean]
    stats.loc[len(stats)] = [b+' female M', female_mean]
    stats.loc[len(stats)] = [b+' male SD', male_SD]
    stats.loc[len(stats)] = [b+' female SD', female_SD]
    stats.loc[len(stats)] = [b+' t', t]
    stats.loc[len(stats)] = [b+' p', p]

stats.loc[len(stats)] = ['Gender differences in self report', '']
for k in SELF_REPORTS:
    male_mean = males[('Questions','','',k)].mean()
    female_mean = females[('Questions','','',k)].mean()
    male_SD = males[('Questions','','',k)].std()
    female_SD = females[('Questions','','',k)].std()
    t, p = ttest_ind(males[('Questions','','',k)], females[('Questions','','',k)])

    stats.loc[len(stats)] = [k+' male M', male_mean]
    stats.loc[len(stats)] = [k+' female M', female_mean]
    stats.loc[len(stats)] = [k+' male SD', male_SD]
    stats.loc[len(stats)] = [k+' female SD', female_SD]
    stats.loc[len(stats)] = [k+' t', t]
    stats.loc[len(stats)] = [k+' p', p]

#%% correlation with age
stats.loc[len(stats)] = ['Typicality correlations with age', '']

for b in BLOCK_TYPE:
    r, p = analysis_func.CALCULATE_PEARSON_R(by_subj_summary[('Typicality',b,'z','all')], by_subj_summary[('Questions','','','age')])
    stats.loc[len(stats)] = [b+' r', r]
    stats.loc[len(stats)] = [b+' p', p]

stats.loc[len(stats)] = ['Self report correlations with age', '']
for k in SELF_REPORTS:
    r, p = analysis_func.CALCULATE_PEARSON_R(by_subj_summary[('Questions','','',k)], by_subj_summary[('Questions','','','age')])
    stats.loc[len(stats)] = [k+' r', r]
    stats.loc[len(stats)] = [k+' p', p]

#%% residual analysis
########  ########  ######  #### ########  ##     ##    ###    ##        ######
##     ## ##       ##    ##  ##  ##     ## ##     ##   ## ##   ##       ##    ##
##     ## ##       ##        ##  ##     ## ##     ##  ##   ##  ##       ##
########  ######    ######   ##  ##     ## ##     ## ##     ## ##        ######
##   ##   ##             ##  ##  ##     ## ##     ## ######### ##             ##
##    ##  ##       ##    ##  ##  ##     ## ##     ## ##     ## ##       ##    ##
##     ## ########  ######  #### ########   #######  ##     ## ########  ######

SUBJS = list(organized_resp['A'].index)

r_squared_dict = {}
residuals = pd.DataFrame(index=SUBJS, columns=organized_resp['A'].columns)
for s in SUBJS:
    x = np.array(organized_resp['B'].loc[s]).reshape(-1, 1)
    y_obsv = np.array(organized_resp['A'].loc[s])
    y = np.array(y_obsv).reshape(-1, 1)
    linear_regressor = sklearn.linear_model.LinearRegression()
    reg = linear_regressor.fit(x, y)
    y_pred = linear_regressor.predict(x)
    y_pred = y_pred.reshape(len(y_pred))
    r_squared_dict[s] = reg.score(x, y)
    residuals.loc[s] = y_obsv - y_pred

transposed_residuals = residuals.copy().transpose()
transposed_residuals.columns = ['S'+str(x) for x in transposed_residuals.columns]
r_squared = pd.DataFrame.from_dict(r_squared_dict, orient='index', columns=['r_squared'])

#%% means and sds by subject and condition
transposed_residuals = transposed_residuals.astype('float')
trial_type_column = [x[-2:] for x in list(transposed_residuals.index)]
transposed_residuals['trialType'] = trial_type_column
grouped_residuals = transposed_residuals.groupby(by=['trialType'])
residual_means = grouped_residuals.mean().transpose()
residual_sds = grouped_residuals.std().transpose()

#%% calculate typicality from residuals
residuals = residuals.astype('float')
RESIDUAL_COLUMNS_TUPLES = [
    ('Typicality','A','residual','all'),
    ('Typicality','A','residual','T1'),
    ('Typicality','A','residual','T2'),
    ('Typicality','A','residual','T3'),
    ('Typicality','A','residual','T4')]
RESIDUAL_MULTILEVEL_COLUMNS = pd.MultiIndex.from_tuples(RESIDUAL_COLUMNS_TUPLES, names=('Measure', 'Type', 'DV', 'TrialType'))
residuals_typicality = pd.DataFrame(index=by_subj_summary.index, columns=RESIDUAL_MULTILEVEL_COLUMNS)
for left_out_s in by_subj_summary.index:
    # overall
    left_out_data = residuals.loc[left_out_s].copy()
    group = residuals.drop(left_out_s)
    means = group.mean()
    residuals_typicality.loc[left_out_s, ('Typicality','A','residual','all')] = means.corr(left_out_data)

    # each trial type
    left_out_data = left_out_data.to_frame()
    left_out_data.columns = ['leftout']
    means = means.to_frame()
    means.columns = ['mean']
    correlation_data = left_out_data.join(means)
    correlation_data['trialType'] = [x[-2:] for x in correlation_data.index]

    correlation_data = correlation_data.groupby(by=['trialType'])
    typicality_trial_type = correlation_data.corr(method='pearson')['leftout'].unstack()['mean']
    for t in TRIAL_TYPE:
        residuals_typicality.loc[left_out_s, ('Typicality','A','residual',t)] = typicality_trial_type[t]
by_subj_summary = by_subj_summary.join(residuals_typicality)

#%% split-half reliability of typicality
ITERATION_N = 10000
print('Start iterations...')
reliabilities = []
sums = residuals.sum()
subject_indices = residuals.index
n_leave_one_out = len(subject_indices) - 1
for i in range(ITERATION_N):
    if i % 100 == 0:
        print('Iteration '+str(i)+'...')
    split_half = pd.DataFrame(columns=['set1','set2'], index=subject_indices)
    for left_out_s in subject_indices:
        # raw ratings
        left_out = residuals.loc[left_out_s].copy()
        means = (sums - left_out) / n_leave_one_out

        left_out_1 = left_out.sample(n=int(len(left_out)/2))
        left_out_2 = left_out.drop(left_out_1.index)
        means_1 = means.drop(left_out_2.index)
        means_2 = means.drop(left_out_1.index)
        split_half.loc[left_out_s, 'set1'] = means_1.corr(left_out_1)
        split_half.loc[left_out_s, 'set2'] = means_2.corr(left_out_2)

    # listwise deletion for zero variance
    split_half = split_half.dropna().astype('float')
    r = split_half['set1'].corr(split_half['set2'])
    reliabilities.append(analysis_func.SPLIT_HALF_RELIABILITY_CORRECTION(r))
print('Done with iterations...')
m = round(statistics.mean(reliabilities), 3)
sd = round(statistics.stdev(reliabilities), 3)

stats.loc[len(stats)] = ['Residual Analysis', '']
stats.loc[len(stats)] = ['Typicality Reliability M', m]
stats.loc[len(stats)] = ['Typicality Reliability SD', sd]

#%% correlation with self reports
for k in SELF_REPORTS:
    r, p = analysis_func.CALCULATE_PEARSON_R(by_subj_summary[('Typicality','A','residual','all')], by_subj_summary[('Questions','','',k)])
    stats.loc[len(stats)] = ['Residual typicality & '+k+' r', r]
    stats.loc[len(stats)] = ['Residual typicality & '+k+' p', p]

# %% save tabs
if SAVE_TABS:
    r_squared.to_excel(writer, sheet_name=REGRESSION_RESULTS_TAB_NAME)
    residuals.to_excel(writer, sheet_name=RESIDUALS_TAB_NAME)
    residual_means.to_excel(writer, sheet_name=RESIDUAL_MEAN_TAB_NAME)
    residual_sds.to_excel(writer, sheet_name=RESIDUAL_SD_TAB_NAME)
    stats.to_excel(writer, sheet_name=STATISTICS_TAB_NAME, index=False)
    by_subj_summary.to_excel(writer, sheet_name=SUMMARY_TAB_NAME)
    writer.save()
    writer.close()

#%% plot typicality distribution
########  ##        #######  ########
##     ## ##       ##     ##    ##
##     ## ##       ##     ##    ##
########  ##       ##     ##    ##
##        ##       ##     ##    ##
##        ##       ##     ##    ##
##        ########  #######     ##
%matplotlib inline

def RGBScaler(rgb):
    RGB = rgb[:3]
    A = rgb[3]
    return [x/255 for x in RGB] + [A]

A_COLOR = RGBScaler([15, 113, 115, 1])
B_COLOR = RGBScaler([240, 93, 94, 1])
TWO_D_COLOR = RGBScaler([93, 41, 148, 1])

TICKS = np.arange(-1, 1.1, 0.5)
LARGE_FONT = 30
SMALL_FONT = 25
LABEL_PAD = 20
LINE_WIDTH = 4

sns.set_style('dark')
sns.set_context('paper')
sns.set(rc={'figure.figsize':(11.7,8.27)})

A_TYPICALITY = by_subj_summary[('Typicality','A','z','all')]
B_TYPICALITY = by_subj_summary[('Typicality','B','z','all')]

fig = plt.figure()
ax = fig.add_subplot(111)
sns.kdeplot(A_TYPICALITY, data2=B_TYPICALITY, shade=True, shade_lowest=False, ax=ax, color=TWO_D_COLOR)
ax.set_xticks(TICKS)
ax.set_yticks(TICKS)
ax.tick_params(labelsize=SMALL_FONT)
plt.ylabel('B Typicality', fontsize=LARGE_FONT, labelpad=LABEL_PAD)
plt.xlabel('A Typicality', fontsize=LARGE_FONT, labelpad=LABEL_PAD)
fig.savefig('typicality_kde2d_'+TODAY_DATE+'.png', bbox_inches='tight')

fig_1d = plt.figure()
ax_1d = fig_1d.add_subplot(111)
sns.distplot(A_TYPICALITY, hist=False, rug=True, kde=True, color=A_COLOR, ax=ax_1d, label='A', kde_kws=dict(linewidth=LINE_WIDTH))
sns.distplot(B_TYPICALITY, hist=False, rug=True, kde=True, color=B_COLOR, ax=ax_1d, label='B', kde_kws=dict(linewidth=LINE_WIDTH))
ax_1d.set_xticks(TICKS)
ax_1d.legend(loc='upper left', fontsize=SMALL_FONT)
ax_1d.tick_params(labelsize=SMALL_FONT)
plt.ylabel('Density', fontsize=LARGE_FONT, labelpad=LABEL_PAD)
plt.xlabel('Typicality', fontsize=LARGE_FONT, labelpad=LABEL_PAD)
fig_1d.savefig('typicality_kde1d_'+TODAY_DATE+'.png', bbox_inches='tight')

#%% plot typicality correlation
EDGE_COLOR = '#333333'

scatter_plot_data = by_subj_summary[[
    ('Typicality','A','z','all'),
    ('Typicality','B','z','all')]].copy()
scatter_plot_data.columns = ['A','B']
scatter_plot_data = scatter_plot_data.astype('float')

# texform
fig_scatter = plt.figure(figsize=(5, 4.5), dpi=120)
fig_scatter.patch.set_alpha(0)
ax = fig_scatter.add_subplot(111)

ax.tick_params(axis='y', which='major', pad=7, length=5, width=2)
ax.tick_params(axis='x', which='major', pad=12, length=5, width=2)

ax.patch.set_facecolor('#ebebeb')
ax.patch.set_alpha(1)
ax.xaxis.label.set_color(EDGE_COLOR)
ax.yaxis.label.set_color(EDGE_COLOR)
ax.tick_params(axis='both', colors=EDGE_COLOR)
ax.set(ylim=(-0.2, 1), xlim=(-.2, 1))
ax = sns.regplot(x='A', y='B', data=scatter_plot_data, ax=ax)
handles, labels = ax.get_legend_handles_labels()

plt.xlabel('A Typicality', fontsize=LARGE_FONT)
plt.ylabel('B Typicality', fontsize=LARGE_FONT)

fig_scatter.savefig('typicalities_'+TODAY_DATE+'.png', bbox_inches='tight')

#%% plot residual typicality distribution
fig_resid = plt.figure()
ax_resid = fig_resid.add_subplot(111)
sns.distplot(by_subj_summary[('Typicality','A','residual','all')], hist=False, rug=True, kde=True, color=A_COLOR, ax=ax_resid, label='A', kde_kws=dict(linewidth=LINE_WIDTH))
ax_resid.set_xticks(TICKS)
ax_resid.get_legend().set_visible(False)
ax_resid.tick_params(labelsize=SMALL_FONT)
plt.ylabel('Density', fontsize=LARGE_FONT, labelpad=LABEL_PAD)
plt.xlabel('Typicality', fontsize=LARGE_FONT, labelpad=LABEL_PAD)
fig_resid.savefig('residualTypicality_kde_'+TODAY_DATE+'.png', bbox_inches='tight')

#%%