import itertools

import pandas as pd
import numpy as np
import scipy.stats

import analysis_func


def NORMALIZED_SQUARED_SUM(input_list):
    return sum(input_list)**2/len(input_list)


def FLATTEN_DATAFRAME(frame):
    if isinstance(frame, np.floating):
        results = [frame]
    else:
        results = frame.to_numpy().flatten()
    return results


def CALCULATE_MAIN_EFFECTS_SS_DF(
    factor, factors, data,
    factor_level_dict, all_normalized_squared_sum):
    OTHER_FACTORS = factors[:]
    OTHER_FACTORS.remove(factor)
    this_data = data.reorder_levels([factor] + OTHER_FACTORS, axis=1)
    SS = sum([
        NORMALIZED_SQUARED_SUM(FLATTEN_DATAFRAME(this_data[c]))
        for c in factor_level_dict[factor]
        ]) - all_normalized_squared_sum
    DF = len(factor_level_dict[factor]) - 1
    return SS, DF


def FIND_PARAMETERS_FOR_INTERACTIONS(
    effect, factors, data, factor_level_dict, error):

    LOWER_EFFECTS = []
    for i in range(1,len(effect)):
        LOWER_EFFECTS += list(itertools.combinations(effect, i))
    LOWER_EFFECTS = [
        ' x '.join(x)
        if len(x) != 1 else x[0]
        for x in LOWER_EFFECTS
        ]
    CONDITION_FACTORS = list(effect[:])
    if error:
        CONDITION_FACTORS.remove('S')
    CONDITIONS = list(itertools.product(*[
        factor_level_dict[x]
        for x in CONDITION_FACTORS
        ]))
    OTHER_FACTORS = list(set(factors) - set(effect))
    this_data = data.reorder_levels(list(CONDITION_FACTORS) + OTHER_FACTORS, axis=1)
    return LOWER_EFFECTS, CONDITION_FACTORS, CONDITIONS, this_data


def CALCULATE_INTERACTION_EFFECTS_SS_DF(
    effect, factors, data, table,
    factor_level_dict, all_normalized_squared_sum):

    LOWER_EFFECTS, _, CONDITIONS, this_data = FIND_PARAMETERS_FOR_INTERACTIONS(
        effect, factors, data, factor_level_dict, False
        )
    SUM_VALUE = sum([
        NORMALIZED_SQUARED_SUM(FLATTEN_DATAFRAME(this_data[c]))
        for c in CONDITIONS
        ])
    SUBTRACT_VALUE = all_normalized_squared_sum + sum([
        table.loc[x,'SS']
        for x in LOWER_EFFECTS
        ])
    SS = SUM_VALUE - SUBTRACT_VALUE
    DF = np.prod([
        len(factor_level_dict[f]) - 1
        for f in effect
        ])
    return SS, DF


def CALCULATE_INTERACTION_ERRORS_SS_DF(
    effect, factors, data, table,
    factor_level_dict, all_normalized_squared_sum):

    LOWER_EFFECTS, CONDITION_FACTORS, CONDITIONS, this_data = FIND_PARAMETERS_FOR_INTERACTIONS(
        effect, factors, data, factor_level_dict, True
        )
    SUBJ_N = len(data.index)
    SUM_VALUE = sum([
        NORMALIZED_SQUARED_SUM(FLATTEN_DATAFRAME(this_data.loc[i,c]))
        for c in CONDITIONS
        for i in this_data.index
        ])
    SUBTRACT_VALUE = all_normalized_squared_sum + sum([
        table.loc[x,'SS']
        for x in LOWER_EFFECTS
        ])
    SS = SUM_VALUE - SUBTRACT_VALUE
    DF = np.prod([
        len(factor_level_dict[f]) - 1
        for f in CONDITION_FACTORS
        ]) * (SUBJ_N - 1)
    return SS, DF


def RM_ANOVA(data):
    SUBJ_N = len(data.index)
    COLUMNS = [list(a) for a in data.columns.levels]
    FACTOR_N = len(COLUMNS)
    FACTOR_NAMES = list(data.columns.names)
    FACTOR_LEVEL_DICT = dict(zip(FACTOR_NAMES, COLUMNS))

    table = pd.DataFrame(columns=['SS','df','MS','F','p','eta^2'])

    ALL_NORMALIZED_SQUARED_SUM = NORMALIZED_SQUARED_SUM(FLATTEN_DATAFRAME(data))

    # main effects SS & df
    for f in FACTOR_NAMES:
        ss, df = CALCULATE_MAIN_EFFECTS_SS_DF(
            f, FACTOR_NAMES, data,
            FACTOR_LEVEL_DICT, ALL_NORMALIZED_SQUARED_SUM
            )
        table.loc[f,'SS'] = ss
        table.loc[f,'df'] = df
    table.loc['S','SS'] = sum([
        NORMALIZED_SQUARED_SUM(FLATTEN_DATAFRAME(row))
        for _, row in data.iterrows()
        ]) - ALL_NORMALIZED_SQUARED_SUM
    table.loc['S','df'] = SUBJ_N - 1

    # interaction effects and errors SS & df
    effects = []
    for i in range(2, FACTOR_N + 2):
        effects += list(itertools.combinations(FACTOR_NAMES + ['S'], i))

    for e in effects:
        if 'S' in e:
            # errors
            ss, df = CALCULATE_INTERACTION_ERRORS_SS_DF(
                e, FACTOR_NAMES, data, table,
                FACTOR_LEVEL_DICT, ALL_NORMALIZED_SQUARED_SUM
                )
            this_index = ' x '.join(e)
            table.loc[this_index,'SS'] = ss
            table.loc[this_index,'df'] = df
        else:
            # interactions
            ss, df = CALCULATE_INTERACTION_EFFECTS_SS_DF(
                e, FACTOR_NAMES, data, table,
                FACTOR_LEVEL_DICT, ALL_NORMALIZED_SQUARED_SUM
                )
            this_index = ' x '.join(e)
            table.loc[this_index,'SS'] = ss
            table.loc[this_index,'df'] = df

    # MS
    for e in table.index:
        table.loc[e,'MS'] = table.loc[e,'SS'] / table.loc[e,'df']

    # F, p, eta^2
    TESTING_EFFECTS = [i for i in table.index if 'S' not in i]
    ERRORS = [i for i in table.index if 'S' in i]
    for e in TESTING_EFFECTS:
        table.loc[e,'F'] = table.loc[e,'MS'] / table.loc[e+' x S','MS']
        table.loc[e,'p'] = 1 - scipy.stats.f.cdf(table.loc[e,'F'], table.loc[e,'df'], table.loc[e+' x S','df'])
        table.loc[e,'eta^2'] = table.loc[e,'SS'] / (table.loc[e,'SS'] + table.loc[e + ' x S','SS'])
    table = table.reindex(TESTING_EFFECTS + ERRORS)

    return table