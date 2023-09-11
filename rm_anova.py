import string
import itertools

import pandas as pd
import numpy as np
import scipy.stats

import analysis_func


def normalized_squared_sum(input_list):
    return sum(input_list) ** 2 / len(input_list)


def flatten_dataframe(frame):
    if isinstance(frame, np.floating):
        results = [frame]
    else:
        results = frame.to_numpy().flatten()
    return results


def calculate_main_effects_ss_df(
    factor, factors, data, factor_level_dict, all_normalized_squared_sum
):
    other_factors = factors[:]
    other_factors.remove(factor)
    if len(other_factors) > 0:
        this_data = data.reorder_levels([factor] + other_factors, axis=1)
    else:
        this_data = data
    ss = (
        sum(
            [
                normalized_squared_sum(flatten_dataframe(this_data[c]))
                for c in factor_level_dict[factor]
            ]
        )
        - all_normalized_squared_sum
    )
    df = len(factor_level_dict[factor]) - 1
    return ss, df


def find_parameters_for_interactions(effect, factors, data, factor_level_dict, error):
    lower_effects = []
    for i in range(1, len(effect)):
        lower_effects += list(itertools.combinations(effect, i))
    lower_effects = [" x ".join(x) if len(x) != 1 else x[0] for x in lower_effects]
    condition_factors = list(effect[:])
    if error:
        condition_factors.remove("S")
    conditions = list(
        itertools.product(*[factor_level_dict[x] for x in condition_factors])
    )
    other_factors = list(set(factors) - set(effect))
    if len(other_factors) > 0:
        this_data = data.reorder_levels(list(condition_factors) + other_factors, axis=1)
    else:
        this_data = data
    return lower_effects, condition_factors, conditions, this_data


def calculate_interaction_effects_ss_df(
    effect, factors, data, table, factor_level_dict, all_normalized_squared_sum
):
    lower_effects, _, conditions, this_data = find_parameters_for_interactions(
        effect, factors, data, factor_level_dict, False
    )
    sum_value = sum(
        [normalized_squared_sum(flatten_dataframe(this_data[c])) for c in conditions]
    )
    subtract_value = all_normalized_squared_sum + sum(
        [table.loc[x, "SS"] for x in lower_effects]
    )
    ss = sum_value - subtract_value
    df = np.prod([len(factor_level_dict[f]) - 1 for f in effect])
    return ss, df


def calculate_interaction_errors_ss_df(
    effect, factors, data, table, factor_level_dict, all_normalized_squared_sum
):
    (
        lower_effects,
        condition_factors,
        conditions,
        this_data,
    ) = find_parameters_for_interactions(effect, factors, data, factor_level_dict, True)
    subj_n = len(data.index)
    sum_value = sum(
        [
            normalized_squared_sum(flatten_dataframe(this_data.loc[i, c]))
            for c in conditions
            for i in this_data.index
        ]
    )
    subtract_value = all_normalized_squared_sum + sum(
        [table.loc[x, "SS"] for x in lower_effects]
    )
    ss = sum_value - subtract_value
    df = np.prod([len(factor_level_dict[f]) - 1 for f in condition_factors]) * (
        subj_n - 1
    )
    return ss, df


def rm_anova(data):
    if data.columns.nlevels > 1:
        data.columns = data.columns.remove_unused_levels()
        columns = [list(a) for a in data.columns.levels]
    else:
        columns = [list(data.columns)]
    factor_n = len(columns)
    factor_names = list(data.columns.names)
    if None in factor_names:
        if factor_n > 1:
            factor_names = string.ascii_uppercase[0:factor_n]
        else:
            factor_names = ["A"]
    factor_level_dict = dict(zip(factor_names, columns))
    subj_n = len(data.index)

    table = pd.DataFrame(columns=["SS", "df", "MS", "F", "p", "eta^2"])

    all_normalized_squared_sum = normalized_squared_sum(flatten_dataframe(data))

    # main effects SS & df
    for f in factor_names:
        ss, df = calculate_main_effects_ss_df(
            f, factor_names, data, factor_level_dict, all_normalized_squared_sum
        )
        table.loc[f, "SS"] = ss
        table.loc[f, "df"] = df
    table.loc["S", "SS"] = (
        sum(
            [
                normalized_squared_sum(flatten_dataframe(row))
                for _, row in data.iterrows()
            ]
        )
        - all_normalized_squared_sum
    )
    table.loc["S", "df"] = subj_n - 1

    # interaction effects and errors SS & df
    effects = []
    for i in range(2, factor_n + 2):
        effects += list(itertools.combinations(factor_names + ["S"], i))

    for e in effects:
        if "S" in e:
            # errors
            ss, df = calculate_interaction_errors_ss_df(
                e,
                factor_names,
                data,
                table,
                factor_level_dict,
                all_normalized_squared_sum,
            )
            this_index = " x ".join(e)
            table.loc[this_index, "SS"] = ss
            table.loc[this_index, "df"] = df
        else:
            # interactions
            ss, df = calculate_interaction_effects_ss_df(
                e,
                factor_names,
                data,
                table,
                factor_level_dict,
                all_normalized_squared_sum,
            )
            this_index = " x ".join(e)
            table.loc[this_index, "SS"] = ss
            table.loc[this_index, "df"] = df

    # MS
    for e in table.index:
        table.loc[e, "MS"] = table.loc[e, "SS"] / table.loc[e, "df"]

    # F, p, eta^2
    testing_effects = [i for i in table.index if "S" not in i]
    errors = [i for i in table.index if "S" in i]
    for e in testing_effects:
        table.loc[e, "F"] = table.loc[e, "MS"] / table.loc[e + " x S", "MS"]
        table.loc[e, "p"] = 1 - scipy.stats.f.cdf(
            table.loc[e, "F"], table.loc[e, "df"], table.loc[e + " x S", "df"]
        )
        table.loc[e, "eta^2"] = table.loc[e, "SS"] / (
            table.loc[e, "SS"] + table.loc[e + " x S", "SS"]
        )
    table = table.reindex(testing_effects + errors)

    return table
