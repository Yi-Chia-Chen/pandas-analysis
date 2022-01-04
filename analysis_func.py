import os
import functools
from datetime import datetime

import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

 ######   ######## ##    ## ######## ########     ###    ##
##    ##  ##       ###   ## ##       ##     ##   ## ##   ##
##        ##       ####  ## ##       ##     ##  ##   ##  ##
##   #### ######   ## ## ## ######   ########  ##     ## ##
##    ##  ##       ##  #### ##       ##   ##   ######### ##
##    ##  ##       ##   ### ##       ##    ##  ##     ## ##
 ######   ######## ##    ## ######## ##     ## ##     ## ########

def CHECK_AND_CREATE_FOLDERS(path):
    PATH = path.strip('/').strip('\\')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

def TODAY_DATE_STRING():
    return str(datetime.now()).split('.')[0].split(' ')[0].replace('-', '.')


 ######  ########    ###    ######## ####  ######  ######## ####  ######   ######
##    ##    ##      ## ##      ##     ##  ##    ##    ##     ##  ##    ## ##    ##
##          ##     ##   ##     ##     ##  ##          ##     ##  ##       ##
 ######     ##    ##     ##    ##     ##   ######     ##     ##  ##        ######
      ##    ##    #########    ##     ##        ##    ##     ##  ##             ##
##    ##    ##    ##     ##    ##     ##  ##    ##    ##     ##  ##    ## ##    ##
 ######     ##    ##     ##    ##    ####  ######     ##    ####  ######   ######

def CALCULATE_SS_FROM_MEAN(number_list):
    count = len(number_list)
    mean = float(sum(number_list))/count
    return functools.reduce(lambda x, y: x+ (y-mean)**2, number_list, 0)

def CALCULATE_SD(number_list):
    count = len(number_list)
    return CALCULATE_SS_FROM_MEAN(number_list) / (count-1)

def CALCULATE_SIGMA(number_list):
    count = len(number_list)
    return CALCULATE_SS_FROM_MEAN(number_list) / count

def SPLIT_HALF_RELIABILITY_CORRECTION(r):
    return 2*r / (1+abs(r))

def EUCLIDEAN_DISTANCE(vector_1, vector_2):
    assert len(vector_1) == len(vector_2), 'ERROR: vectors have unequal lengths'
    return sum([(x_1-x_2)**2 for x_1,x_2 in zip(vector_1,vector_2)]) ** 0.5

def SUM_SQUARES(number_list):
    return functools.reduce(lambda x, y: x+y**2, number_list, 0)

def SUM_X_Y_MULTIPLY(x_list, y_list):
    assert len(x_list) == len(y_list), 'ERROR: input lists have unequal lengths'
    return functools.reduce(lambda x, y: x + y[0]*y[1], zip(x_list,y_list), 0)

def CALCULATE_PEARSON_R(x_list, y_list):
    count = len(x_list)
    assert count == len(y_list), 'ERROR: input lists have unequal lengths'
    mean_x = sum(x_list)/count
    mean_y = sum(y_list)/count
    SS_x = SUM_SQUARES(x_list)
    SS_y = SUM_SQUARES(y_list)
    sum_xy = SUM_X_Y_MULTIPLY(x_list,y_list)
    SD_x = ((SS_x - count*(mean_x**2))) ** 0.5
    SD_y = ((SS_y - count*(mean_y**2))) ** 0.5
    r = (sum_xy - mean_x*mean_y*count) / (SD_x*SD_y)
    if abs(r) != 1:
        tt = abs(r)*((count-2)**0.5)/((1-r**2)**0.5)
        p = stats.t.sf(tt, count-2)*2 # two-tailed p-value
    else:
        p = 0
    return (r, p)


########     ###    ##    ## ########     ###     ######
##     ##   ## ##   ###   ## ##     ##   ## ##   ##    ##
##     ##  ##   ##  ####  ## ##     ##  ##   ##  ##
########  ##     ## ## ## ## ##     ## ##     ##  ######
##        ######### ##  #### ##     ## #########       ##
##        ##     ## ##   ### ##     ## ##     ## ##    ##
##        ##     ## ##    ## ########  ##     ##  ######

def CLEAN_CARRIAGE_RETURN(df):
    df = df.copy()
    last_column_name = list(df.columns)[-1]
    if last_column_name[-1] == '\r':
        new_column_name = last_column_name[:-1]
        df.columns = list(df.columns[:-1]) + [new_column_name]
        df[new_column_name] = df[new_column_name].str.rstrip('\r')
    return df

def DELETE_REPEATING_TITLES(df):
    df = df.copy()
    first_column_name = df.columns[0]
    df = df.drop(df.loc[df[first_column_name] == first_column_name].index, axis=0)
    return df


 ######  ########    ###    ########   #######  ########  ##    ##
##    ## ##         ## ##   ##     ## ##     ## ##     ## ###   ##
##       ##        ##   ##  ##     ## ##     ## ##     ## ####  ##
 ######  ######   ##     ## ########  ##     ## ########  ## ## ##
      ## ##       ######### ##     ## ##     ## ##   ##   ##  ####
##    ## ##       ##     ## ##     ## ##     ## ##    ##  ##   ###
 ######  ######## ##     ## ########   #######  ##     ## ##    ##

def PLOT_DISTRIBUTION(
    data_list, color='#3d879c', hist=True, kde=True, rug=True,
    x_limit=None, y_limit=None,
    x_label=None, y_label=None,
    save=False, today_date=None, dpi=200):

    PLOT_SETTINGS_DIC = {
        'axes.grid': False,
        'font.size':10,
        'grid.color': '.5',
        'font.sans-serif': ['Century Gothic', 'Arial', 'sans-serif'],
        'axes.spines.left': False,
        'axes.spines.bottom': True,
        'axes.spines.right': False,
        'axes.spines.top': False
        }
    with sns.axes_style('darkgrid',rc=PLOT_SETTINGS_DIC):
        plot = sns.distplot(data_list,
                        hist=hist, kde=kde, rug=rug,
                        color=color, kde_kws={'linewidth':4})
        plot.set(yticks=[])
        if x_limit != None:
            plt.xlim(x_limit[0], x_limit[1])
        if y_limit != None:
            plt.ylim(y_limit[0], y_limit[1])
        if x_label != None:
            plt.xlabel(x_label, labelpad=10)
        if y_label != None:
            plt.ylabel(y_label, labelpad=10)
        if save:
            plt.savefig('distribution_'+today_date+'.png', dpi=dpi, bbox_inches='tight')
            plt.savefig('distribution_'+today_date+'.svg', dpi=dpi, bbox_inches='tight')
