import numpy as np
import pandas as pd
import os, json
from typing import List, Dict, Union
from collections import Counter, defaultdict
import dask.dataframe as dd
import swifter
from bokeh.util.logconfig import level
from joblib import Parallel, delayed
from tqdm import tqdm
from pandas.core.frame import DataFrame
from pandas.core.groupby.generic import SeriesGroupBy
from datetime import datetime
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
base_dir = 'E:/芒果台比赛/'
basic_item_fea = ["item_cid", "item_type", "item_duration", "item_assetSource", "item_classify", "item_isIntact", "item_serialno", "sid", "stype"]
item_fea = ["item_cid", "item_type", "item_assetSource", "item_classify", "item_isIntact", "stype"]
seq_fea = ["hash_vid", "sid", "item_cid", "click_day", "hash_vid_ptr", "item_serialno"]
options = {'n_jobs': 16}


def day_id(i: int):
    return str(i) if i >= 10 else '0' + str(i)

def hash_helper(df: DataFrame, col: str):
    codes, uniques = pd.factorize(df[col], sort=False)
    return codes

def merge_helper(attr_dic: Dict, df: DataFrame, key: str, col:str, fill_na:Union[int,float,str], col_name:str = None):
    col_name = col_name or col
    df_tmp = pd.DataFrame({key: attr_dic[col].keys(), col_name: attr_dic[col].values()}).set_index(key)
    df = df.merge(df_tmp, how='left', left_index=True, right_index=True)
    if isinstance(fill_na, int) or isinstance(fill_na, float):
        df.fillna(fill_na, inplace=True)
    elif fill_na == 'mean':
        df.fillna(df_tmp[col_name].mean(), inplace=True)
    elif fill_na == 'most':
        df.fillna(df_tmp[col_name].mode()[0], inplace=True)
    else:
        raise ValueError("incompatible 'fill_na' parameter")
    print(f"{col_name} success")
    return df



def build_init_df_tmp_v1(cols: List[str]):
    df = pd.read_csv(base_dir + 'did_features/did_features_table.csv', usecols=['did'])
    df["show_sum"] = 0; df["click_sum"] = 0; df["play_ratio_sum"] = 0
    for col in cols:
        df[f"{col}_show_sum"] = '{}'; df[f"{col}_click_sum"] = '{}'; df[f"{col}_play_ratio_sum"] = '{}'
    for col in df.columns[4:]:
        df[col] = df[col].swifter.apply(json.loads)
    return df


def update_df_tmp_dict_all(df_today: DataFrame, df_tmp: DataFrame):
    df = df_today[['did', 'vid', 'click', 'ptr']].copy()

    df = df.groupby('did', sort=False).agg(
            show_sum=('vid', 'count'),
            click_sum=('click', 'sum'),
            play_ratio_sum=('ptr', 'sum')
        )

    df_tmp_new = df_tmp[['did', 'show_sum', 'click_sum', 'play_ratio_sum']].copy().set_index('did')
    df_tmp_new = df_tmp_new.add(df, fill_value=0)

    return df_tmp_new


def update_df_tmp_dict_fea(feature: str, df_today: DataFrame, df_tmp: DataFrame):
    df = df_today[['did', 'vid', 'click', 'ptr', feature]].copy()
    df[feature] = df[feature].astype('str')

    agg_df = df.groupby(['did', feature], sort=False, as_index=False).agg(
                    show_sum=('vid', 'count'),
                    click_sum=('click', 'sum'),
                    play_ratio_sum=('ptr', 'sum')
                )

    dict_df = zip_dict_parallel(agg_df.groupby('did'), feature)

    tmp_selected_cols = ['did', f'{feature}_show_sum', f'{feature}_click_sum', f'{feature}_play_ratio_sum']
    df_tmp_new = df_tmp[tmp_selected_cols].copy().merge(dict_df, on='did', how='left')
    for new_col in tmp_selected_cols[1:]:
        base_col = new_col[len(feature)+1:]
        df_tmp_new[new_col] = df_tmp_new.swifter.apply(lambda row: update_sum_dict(row[base_col], row[new_col]), axis=1)
    df_tmp_new = df_tmp_new[tmp_selected_cols].set_index('did')

    return df_tmp_new


def update_df_dict_all(df_tomorrow_merged: DataFrame):
    df_new = df_tomorrow_merged[['show_sum', 'click_sum', 'play_ratio_sum']].copy()
    df_new['avg_ctr'] = df_new.swifter.apply(lambda row: walson_ratio(row['click_sum'], row['show_sum']), axis=1).astype('float32')
    df_new['avg_ptr'] = df_new.swifter.apply(lambda row: walson_ratio(row['play_ratio_sum'], row['click_sum']), axis=1).astype('float32')
    df_new = df_new[['show_sum', 'click_sum', 'avg_ctr', 'avg_ptr']].fillna(0)
    return df_new


def update_df_dict_fea(feature: str, df_tomorrow_merged: DataFrame):
    df_new = df_tomorrow_merged[[f'{feature}_show_sum', f'{feature}_click_sum', f'{feature}_play_ratio_sum']].copy()
    df_new[f'{feature}_ctr'] = df_new.swifter.apply(lambda row: get_walson_ratio_dic(row[f'{feature}_click_sum'], row[f'{feature}_show_sum']), axis=1)
    df_new[f'{feature}_ptr'] = df_new.swifter.apply(lambda row: get_walson_ratio_dic(row[f'{feature}_play_ratio_sum'], row[f'{feature}_click_sum']), axis=1)
    df_new = df_new[[f'{feature}_click_sum', f'{feature}_ctr', f'{feature}_ptr']]
    return df_new


def zip_dict_parallel(df_grouped: SeriesGroupBy, feature: str):
    def zip_dic(key, group, feature):
        show_dict = dict(zip(group[feature], group['show_sum']))
        click_dict = dict(zip(group[feature], group['click_sum']))
        play_dict = dict(zip(group[feature], group['play_ratio_sum']))
        return {
            'did': key,
            'show_sum': show_dict,
            'click_sum': click_dict,
            'play_ratio_sum': play_dict
        }

    results = Parallel(n_jobs = options['n_jobs'])(
        delayed(zip_dic)(key, group, feature) for key, group in df_grouped
    )
    return pd.DataFrame(results)


def update_sum_dict(new_dic: Dict[str, int], main_dic: Dict[str, int]):
    return dict(Counter(main_dic) + Counter(new_dic))


'''
def get_ratio_dic(numerator: Union[Dict[str, int], Dict[str, float]], denominator: Union[Dict[str, int], Dict[str, float]]):
    ratio_dic = {}
    for key in numerator.keys():
        if denominator[key] > 0:
            ratio_dic[key] = numerator[key] / denominator[key]
    return ratio_dic
    
def add_time_fea(df: DataFrame):
    time_tmp = pd.to_datetime(df['click_time'], errors='coerce')
    today = time_tmp[time_tmp.first_valid_index()]
    df['weekday'] = (today.weekday() + 1).astype(int)
    return df    
'''


def walson_ratio(m: Union[int, float], n: Union[int, float]):
    if n == 0:
        return 0
    z = 1.96
    p = m / n
    a = p + z ** 2 / (2 * n)
    b = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * (n ** 2)))
    c = 1 + z ** 2 / n
    return (a - b) / c


def get_walson_ratio_dic(numerator: Union[Dict[str, int], Dict[str, float]], denominator: Union[Dict[str, int], Dict[str, float]]):
    ratio_dic = {}
    for key in numerator.keys():
        if denominator[key] > 0:
            ratio_dic[key] = walson_ratio(numerator[key], denominator[key])
    return ratio_dic





def build_init_attr_fea_2(fea_cols: List[str]):
    df = pd.read_parquet(base_dir + 'did_features/did_fea.parquet', columns=['did']).set_index('did')
    num_row = len(df) ; empty_lists = [[] for _ in range(num_row)]
    for col in fea_cols:
        if col != 'item_serialno':
            df[f"{col}_seq"] = empty_lists
        else:
            df["item_serialno_seq"] = [defaultdict(list) for _ in range(num_row)]
    return df


def update_df_tmp_seq(fea: str, df_today: DataFrame, df_tmp: DataFrame, trunc_num: int = 100):
    if fea == 'click_day':
        df_today['click_day'] = (pd.to_datetime(df_today['click_time']) - pd.to_datetime('2025/3/23')).dt.days
    elif fea == 'hash_vid_ptr':
        df_today['hash_vid_ptr'] = df_today['ptr']

    df_today_fea_agg = df_today.groupby(level=0)[fea].apply(list).rename(f'today_{fea}_seq')
    df_tmp_new = df_tmp.merge(df_today_fea_agg, how='left', left_index=True, right_index=True)
    df_tmp_new[f'{fea}_seq'] = df_tmp_new.swifter.apply(lambda row:(row[f'today_{fea}_seq'] + row[f'{fea}_seq'])[:trunc_num], axis=1)
    del df_tmp_new[f'today_{fea}_seq']
    return df_tmp_new


def update_df_tmp_serialno_seq(df_today: DataFrame, df_tmp: DataFrame, trunc_num: int = 100):
    df_today_fea_agg = df_today.reset_index().groupby(['did', 'item_cid'], as_index=False) \
                               .agg({'item_serialno': list}) \
                               .groupby('did') \
                               .apply(lambda group: dict(zip(group['item_cid'], group['item_serialno']))) \
                               .rename('today_item_serialno_seq')

    df_tmp_new = df_tmp.merge(df_today_fea_agg, how='left', left_index=True, right_index=True)
    df_tmp_new['item_serialno_seq'] = df_tmp_new.swifter.apply(lambda row: update_serialno_dict(row, trunc_num), axis=1)
    del df_tmp_new['today_item_serialno_seq']
    return df_tmp_new


def update_serialno_dict(row, trunc_num):
    new_dic, main_dic = row['today_item_serialno_seq'], row['item_serialno_seq']
    for cid in new_dic:
        main_dic[cid] = (new_dic[cid] + main_dic[cid])[:trunc_num]
    return main_dic


def find_top_click_fea(dic_str: str):
    dic = json.loads(dic_str)
    if not dic:
        res = None
    else:
        res = sorted(dic.items(), key=lambda item: item[1], reverse=True)[0][0]
    return res




def shrink_bound_ls(bound_ls: List[float]):
    shrink_bound_ls = [bound_ls[0]]
    for v in bound_ls:
        if v > shrink_bound_ls[-1]:
            shrink_bound_ls.append(v)
    return shrink_bound_ls

def get_buketed_fea_df(df: DataFrame, bound_ls_dic: Dict[str, List[float]]):
    for col in bound_ls_dic:
        bound_ls = shrink_bound_ls(bound_ls_dic[col])
        df[col] = np.digitize(df[col], bins=bound_ls, right=True)
        #print(f"{col}:{bound_ls}")
        print(f"{col} success")
    return df


def get_is_first_click(df: DataFrame):
    df_click = df[df['click'] == 1].sort_values('click_time').drop_duplicates('did')[['did']]
    df_click['first_click'] = 1
    df = df.merge(df_click, how='left', left_index=True, right_index=True, suffixes=('','_')).drop(columns=['did_'])
    df['first_click'] = df['first_click'].fillna(0).astype('int')
    return df


def get_is_cid_next_episode(df: DataFrame):
    def get_is_cid_next_episode_helper(row):
        dic = json.loads(row['item_serialno_seq'])
        if not dic or row['item_cid'] not in dic:
            return 2
        else:
            last_no = dic[row['item_cid']][0]
            return int(row['item_serialno'] > last_no)

    df['is_cid_next_episode'] = df.swifter.apply(lambda row: get_is_cid_next_episode_helper(row), axis=1)
    return df


