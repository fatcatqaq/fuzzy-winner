import json
import os.path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
import pandas as pd
#from ydata_profiling import ProfileReport
#from modelscope.pipelines import pipeline
#from modelscope.utils.constant import Tasks
from tqdm import tqdm
import fea_utils
from datetime import datetime
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
base_dir = 'E:/芒果台比赛/'
basic_item_fea = ["item_cid", "item_type", "item_duration", "item_assetSource", "item_classify", "item_isIntact", "item_serialno", "sid", "stype"]
item_fea = ["item_cid", "item_type", "item_assetSource", "item_classify", "item_isIntact", "stype"]
seq_fea = ["hash_vid", "sid", "item_cid", "click_day", "hash_vid_ptr", "item_serialno"]


# def report(data, html_name=''):
#     profile = ProfileReport(data,
#                             title='Fast Report for EDA',
#                             html={'style': {'full_width': True}})
#
#     profile.to_file(rf"C:\Users\30443\Desktop\{html_name}.html")

'''
def comp_test_and_train():
    df_test = pd.read_csv(base_dir + 'A榜用户曝光数据/testA_did_show.csv')
    set_test_did = set(df_test['did'].tolist())
    set_test_vid = set(df_test['vid'].tolist())
    set_train_did = set(pd.read_csv(base_dir + f'did_features/did_features_table.csv')['did'].tolist())
    set_train_vid = set(pd.read_csv(base_dir + f'vid_info/vid_info_table.csv')['vid'].tolist())
    did_intersect = set_test_did & set_train_did
    vid_intersect = set_test_vid & set_train_vid
    print(f"未在训练集中现过的did在测试集中占比：{1 - len(did_intersect)/len(set_test_did)}\n"
          f"未在训练集中现过的vid在测试集中占比：{1 - len(vid_intersect)/len(set_test_vid)}")

def compress_bullet_chat_file():
    df = pd.DataFrame({'vid': [], 'content': [], 'ctime': [], 'etime': []})
    for i in range(1, 1 + 20):
        df_tmp = pd.read_excel(base_dir + f'弹幕文本数据/{i}.xlsx', sheet_name='Sheet1',
                               usecols=['videoid','content','ctime','etime']).rename(columns={"videoid": "vid"})
        df = pd.concat([df, df_tmp], axis=0)
        print(i)
    for col in df.columns:
        df[col] = df[col].astype(int) if col != 'content' else df['content'].astype(str).fillna("")
    df.to_parquet(base_dir + f'弹幕文本数据/bullets.parquet')

def analyze_bullet_chat_emotion():
    df = pd.read_parquet(base_dir + '弹幕文本数据/bullet.parquet')
    vids = pd.read_csv(base_dir + 'vid_info/vid_info_table.csv', usecols=['vid'])
    df = df[df['vid'].isin(vids['vid'])]\
        .groupby('vid', as_index=False)\
        .agg(content=('content', '。'.join),
             bullet_cnt=('content', 'size'))
    df['bullet_len'] = df['content'].apply(lambda x: len(x))
    df['model_res'] = None
    classifier = pipeline(Tasks.text_classification, 'iic/nlp_structbert_sentiment-classification_chinese-base')

    def get_emotion(index, row):
        res = classifier(input=row['content'])
        return index, dict(zip(res['labels'], res['scores']))

    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(get_emotion, index, row) for index, row in df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Rows"):
            index, dic = future.result()
            df.loc[index, 'model_res'] = dic

    df.to_parquet(base_dir + '弹幕文本数据/bullet_fea.parquet')
    df.to_csv(base_dir + '弹幕文本数据/bullet_fea.csv', index=False, encoding='utf_8_sig')

def further_analysis_of_bullet_chat():
    df = pd.read_parquet(base_dir + '弹幕文本数据/bullet.parquet')
    vid_info = pd.read_csv(base_dir + 'vid_info/vid_info_table.csv', usecols=['vid','item_duration'])
    df = df[df['vid'].isin(vid_info['vid'])]
    df['bullet_time'] = df['etime'].astype(float) / 1000 / df['item_duration'].astype(float)
    df = df.groupby('vid', sort=False, as_index=False)[['bullet_time']].agg(
        bullet_time_mean = ('bullet_time', 'mean'),
        bullet_time_std = ('bullet_time', 'std')
    ).fillna(0)
    df.to_parquet(base_dir + '弹幕文本数据/bullet_fea_2.parquet')
'''


def merge_and_hash_did_fea():
    def merge_helper(df, col, fill_na, col_name=None):
        return fea_utils.merge_helper(attr_dic, df, 'did', col, fill_na, col_name)

    df = pd.read_csv(base_dir + 'did_features/did_features_table.csv').fillna(0).set_index('did')
    attr_dic = pd.read_pickle(base_dir + '特征补充/train_feature_data_dict.pkl')

    df = merge_helper(df, 'did_active_days', 0.0)
    df = merge_helper(df, 'did_avg_daily_views', 0.0)
    df = merge_helper(df, 'did_click_unique_vid', 0.0)
    df = merge_helper(df, 'did_click_unique_item_cid', 0.0)
    df = merge_helper(df, 'did_click_avg_watch_time', 0.0)
    df = merge_helper(df, 'did_click_total_watch_time', 0.0)
    df = merge_helper(df, 'did_click_most_hour', 'most')
    df = merge_helper(df, 'did_click_hour_std', 0.0)
    df = merge_helper(df, 'did_click_long_play_ratio', 0.0)
    df = merge_helper(df, 'did_click_new_content_ratio', 0.0)
    df.reset_index(inplace=True)

    # 以下did特征需要哈希
    for col in ['f0','f68','f69','f87']:
        df[col] = fea_utils.hash_helper(df, col)
    df['hash_did'] = fea_utils.hash_helper(df, 'did')
    # 删除这个没用的常数特征
    del df['f67']

    df.to_parquet(base_dir + 'did_features/did_fea.parquet', index=False)
    return df


def merge_and_hash_vid_fea():
    def merge_helper(df, key, col, fill_na, col_name=None):
        return fea_utils.merge_helper(attr_dic, df, key, col, fill_na, col_name)

    df = pd.read_csv(base_dir + 'vid_info/vid_info_table.csv')
    attr_dic = pd.read_pickle(base_dir + '特征补充/feature_data_dict.pkl')

    # 加cid统计特征
    df = df.set_index('item_cid')
    df = merge_helper(df, 'item_cid', 'collection_retention_rate', 0.0)
    df = merge_helper(df, 'item_cid', 'collection_vid_diversity', 0.0)
    df = merge_helper(df, 'item_cid', 'collection_avg_completion', 0.0)

    # 加vid统计特征
    df = df.reset_index.set_index('vid')
    df = merge_helper(df, 'vid', 'vid_click_count', 0.0)
    df = merge_helper(df, 'vid', 'vid_avg_play_time', 0.0)
    df = merge_helper(df, 'vid', 'vid_play_time_std', 0.0)
    df = merge_helper(df, 'vid', 'vid_ctr', 0.0)
    df = merge_helper(df, 'vid', 'vid_avg_completion_rate', 0.0)
    df = merge_helper(df, 'vid', 'vid_repeat_rate', 0.0)
    df = merge_helper(df, 'vid', 'vid_rank_in_collection', 0.0)
    # 加弹幕特征
    df = merge_helper(df, 'vid', 'vid_danmu_count', 0.0)
    df = merge_helper(df, 'vid', 'vid_danmu_density', 0.0)
    df = merge_helper(df, 'vid', 'vid_emotion_judgement', 1)
    df = merge_helper(df, 'vid', 'vid_emotion_entropy', 'mean')
    df = merge_helper(df, 'vid', 'high_quality_ratio', 0.0)

    # 再加弹幕特征
    df.reset_index(inplace=True)
    df_tmp = pd.read_parquet(base_dir + '弹幕文本数据/bullet_fea_2.parquet') \
                .rename(columns={"bullet_time_mean": "vid_danmu_time_mean", "bullet_time_std": "vid_danmu_time_std"})
    df = df.merge(df_tmp, how='left', on='vid').fillna(0)
    # 加dist,cos特征
    df_tmp = pd.read_csv(base_dir + '特征补充/cos_dist/train_data_features.csv')
    df = df.merge(df_tmp, how='left', on='vid').fillna(0)

    # 以下vid特征需要哈希
    for col in ['item_cid', 'sid']:
        df[col] = fea_utils.hash_helper(df, col)
    df['hash_vid'] = fea_utils.hash_helper(df, 'vid')

    df.to_parquet(base_dir + 'vid_info/vid_fea.parquet', index=False)
    return df


def merge_history_df():
    for i in range(1, 30+1):
        day_id =  fea_utils.day_id(i)
        df_play = pd.read_csv(base_dir + '用户历史播放数据/' + f'day{day_id}/day{day_id}_data.csv', usecols=['did','vid','play_time'])
        df_click = pd.read_csv(base_dir + '用户历史点击数据/' + f'day{day_id}/day{day_id}_data.csv')
        df_show = pd.read_csv(base_dir + '用户历史曝光数据/' + f'day{day_id}/did_show_data{day_id}.csv')

        df_show.merge(df_click, on=['did','vid'], how='outer') \
               .merge(df_play, on=['did','vid'], how='outer') \
               .sort_values(by=['did', 'vid']).drop_duplicates(subset=['did', 'vid']) \
               .to_csv(base_dir + '用户历史日志/' + f'user_history_day{day_id}.csv', index=False)
        print(f"day{day_id} processed successfully")


def merge_hashed_item_fea_to_history_df():
    select_cols = ['vid', 'hash_vid'] + basic_item_fea
    item_fea = pd.read_parquet(base_dir + 'vid_info/vid_fea.parquet', columns=select_cols)

    def merge_item_fea(i: int):
        day_id = fea_utils.day_id(i)
        input_path = base_dir + f'用户历史日志/user_history_day{day_id}.csv'
        output_path = base_dir + f'用户历史日志_含特征/user_history_day{day_id}.parquet'

        df = pd.read_csv(input_path)
        df = df.merge(item_fea, on='vid', how='left') \
            .drop(columns=['item_cid_x']).rename(columns={'item_cid_y': 'item_cid'})

        df['click'] = (df['click_time'].notna()).astype('int')
        df['ptr'] = (df['play_time']/df['item_duration']).fillna(0).astype('float32')

        df.to_parquet(output_path, index=False)
        return day_id

    futures = []
    with ThreadPoolExecutor(max_workers=None) as executor:
        for i in range(1, 1+30):
            futures.append(executor.submit(merge_item_fea, i))

        for future in as_completed(futures):
            day_id = future.result()
            print(f"day{day_id} processed successfully")



def build_seq_fea_v1(item_fea: List[str], start_idx: int = 1, end_idx: int = 30, init = True):
    log_path = base_dir + f'用户历史序列特征_v1/log1.txt'
    tmp_path = base_dir + f'用户历史序列特征_v1/temp1.pkl'
    if os.path.exists(tmp_path) and not init:
        df_tmp = pd.read_pickle(tmp_path)
    else:
        df_tmp = fea_utils.build_init_df_tmp_v1(item_fea)
        df_tmp.to_pickle(tmp_path)
    print("initiated successfully")

    fea_cols = ['all'] + item_fea

    def update_df_tmp_dict_bind(fea: str):
        if fea == 'all':
            df_tmp_new = fea_utils.update_df_tmp_dict_all(df_today, df_tmp_to_update)
        else:
            df_tmp_new = fea_utils.update_df_tmp_dict_fea(fea, df_today, df_tmp_to_update)
        return df_tmp_new

    def update_df_dict_bind(fea: str):
        if fea == 'all':
            df_new = fea_utils.update_df_dict_all(df_tomorrow_merged)
        else:
            df_new = fea_utils.update_df_dict_fea(fea, df_tomorrow_merged)
        return df_new

    for i in range(start_idx, end_idx):
        today_id, tomorrow_id = fea_utils.day_id(i), fea_utils.day_id(i+1)
        init_path = base_dir + f'用户历史日志_含特征/user_history_day{today_id}.parquet'
        tomorrow_path = base_dir + f'用户历史日志_含特征/user_history_day{tomorrow_id}.parquet'
        output_path = base_dir + f'用户历史序列特征_v1/seq_fea_{tomorrow_id}.parquet'
        df_today = pd.read_parquet(init_path) if i == start_idx else df_tomorrow

        # 筛选今天有交互行为的did，利用df_today中的交互信息更新df_tmp_to_update，进而得到df_tmp_new
        filter_row = df_tmp['did'].isin(df_today['did'])
        df_tmp_unchanged, df_tmp_to_update = df_tmp[~filter_row], df_tmp[filter_row]
        # 最后合并得到全新的df_tmp
        df_tmp_new = df_tmp_to_update[['did']].set_index('did')
        for col in fea_cols:
            df_tmp_new = df_tmp_new.merge(update_df_tmp_dict_bind(col), how='left', left_index=True, right_index=True)
            print(f"metrics group by {col if col!='all' else 'none'} have been updated on {today_id}'s df_tmp successfully")
        df_tmp_new.reset_index(inplace=True)
        df_tmp = pd.concat([df_tmp_new, df_tmp_unchanged], ignore_index=True)

        # 筛选明天的unique did，并拼接上df_tmp的信息，进而计算cnt,ctr,ptr等特征
        df_tomorrow = pd.read_parquet(tomorrow_path).set_index('did')
        df_tomorrow_merged = pd.DataFrame(df_tomorrow.index.drop_duplicates()).merge(df_tmp, on='did', how='left').set_index('did')
        for col in fea_cols:
            df_tomorrow = df_tomorrow.merge(update_df_dict_bind(col), how='left', left_index=True, right_index=True)
            print(f"metrics group by {col if col!='all' else 'none'} have been merged into {tomorrow_id}'s df successfully")
        df_tomorrow.reset_index(inplace=True)
        for col in df_tomorrow.columns:
            if isinstance(df_tomorrow.loc[0,col], dict):
                df_tomorrow[col] = df_tomorrow[col].swifter.apply(json.dumps)

        df_tomorrow.to_parquet(output_path, index=False)

        print("---"*20 + f"day{today_id} is over"+ "---"*20)

        df_tmp.to_pickle(tmp_path)
        with open(log_path, 'a') as f:
            f.write(f"{datetime.now()}: day{today_id} is over\n")

    '''
    # 手动更新最后一天
    tmp_path = base_dir + f'用户历史序列特征_v1/temp1_final.pkl'
    today_id = fea_utils.day_id(30)
    today_path = base_dir + f'用户历史日志_含特征/user_history_day{today_id}.parquet'
    df_today = pd.read_parquet(today_path)
    filter_row = df_tmp['did'].isin(df_today['did'])
    df_tmp_unchanged, df_tmp_to_update = df_tmp[~filter_row], df_tmp[filter_row]
    df_tmp_new = df_tmp_to_update[['did']].set_index('did')
    for col in fea_cols:
        df_tmp_new = df_tmp_new.merge(update_df_tmp_dict_bind(col), how='left', left_index=True, right_index=True)
        print(f"metrics group by {col if col!='all' else 'none'} have been updated on {today_id}'s df_tmp successfully")
    df_tmp_new.reset_index(inplace=True)
    df_tmp = pd.concat([df_tmp_new, df_tmp_unchanged], ignore_index=True)
    # 猥琐地手动操作
    # df_tmp['avg_ctr'] = df_tmp.swifter.apply(lambda row: fea_utils.walson_ratio(row['click_sum'], row['show_sum']), axis=1).astype('float32')
    # df_tmp['avg_ptr'] = df_tmp.swifter.apply(lambda row: fea_utils.walson_ratio(row['play_ratio_sum'], row['click_sum']), axis=1).astype('float32')
    # for feature in fea_cols[1:]:
    #     df_tmp[f'{feature}_ctr'] = df_tmp.swifter.apply(lambda row: fea_utils.get_walson_ratio_dic(row[f'{feature}_click_sum'], row[f'{feature}_show_sum']), axis=1)
    #     df_tmp[f'{feature}_ptr'] = df_tmp.swifter.apply(lambda row: fea_utils.get_walson_ratio_dic(row[f'{feature}_play_ratio_sum'], row[f'{feature}_click_sum']), axis=1)
    # dropped_columns = [col for col in df.columns if col.endswith(('show_sum','play_ratio_sum'))] + ['show_sum']
    # df_tmp = df_tmp.drop(columns=dropped_columns)
    # 保存
    df_tmp.to_pickle(tmp_path)
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now()}: day{today_id} is over\n")
    '''



def build_seq_fea_v2(item_fea: List[str], seq_fea: List[str], start_idx: int = 1, end_idx: int = 30, init = True):
    log_path = base_dir + f'用户历史序列特征_v2/log2.txt'
    tmp_path = base_dir + f'用户历史序列特征_v2/temp2.pkl'
    if os.path.exists(tmp_path) and not init:
        df_tmp = pd.read_pickle(tmp_path)
    else:
        df_tmp = fea_utils.build_init_attr_fea_2(seq_fea)
    print("initiated successfully")

    for i in range(start_idx, end_idx):
        today_id, tomorrow_id = fea_utils.day_id(i), fea_utils.day_id(i+1)
        init_path = base_dir + f'用户历史日志_含特征/user_history_day{today_id}.parquet'
        tomorrow_path = base_dir + f'用户历史序列特征_v1/seq_fea_{tomorrow_id}.parquet'
        output_path = base_dir + f'用户历史序列特征_v2/seq_fea_{tomorrow_id}.parquet'
        df_today = pd.read_parquet(init_path).set_index('did') if i == start_idx else df_tomorrow
        df_today = df_today[df_today['click_time'].notna()]       # 只取今天有点击的行

        # 筛选今天有点击行为的did，利用df_today中的点击物品信息更新df_tmp的点击序列
        filter_row = df_tmp.index.isin(df_today.index)
        df_tmp_unchanged, df_tmp_new = df_tmp[~filter_row], df_tmp[filter_row]
        # 最后合并得到全新的df_tmp
        for col in seq_fea:
            if col != 'item_serialno':
                df_tmp_new = fea_utils.update_df_tmp_seq(col, df_today, df_tmp_new)
            else:
                df_tmp_new = fea_utils.update_df_tmp_serialno_seq(df_today, df_tmp_new)
            print(f"{col}_seq has been updated on {today_id}'s df_tmp successfully")
        df_tmp = pd.concat([df_tmp_new, df_tmp_unchanged])

        # 拼接序列特征
        df_tomorrow = pd.read_parquet(tomorrow_path).set_index('did')
        df_tomorrow = df_tomorrow.merge(df_tmp, how='left', left_index=True, right_index=True)
        for col in seq_fea:
            df_tomorrow[f'{col}_seq'] = df_tomorrow[f'{col}_seq'].apply(json.dumps)
        print(f"seq_feas have been merged into {tomorrow_id}'s df successfully")

        # 添加点击排名靠前的item_fea
        input_cols = [f"{fea}_click_sum" for fea in item_fea]
        df_find_top = df_tomorrow[~df_tomorrow.index.duplicated()][input_cols].copy()
        for col in item_fea:
            df_find_top[f"top_click_{col}"] = df_find_top[f"{col}_click_sum"].swifter.apply(fea_utils.find_top_click_fea)
            print(f"top_click_{col} has been merged into {tomorrow_id}'s df successfully")
        output_cols = [f"top_click_{fea}" for fea in item_fea]
        df_find_top = df_find_top[output_cols]
        df_tomorrow = df_tomorrow.merge(df_find_top, how='left', left_index=True, right_index=True)

        df_tomorrow.reset_index().to_parquet(output_path, index=False)
        print("---"*20 + f"day{today_id} is over"+ "---"*20)
        df_tmp.to_pickle(tmp_path)
        with open(log_path, 'a') as f:
            f.write(f"{datetime.now()}: day{today_id} is over\n")

    ''' 
    tmp_path = base_dir + f'用户历史序列特征_v2/temp2_final.pkl'
    today_id = fea_utils.day_id(30)
    init_path = base_dir + f'用户历史日志_含特征/user_history_day{today_id}.parquet'
    df_today = pd.read_parquet(init_path).set_index('did')
    df_today = df_today[df_today['click_time'].notna()]       
    
    filter_row = df_tmp.index.isin(df_today.index)
    df_tmp_unchanged, df_tmp_new = df_tmp[~filter_row], df_tmp[filter_row]
    for col in seq_fea:
        if col != 'item_serialno':
            df_tmp_new = fea_utils.update_df_tmp_seq(col, df_today, df_tmp_new)
        else:
            df_tmp_new = fea_utils.update_df_tmp_serialno_seq(df_today, df_tmp_new)
        print(f"{col}_seq has been updated on {today_id}'s df_tmp successfully")
    df_tmp = pd.concat([df_tmp_new, df_tmp_unchanged]).reset_index()
    for col in item_fea:
        df_tmp[f"top_click_{col}"] = df_tmp[f"{col}_click_sum"].swifter.apply(fea_utils.find_top_click_fea)
    df_tmp.to_pickle(tmp_path)
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now()}: day{today_id} is over\n")
    '''



def get_bucket_did_fea():
    df = pd.read_parquet(base_dir + 'did_features/did_fea.parquet')
    percentiles = [i / 10 for i in range(1, 10)]
    pct_dic = {}
    for col in df.columns:
        if df[col].dtype == 'float' or \
            col in {'f29','f31','f32','f34','f35','f37','f38','f40','f41','f43','f44','f46','f47','f48','f61',
                    'f63','f64','f66','f71','f72','f74','f76','f79','f80','f83'}:
            pct_dic[col] = df[col].quantile(percentiles).tolist()

    df = fea_utils.get_buketed_fea_df(df, pct_dic)
    df.to_parquet(base_dir + 'did_features/did_fea_bucket.parquet', index=False)


def get_bucket_vid_fea():
    df = pd.read_parquet(base_dir + 'vid_info/vid_fea.parquet')
    percentiles = [i / 10 for i in range(1, 10)]
    pct_dic = {}
    for col in df.columns:
        if df[col].dtype == 'float64' or \
            col in {'item_duration','item_serialno', 'vid_click_count', 'vid_avg_play_time', 'vid_play_time_std',
                    'vid_ctr', 'vid_avg_completion_rate', 'vid_repeat_rate', 'vid_rank_in_collection',
                    'vid_danmu_count', 'vid_danmu_density'}:
            pct_dic[col] = df[col].quantile(percentiles).tolist()
    df = fea_utils.get_buketed_fea_df(df, pct_dic)
    df.to_parquet(base_dir + 'vid_info/vid_fea_bucket.parquet', index=False)


def get_bucket_seq_fea():
    def normalize(ls: List[float]):
        z = sum(ls)
        return list(map(lambda x: x/z, ls)) if z > 0 else []

    for i in range(2, 30 + 1):
        day_id = fea_utils.day_id(i)
        input_path = base_dir + f'用户历史序列特征_v2/seq_fea_{day_id}.parquet'
        output_path = base_dir + f'用户历史序列特征_v2/seq_fea_{day_id}_bucket.parquet'
        df = pd.read_parquet(input_path)
        # 分桶+归一化
        percentiles = [i / 10 for i in range(1, 10)]
        pct_dic = {col: df[col].quantile(percentiles).tolist() for col in ['avg_ctr', 'avg_ptr']}
        df = fea_utils.get_buketed_fea_df(df, pct_dic)
        # 解耦dict字段为k,v
        selected_cols = ['did'] + [col for col in df.columns if col.endswith(('_ctr','_ptr')) and not col.startswith('avg_')]
        df_tmp = pd.DataFrame(df[selected_cols].drop_duplicates()).reset_index(drop=True)
        for col in selected_cols[1:]:
            df_tmp[col] = df_tmp[col].swifter.apply(json.loads)
            df_tmp[f'{col}_fea'] = df_tmp[col].swifter.apply(lambda dic: list(dic.keys()))
            df_tmp[f'{col}_weight'] = df_tmp[col].swifter.apply(lambda dic: normalize(list(dic.values())))   # 权重归一化
            # 删掉，避免后续merge时有重复字段
            del df_tmp[col]
            print(f"day_{day_id} {col} success")

        for col in df_tmp.columns:
            if col != 'did':
                df_tmp[col] = df_tmp[col].swifter.apply(json.dumps)

        df = df.merge(df_tmp, on='did', how='left')
        df.to_parquet(output_path, index=False)
        print("---" * 20 + f"day{day_id} is over" + "---" * 20)



def get_train_data(bucket=False):
    suffix, output_suffix = '_bucket' if bucket else '', '_dl' if bucket else ''
    did_df = pd.read_parquet(base_dir + f'did_features/did_fea{suffix}.parquet')
    vid_df = pd.read_parquet(base_dir + f'vid_info/vid_fea{suffix}.parquet')

    for i in range(2, 30+1):
        day_id = fea_utils.day_id(i)
        seq_fea_path = base_dir + f'用户历史序列特征_v2/seq_fea_{day_id}{suffix}.parquet'
        output_path = base_dir + f'train_data/train_day{day_id}{output_suffix}.parquet'

        dropped_cols = basic_item_fea + ['hash_vid', 'show_sum', 'click_sum']
        seq_df = pd.read_parquet(seq_fea_path).drop(columns=dropped_cols)
        seq_df = seq_df.merge(did_df, on='did', how='left').merge(vid_df, on='vid', how='left')

        # 补充内容
        # ptr过滤
        seq_df['ptr'] = seq_df['ptr'].apply(lambda x: x if x > 0.01 else 0)
        # 额外构造标签 first_click
        seq_df = fea_utils.get_is_first_click(seq_df)
        # 额外构造字段 is_cid_next_episode
        seq_df = fea_utils.get_is_cid_next_episode(seq_df)

        seq_df.to_parquet(output_path, index=False)
        print("---" * 20 + f"day{day_id} is over" + "---" * 20)


def get_test_data(bucket=False):
    suffix, output_suffix = '_bucket' if bucket else '', '_dl' if bucket else ''
    input_path = base_dir + 'A榜用户曝光数据/testA_did_show.csv'
    output_path = base_dir + f'test_data/test_A{output_suffix}.parquet'
    base_df = pd.read_csv(input_path)
    did_df = pd.read_parquet(base_dir + f'did_features/did_fea{suffix}.parquet')
    vid_df = pd.read_parquet(base_dir + f'vid_info/vid_fea{suffix}.parquet')
    seq_df_1 = pd.read_pickle(base_dir + f'用户历史序列特征_v1/temp1_final.pkl')
    seq_df_2 = pd.read_pickle(base_dir + f'用户历史序列特征_v2/temp2_final.pkl').reset_index()
    seq_df = seq_df_1.merge(seq_df_2, on='did', how='left')
    del seq_df_1, seq_df_2
    # json_dumps
    for col in seq_df.columns:
        if col not in {'did','avg_ctr','avg_ptr'}:
            seq_df[col] = seq_df[col].swifter.apply(json.dumps)
    # merge
    df = base_df.merge(seq_df, on='did', how='left').merge(did_df, on='did', how='left').merge(vid_df, on='vid', how='left')
    del base_df, vid_df, did_df
    # 补top特征
    for col in item_fea:
        df[f"top_click_{col}"] = df[f"{col}_click_sum"].swifter.apply(fea_utils.find_top_click_fea)
    # 补字段 is_cid_next_episode
    df = fea_utils.get_is_cid_next_episode(df)
    # 保存
    df.to_parquet(output_path, index=False)
    print(df.shape)



if __name__ == '__main__':
    pass
    # df = merge_and_hash_did_fea()
    # print(df.head(10))
    # df = merge_and_hash_vid_fea()
    # print(df.head(10))
    # merge_hashed_item_fea_to_history_df()
    # build_seq_fea_v1(item_fea, 1, 30)
    # build_seq_fea_v2(item_fea, seq_fea, 1, 30)
    # get_bucket_did_fea()
    # get_bucket_vid_fea()
    # get_bucket_seq_fea()
    # get_train_data(False);get_train_data(True)
    # get_test_data(False);get_test_data(True)






