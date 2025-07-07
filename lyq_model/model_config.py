labels = ['click', 'first_click', 'ptr']

sequence_features = ['hash_vid_seq', 'sid_seq', 'item_cid_seq']
sequence_features_map = {fea: fea[:-4] for fea in sequence_features}

category_features = ['hash_did','hash_vid','avg_ctr','avg_ptr',
'f0','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11',
'f12','f13','f14','f15','f16','f17','f18','f19','f20','f21',
'f22','f23','f24','f25','f26','f27','f28','f29','f30','f31',
'f32','f33','f34','f35','f36','f37','f38','f39','f40','f41',
'f42','f43','f44','f45','f46','f47','f48','f49','f50','f51',
'f52','f53','f54','f55','f56','f57','f58','f59','f60','f61',
'f62','f63','f64','f65','f66','f68','f69','f70','f71','f72',
'f73','f74','f75','f76','f77','f78','f79','f80','f81','f82',
'f83','f84','f85','f86','f87','did_active_days',
'did_avg_daily_views','did_click_unique_vid',
'did_click_unique_item_cid','did_click_avg_watch_time',
'did_click_total_watch_time','did_click_most_hour',
'did_click_hour_std','did_click_long_play_ratio',
'did_click_new_content_ratio','item_cid','item_type',
'item_duration','item_assetSource','item_classify',
'item_isIntact','item_serialno','sid','stype',
'collection_retention_rate','collection_vid_diversity',
'collection_avg_completion','vid_click_count',
'vid_avg_play_time','vid_play_time_std','vid_ctr',
'vid_avg_completion_rate','vid_repeat_rate',
'vid_rank_in_collection','vid_danmu_count',
'vid_danmu_density','vid_emotion_judgement',
'vid_emotion_entropy','high_quality_ratio',
'vid_danmu_time_mean','vid_danmu_time_std',
'vid_audience_cos_dist','vid_audience_var_dist',
'is_cid_next_episode']


dict_features = {'item_cid_ctr_fea':'item_cid_ctr_weight',
'item_cid_ptr_fea':'item_cid_ptr_weight',
'item_type_ctr_fea':'item_type_ctr_weight',
'item_type_ptr_fea':'item_type_ptr_weight',
'item_assetSource_ctr_fea':'item_assetSource_ctr_weight',
'item_assetSource_ptr_fea':'item_assetSource_ptr_weight',
'item_classify_ctr_fea':'item_classify_ctr_weight',
'item_classify_ptr_fea':'item_classify_ptr_weight',
'item_isIntact_ctr_fea':'item_isIntact_ctr_weight',
'item_isIntact_ptr_fea':'item_isIntact_ptr_weight',
'stype_ctr_fea':'stype_ctr_weight',
'stype_ptr_fea':'stype_ptr_weight'}
dict_features_map = {fea: fea[:-8] for fea in dict_features}



DEFAULT_EMB_NUM = 100
emb_num_dict = {'f68':500,'f69':500,'item_cid':2500,
'hash_did':1500000,'f0':160000,'hash_vid':17000,'sid':20000}

DEFAULT_EMB_DIM = 8
emb_dim_dict = {'hash_vid':16,'hash_did':16,'sid':16,'item_cid':16}

CATEGORY_FEA_TOTAL_EMB_DIM = sum([emb_dim_dict.get(fea, DEFAULT_EMB_DIM) for fea in category_features])
DICT_FEA_TOTAL_EMB_DIM = sum([emb_dim_dict.get(dict_features_map[fea], DEFAULT_EMB_DIM) for fea in dict_features_map])
SEQ_FEA_TOTAL_EMB_DIM = sum([emb_dim_dict.get(sequence_features_map[fea], DEFAULT_EMB_DIM) for fea in sequence_features])
TOTAL_EMB_DIM = CATEGORY_FEA_TOTAL_EMB_DIM + DICT_FEA_TOTAL_EMB_DIM + SEQ_FEA_TOTAL_EMB_DIM


bottom_hidden_dims = [512, 256, 188, 128]
tower_hidden_dims = [96, 64, 16]


id_features = ['hash_vid','hash_did']
