
''' ------- 准备数据 ------
    def sample 
    输入 input1 vocab/user_vocab.pkl, input2 dataset/rec_train.pkl
    输出有history点击的data
'''
from collections import Counter
import numpy as np
import pandas as pd

from utils import const


class Sampler(object):
    def __init__(self, data_path, search, user_vocab) -> None:
        self.search = search
        self.user_vocab = user_vocab #读用户字典-input1 vocab/user_vocab.pkl
        raw_data = pd.read_pickle(data_path) #读数据表-input2 dataset/rec_train.pkl
        raw_data = self._filter_zero_entries(raw_data)
        raw_data = self._filter_short_users(raw_data)
        self.virtual_rec_items = self._build_virtual_rec_items(raw_data)
        self.virtual_src_sessions = self._build_virtual_src_sessions(user_vocab)
        if {'rec_his', 'src_session_his'}.issubset(raw_data.columns):
            rec_zero = int((raw_data['rec_his'] <= 0).sum())
            src_zero = int((raw_data['src_session_his'] <= 0).sum())
            if rec_zero > 0 or src_zero > 0:
                print(f"[Sampler] {data_path} rec_his为0 {rec_zero} 条, "
                      f"src_session_his为0 {src_zero} 条，将确保至少1条非0历史")
            if rec_zero > 0 and not self.virtual_rec_items:
                print(f"[Sampler] {data_path} 虚拟rec历史为空，无法填充rec_his")
            if src_zero > 0 and not self.virtual_src_sessions:
                print(f"[Sampler] {data_path} 虚拟src历史为空，无法填充src_session_his")
        self.data = raw_data


    def sample(self, index): 
        '''
        DATA格式 for rec
            user_id  item_id     timestamp  click  rec_his  src_session_his  src_his   all_his   g_s_id  neg_items  
        0    10144   237343  1.684739e+12      1        5                1        1       6    2497469  [427877, 121932, 435611, 335915]  
        1    14790   563181  1.684739e+12      1        1                1        1       2    3045286  [215542, 472737, 60392, 357025]  
        DATA格式 for search 
            加search_session_id & keyword 
            user_id  item_id     timestamp  search_session_id   keyword                         click  rec_his  src_session_his  src_his   all_his   g_s_id  neg_items  
        0    8495    237343  1.684739e+12          256     [140663, 272949, 348145, 238582]       1        5                1        1       6    2497469  [427877, 121932, 435611, 335915]  
        1    8495    563181  1.684739e+12          256     [140663, 272949, 348145, 238582]       1        1                1        1       2    3045286  [215542, 472737, 60392, 357025]    
        一个用户一个item一行，取neg_items
        src_his # 该用户有n条搜索历史  
        rec_his # 该用户有n条推荐历史
        src_session_his # 该用户有n条搜索session历史         
        '''

        feed_dict = {}
        line = self.data.iloc[index] #取一行data 有’user_id‘，’item_id‘，’neg_items‘等字段

        user = int(line['user_id'])
        feed_dict['user'] = [user] 
        feed_dict['item'] = [int(line['item_id'])]
        feed_dict['neg_items'] = [line['neg_items']]
        feed_dict['search'] = self.search
        # 调试：仅在出现越界时输出
        item_val = feed_dict['item'][0]
        neg_vals = feed_dict['neg_items'][0]
        if item_val >= const.item_id_num or item_val < 0:
            print(f"[Sampler] idx={index} item越界:{item_val}")
        if neg_vals:
            neg_min, neg_max = min(neg_vals), max(neg_vals)
            if neg_min < 0 or neg_max >= const.item_id_num:
                print(f"[Sampler] idx={index} neg_items范围异常 min={neg_min} max={neg_max}")
        if self.search:
            query = self.get_pad_query(line['keyword']) # query的padding
            feed_dict['query'] = list([query])
        rec_his_num = int(line['rec_his'])
        src_session_his_num = int(line['src_session_his'])
        
        feed_dict.update(
            self.get_all_his(user, rec_his_num,
                             src_session_his_num)) # 读取 user_vocab[user] 中保存的所有历史轨迹信息

        return feed_dict

    def get_all_his(self, user, rec_his_num, src_his_num):
        # --recommend 
        # 读取user_vocab[user] 中当前历史个数rec_his_num 保留max_rec_his_len个
        rec_his_item = self.user_vocab[user]['rec_his'][:rec_his_num][
            -const.max_rec_his_len:] if rec_his_num > 0 else []
        rec_his_ts = self.user_vocab[user]['rec_his_ts'][:rec_his_num][
            -const.max_rec_his_len:] if rec_his_num > 0 else []
        # 长度不足时padding
        if len(rec_his_item) < const.max_rec_his_len:
            rec_his_item += [0] * (const.max_rec_his_len - len(rec_his_item)) #padding
            rec_his_ts += [np.inf] * (const.max_rec_his_len - len(rec_his_ts))
        rec_his_item, rec_his_ts = self._ensure_min_one_nonzero(
            rec_his_item, rec_his_ts, const.max_rec_his_len, source='rec')
        rec_his_type = [1] * len(rec_his_item)
        rec_his = list(zip(rec_his_item, rec_his_ts, rec_his_type))

        # --search
        # 读取user_vocab[user] 中当前历史session个数src_his_num 保留max_src_session_his_len个
        src_his_item = self.user_vocab[user]['src_session_his'][:src_his_num][
            -const.max_src_session_his_len:] if src_his_num > 0 else []
        src_his_ts = self.user_vocab[user]['src_session_his_ts'][:src_his_num][
            -const.max_src_session_his_len:] if src_his_num > 0 else []
        # 长度不足时padding
        if len(src_his_item) < const.max_src_session_his_len:
            src_his_item += [0] * \
                (const.max_src_session_his_len - len(src_his_item))
            src_his_ts += [np.inf] * \
                (const.max_src_session_his_len - len(src_his_ts))
        src_his_item, src_his_ts = self._ensure_min_one_nonzero(
            src_his_item, src_his_ts, const.max_src_session_his_len, source='src')
        src_his_type = [2] * len(src_his_item)
        src_his = list(zip(src_his_item, src_his_ts, src_his_type))

        all_his = rec_his + src_his

        sorted_all_his = sorted(all_his, key=lambda x: x[1])
        sorted_all_his_item = [x[0] for x in sorted_all_his]
        sorted_all_his_time = [x[1] for x in sorted_all_his]
        sorted_all_his_type = [x[2] for x in sorted_all_his]
        # 调试：all_his 范围
        if sorted_all_his_item:
            ah_min, ah_max = min(sorted_all_his_item), max(sorted_all_his_item)
            if ah_min < 0 or ah_max >= const.item_id_num:
                print(f"[Sampler] all_his item 越界 min={ah_min} max={ah_max}")

        return {
            "all_his": [sorted_all_his_item],
            "all_his_ts": [sorted_all_his_time],
            "all_his_type": [sorted_all_his_type]
        }
        # -----Output----
        #all_his 所有历史item ID（search+rec混合）
        #all_his_ts 历史item时间戳（升序排列）
        #all_his_type 历史item类型（1=rec, 2=search）
        #

    '''query的padding'''
    def get_pad_query(self, query): 
        if type(query) == str:
            query = eval(query)
        if type(query) == int:
            query = [query]
        query = query[:const.max_query_word_len]
        if len(query) < const.max_query_word_len:
            query += [0] * (const.max_query_word_len - len(query))
        return query

    def _build_virtual_rec_items(self, raw_data):
        if 'item_id' not in raw_data.columns:
            return []
        counts = raw_data['item_id'].value_counts()
        popular_items = [int(i) for i in counts.index if int(i) > 0]
        return popular_items

    def _build_virtual_src_sessions(self, user_vocab):
        counter = Counter()
        for _, info in user_vocab.items():
            src_list = info.get('src_session_his', [])
            for sid in src_list:
                if int(sid) > 0:
                    counter[int(sid)] += 1
        if not counter:
            return []
        popular_sessions = [sid for sid, _ in counter.most_common()]
        return popular_sessions

    def _get_virtual_items(self, num_needed, source):
        if source == 'rec':
            pool = self.virtual_rec_items
        elif source == 'src':
            pool = self.virtual_src_sessions
        else:
            raise ValueError(f"unknown virtual history source: {source}")
        if not pool:
            raise ValueError(f"virtual {source} history is empty")
        items = []
        for i in range(num_needed):
            items.append(pool[i % len(pool)])
        return items

    def _ensure_min_one_nonzero(self, items, ts, max_len, source):
        has_nonzero = any(int(it) > 0 for it in items)
        if has_nonzero:
            return items, ts
        virtual_item = self._get_virtual_items(1, source)[0]
        items = list(items)
        ts = list(ts)
        items[0] = virtual_item
        ts[0] = -1.0
        if len(items) != max_len:
            raise ValueError(f"{source} history length mismatch: {len(items)}/{max_len}")
        return items, ts

    def _filter_zero_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除含有全零 embedding 的点击 / query，避免后续 gate 将其当作有效信号。
        - 搜索点击 item: item_id == 0 直接删除
        - 搜索 query: keyword 全 0（或为空列表）删除
        - 推荐侧 query（若存在 keyword 字段）同样删除
        """
        before = len(df)
        if 'item_id' in df.columns:
            df = df[df['item_id'] != 0]

        if 'keyword' in df.columns:
            def _valid_kw(kw):
                try:
                    if isinstance(kw, str):
                        kw = eval(kw)
                    if isinstance(kw, int):
                        kw = [kw]
                    if not isinstance(kw, (list, tuple)):
                        return True
                    return any(int(x) != 0 for x in kw)
                except Exception:
                    return True
            df = df[df['keyword'].apply(_valid_kw)]

        removed = before - len(df)
        if removed > 0:
            print(f\"[Sampler] 过滤零 embedding 行 {removed}/{before} 条 -> {len(df)}\")
        return df

    def _filter_short_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除总历史（rec+src）长度 < 5 的用户对应的所有样本。"""
        before = len(df)
        if 'user_id' not in df.columns:
            return df

        def _total_len(uid: int) -> int:
            info = self.user_vocab.get(uid, {})
            rec_len = len(info.get('rec_his', []))
            src_len = len(info.get('src_session_his', []))
            return rec_len + src_len

        valid_users = {uid for uid in df['user_id'].unique()
                       if _total_len(int(uid)) >= 5}
        df = df[df['user_id'].isin(valid_users)]

        removed = before - len(df)
        if removed > 0:
            print(f\"[Sampler] 删除历史<5的用户样本 {removed}/{before} 条，剩余 {len(df)}\")
        return df
