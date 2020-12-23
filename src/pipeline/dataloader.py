import copy
import time
from functools import reduce

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Dataloader(object):
    def __init__(self):
        abs_file = __file__
        seg_symbol = '/'
        if abs_file.rfind('\\') is not -1:
            seg_symbol = '\\'
        abs_path = abs_file[:abs_file.rfind(seg_symbol)] + '/../data/'

        self.file_path = abs_path + 'round1_ijcai_18_train_20180301.txt'
        self.csv_path = abs_path + 'all_data.csv'
        # self.train_csv_path = abs_path + 'train_data.csv'
        # self.test_csv_path = abs_path + 'test_data.csv'
        self.rand_seed = 42
        self.test_ratio = 0.2
        self.ori_column = ['instance_id', 'item_id', 'item_category_list', 'item_property_list', 'item_brand_id',
                           'item_city_id',
                           'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id',
                           'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id',
                           'context_timestamp',
                           'context_page_id', 'predict_category_property', 'shop_id', 'shop_review_num_level',
                           'shop_review_positive_rate',
                           'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                           'is_trade']
        self.long_type_column = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'context_id',
                                 'context_timestamp',
                                 'shop_id']
        self.int_type_column = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                                'item_price_level',
                                'item_sales_level', 'item_collected_level', 'item_pv_level', 'shop_review_num_level',
                                'shop_star_level', 'is_trade']
        self.double_type_column = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
                                   'shop_score_description']
        self.item_column = ['item_id', 'item_category_list', 'item_brand_id', 'item_city_id', 'item_price_level',
                            'item_sales_level', 'item_collected_level', 'item_pv_level']
        self.user_column = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
        self.context_column = ['context_id', 'context_timestamp', 'context_page_id', 'predict_category_property']
        self.shop_column = ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
                            'shop_score_service', 'shop_score_delivery', 'shop_score_description']
        self.encode_column = ['shop_id', 'item_id', 'user_id', 'item_brand_id', 'item_city_id', 'user_gender_id',
                              'item_property_list', 'predict_category_property',
                              'user_occupation_id', 'context_page_id', 'predict_query_1', 'predict_query']
        self.drop_column = ['instance_id', 'context_id', 'context_timestamp']
        self.continuous_column = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
                                  'shop_score_description']

    def load_file(self):
        ori_data = pd.read_csv(self.file_path, sep='\s+')
        ori_data[self.long_type_column] = ori_data[self.long_type_column].astype('long')
        ori_data[self.int_type_column] = ori_data[self.int_type_column].astype('int')
        ori_data[self.double_type_column] = ori_data[self.double_type_column].astype('double')
        data = copy.copy(ori_data)
        data = data.drop_duplicates(subset='instance_id')
        data = data.reset_index(drop=True)
        data['context_timestamp'] = data['context_timestamp'].apply(
            lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
        real_time = pd.to_datetime(data['context_timestamp'])
        data['day'] = real_time.dt.day
        data['hour'] = real_time.dt.hour

        self.ori_data = ori_data
        self.data = data

    def pipeline(self):
        self.__fill_missing_value()
        self.__process_category()
        self.__preprocess_star()
        self.__encode()
        self.data = self.data.drop(columns=self.drop_column)
        for col in self.continuous_column:
            df = self.data[col]
            self.data = self.data.drop(columns=[col])
            self.data.insert(0, col, df)

    def save_to_csv(self):
        self.data.to_csv(self.csv_path, index=False)

    def load_from_csv(self):
        self.data = pd.read_csv(self.csv_path)

    def spilt_train_test(self):
        feature = self.data.drop(columns=['is_trade'])
        target = self.data['is_trade'].astype('float')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(feature, target,
                                                                                random_state=self.rand_seed)

    def __fill_missing_value(self):
        for col in self.int_type_column:
            self.__fill_with_mode(col)
        for col in self.long_type_column:
            self.__fill_with_mode(col)
        for col in self.double_type_column:
            self.__fill_with_mean(col)

    def __fill_with_mode(self, column_name):
        mode = self.data[column_name].mode()
        self.data[column_name] = self.data[column_name].replace(-1, mode[0])

    def __fill_with_mean(self, column_name):
        mean = self.data[column_name].mean()
        self.data[column_name] = self.data[column_name].replace(-1.0, mean)

    def __process_category(self):
        data = self.data
        data['same_cate'] = data.apply(same_cate, axis=1)  # 相同类别数
        data['same_property'] = data.apply(same_property, axis=1)  # 相同属性数
        data['property_num'] = data['item_property_list'].apply(lambda x: len(x.split(';')))  # 属性的数目
        data['pred_cate_num'] = data['predict_category_property'].apply(lambda x: len(x.split(';')))  # query的类别数目

        def f(x):
            try:
                return len([i for i in reduce((lambda x, y: x + y), [i.split(':')[1].split(',') for i in x.split(';') if
                                                                     len(i.split(':')) > 1]) if i != '-1'])
            except:
                return 0

        data['pred_prop_num'] = data['predict_category_property'].apply(f)  # query的属性数目
        data['predict_query_1'] = data['predict_category_property'].apply(
            lambda x: x.split(';')[0].split(':')[0])  # query第一个类别
        data['predict_query'] = data['predict_category_property'].apply(
            lambda x: '-'.join(sorted([i.split(':')[0] for i in [i for i in x.split(';')]])))  # query的全部类别
        data['item_category_list'] = data['item_category_list'].apply(lambda x: x.split(';')[1])
        self.data = data

    def __preprocess_star(self):
        data = self.data
        data['user_age_level'] = data['user_age_level'] - 1000
        data['user_star_level'] = data['user_star_level'] - 3000
        data['shop_star_level'] = data['shop_star_level'] - 5000
        self.data = data

    def __encode(self):
        data = self.data
        for feature in self.encode_column:
            data[feature] = LabelEncoder().fit_transform(data[feature])
        self.data = data


def same_cate(x):
    cate = set(x['item_category_list'].split(';'))
    cate2 = set([i.split(':')[0] for i in x['predict_category_property'].split(';')])
    return len(cate & cate2)


def same_property(x):
    property_a = set(x['item_property_list'].split(';'))
    a = []
    for i in [i.split(':')[1].split(',') for i in x['predict_category_property'].split(';') if
              len(i.split(':')) > 1]:
        a += i
    property_b = set(a)
    return len(property_a & property_b)


if __name__ == '__main__':
    data_loader = Dataloader()

    data_loader.load_file()
    data_loader.pipeline()
    data_loader.save_to_csv()

    data_loader.load_from_csv()
    data_loader.spilt_train_test()
    data = data_loader.data
    print(data.columns)
