import copy
import datetime

import numpy as np
import pandas as pd


class My_DataLoader(object):
    def __init__(self):
        self.rand_seed = 42
        self.window_size = 7

        abs_file = __file__
        seg_symbol = '/'
        # if abs_file.rfind('\\') is not -1:
        #     seg_symbol = '\\'
        self.root_path = abs_file[:abs_file.rfind(seg_symbol)] + '/../'
        self.abs_path = self.root_path + 'data/'
        self.processed_path = self.abs_path + 'processed/'
        self.D_path = self.abs_path + 'time_series_covid19_confirmed_global.csv'
        self.A_path = self.abs_path + 'time_series_covid19_deaths_global.csv'
        self.R_path = self.abs_path + 'time_series_covid19_recovered_global.csv'
        self.additional_path = self.abs_path + 'owid-covid-data.csv'

        # additional_path = '/tf/Lecture/Covid_19_Prediction_Stacking/data/owid-covid-data.csv'
        # D_path = '/tf/Lecture/Covid_19_Prediction_Stacking/data/time_series_covid19_confirmed_global.csv'
        # A_path = '/tf/Lecture/Covid_19_Prediction_Stacking/data/time_series_covid19_deaths_global.csv'
        # R_path = '/tf/Lecture/Covid_19_Prediction_Stacking/data/time_series_covid19_recovered_global.csv'

        hard_hit_country = ['US', 'India', 'Brazil', 'Russia', 'France', 'Italy']
        relapse_country = ['Italy', 'Germany', 'Denmark', 'Switzerland', 'Russia', 'Austria']
        lack_data_countries = ['Brazil', 'Germany']
        # self.interested_countries = list((set(hard_hit_country) | set(relapse_country)) - set(lack_data_countries))
        self.interested_countries = list(set(relapse_country) - set(lack_data_countries))

        self.good_features = [
            # 'stringency_index',
            'population',
            'population_density',
            'median_age',
            'aged_65_older',
            'aged_70_older',
            'gdp_per_capita',
            # 'extreme_poverty',
            'cardiovasc_death_rate',
            'diabetes_prevalence',
            'male_smokers',
            'female_smokers',
            'hospital_beds_per_thousand',
            'life_expectancy',
            'human_development_index']

        self.good_time_sequences_features = ['new_tests', 'new_tests_per_thousand',
                                             'total_tests', 'total_tests_per_thousand', 'positive_rate']
        self.input_features = [
            'date',
            # 'start_days',
            # 'date_days'
            'total_cases',
            # 'total_cases_smoothed',
            # 'total_cases_per_million',

            'new_cases',
            # 'new_cases_smoothed',
            # 'new_cases_per_million',
            # 'new_cases_smoothed_per_million',
            'total_deaths',
            'new_deaths',
            # 'new_deaths_smoothed',
            # 'new_deaths_per_million',
            # 'new_deaths_smoothed_per_million',
            'total_recovered',
            'new_recovered',
            # 'new_recovered_per_million',
            'total_tests',
            'new_tests',
            # 'new_tests_per_thousand',
            # 'total_tests_per_thousand',
            'death_rate',
            'recovery_rate',
            'positive_rate',

            'current_patients',
            'day_of_the_week'
        ]

    def load_file(self):
        addition_data = pd.read_csv(self.additional_path)
        addition_data['location'].replace('United States', 'US', inplace=True)
        addition_data = addition_data[addition_data['location'].isin(self.interested_countries)].reset_index(drop=True)
        addition_data['datetime'] = 1

        addition_data = addition_data.apply(generate_date_time, axis=1)

        addition_seq_df = addition_data[['location', 'date'] + self.good_time_sequences_features]
        addition_seq_df = addition_seq_df[addition_data['total_cases'] > 0].reset_index(drop=True)

        feature = 'new_tests'
        for country in self.interested_countries:
            # print(country)
            t_data = addition_seq_df[addition_seq_df['location'] == country][feature]
            # print(t_data.isnull().value_counts())
            # print(t_data.value_counts())
            t_filled_data = t_data.fillna(method='ffill').fillna(t_data.min())
            # print(t_filled_data.value_counts())
            # for i, v in enumerate(t_filled_data):
            #     print(i, v)

            for i in t_filled_data.index.values:
                addition_seq_df.at[i, feature] = t_filled_data[i]

            # t_data = addition_seq_df[addition_seq_df['location']==country][feature]
            # print(t_data.isnull().value_counts())

        # feature = 'tests_per_case'
        # feature_df = interested_data[['location', 'date', feature]]
        # feature_df = feature_df.reset_index()
        # feature_with_data = feature_df[feature_df[feature].isnull()==False]
        # feature_with_data['location'].value_counts()

        D_data = pd.read_csv(self.D_path)
        D_interested = D_data[
            (D_data['Country/Region'].isin(self.interested_countries)) & (D_data['Province/State'].isnull())]
        D_interested = D_interested.reset_index(drop=True).drop(columns=['Province/State', 'Lat', 'Long'])
        D_interested = D_interested.rename(columns={'Country/Region': 'location'})
        D_interested = D_interested.set_index('location')
        D_interested = D_interested.transpose()

        A_data = pd.read_csv(self.A_path)
        A_interested = A_data[
            (A_data['Country/Region'].isin(self.interested_countries)) & (A_data['Province/State'].isnull())]
        A_interested = A_interested.reset_index(drop=True).drop(columns=['Province/State', 'Lat', 'Long'])
        A_interested = A_interested.rename(columns={'Country/Region': 'location'})
        A_interested = A_interested.set_index('location')
        A_interested = A_interested.transpose()

        R_data = pd.read_csv(self.R_path)
        R_interested = R_data[
            (R_data['Country/Region'].isin(self.interested_countries)) & (R_data['Province/State'].isnull())]
        R_interested = R_interested.reset_index(drop=True).drop(columns=['Province/State', 'Lat', 'Long'])
        R_interested = R_interested.rename(columns={'Country/Region': 'location'})
        R_interested = R_interested.set_index('location')
        R_interested = R_interested.transpose()

        demography_data = addition_data[['location'] + self.good_features].drop_duplicates().reset_index(
            drop=True).set_index(
            'location')
        demography_data['first_discovery_date'] = '1/22/20'
        # country = 'France'
        # D_interested[country][D_interested[country] > 0.0].index[0]
        for country in self.interested_countries:
            first_D_data = D_interested[country][D_interested[country] > 0.0].index[0]
            demography_data.at[country, 'first_discovery_date'] = first_D_data

        self.countries_data = {}
        self.norm_countries_data = {}
        self.norm_param_countries_data = {}
        for country in self.interested_countries:
            discovery_data = demography_data['first_discovery_date'][country]
            days = len(D_interested[country][discovery_data:])
            df = pd.DataFrame(np.zeros(shape=(days, len(self.input_features))), columns=self.input_features)

            df['date'] = D_interested[country][discovery_data:].index.values
            df['total_cases'] = D_interested[country][discovery_data:].values.astype('float')
            df['total_deaths'] = A_interested[country][discovery_data:].values.astype('float')
            df['total_recovered'] = R_interested[country][discovery_data:].values.astype('float')
            t_addition = addition_seq_df[addition_seq_df['location'] == country]
            for i in range(days):
                df.at[i, 'new_tests'] = t_addition[t_addition['date'] == df['date'][i]]['new_tests'].astype('float')

            df.at[0, 'new_cases'] = df['total_cases'][0]
            df.at[0, 'new_deaths'] = df['total_deaths'][0]
            df.at[0, 'new_recovered'] = df['total_recovered'][0]
            df.at[0, 'total_tests'] = df['new_tests'][0]
            df.at[0, 'death_rate'] = 0
            df.at[0, 'recovery_rate'] = 0
            df.at[0, 'positive_rate'] = 0

            for i in range(1, days):
                df.at[i, 'new_cases'] = df['total_cases'][i] - df['total_cases'][i - 1]
                df.at[i, 'new_deaths'] = df['total_deaths'][i] - df['total_deaths'][i - 1]
                df.at[i, 'new_recovered'] = df['total_recovered'][i] - df['total_recovered'][i - 1]
                df.at[i, 'total_tests'] = df['total_tests'][i - 1] + df['new_tests'][i]

                df.at[i, 'current_patients'] = df['total_cases'][i - 1] - df['total_deaths'][i - 1] - \
                                               df['total_recovered'][i - 1]

                df.at[i, 'death_rate'] = df['new_deaths'][i] / df.at[i, 'current_patients']
                df.at[i, 'recovery_rate'] = df['new_recovered'][i] / df.at[i, 'current_patients']

            df['positive_rate'] = df['new_cases'] / df['new_tests']

            df = df.fillna(0.0)

            df = df.apply(add_day_of_the_week, axis=1)

            norm_df = copy.copy(df).drop(columns=['day_of_the_week'])
            norm_df['day_of_the_week_SIN'] = np.sin(2.0 * np.pi * (df['day_of_the_week'] / 6.0))
            norm_df['day_of_the_week_COS'] = np.cos(2.0 * np.pi * (df['day_of_the_week'] / 6.0))
            norm_param_df = pd.DataFrame(np.zeros(shape=(2, norm_df.shape[1])), columns=norm_df.columns.values)
            for col in norm_df.columns.values:
                if col == 'date' or col == 'day_of_the_week_SIN' or col == 'day_of_the_week_COS':
                    continue
                mean = df[col].mean()
                std = df[col].std()
                norm_param_df.at[0, col] = mean
                norm_param_df.at[1, col] = std
                norm_df[col] = (df[col] - mean) / std

            df['day_of_the_week_SIN'] = np.sin(2.0 * np.pi * (df['day_of_the_week'] / 6.0))
            df['day_of_the_week_COS'] = np.cos(2.0 * np.pi * (df['day_of_the_week'] / 6.0))


            self.countries_data[country] = df
            self.norm_countries_data[country] = norm_df
            self.norm_param_countries_data[country] = norm_param_df

    # def pipeline(self):

    def save_to_csv(self):
        for country in self.interested_countries:
            path = self.processed_path + 'processed_{}.csv'.format(country)
            self.countries_data[country].to_csv(path, index=False)

            norm_path = self.processed_path + 'processed_norm_{}.csv'.format(country)
            self.norm_countries_data[country].to_csv(norm_path, index=False)

            norm_param_path = self.processed_path + 'processed_norm_param_{}.csv'.format(country)
            self.norm_param_countries_data[country].to_csv(norm_param_path, index=False)

    def load_from_csv(self):
        self.countries_data = {}
        self.norm_countries_data = {}
        self.norm_param_countries_data = {}
        for country in self.interested_countries:
            path = self.processed_path + 'processed_{}.csv'.format(country)
            df = pd.read_csv(path)
            self.countries_data[country] = df

            norm_path = self.processed_path + 'processed_norm_{}.csv'.format(country)
            norm_df = pd.read_csv(norm_path)
            self.norm_countries_data[country] = norm_df

            norm_param_path = self.processed_path + 'processed_norm_param_{}.csv'.format(country)
            norm_param_df = pd.read_csv(norm_param_path)
            self.norm_param_countries_data[country] = norm_param_df


def generate_date_time(row):
    date_str = row['date'].split('-')
    year = int(date_str[0])
    month = int(date_str[1])
    day = int(date_str[2])
    date_time = int(datetime.date(year, month, day).strftime('%j'))
    if year > 2020:
        date_time += 366
    row['datetime'] = date_time
    row['date'] = '{}/{}/{}'.format(month, day, year - 2000)
    return row


def add_day_of_the_week(row):
    date_str = row['date'].split('/')
    year = int(date_str[2]) + 2000
    month = int(date_str[0])
    day = int(date_str[1])
    # day_of_the_week_str = datetime.date(year, month, day).strftime('%a')
    day_of_the_week_str = datetime.date(year, month, day).strftime('%w')
    day_of_the_week = int(day_of_the_week_str)
    row['day_of_the_week'] = day_of_the_week
    return row


if __name__ == '__main__':
    data_loader = My_DataLoader()

    data_loader.load_file()
    # data_loader.pipeline()
    data_loader.save_to_csv()

    data_loader.load_from_csv()
    countries_data = data_loader.countries_data
    norm_countries_data = data_loader.norm_countries_data
    print(len(countries_data))
