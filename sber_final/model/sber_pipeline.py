import dill
import pandas as pd
import requests
import datetime
from requests.adapters import HTTPAdapter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from urllib3 import Retry


def merge_dfs():

    print('start: ', datetime.datetime.now())

    df_sessions = pd.read_csv('data/ga_sessions.csv', dtype={'client_id': 'object'})
    df_hits = pd.read_csv('data/ga_hits.csv')

    df_sessions = df_sessions.drop('visit_number', axis=1)  # порядковый номер визита клиента удаляем

    df_hits.event_value = df_hits.event_action.apply(lambda x: 1 if ((x == 'sub_car_claim_click') |
                                                                     (x == 'sub_car_claim_submit_click') |
                                                                     (x == 'sub_open_dialog_click') |
                                                                     (x == 'sub_custom_question_submit_click') |
                                                                     (x == 'sub_call_number_click') |
                                                                     (x == 'sub_callback_submit_click') |
                                                                     (x == 'sub_submit_success') |
                                                                     (x == 'sub_car_request_submit_click')) else 0)
    df_hits = df_hits.drop(['hit_type', 'hit_time', 'hit_referer', 'hit_page_path',
                            'event_label', 'event_category', 'event_action'], axis=1)

    stats = df_hits.groupby(['session_id'], as_index=False).agg({'hit_date': 'first',
                                                                 'hit_number': 'mean', 'event_value': 'max'})
    df_hits = stats.copy()

    df = pd.merge(left=df_sessions, right=df_hits, on='session_id')

    index_list = df[(df.geo_country == '(not set)') & (df.geo_city == '(not set)')].index.tolist()
    df = df.drop(index_list, axis=0)

    return df


def filter_data(df):

    df = df.copy()

    columns_to_drop = ['session_id', 'client_id', 'utm_source', 'utm_campaign',
                       'utm_adcontent', 'utm_keyword', 'device_model']

    return df.drop(columns_to_drop, axis=1)


def device_brand_fill(df):
    df = df.copy()

    tmp_df = df[(df.device_brand.isna()) & (df.device_os == 'Windows')]
    tmp_df.loc[:, 'device_brand'] = 'Windows'
    df[(df.device_brand.isna()) & (df.device_os == 'Windows')] = tmp_df

    tmp_df = df[(df.device_brand.isna()) & (df.device_os == 'Macintosh')]
    tmp_df.loc[:, 'device_brand'] = 'Apple'
    df[(df.device_brand.isna()) & (df.device_os == 'Macintosh')] = tmp_df

    # для владельцев операционной системы Linux установим device_brand = Windows
    tmp_df = df[(df.device_brand.isna()) & (df.device_os == 'Linux')]
    tmp_df.loc[:, 'device_brand'] = 'Windows'
    df[(df.device_brand.isna()) & (df.device_os == 'Linux')] = tmp_df

    df.device_brand = df.device_brand.fillna('other')  # все остальные пропуски заполним значением other
    df.device_brand = df.device_brand.replace('(not set)', 'other')

    df = df.drop('device_os', axis=1)

    df.device_brand = df.device_brand.apply(lambda x: x.lower())

    stats = df.groupby(['device_brand'], as_index=False).agg({'device_category': 'count'}).sort_values(
        ascending=False, by='device_category')
    stats = stats.rename(columns={'device_category': 'count'})
    device_brands_list = stats[stats['count'] < 1000].loc[:, 'device_brand'].unique().tolist()

    df.device_brand = df.device_brand.replace(device_brands_list, 'other')  # заменим не заданные значения на other

    return df


def device_screen_fill(df):
    df = df.copy()

    # заменим device_screen_resolution на площадь экрана
    df_w = df.device_screen_resolution.apply(lambda x: x.split('x')[0])
    df_w = df_w.astype('int')

    df_h = df.device_screen_resolution.apply(lambda x: x.split('x')[1])
    df_h = df_h.astype('int')

    df['device_screen'] = df_w * df_h

    df = df.drop(['device_screen_resolution'], axis=1)

    return df


def delete_outliers(df):
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

        return boundaries

    df = df.copy()

    boundaries = calculate_outliers(df.hit_number)
    df.loc[df.hit_number > boundaries[1], 'hit_number'] = df.hit_number.mode()[0]

    boundaries = calculate_outliers(df.device_screen)
    df.loc[df.device_screen > boundaries[1], 'device_screen'] = boundaries[1]
    df.loc[df.device_screen < boundaries[0], 'device_screen'] = boundaries[0]  # df_check.device_screen.mean()

    return df


def utm_medium_fill(df):
    df = df.copy()

    # преобразуем utm_medium, чтобы уменьшить количество уникальных значений
    df.utm_medium = df.utm_medium.apply(lambda x: x.lower())

    # заменим не заданные значения на other
    df.utm_medium = df.utm_medium.replace('(not set)', 'other')

    # заменим (none) на organic, так как это тоже органический трафик
    df.utm_medium = df.utm_medium.replace('(none)', 'organic')

    return df


def devise_browser_fill(df):
    df = df.copy()

    df.device_browser = df.device_browser.replace('(not set)', 'other')  # заменим не заданные значения на other
    df.device_browser = df.device_browser.apply(lambda x: x.lower().split(' ')[0])  # возьмем первое слово до пробела

    return df


def date_fill(df):
    df = df.copy()

    df['date'] = df.visit_date  # колонки равны, значит можно удалить дублирующуюся и переименовать колонку в date
    df = df.drop(['visit_date', 'hit_date'], axis=1)

    df.date = pd.to_datetime(df.date, utc=True)
    df['dayofweek'] = df.date.dt.weekday
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year

    df = df.drop('year', axis=1)  # удалим год, так как одно уникальное значение
    df = df.drop(['visit_time', 'date'], axis=1)

    return df


def geo_features(df):

    df = df.copy()

    def get_lat_lon(address):
        url = f'https://nominatim.openstreetmap.org/search?q={address}&format=json&accept-language=en&NOMINATIM_REQUEST_TIMEOUT=None'
        headers = {'user-agent': 'my-app/0.0.1'}

        # Конфигурируем сессию requests
        session = requests.Session()

        # Устанавливаем стратегию повторных попыток
        retries = Retry(total=7,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504],
                        allowed_methods=frozenset(['GET', 'POST']))

        # Связываем стратегию с сессией requests
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # Осуществляем запрос
        response = session.get(url=url, headers=headers)
        response.raise_for_status()

        if response.status_code != 204:
            result_json = response.json()
            if result_json:  # result_json != []:
                return result_json[0]['lat'], result_json[0]['lon']
            else:
                return 0, 0  # np.NaN

    def lat_lon_dict(data):
        country_list = data.full_address.tolist()
        lat_lon_list = data.lat_lon.tolist()

        ll_dict = {}
        for k, v in zip(country_list, lat_lon_list):
            ll_dict[k] = v

        return ll_dict

    # index_list = df[(df.geo_country == '(not set)') & (df.geo_city == '(not set)')].index.tolist()
    # df = df.drop(index_list, axis=0)   # вынесла в merge_dfs()

    df.geo_city = df.geo_city.replace('(not set)', '')
    df.geo_city = df.geo_city.replace("'", "")

    df['full_address'] = df.geo_country.str.cat(others=df.geo_city, sep=', ', na_rep='')
    df.loc[:, 'lat_lon'] = df.loc[:, 'full_address']
    df = df.drop(['geo_country', 'geo_city'], axis=1)

    tmp_df = df.loc[:, ['full_address']]
    tmp_df = tmp_df.drop_duplicates()

    tmp_df['lat_lon'] = tmp_df['full_address'].apply(lambda x: get_lat_lon(x))

    df.lat_lon = df.lat_lon.apply(lambda x: lat_lon_dict(tmp_df).get(x))

    df.loc[:, 'lat'] = df.lat_lon.apply(lambda x: round(float(x[0]), 4))
    df.loc[:, 'lon'] = df.lat_lon.apply(lambda x: round(float(x[1]), 4))

    df = df.drop(['full_address', 'lat_lon'], axis=1)

    return df


def main():
    df = merge_dfs()

    X = df.drop('event_value', axis=1)
    y = df['event_value']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    feature_engineering = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('screen_resolution', FunctionTransformer(device_screen_fill)),
        ('delete_outliers', FunctionTransformer(delete_outliers)),
        ('utm_medium', FunctionTransformer(utm_medium_fill)),
        ('device_branding', FunctionTransformer(device_brand_fill)),
        ('device_browser', FunctionTransformer(devise_browser_fill)),
        ('geo', FunctionTransformer(geo_features)),
        ('date_filling', FunctionTransformer(date_fill))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=['object']))
    ])

    preprocessor = Pipeline(steps=[
        ('feature_engineering', feature_engineering),
        ('column_transformer', column_transformer)
    ])

    models = [
        LogisticRegression(max_iter=700),
        RandomForestClassifier(),
        MLPClassifier(max_iter=700)
    ]

    best_roc_auc_score = 0.0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ]).fit(x_train, y_train)

        score = roc_auc_score(y_test, pipe.predict_proba(x_test)[:, 1])
        print(f'model: {type(model).__name__}, roc_auc_score: {score:.4f}, \n')

        if score > best_roc_auc_score:
            best_roc_auc_score = score
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc_score: {best_roc_auc_score:.4f}')

    best_pipe.fit(X, y)
    with open('sber_podpiska_predict_model.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Sber Autopodpiska Model',
                'author': 'Diana Nigm',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc_score': best_roc_auc_score
            }
        }, file, recurse=True)


if __name__ == '__main__':
    main()
