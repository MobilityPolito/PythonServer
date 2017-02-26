## compute_input.py

import sys
import json
import datetime

import numpy as np
import pandas as pd

sys.path.append("/Users/anr.putina/Desktop/REPORT/turin-mobility/Architecture")
from DataBaseProxy import DataBaseProxy
dbp = DataBaseProxy()

from Graphics import Graphics
g = Graphics()


# Read data from stdin
def read_in():
    lines = sys.stdin.readlines()
    #Since our input would only be having one line, parse our JSON data from that
    return json.loads(lines[0])

def filter_df(df, filter_col):
    return df.loc[df[filter_col] == True]

def main():


    #get our data as an array from read_in()
    lines = read_in()

    # data = {
    #     'test': lines['start']
    # }

    # print (json.dumps(data))

    try:
        start = datetime.datetime.strptime(str(lines['start']),"%Y-%m-%dT%H:%M:%S.%fZ")
        end = datetime.datetime.strptime(str(lines['end']), "%Y-%m-%dT%H:%M:%S.%fZ")
    except:
        print ('error in start/end')
        sys.exit()

    enjoy_df = dbp.query_books_df_filtered_v3("enjoy", "torino", start, end)
    car2go_df = dbp.query_books_df_filtered_v3("car2go", "torino", start, end)

    s = enjoy_df["duration"].dropna()
    s1 = s[(s >= s.quantile(q=0.01)) & (s <= s.quantile(q=0.99))]
    s = enjoy_df["distance"].dropna()
    s2 = s[(s >= s.quantile(q=0.01)) & (s <= s.quantile(q=0.99))]
    enjoy_quant = enjoy_df.loc[s1.index.intersection(s2.index)]

    s = car2go_df["duration"].dropna()
    s1 = s[(s >= s.quantile(q=0.01)) & (s <= s.quantile(q=0.99))]
    s = car2go_df["distance"].dropna()
    s2 = s[(s >= s.quantile(q=0.01)) & (s <= s.quantile(q=0.99))]
    car2go_quant = car2go_df.loc[s1.index.intersection(s2.index)]
    #
    #
    enjoy_daily = filter_df(enjoy_quant, 'max_daily')
    car2go_daily = filter_df(car2go_quant, 'max_daily')

    enjoy_reservation = filter_df(enjoy_daily, 'reservations')
    car2go_reservation = filter_df(car2go_daily, 'reservations')

    enjoy_ride= filter_df(enjoy_daily, 'ride')
    car2go_ride = filter_df(car2go_daily, 'ride')

    enjoy_long = filter_df(enjoy_ride, 'long_trips')
    car2go_long = filter_df(car2go_ride, 'long_trips')

    enjoy_short = filter_df(enjoy_ride, 'short_trips')
    car2go_short = filter_df(car2go_ride, 'short_trips')


    ### NUMBER OF BOOKINGS ###
    enjoy = len(enjoy_short)
    car2go = len(car2go_short)
    # enjoy_fleet = len(enjoy_short.plate.unique())
    # car2go_fleet = len(car2go_short.plate.unique())
    days = (end-start).days + 1

    info_bookings = {
        'enjoy': enjoy,
        'car2go': car2go,
        # 'enjoy_fleet': enjoy_fleet,
        # 'car2go_fleet': car2go_fleet,
        'days': days
    }

    # ### AGGREGATED
    freq="360Min"

    df_ = car2go_short.set_index("start")
    s1 = df_['duration'].dropna()
    s1 = s1.resample(freq).count()
    # s.plot(marker='o', figsize=figsize, label='car2go', color='blue')

    df_ = enjoy_short.set_index("start")
    s2 = df_['duration'].dropna()
    s2 = s2.resample(freq).count()
    # s.plot(marker='o', figsize=figsize, label='enjoy', color='red')

    # amchart = []
    # for i in range(len(s1)):
    #     sample = {
    #         ''
    #     }

    dates = []
    for date in s1.index.to_datetime():
        dates.append(str(date))

    agg_duration = {
        'name': 'aggregated duration',
        'index': dates,
        'enjoy': s2.tolist(),
        'car2go': s1.tolist()
    }

    # result = [
    #     {
    #         'name': 'Aggregated By 360Min',
    #         'index': s1.index.to_datetime().tolist(),
    #         'enjoy': s1.tolist(),
    #         'car2go': s2.tolist()
    #     }
    # ]

    # print json.dumps(result)

    ### AVG DURATION business and weekend ###
    
    df_ = car2go_short[car2go_short['weekend']].set_index("start")
    s = df_['duration'].dropna()
    div = float(len(s.groupby(s.index.map(lambda t: t.date))))
    s = s.groupby(s.index.map(lambda t: t.hour)).count()
    s_ = s/div

    df_ = enjoy_short[enjoy_short['weekend']].set_index("start")
    s = df_['duration'].dropna()
    div = float(len(s.groupby(s.index.map(lambda t: t.date))))
    s = s.groupby(s.index.map(lambda t: t.hour)).count()
    s_2 = s/div

    df_ = enjoy_short[enjoy_short['business']].set_index("start")
    s = df_['duration'].dropna()
    div = float(len(s.groupby(s.index.map(lambda t: t.date))))
    s = s.groupby(s.index.map(lambda t: t.hour)).count()
    s_3 = s/div

    df_ = car2go_short[car2go_short['business']].set_index("start")
    s = df_['duration'].dropna()
    div = float(len(s.groupby(s.index.map(lambda t: t.date))))
    s = s.groupby(s.index.map(lambda t: t.hour)).count()
    s_4 = s/div

    if (len(s_)==0):
        s_ = [0] * 24
        s_2 = [0] * 24

        avg_duration_bussines_week = {
            'name': 'Durations by hours',
            'car2go_weekend': s_,
            'enjoy_weekend': s_2,
            'enjoy_business': s_3.tolist(),
            'car2go_business': s_4.tolist()
        }

    elif (len(s_3)==0):
        s_3 = [0] * 24
        s_4 = [0] * 24

        avg_duration_bussines_week = {
            'name': 'Durations by hours',
            'car2go_weekend': s_.tolist(),
            'enjoy_weekend': s_2.tolist(),
            'enjoy_business': s_3,
            'car2go_business': s_4
        }

    else:
        avg_duration_bussines_week = {
            'name': 'Durations by hours',
            'car2go_weekend': s_.tolist(),
            'enjoy_weekend': s_2.tolist(),
            'enjoy_business': s_3.tolist(),
            'car2go_business': s_4.tolist()
        }

    # result = [
    #     {
    #         'name': 'Durations by hours',
    #         'car_weekend': s_.tolist(),
    #         'enj_weekend': s_2.tolist(),
    #         'enj_bus': s_3.tolist(),
    #         'car_bus': s_4.tolist()
    #     }
    # ]

    # print json.dumps(result)
    #### END AVG DURATION #######

    #### BAR VS PT
    df = g.slotted_df(enjoy_short)
    df2 = g.slotted_df(car2go_short)

    a = df.groupby('slot')._id.count().apply(lambda x: x/float(len(df)))
    b = df2.groupby('slot')._id.count().apply(lambda x: x/float(len(df2)))

    # result = [
    #     {
    #         'name': 'Probability car vs duration pt',
    #         'index': a.index.tolist(),
    #         'enjoy': a.tolist(),
    #         'car2go': b.tolist()
    #     }
    # ]

    bar_prob_vs_pt = {
            'name': 'Probability car vs duration pt',
            'index': a.index.tolist(),
            'enjoy': a.tolist(),
            'car2go': b.tolist()
    }

    # # result = {
    # #         'key': 'come va'
    # #     }

    # print json.dumps(result)
    #### END BAR VS PT #####

    #### START BILL #####
    freq = '1440Min'
    df_ = enjoy_short.set_index("start")
    s = df_['min_bill'].dropna()
    s_enj_min = s.resample(freq).sum()

    s = df_['max_bill'].dropna()
    s_enj_max= s.resample(freq).sum()

    df_ = car2go_short.set_index("start")
    s = df_['min_bill'].dropna()
    s_car_min = s.resample(freq).sum()

    s = df_['max_bill'].dropna()
    s_car_max = s.resample(freq).sum()

    dates = []
    for date in s_enj_max.index:
        dates.append(str(date))

    bill = {
        'name': 'Bill',
        'index': dates,
        'enjoy_min': s_enj_min.tolist(),
        'enjoy_max': s_enj_max.tolist(),
        'car2go_min': s_car_min.tolist(),
        'car2go_max': s_car_max.tolist()        
    }

    # result = [
    #     {
    #         'name': 'Bill',
    #         'enjoy_min': s_enj_min.tolist(),
    #         'enjoy_max': s_enj_max.tolist(),
    #         'car2go_min': s_car_min.tolist(),
    #         'car2go_max': s_car_max.tolist()
    #     }
    # ]

    # print json.dumps(result)

    #### END BILL ####


    result = [
        avg_duration_bussines_week,
        bar_prob_vs_pt, 
        bill,
        agg_duration,
        info_bookings
    ]

    print json.dumps(result)

    # ################ start working #####################
    # # start = datetime.datetime(2017, 1, 10, 0, 0, 0)
    # # end = datetime.datetime(2017, 1, 17, 23, 59, 59)

    # enjoy_df = dbp.query_books_df_filtered_v3("enjoy", "torino", start, end)

    # df = enjoy_df[(enjoy_df['ride'] == True) & \
    #          (enjoy_df['short_trips'] == True) & \
    #          (enjoy_df['tot_duration_google_transit'].isnull() == False)]

    # df_ = df.set_index("start")

    # car = df_.groupby(df_.index.map(lambda t: t.hour)).duration.mean()
    # pt = df_.groupby(df_.index.map(lambda t: t.hour)).tot_duration_google_transit.mean()

    # prova = [
    #     {
    #         "name": "Car_vs_pt",
    #         "car": car.tolist(),
    #         "pt": pt.tolist()
    #     }
    # ]

    # # return the sum to the output stream
    # print json.dumps(prova)
    # ############### END WORK PROGRAM ################

#start process
if __name__ == '__main__':
    main()
