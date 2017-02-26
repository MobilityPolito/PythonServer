#import json
import datetime

from shapely.geometry import Point
#from shapely.geometry import LineString
from sklearn import linear_model
import numpy as np

from scipy.spatial import ConvexHull


from scipy import ndimage

import pandas as pd
import geopandas as gpd

import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from pandas.tools.plotting import scatter_matrix

from Graphics import Graphics
g = Graphics()

from DataBaseProxy import DataBaseProxy, haversine
dbp = DataBaseProxy()

from EnjoyProvider import Enjoy
enjoy = Enjoy()

from Car2GoProvider import Car2Go
car2go = Car2Go()

from area_enjoy import create_zones_enjoy
from area_car2go import create_zones_car2go

"""
Load data structure
"""

start = datetime.datetime(2016, 12, 10, 0, 0, 0)
end = datetime.datetime(2017, 1, 30, 23, 59, 59)

#enjoy_parks = dbp.query_parks_df_filtered_v3("enjoy", "torino", start, end)
#car2go_parks = dbp.query_parks_df_filtered_v3("car2go", "torino", start, end)
#
#enjoy_ = enjoy_parks[(enjoy_parks['end'] <= end) & \
#                             (enjoy_parks['duration'] <= 10000)&\
#                             (enjoy_parks['duration'] >= 10)]
#
#car2go_ = car2go_parks[(car2go_parks['end'] <= end) & \
#                             (car2go_parks['duration'] <= 10000) &\
#                             (car2go_parks['duration'] >= 10)]
#
#
#enjoy_filtered = enjoy_[(enjoy_['duration'] >= enjoy_['duration'].quantile(0.01)) &\
#                             (enjoy_['duration'] <= enjoy_['duration'].quantile(0.99))]
#
#
#car2go_filtered = car2go_[(car2go_['duration'] >= car2go_['duration'].quantile(0.01)) &\
#                             (car2go_['duration'] <= car2go_['duration'].quantile(0.99))]

enjoy_df = dbp.query_books_df_filtered_v3("enjoy", "torino", start, end)
car2go_df = dbp.query_books_df_filtered_v3("car2go", "torino", start, end)

def filter_df(df, filter_col):
    return df.loc[df[filter_col] == True]

#
#zones = gpd.read_file("../../SHAPE/Zonizzazione.dbf").to_crs({"init": "epsg:4326"})\
#                     .sort_values("Denom_GTT").reset_index().drop("index", axis=1)
#
#centroids = gpd.read_file("../../SHAPE/Centroidi.dbf").to_crs({"init": "epsg:4326"})\
#                     .sort_values("Denom_GTT").reset_index().drop("index", axis=1)
#
#demography = gpd.read_file("../../dati_torino/zonestat_popolazione_residente_2015_geo.dbf").to_crs({"init": "epsg:4326"})
#
#enjoy_operational_zones = create_zones_enjoy()
#car2go_operational_zones = create_zones_car2go('zones')
#car2go_airport = create_zones_car2go('airport')
#car2go_ikea = create_zones_car2go('ikea')
#
#frames = [enjoy_operational_zones, 
#          car2go_operational_zones, 
#          car2go_airport, 
#          car2go_ikea]
#
#operational_zones = gpd.GeoDataFrame(pd.concat(frames))

#"""
#Fleet
#"""
#
#enjoy_fleetsize_series = enjoy.get_fleetsize_info().dropna()
#car2go_fleetsize_series = car2go.get_fleetsize_info().dropna()
#
#fig, axs = plt.subplots(1,1)
#enjoy_fleetsize_series.plot(figsize=(13,6), marker='o', ax=axs, label="Enjoy")
#car2go_fleetsize_series.plot(figsize=(13,6), marker='o', ax=axs, label="Car2Go")
#plt.legend()
#plt.title("Fleet size evolution")
##plt.savefig('fleet.png')
#plt.show()

#
#"""
#Sketch
#"""
## * SAMPLES *
#
#filter_name = "all"
#col = "distance"
##col = "tot_duration_in_traffic"
#col = "riding_time"
#col = "duration"
#
#plt.figure()
#g.plot_samples(car2go_df, col, filter_name, "car2go")
#plt.figure()
#g.plot_samples(car2go_df, col, filter_name, "car2go", quantile=0.01)
#plt.figure()
#g.plot_samples(car2go_df, col, filter_name, "car2go", quantile=0.05)
##
#plt.figure()
#g.plot_samples(enjoy_df, col, filter_name, "enjoy")
#plt.figure()
#g.plot_samples(enjoy_df, col, filter_name, "enjoy", quantile=0.01)
#plt.figure()
#g.plot_samples(enjoy_df, col, filter_name, "enjoy", quantile=0.05)
#
#
#g.plot_samples_vs(enjoy_df, car2go_df, col, filter_name)

#g.plot_samples_vs(enjoy_df, car2go_df, col, filter_name, quantile=0.01)
#g.plot_samples_vs(enjoy_df, car2go_df, col, filter_name, quantile=0.05)
#
## HISTOGRAMS 
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
##
##
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
#
#cars = car2go_short.plate.unique()
#
#parcheggi=pd.DataFrame()
#for car in cars:
##    
#    data = car2go_short[car2go_short['plate']==car]
#    
#    for i in (1, len(data)-1):
#        
#    
#st = datetime.datetime(2016,12,14,0,0) 
#en = datetime.datetime(2016,12,15,0,0) 

#a = car[(car['start'] > st) & (car['end'] < en)]

#figsize=(13,6)
#bins=1000
#cumulative=True
#plt.figure()
#plt.title('Cumulative Density Function Duration')
#plt.xlabel('Duration [min]')
#plt.ylabel("Probability [p]")
#plt.axis((0,80,0,1.05))
#enjoy_short.duration.hist(figsize=figsize, label='enjoy', color='red',\
#                          cumulative=cumulative, normed = True,\
#                          bins=bins, histtype='step')        
#car2go_short.duration.hist(figsize=figsize, label='car2go',\
#                           color='blue', cumulative=cumulative,\
#                           normed = True, bins=bins,\
#                           histtype='step') 
#plt.legend()
#plt.savefig('CDF_Duration.png')
#
##enjoy_short.distance = enjoy_short.distance + [np.inf]
#
##n, bins, patches = plt.hist(enjoy_short.distance, normed=True, histtype='step', cumulative=True, 
##                            bins=bins)
#
#bins = sorted(enjoy_short.duration) + [np.inf]
#
#figsize=(13,6)
#bins=100
#cumulative=True
#plt.figure()
#plt.title('Cumulative Density Function Distance')
#plt.xlabel('Distance [km]')
#plt.ylabel("Probability [p]")
#plt.axis((0,20,0,1.05))
#
#
#enjoy_short.distance.hist(figsize=figsize, label='enjoy', color='red',\
#                          cumulative=cumulative, normed = True,\
#                          bins=bins, histtype='step')        
#car2go_short.distance.hist(figsize=figsize, label='car2go',\
#                           color='blue', cumulative=cumulative,\
#                           normed = True, bins=bins,\
#                           histtype='step') 
#plt.legend()
#plt.savefig('CDF_Distance.png')


figsize=(13,6)
#bins=100
#cumulative=True
#plt.figure()
#plt.xlabel('Duration [min]')
#plt.ylabel("Probability [p]")
##plt.axis((0,80,0,1))
#enjoy_long.duration.hist(figsize=figsize, label='enjoy', color='red',\
#                          cumulative=cumulative, normed = True,\
#                          bins=bins, histtype='step')        
#car2go_long.duration.hist(figsize=figsize, label='car2go',\
#                           color='blue', cumulative=cumulative,\
#                           normed = True, bins=bins,\
#                           histtype='step') 
#plt.legend()
#
#col = "duration"


#g.hist(enjoy_df, col, filter_name, "Enjoy", "red")
#g.hist(enjoy_df, col, filter_name, "Enjoy", "red", quantile=0.05)
#g.hist(enjoy_df, col, filter_name, "Enjoy", "red", quantile=0.05, cumulative= True)

#g.hist(car2go_df, col, filter_name, "Car2Go", "blue")
#g.hist(car2go_df, col, filter_name, "Car2Go", "blue", quantile=0.05)
#g.hist(car2go_df, col, filter_name, "Car2Go", "blue", quantile=0.05, cumulative= True)



#
##### CDF WEEKS ###
#plt.figure()
#g.cdf_weeks_duration(enjoy_df, car2go_df)
#g.cdf_weeks_distance(enjoy_df, car2go_df)
##
##### CDF BUSINESS VS WEEKEND ###
#g.cdf_business_weekend(enjoy_df)
#g.cdf_business_weekend(car2go_df)

    
#plt.figure()
#fig, ax = plt.subplots(figsize=(13, 6))
#plt.title("CDF duration Business days vs Weekend days")
#
#bd_df = enjoy_short[(enjoy_short['business'] == True)]
#n, bins, patches = ax.hist(bd_df.duration, 1000, histtype='step',
#                           cumulative=True, label="Enjoy Business",
#                           rwidth=2.0, color='red', normed=1)
#
#we_df = enjoy_short[(enjoy_short['weekend'] == True)]
#n, bins, patches = ax.hist(we_df.duration, 1000, histtype='step',
#                           cumulative=True, label="Enjoy Weekend", linestyle="--",
#                           rwidth=2.0, color='red', normed=1)
#bd_df = car2go_short[(car2go_short['business'] == True)]
#n, bins, patches = ax.hist(bd_df.duration, 1000, histtype='step',
#                           cumulative=True, label="Car2Go Business",
#                           rwidth=2.0, color='blue', normed=1)
#
#we_df = car2go_short[(car2go_short['weekend'] == True)]
#n, bins, patches = ax.hist(we_df.duration, 1000, histtype='step',
#                           cumulative=True, label="Car2Go Weekend", linestyle="--",
#                           rwidth=2.0, color='blue', normed=1)
#
#ax.legend(loc=4)
#ax.set_xlabel('Durations [min]')
#ax.set_ylabel('probability [p]')
##plt.savefig('CDF_Business_Weekend_Duration.png')
#plt.show()

#plt.figure()
#fig, ax = plt.subplots(figsize=(13, 6))
#plt.title("CDF distances Business days vs Weekend days")
#
#bd_df = enjoy_short[(enjoy_short['business'] == True)]
#n, bins, patches = ax.hist(bd_df.distance, 1000, histtype='step',
#                           cumulative=True, label="Enjoy Business",
#                           rwidth=2.0, color='red', normed=1)
#
#we_df = enjoy_short[(enjoy_short['weekend'] == True)]
#n, bins, patches = ax.hist(we_df.distance, 1000, histtype='step',
#                           cumulative=True, label="Enjoy Weekend", linestyle="--",
#                           rwidth=2.0, color='red', normed=1)
#bd_df = car2go_short[(car2go_short['business'] == True)]
#n, bins, patches = ax.hist(bd_df.distance, 1000, histtype='step',
#                           cumulative=True, label="Car2Go Business",
#                           rwidth=2.0, color='blue', normed=1)
#
#we_df = car2go_short[(car2go_short['weekend'] == True)]
#n, bins, patches = ax.hist(we_df.distance, 1000, histtype='step',
#                           cumulative=True, label="Car2Go Weekend", linestyle="--",
#                           rwidth=2.0, color='blue', normed=1)
#plt.axis([0, 8, 0, 1.1])
#ax.legend(loc=4)
#ax.set_xlabel('Durations [min]')
#ax.set_ylabel('probability [p]')
##plt.savefig('CDF_Business_Weekend_Duration.png')
#plt.show()




## AGGREGATED PLOTS
#col = "duration"
#g.plot_aggregated_count_vs(enjoy_df, car2go_df, col, filter_name, quantile=0.01)
#g.plot_aggregated_mean_vs(enjoy_df, car2go_df, col, "ride")
#g.plot_aggregated_mean_vs(enjoy_df, car2go_df, col, "all", quantile=0.01)
##


######################################
#freq="360Min"
#plt.figure()
#plt.title("Number of bookings aggregated every " + freq)
#
#df_ = car2go_short.set_index("start")
#s = df_['duration'].dropna()
#s = s.resample(freq).count()
#s.plot(marker='o', figsize=figsize, label='car2go', color='blue')
#
#df_ = enjoy_short.set_index("start")
#s = df_['duration'].dropna()
#s = s.resample(freq).count()
#s.plot(marker='o', figsize=figsize, label='enjoy', color='red')
#
#plt.xlabel('Date')
#plt.ylabel("Number of bookings [k]")
#plt.legend()   
##plt.savefig('Number_bookings_aggregated_360.png')
#plt.show()    

#####################################
#plt.figure()
#plt.title("Aggregated Duration every "+freq)
#
#
#df_ = enjoy_short.set_index("start")
#s = df_['duration'].dropna()
#s = s.resample(freq).mean()
#s.plot(marker='o', figsize=figsize, label='enjoy', color='red')
#
#df_ = car2go_short.set_index("start")
#s = df_['duration'].dropna()
#s = s.resample(freq).mean()
#s.plot(marker='o', figsize=figsize, label='car2go', color='blue')
#
#plt.xlabel('Date')
#plt.ylabel('Average duration [min]')
#plt.legend()  
##plt.savefig('Aggregated_Duration_60.png')
#
#plt.figure()
#a.duration.hist(bins=100, normed=True, cumulative=True)
#plt.axis((0,30,0,1))

### DAILY
###
#col='duration'
#filter_name='short_trips'

#g.plot_daily_count_vs(enjoy_df, car2go_df, col, filter_name, quantile=0.01)
#g.plot_daily_mean_vs(enjoy_df, car2go_df, col, filter_name, quantile=0.01)

#flotta_enjoy = dbp.query_fleetsize_series('enjoy','torino')

#flotta_car2go = dbp.query_fleetsize_series('car2go','torino')
##################################
plt.figure()
plt.title("Average Number of daily bookings - per hours")

df_ = enjoy_short.set_index("start")
s = df_['duration'].dropna()
div = float(len(s.groupby(s.index.map(lambda t: t.date))))
print "GIORNI + " + str(div)
s = s.groupby(s.index.map(lambda t: t.hour)).count()
s_ = s/div
s_.plot(marker='o', figsize=figsize, label='enjoy overall', color='red')

df_ = car2go_short.set_index("start")
s = df_['duration'].dropna()
div = float(len(s.groupby(s.index.map(lambda t: t.date))))
print "GIORNI + " + str(div)
s = s.groupby(s.index.map(lambda t: t.hour)).count()
s_ = s/div
s_.plot(marker='o', figsize=figsize, label='car2go overall', color='blue')

df_ = car2go_short[car2go_short['weekend']].set_index("start")
s = df_['duration'].dropna()
div = float(len(s.groupby(s.index.map(lambda t: t.date))))
print "GIORNI + " + str(div)
s = s.groupby(s.index.map(lambda t: t.hour)).count()
s_ = s/div
s_.plot(marker='d', figsize=figsize, label='car2go weekend', color='blue', linestyle='--')

df_ = enjoy_short[enjoy_short['weekend']].set_index("start")
s = df_['duration'].dropna()
div = float(len(s.groupby(s.index.map(lambda t: t.date))))
print "GIORNI + " + str(div)
s = s.groupby(s.index.map(lambda t: t.hour)).count()
s_ = s/div
s_.plot(marker='d', figsize=figsize, label='enjoy weekend', color='red', linestyle='--')

df_ = enjoy_short[enjoy_short['business']].set_index("start")
s = df_['duration'].dropna()
div = float(len(s.groupby(s.index.map(lambda t: t.date))))
print "GIORNI + " + str(div)
s = s.groupby(s.index.map(lambda t: t.hour)).count()
s_ = s/div
s_.plot(marker='x', figsize=figsize, label='enjoy business', color='red', linestyle='-.')

df_ = car2go_short[car2go_short['business']].set_index("start")
s = df_['duration'].dropna()
div = float(len(s.groupby(s.index.map(lambda t: t.date))))
print "GIORNI + " + str(div)
s = s.groupby(s.index.map(lambda t: t.hour)).count()
s_ = s/div
s_.plot(marker='x', figsize=figsize, label='car2go business', color='blue', linestyle='-.')

plt.xlabel("Daily hours [h]")
plt.ylabel("Average number of bookings [k]")
plt.legend(loc=4)        
#plt.savefig('avg_books_per_hours.png')
plt.show()

#plt.figure()
#plt.title("Daily Fleet Utilization - per hours")
#
#df_ = enjoy_short.set_index("start")
#s = df_['duration'].dropna()
#div = float(len(s.groupby(s.index.map(lambda t: t.date))))
#print "GIORNI + " + str(div)
#s = s.groupby(s.index.map(lambda t: t.hour)).count()
#s_ = s/div/flotta_enjoy.mean()
#s_.plot(marker='o', figsize=figsize, label='enjoy', color='red')
#
#df_ = car2go_short.set_index("start")
#s = df_['duration'].dropna()
#div = float(len(s.groupby(s.index.map(lambda t: t.date))))
#print "GIORNI + " + str(div)
#s = s.groupby(s.index.map(lambda t: t.hour)).count()
#s_ = s/div/flotta_car2go.mean()
#s_.plot(marker='o', figsize=figsize, label='car2go', color='blue')
#plt.xlabel("Daily hours [h]")
#plt.ylabel("Utilization of the fleet [%]")
#plt.legend()        
#plt.savefig('utilization_fleet.png')
#plt.show()

#plt.figure()
#plt.title("Average Duration of daily bookings - per hours")
#
#df_ = enjoy_short.set_index("start")
#s = df_['duration'].dropna()
#s = s.groupby(s.index.map(lambda t: t.hour)).mean()
#s.plot(marker='o', figsize=figsize, label='enjoy', color='red')
#
#df_ = car2go_short.set_index("start")
#s = df_['duration'].dropna()
#s = s.groupby(s.index.map(lambda t: t.hour)).mean()
#s.plot(marker='o', figsize=figsize, label='car2go', color='blue')
#
#plt.xlabel("Daily hours [h]")
#plt.ylabel("Average duration [min]")
#plt.legend()        
#plt.savefig('avg_duration_per_hours.png')
#plt.show()

#####PARKS HIST###
#figsize=(13,6)
#bins=100
#cumulative=True
#plt.figure()
#plt.title('Cumulative Density Function Parks Duration')
#plt.xlabel('Duration [min]')
#plt.ylabel("Probability [p]")
##plt.axis((0,80,0,1.05))
#enjoy_filtered.duration.hist(figsize=figsize, label='enjoy', color='red',\
#                          cumulative=cumulative, normed = True,\
#                          bins=bins, histtype='step')        
#car2go_filtered.duration.hist(figsize=figsize, label='car2go',\
#                           color='blue', cumulative=cumulative,\
#                           normed = True, bins=bins,\
#                           histtype='step') 
#plt.legend()
#plt.savefig('CDF_Duration_Parks.png')
#
#plt.show()
#
#plt.figure()
#plt.title("Average Duration of daily parkings - per hours")
#
#df_ = enjoy_filtered.set_index("start")
#s = df_['duration'].dropna()
#s = s.groupby(s.index.map(lambda t: t.hour)).mean()
#s.plot(marker='o', figsize=figsize, label='enjoy', color='red')
#
#df_ = car2go_filtered.set_index("start")
#s = df_['duration'].dropna()
#s = s.groupby(s.index.map(lambda t: t.hour)).mean()
#s.plot(marker='o', figsize=figsize, label='car2go', color='blue')
#
#plt.xlabel("Daily hours [h]")
#plt.ylabel("Average duration [min]")
#plt.legend()        
##plt.savefig('avg_duration_parks_per_hours.png')
#plt.show()

"""
 * GOOGLE RESULTS *
"""

#g.car_vs_google(enjoy_df)
#g.car_vs_google(car2go_df)
#g.car_vs_google_comparison(enjoy_df, car2go_df)

##############################
#plt.figure()
#
#df1 = enjoy_short
#df2 = car2go_short
#
#bis_y = bis_x = range(1,int(df1.duration.max()))
#fig, ax = plt.subplots(figsize=(13, 6))
#
#plt.title ("Duration vs Google forecast driving time")
#ax.set_xlabel('Duration')
#ax.set_ylabel('Google Duration')
#ax.scatter(df1['duration_driving'],df1['duration'],
#           s=1, label= "Enjoy Trips", color='red')
#ax.scatter(df2['duration_driving'],df2['duration'],
#           s=1, label= "Car2go Trips", color= 'blue')
#ax.plot(bis_x,bis_y,linewidth=1.0, linestyle='--', 
#        label= "Equal time bisector", color= "black")
#ax.set_xlabel("Google Forecast [min]")
#ax.set_ylabel("Measured Duration [min]")
#plt.axis([0, 50, 0, 50])
#plt.legend(loc=4)
#plt.savefig('duration_vs_google_driving.png')
#plt.show()


#g.car_vs_transit(enjoy_df)
#g.car_vs_transit(car2go_df)

#plt.figure()
#df_ = enjoy_short[(enjoy_short['tot_duration_google_transit'].isnull() == False)]  
#fig, ax = plt.subplots(figsize=(13, 6))
#plt.title ("Duration Enjoy vs Public Transport")
#ax.set_xlabel('Tbus [min]')
#ax.set_ylabel('Tcar [min]')
#ax.axis([0,100,0,45])                                       
#ax.scatter(df_.tot_duration_google_transit, df_.duration, color='red',s=0.5)
#bis_y = bis_x = range(1,int(df_.duration.max()))
#ax.plot(bis_x, bis_y, color="black", linestyle = '--', label = 'time bisector')
#
#x = df_.tot_duration_google_transit.values
#y = df_.duration.values
#x = x.reshape(len(x), 1)
#y = y.reshape(len(y), 1)
#regr = linear_model.LinearRegression()
#regr.fit(x, y)
## Train the model using the training sets
#ax.plot(x, regr.predict(x), color='black', linewidth=1, linestyle='-', label = 'linear regression')
#plt.legend(loc=4)
#plt.savefig('enjoy_duration_vs_pt.png')
#plt.show()

#plt.figure()
#df_ = car2go_short[(car2go_short['tot_duration_google_transit'].isnull() == False)]  
#fig, ax = plt.subplots(figsize=(13, 6))
#plt.title ("Duration Car2Go vs Public Transport")
#ax.set_xlabel('Tbus [min]')
#ax.set_ylabel('Tcar [min]')
#ax.axis([0,100,0,45])                                       
#ax.scatter(df_.tot_duration_google_transit, df_.duration, color='blue',s=0.5)
#bis_y = bis_x = range(1,int(df_.duration.max()))
#ax.plot(bis_x, bis_y, color="black", linestyle = '--', label = 'time bisector')
#
#x = df_.tot_duration_google_transit.values
#y = df_.duration.values
#x = x.reshape(len(x), 1)
#y = y.reshape(len(y), 1)
#regr = linear_model.LinearRegression()
#regr.fit(x, y)
## Train the model using the training sets
#ax.plot(x, regr.predict(x), color='black', linewidth=1, linestyle='-', label = 'linear regression')
#plt.legend(loc=4)
#plt.savefig('car2go_duration_vs_pt.png')
#plt.show()


#g.car_vs_transit_bar(enjoy_df)
#g.car_vs_transit_bar(car2go_df)
#
#
#########################
#plt.figure()
#fig, ax = plt.subplots(figsize=(13, 6))     
#
#ax = plt.subplot(111)
#plt.title('Probability of car choice with respect to PT duration')
#plt.xlabel('Public Transport duration [min]')
#plt.ylabel('Booking Probability [p]') 
#
#df = g.slotted_df(enjoy_short)
#df2 = g.slotted_df(car2go_short)
#
#a = df.groupby('slot')._id.count().apply(lambda x: x/float(len(df)))
#b = df2.groupby('slot')._id.count().apply(lambda x: x/float(len(df2)))
##ax.plot.bar(color='blue')
#plt.xlabel('Google transit durations slots [m]')
#plt.ylabel('Booking Probability') 
#
#ax.bar(a.index-1.5, a,color='r', width=1.5, label='enjoy')
##ax.bar(x, z,width=0.2,color='g',align='center')
#ax.bar(b.index, b,color='b', width=1.5, label='car2go')
#plt.legend()
#plt.savefig('Car_choice_VS_PT.png')
#plt.show()

##
##
#g.car_vs_transit_resampled(enjoy_df)
#g.car_vs_transit_resampled(car2go_df)
#
#g.faster_PT_hours(enjoy_df)
##  night problem
#g.faster_PT_hours(car2go_df)
#
#g.faster_car_hours(enjoy_df)
#g.faster_car_hours(car2go_df)
##
#g.faster_car_PTtime_hours(enjoy_short)
#g.faster_car_PTtime_hours(car2go_df)
#
#
#g.car_pt(enjoy_df)
#g.car_pt(car2go_df)
#


#g.car_pt_vs(enjoy_df,car2go_df)
#######################
#plt.figure()
#plt.title('Comparison among avg aggregated durations')
#df = enjoy_short[(enjoy_short['tot_duration_google_transit'].isnull() == False)]   
#df_ = df.set_index("start")
#df_.groupby(df_.index.map(lambda t: t.hour)).duration.mean().plot(figsize=(13, 6), marker='o', color='red', label = 'avg enjoy duration')
#
#_df = car2go_short[(car2go_short['tot_duration_google_transit'].isnull() == False)]   
#__df = _df.set_index("start")
#__df.groupby(__df.index.map(lambda t: t.hour)).duration.mean().plot(figsize=(13, 6), marker='o', color='blue', label = 'avg car2go duration')
#        
#dur_pt = (df_.groupby(df_.index.map(lambda t: t.hour)).tot_duration_google_transit.mean()\
# + __df.groupby(__df.index.map(lambda t: t.hour)).tot_duration_google_transit.mean())/2.0
#dur_pt.plot(figsize=(13, 6), marker='o', color='orange', label = 'avg PT duration')
#
#dur_pt = (df_.groupby(df_.index.map(lambda t: t.hour)).duration_driving.mean()\
# + __df.groupby(__df.index.map(lambda t: t.hour)).duration_driving.mean())/2.0
#dur_pt.plot(figsize=(13, 6), marker='o', color='purple', label = 'avg google forecast driving duration')
#
#plt.xticks(np.arange(0,23+1, 1.0))
#plt.xlabel('Daily hours [h]')
#plt.ylabel('Average Duration [min]')
#plt.legend()  
#plt.savefig('Comparison_among_avg_aggregated_durations.png')
#plt.show()

##
#
##
#


"""
Bills
"""

#g.plot_aggregated_sum_vs(enjoy_df, car2go_df, "min_bill", "short_trips", quantile=0.01)
#g.plot_aggregated_sum_vs(enjoy_df, car2go_df, "max_bill", "short_trips", quantile=0.01)
#
#g.plot_daily_sum_vs(enjoy_df, car2go_df, "min_bill", "ride", quantile=0.01)
#g.plot_daily_sum_vs(enjoy_df, car2go_df, "max_bill", "ride", quantile=0.01)

#plt.figure()
#plt.title('Bill comparison aggregated per day')
#freq = '1440Min'
#df_ = enjoy_short.set_index("start")
#s = df_['min_bill'].dropna()
#s = s.resample(freq).sum()
#s.plot(marker='o', figsize=figsize, label='enjoy min', color='red')
#
#s = df_['max_bill'].dropna()
#s = s.resample(freq).sum()
#s.plot(marker='o', figsize=figsize, label='enjoy max', color='red', linestyle="--")
#
#df_ = car2go_short.set_index("start")
#s = df_['min_bill'].dropna()
#s = s.resample(freq).sum()
#s.plot(marker='o', figsize=figsize, label='car2go min', color='blue')
#
#s = df_['max_bill'].dropna()
#s = s.resample(freq).sum()
#s.plot(marker='o', figsize=figsize, label='car2go max', color='blue', linestyle="--")
#
#plt.xlabel('date')
#plt.ylabel('Bill [euro]')
#plt.legend()
#plt.savefig('Bill_comparison.png')
#plt.show()


"""
Isocronous / Isocost
"""
#pos_piazzaVittorio = [45.0650653, 7.6936148]
#pos_PortaNuova = [45.0620829, 7.6762908]
##g.isocrono(enjoy_short, pos_piazzaVittorio)
##g.isocost(enjoy_short, pos_piazzaVittorio)
#
#####CRONO
#zones = gpd.read_file("../../../SHAPE/Zonizzazione.dbf")\
#  .to_crs({"init": "epsg:4326"})
#zones_geo = zones["geometry"]
#
#lat_s = pos_PortaNuova[0]
#lon_s = pos_PortaNuova[1]
#
#df_isoc = enjoy_short
#   
#df_isoc['eucl_dist'] = df_isoc[['start_lat', 'start_lon', 'end_lat', 'end_lon']].apply\
#   (lambda x : haversine(x['start_lat'],x['start_lon'], lat_s, lon_s),axis=1)
#   
#df_isoc = df_isoc[(df_isoc["eucl_dist"] <= 0.5)]
#
#fig, ax = plt.subplots(1,1,figsize=(10,10))
#
#zones_geo.plot(color="gray",ax=ax)
#ax.set_xlim([7.56, 7.74])
#ax.set_ylim([45.0, 45.12])
#
#colors=['red','green','orange', 'blue', 'yellow', 'gray']
#hull= ConvexHull(df_isoc[['start_lon','start_lat']])
#for simplex in hull.simplices:
#    plt.plot(df_isoc['start_lon'].iloc[simplex], df_isoc['start_lat'].iloc[simplex], color='red', linewidth=3, label='_nolegend_' )
#
#for t in range(10,60,10):
#    df_isoc_time = df_isoc[(df_isoc['riding_time'] <= t) & (df_isoc['riding_time']>(t-10))]
#    print str(t) + ' abbiamo '+ str(len(df_isoc_time))
#    if(len(df_isoc_time) > 0):
#        print 'in {} minuti : {}'.format(t, len(df_isoc_time))
#        if len(df_isoc_time) >= 3:
#            hull= ConvexHull(df_isoc_time[['end_lon','end_lat']])
#            for simplex in hull.simplices:
#                plt.plot(df_isoc_time['end_lon'].iloc[simplex], df_isoc_time['end_lat'].iloc[simplex],color=colors[t/10], linewidth=3, label='_nolegend_' )
#        df_isoc_time.plot.scatter(x="end_lon", y='end_lat', label=str(t)+' minutes', s=100, ax=ax, color=colors[t/10])
#
#plt.title('Isochrone enjoy starting from Porta Nuova')
#plt.xlabel('latitude')
#plt.ylabel('longitude')
##plt.savefig('isochrone_enjoy.png')
#plt.legend()
#####################


#######################COSTO
#lat_s = pos_PortaNuova[0]
#lon_s = pos_PortaNuova[1]
#
#df_isoc = enjoy_short
#   
#df_isoc['eucl_dist'] = df_isoc[['start_lat', 'start_lon', 'end_lat', 'end_lon']].apply\
#   (lambda x : haversine(x['start_lat'],x['start_lon'], lat_s, lon_s),axis=1)
#   
#df_isoc = df_isoc[(df_isoc["eucl_dist"] <= 0.5)]
#
#fig, ax = plt.subplots(1,1,figsize=(10,10))
#
#zones_geo.plot(color="gray",ax=ax)
#ax.set_xlim([7.56, 7.74])
#ax.set_ylim([45.0, 45.12])
#
#colors=['red','green','orange', 'blue', 'purple', 'yellow']
#hull= ConvexHull(df_isoc[['start_lon','start_lat']])
#for simplex in hull.simplices:
#    plt.plot(df_isoc['start_lon'].iloc[simplex], df_isoc['start_lat'].iloc[simplex], color='red', linewidth=3, label='_nolegend_' )
#
#for t in range(2,10,2):
#   df_isoc_time = df_isoc[(df_isoc['min_bill'] <= t) & (df_isoc['min_bill']>(t-2))]
#   if(len(df_isoc_time) > 0):
#       print 'in {} minuti : {}'.format(t, len(df_isoc_time))
#       if len(df_isoc_time) >=3:
#           hull= ConvexHull(df_isoc_time[['end_lon','end_lat']])
#           for simplex in hull.simplices:
#               plt.plot(df_isoc_time['end_lon'].iloc[simplex], df_isoc_time['end_lat'].iloc[simplex],color=colors[t/2], linewidth=3, label='_nolegend_' )
#       df_isoc_time.plot.scatter(x="end_lon", y='end_lat', label=str(t)+' euro', s=100, ax=ax, color=colors[t/2])
# 
#plt.title('Isocosto car2go starting from Porta Nuova')
#plt.xlabel('latitude')
#plt.ylabel('longitude')
#plt.savefig('isocosto_enjoy.png')
#plt.legend()
#


####PDF QUELLI SOTTO GOOGLE ###
#plt.figure()
#plt.title('PDF duration less than duration_driving')
#
#df = enjoy_short[(enjoy_short['duration']-enjoy_short['duration_driving']) < 0]
#df_ = df['duration_driving']-df['duration']
##df_.hist(bins=100, normed=True, label='Enjoy')
#
#df2 = car2go_short[(car2go_short['duration']-car2go_short['duration_driving']) < 0]
#df2_ = df2['duration_driving']-df2['duration']
#
##df_finale = pd.concat([df_,df2_])
#
#df2_.hist(bins=100, normed=True, label='car2go', color='blue', alpha=1, cumulative=True)
#df_.hist(bins=100, normed=True, label='enjoy', color='red', alpha=0.5, cumulative=True)
#
##df_finale.hist(bins=100, normed=True)
##df_.hist(bins=100, normed=True,histtype='step', stacked=True, label='Car2Go')
##
#plt.axis((0,25,0,1.1))
#plt.xlabel('Duration [min]')
#plt.ylabel('Probability [p]')
#plt.legend()
##plt.savefig('CDF_duration_less_google.png')
#plt.show()
#
#
#plt.figure()
#plt.title('CDF duration less than duration_driving')
#plt.xlabel('Duration [min]')
#plt.ylabel('Probability [p]')
#plt.axis((0,25,0,1.1))
#df_finale.hist(bins=100, normed=True, cumulative=True)
#plt.savefig('CDF_duration_less_google.png')
#plt.show()

#plt.figure()
#plt.title('PDF (measured duration - google forecast)')
#
#df = pd.concat([enjoy_short, car2go_short])
#
#df_plot = df['duration'] - df['duration_driving']
#
#df_enjoy = enjoy_short['duration']-enjoy_short['duration_driving']
#df_car2go = car2go_short['duration']-car2go_short['duration_driving']
#
#plt.xlabel('Duration [min]')
#plt.ylabel('Probability [p]')
##df_plot.hist(bins=100, normed=True)
#
#df_car2go.hist(bins=100, normed=True, color='blue')
#df_enjoy.hist(bins=100, normed=True, color='red', alpha=0.6)
#
#plt.axis((-20,70,0,0.06))
#plt.legend()
##plt.savefig('measured-google.png')
#plt.show()

#plt.figure()
#plt.title('CDF (measured duration - google forecast)')
#
#df = pd.concat([enjoy_short, car2go_short])
#
#df_plot = df['duration'] - df['duration_driving']
#plt.xlabel('Duration [min]')
#plt.ylabel('Probability [p]')
#df_plot.hist(bins=100, normed=True, cumulative=True)
#plt.axis((-20,80,0,1.1))
#plt.legend()
##plt.savefig('CDF-measured-google.png')
#plt.show()


"""
Heatmap
"""

#g.heatmaps_per_hour(car2go_df)
#g.heatmaps_per_hour(enjoy_df)

"""
OD matrix
"""

#destinazioni = DestinationFromOrigin(enjoy_short,zones, 53)


#def DestinationFromOrigin(books_df, zones, origin):
#    
#    origins = books_df[["start_lat","start_lon","end_lat","end_lon"]]
#    origins.loc[:,"geometry"] = origins.apply\
#        (lambda row: Point(row.loc["start_lon"], row.loc["start_lat"]), 
#         axis=1)  
#    origins.loc[:,"geometry_end"] = origins.apply\
#    (lambda row: Point(row.loc["end_lon"], row.loc["end_lat"]), 
#     axis=1)  
#    
#    origin_zone = gpd.GeoDataFrame(zones.loc[origin]).T
#
#    origins['contained'] = origins['geometry'].apply(lambda w: origin_zone.contains(w))
#    origins = origins[origins['contained']]
#    origins = origins.reset_index()
#
#    dest = [0] * len(zones)
#    for i in range(len(origins)):
#        intersect = zones.contains(origins['geometry_end'][i])
#        z = intersect[intersect].index.values[0]
#        dest[z]+=1
#
#    return dest
    
#def getODmatrix (books_df, zones):
#    
#    origins = books_df[["start_lat","start_lon"]]
#    origins.loc[:,"geometry"] = origins.apply\
#        (lambda row: Point(row.loc["start_lon"], row.loc["start_lat"]), 
#         axis=1)    
#
#    destinations = books_df[["end_lat","end_lon"]]
#    destinations.loc[:,"geometry"] = destinations.apply\
#        (lambda row: Point(row.loc["end_lon"], row.loc["end_lat"]), 
#         axis=1)    
#
#    OD = pd.DataFrame(0.0, index = zones.index, columns = zones.index)
#
#    for i in range(len(books_df)):
#        try:
#            if i%10000 == 0:
#                print i
#            o = origins.ix[i, "geometry"]
#            d = destinations.ix[i, "geometry"]
#            intersect_o = zones.contains(o)
#            intersect_d = zones.contains(d)
#            zo = intersect_o[intersect_o == True].index.values[0]
#            zd = intersect_d[intersect_d == True].index.values[0]
#            OD.loc[zo, zd] += 1
#        except:
#            "Book" + str(i) + "has points not contained in any zone!"
#
#    return origins, destinations, OD
#    
#def filter_quantile(df, col, filter_col, quantile):
#    
#    s = df.loc[df[filter_col] == True, col].dropna()
#    s = s[(s >= s.quantile(q=quantile)) & (s <= s.quantile(q=1.0-quantile))]
#    return df.loc[s.index]
#
#origins, destinations, car2go_od = getODmatrix\
#    (filter_quantile(car2go_df, "start_lat", "ride", 0.001), zones)
#origins, destinations, enjoy_od = getODmatrix\
#    (filter_quantile(enjoy_df, "start_lat", "ride", 0.001), zones)
#
#def drop_od (od):
#    dropped_od = od
#    for zone in od:
#        zone_as_origin = od.iloc[zone]
#        zone_as_dest = od.iloc[:,zone]
#        if not zone_as_origin.sum() and not zone_as_dest.sum():
#            dropped_od = od.drop(zone, axis=0)
#            dropped_od = od.drop(zone, axis=1)
#    return dropped_od
#
#def standardize (x):
#    return (x-x.mean())/x.std()
#
#def force_positive (x):
#    return x+abs(x.min())
#
#dropped_car2go_od = drop_od(car2go_od)
#dropped_enjoy_od = drop_od(enjoy_od)
#
#zones["car2go_d_tot"] = dropped_car2go_od.sum(axis=1)
#zones["car2go_o_tot"] = dropped_car2go_od.sum(axis=0)
#zones["car2go_tot"] = (zones["car2go_o_tot"]-zones["car2go_d_tot"])*(zones["car2go_o_tot"]+zones["car2go_d_tot"])
#
#zones["enjoy_d_tot"] = dropped_enjoy_od.sum(axis=1)
#zones["enjoy_o_tot"] = dropped_enjoy_od.sum(axis=0)
#zones["enjoy_tot"] = (zones["enjoy_o_tot"]-zones["enjoy_d_tot"])*(zones["enjoy_o_tot"]+zones["enjoy_d_tot"])
#
#fig, axs = plt.subplots(2, 2, figsize=(18,18))
#zones.plot(column='car2go_o_tot', cmap='Blues', ax=axs[0][0])
#zones.plot(column='car2go_d_tot', cmap='Blues', ax=axs[0][1])
#zones.where((zones.car2go_d_tot>0) & (zones.car2go_o_tot>0)).dropna()\
#           .plot(column='car2go_o_tot', cmap='Blues', ax=axs[1][0])
#zones.where((zones.car2go_d_tot>0) & (zones.car2go_o_tot>0)).dropna()\
#           .plot(column='car2go_d_tot', cmap='Blues', ax=axs[1][1])
#
#fig, axs = plt.subplots(2, 2, figsize=(18,18))
#zones.plot(column='enjoy_o_tot', cmap='OrRd', ax=axs[0][0])
#zones.plot(column='enjoy_d_tot', cmap='OrRd', ax=axs[0][1])
#zones.where((zones.enjoy_d_tot>0) & (zones.enjoy_o_tot>0)).dropna()\
#           .plot(column='enjoy_o_tot', cmap='OrRd', ax=axs[1][0])
#zones.where((zones.enjoy_d_tot>0) & (zones.enjoy_o_tot>0)).dropna()\
#           .plot(column='enjoy_d_tot', cmap='OrRd', ax=axs[1][1])
