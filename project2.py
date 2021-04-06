from traffic.data import samples
from geopy.distance import geodesic
from numpy.random import default_rng
from traffic.core.projection import Amersfoort, GaussKruger, Lambert93, PlateCarree
import pyproj
import copy

#used for the KalmanFilter
from pykalman import KalmanFilter
import numpy as np

from traffic.data.samples import airbus_tree, quickstart, belevingsvlucht

import matplotlib.pyplot as plt

from traffic.core.projection import Amersfoort, GaussKruger, Lambert93, PlateCarree
from traffic.drawing import countries

import pandas as pd
# returns a list of flights with the original GPS data
def get_ground_truth_data():
    names=['liguria', 'pixair_toulouse', 'indiana', 'texas', 'georeal_fyn_island', 'ign_mercantour', 'ign_fontainebleau', 'mecsek_mountains', 'ign_lot_et_garonne', 'inflight_refuelling', 'aircraft_carrier', 'luberon', 'alto_adige', 'franconia', 'danube_valley', 'cevennes', 'oxford_cambridge', 'alpi_italiane', 'rega_zh', 'samu31', 'rega_sg', 'monastir', 'guatemala', 'london_heathrow', 'cardiff', 'sydney', 'brussels_ils', 'ajaccio', 'toulouse', 'noumea', 'london_gatwick', 'perth', 'kota_kinabalu', 'montreal', 'funchal', 'nice', 'munich', 'vancouver', 'lisbon', 'liege_sprimont', 'kiruna', 'bornholm', 'kingston', 'brussels_vor', 'vienna', 'border_control', 'dreamliner_boeing', 'texas_longhorn', 'zero_gravity', 'qantas747', 'turkish_flag', 'airbus_tree', 'easter_rabbit', 'belevingsvlucht', 'anzac_day', 'thankyou', 'vasaloppet']
    return [samples.__getattr__(x) for x in names]

# needed for set_lat_lon_from_x_y below
# is set by get_radar_data()
projection_for_flight = {}

# Returns the same list of flights as get_ground_truth_data(), but with the position data modified as if it was a reading from a radar
# i.e., the data is less accurate and with fewer points than the one from get_ground_truth_data()
# The flights in this list will have x, y coordinates set to suitable 2d projection of the lat/lon positions.
# You can access these coordinates in the Flight.data attribute, which is a Pandas DataFrame.
def get_radar_data():
    rng = default_rng()
    radar_error = 0.1  # in kilometers
    radar_altitude_error = 330  # in feet ( ~ 100 meters)
    gt = get_ground_truth_data()
    radar_data = []
    for flight in gt:
        print("flight: %s" % (str(flight)))
        flight_radar = flight.resample("10s")
        for i in range(len(flight_radar.data)):
            point = geodesic(kilometers=rng.normal() * radar_error).destination(
                (flight_radar.data.at[i, "latitude"], flight_radar.data.at[i, "longitude"]), rng.random() * 360)
            (flight_radar.data.at[i, "latitude"], flight_radar.data.at[i, "longitude"]) = (
            point.latitude, point.longitude)
            flight_radar.data.at[i, "altitude"] += rng.normal() * radar_altitude_error
            # print("after: %f, %f" % (flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]))
        projection = pyproj.Proj(proj="lcc", ellps="WGS84", lat_1=flight_radar.data.latitude.min(),
                                 lat_2=flight_radar.data.latitude.max(), lat_0=flight_radar.data.latitude.mean(),
                                 lon_0=flight_radar.data.longitude.mean())
        flight_radar = flight_radar.compute_xy(projection)
        flightid = flight_radar.callsign + str(flight_radar.start)
        if flightid in projection_for_flight:
            print("ERROR: duplicate flight ids: %s" % (flightid))
        projection_for_flight[flight_radar.callsign + str(flight_radar.start)] = projection
        radar_data.append(flight_radar)
    return radar_data


# returns the same flight with latitude and longitude changed to reflect the x, y positions in the data
# The intended use of this function is to:
#  1. make a copy of a flight that you got from get_radar_data
#  2. use a kalman filter on that flight and set the x, y columns of the data to the filtered positions
#  3. call set_lat_lon_from_x_y() on that flight to set its latitude and longitude columns according to the filitered x,y positions
# Step 3 is necessary, if you want to plot the data, because plotting is based on the lat/lon coordinates.
def set_lat_lon_from_x_y(flight):
    flightid = flight.callsign + str(flight.start)
    projection = projection_for_flight[flightid]
    if projection is None:
        print("No projection found for flight %s. You probably did not get this flight from get_radar_data()." % (
            flightid))

    lons, lats = projection(flight.data["x"], flight.data["y"], inverse=True)
    flight.data["longitude"] = lons
    flight.data["latitude"] = lats
    return flight

#visualization for two flights (prediced route, actual route)
def visualization(flight1, flight2):
    #visualization
    with plt.style.context("traffic"):

        fig = plt.figure()

        # Choose the projection type
        ax0 = fig.add_subplot(221, projection=PlateCarree())
        ax1 = fig.add_subplot(222, projection=PlateCarree())

        for ax in [ax0, ax1]:
            ax.add_feature(countries())
            # Maximum extent for the map
            ax.set_global()
            # Remove border and set transparency for background
            ax.spines['geo'].set_visible(False)
            ax.background_patch.set_visible(False)

        ret, *_ = flight1.plot(ax0)
        ret, *_ = flight2.plot(ax1)

        params = dict(fontname="Ubuntu", fontsize=18, pad=12)

        ax0.set_title("predicted()", **params)
        ax1.set_title("Expected()", **params)

        fig.tight_layout()
        plt.show()


#visualization for all flight, first predicted then expected
def visualization2(flights1, flights2 ):
    #visualization
    with plt.style.context("traffic"):

        fig = plt.figure()

        # Choose the projection type
        ax0 = fig.add_subplot(221, projection=PlateCarree())
        ax1 = fig.add_subplot(222, projection=PlateCarree())
        ax2 = fig.add_subplot(223, projection=PlateCarree())

        for ax in [ax0, ax1, ax2]:
            ax.add_feature(countries())
            # Maximum extent for the map
            ax.set_global()
            # Remove border and set transparency for background
            ax.spines['geo'].set_visible(False)
            ax.background_patch.set_visible(False)


        for flight in flights1:
            ret, *_ = flight.plot(ax0)
            ret, *_ = flight.plot(ax1)

        for flight in flights2:
            ret, *_ = flight.plot(ax0)
            ret, *_ = flight.plot(ax2)


        params = dict(fontname="Ubuntu", fontsize=18, pad=12)

        ax0.set_title("both()", **params)
        ax1.set_title("Predicted()", **params)
        ax2.set_title("Expected()", **params)

        fig.tight_layout()
        plt.show()



#input: flight (from filter), flight (from ground Data)
#output: [Rate of errors, avg distance of errors, maximum distance of one error]
def errorCalculations(predictedFllight, actualFlight):

    #Rate of errors,
    #error distance (used to calculate avrage distance
    #biggest error distance
    errorRate = 1
    errorDistance = 0
    maxErrorDistance = 0
    #if there is more info in the "PredictedFlights" go through every route in there
    #else if there is more in the "ActualFlights" go through every route in there
    #(since they are both calibrated for every 10 seconds)
    if(len(predictedFllight.data['longitude']) > len(actualFlight.data['longitude'])):
        lenValues = len(actualFlight.data['longitude'])
    else:
        lenValues = len(predictedFllight.data['longitude'])


    #for every longitude/latitude value
    for index in range(lenValues):
        #TO DO, USE GEOPY TO CALCULATE ACTUAL DISTANCE FOR ERRORDISTANCE

        #calculate the difference between the x and y of every flight
        error = abs(predictedFllight.data['longitude'][index] - actualFlight.data['longitude'][index]) + \
                abs(predictedFllight.data['latitude'][index] - actualFlight.data['latitude'][index])
        #if there was any error
        if(error != 0 and ~np.isnan(error)):
            errorRate += 1
            #print('error' , error)
            errorDistance += error
            #if the error is bigger then the last biggest error
            if(maxErrorDistance < error):
                maxErrorDistance = error
        else:
            print("NAAAAAAAN")
            #print(predictedFllight.data['longitude'][index])
            #print(predictedFllight.data['latitude'][index])
            #print(actualFlight.data)
            #print(actualFlight.data.columns)

            #rate calculated (nr.errors/allFlights), avg. Distance of errors, Maximum Distance of one error
    return [errorRate/lenValues, errorDistance/errorRate, maxErrorDistance]



def testing(sigmaP, sigmaO ):
    flights = []    #predicted locations
    flights2 = []   #actual locations

    radarData = get_radar_data()    #Data From the Radar
    true_data = get_ground_truth_data() #Actual Data


    t = 10 #Delta Time
    standardDeviation_covariance = sigmaO **2
    standardDeviation_transition = sigmaP **2
    #standardDeviation_transition = np.random.uniform(0.15, 0.3)   #when you want sigmaP to be a random nr. between 0.15 and 0.3
    for flight1 in radarData:
        flight = copy.copy(flight1) #copy flight

        #kalman Filter from our implementation
        kf = KalmanFilter(transition_matrices=[[1*(standardDeviation_transition), 0, t**2  * (standardDeviation_transition), 0],
                                               [0, 1*(standardDeviation_transition), 0, t**2 * (standardDeviation_transition)],
                                               [t**2  * (standardDeviation_transition), 0, 1*(standardDeviation_transition), 0],
                                               [0, t**2 * (standardDeviation_transition), 0, 1*(standardDeviation_transition)]],
                          observation_matrices=[[1 *standardDeviation_covariance, 0, 0, 0],
                                                [0, 1 *standardDeviation_covariance, 0, 0],
                                                [0,0,0,0],  #originally this was a 2x4 matix but Pykal would not accept that, so 2 lines were added to make this a 4x4 matrix
                                                [0,0,0,0]],
                          transition_covariance=[[(t ** 4 / 4)* (standardDeviation_transition), 0, (t ** 3 / 2)*(standardDeviation_transition), 0],
                                                 [0, (t ** 4 / 4)*(standardDeviation_transition), 0, (t ** 3 / 2)*(standardDeviation_transition)],
                                                 [(t ** 3 / 2)*(standardDeviation_transition), 0, (t ** 2)*(standardDeviation_transition), 0],
                                                 [0, (t ** 3 / 2)*(standardDeviation_transition), 0, (t ** 2)*(standardDeviation_transition)]],
                          observation_covariance=[[standardDeviation_covariance, 0, t * (standardDeviation_covariance), 0],
                                                  [0, standardDeviation_covariance, t * (standardDeviation_covariance), 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]
                          )
        #avrage acceleration calculation
        xv = (flight.data['x'][len(flight.data['x'])-1] - flight.data['x'][0])/ len(flight.data['x'])*10
        yv = (flight.data['y'][len(flight.data['y'])-1] - flight.data['y'][0])/ len(flight.data['y'])*10

        #acceleration per point
        accelerationX = []
        accelerationY = []
        for i in range(len(flight.data['x'])):
            if(i == 0):
                #if it's the first in the list then time is 0 so the acceleration is 0 for both
                accelerationX.append(0)
                accelerationY.append(0)
            else:
                #acceleration calculated per point (x2 -x1)/time, (y2 -y1)/time
                accelerationX.append((flight.data['x'][i] - flight.data['x'][i-1])/10)
                accelerationY.append((flight.data['y'][i] - flight.data['y'][i-1])/10)



        #locations from the flight, [x,y,vel.x, vel.y]
        #locations = np.stack((flight.data['x'], flight.data['y'], [xv]*flight.data['x'].size, [yv]*flight.data['y'].size ), axis=1)
        locations = np.stack((flight.data['x'], flight.data['y'], accelerationX, accelerationY ), axis=1)

        #mesurments from the KalmanFilter
        mesurements = kf.filter(locations) #returns [[x,y,velX,velY],[x,y,velX,velY]]

        xEs = []    #X's from the mesurements
        yEs =[]     #y's from the mesurements
        currIndex = -1 #to hold the index

        #taking the mesurements from the KalmanFilter and putting the X coordinates in the xEs list and y in the yEs list
        for i in mesurements[0]:
            currIndex += 1
            xEs.append(i[0] + flight.data['x'][currIndex])
            yEs.append(i[1] + flight.data['y'][currIndex])
        #replace old 'x', 'y' with the prediced 'x' and 'y' from the KalmanFilter
        flight.data['x'] = flight.data['x'].replace([flight.data['x']], [pd.Series(xEs)])
        flight.data['y'] = flight.data['y'].replace([flight.data['y']], [pd.Series(yEs)])

        #set longitude and latitude from the new x and y coordinates
        set_lat_lon_from_x_y(flight)


        #put the new predicted flight in the "flights" list
        flights.append(flight)

    #put actual rout of flights in "flights2" list
    for flight in true_data:
        flight.resample("10s") #get data for every 10 seconds
        flights2.append(flight)



    errorDistances = 0
    maxErrorDistance = 0
    errorRates = 0
    worstFlightPredicted = flight
    worstFlightExpected = flight

    #error calculation for each flight
    for index in range(len(flights)):
        #try:
            errorData = errorCalculations(flights[index], flights2[index])
            errorDistances += errorData[1].item()
            errorRates += errorData[0]
            if(errorData[2] > maxErrorDistance):
                maxErrorDistance = errorData[2]

                visualization(flights[index], flights2[index])
            #print(flights[index].callsign, ' has error rate     ', errorData[0], '   max error ', errorData[2], '    avg error distance     ', errorDistances)
        #except:
            #if flight could not be found or processed then print it out
         #   try:
          #      print(flights[index].data['latitude'])
          #      print(flights[index].data['longitude'])
          #      print(flights2[index].data['latitude'])
          #      print(flights2[index].data['longitude'])
          #      visualization(flights[index], flights2[index])
          #  except:
          #      print("no flight found ")
    print("in conclusion ")
    print("mean error distance = ", errorDistances/len(flights))
    print("max distance error = ", maxErrorDistance)
    print("rate of errors over all = ", errorRates/len(flights))
    #visualization of all flights (predicted route flights, actual route flights)
    visualization2(flights, flights2)
    return "bla"

for i in [[ 0.3, 50] , [0.15, 50], [0.3, 25], [0.15, 25]]:#, [50, 0.20], [50, 0.25], [50, 0.3], [50, 0.2], [50, 0.2], [50, 0.2]]:  #sigma changes, for testing
    testing(i[0], i[1]) #implementation of the kalman filter

