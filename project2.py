from traffic.data import samples
from geopy.distance import geodesic
from numpy.random import default_rng
import pyproj
import copy

#used for the KalmanFilter
from pykalman import KalmanFilter
import numpy as np

from traffic.data.samples import airbus_tree, quickstart, belevingsvlucht

import matplotlib.pyplot as plt

from traffic.core.projection import Amersfoort, GaussKruger, Lambert93, EuroPP
from traffic.drawing import countries

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
    radar_error = 0.1 # in kilometers
    gt = get_ground_truth_data()
    radar_data = []
    for flight in gt:
        #print("flight: %s" % (str(flight)))
        flight_radar = flight.resample("10s")
        for i in range(len(flight_radar.data)-1):
            point = geodesic(kilometers=rng.normal()*radar_error).destination((flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]), rng.random()*360)
            (flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]) = (point.latitude, point.longitude)
            #print("after: %f, %f" % (flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]))
        projection = pyproj.Proj(proj="lcc", ellps="WGS84", lat_1=flight_radar.data.latitude.min(), lat_2=flight_radar.data.latitude.max(), lat_0=flight_radar.data.latitude.mean(), lon_0=flight_radar.data.longitude.mean())
        flight_radar = flight_radar.compute_xy(projection)
        projection_for_flight[flight_radar.callsign]=projection
        radar_data.append(flight_radar)
    return radar_data

# returns the same flight with latitude and longitude changed to reflect the x, y positions in the data
# The intended use of this function is to:
#  1. make a copy of a flight that you got from get_radar_data
#  2. use a kalman filter on that flight and set the x, y columns of the data to the filtered positions
#  3. call set_lat_lon_from_x_y() on that flight to set its latitude and longitude columns according to the filitered x,y positions
# Step 3 is necessary, if you want to plot the data, because plotting is based on the lat/lon coordinates.
def set_lat_lon_from_x_y(flight):
    projection = projection_for_flight[flight.callsign]
    if projection is None:
        print("No projection found for flight %s. You probably did not get this flight from get_radar_data()." % (str(flight.flight_id)))
    
    lons, lats = projection(flight.data["x"], flight.data["y"], inverse=True)
    flight.data["longitude"] = lons
    flight.data["latitude"] = lats
    return flight


def visualization(flights):
    #visualization
    with plt.style.context("traffic"):

        fig = plt.figure()

        # Choose the projection type
        ax0 = fig.add_subplot(221, projection=EuroPP())

        for ax in [ax0]:
            ax.add_feature(countries())
            # Maximum extent for the map
            ax.set_global()
            # Remove border and set transparency for background
            ax.spines['geo'].set_visible(False)
            ax.background_patch.set_visible(False)
            for flight in flights:
                ret, *_ = flight.plot(ax0)

        # We reduce here the extent of the EuroPP() map
        # between 8°W and 18°E, and 40°N and 60°N
        ax0.set_extent((-8, 18, 40, 60))

        params = dict(fontname="Ubuntu", fontsize=18, pad=12)

        ax0.set_title("EuroPP()", **params)

        fig.tight_layout()
        plt.show()


#visualization but for only 2 flights
def visualization2(flights1, flights2 ):
    #visualization
    with plt.style.context("traffic"):

        fig = plt.figure()

        # Choose the projection type
        ax0 = fig.add_subplot(221, projection=EuroPP())
        ax1 = fig.add_subplot(222, projection=EuroPP())
        ax2 = fig.add_subplot(223, projection=EuroPP())

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

        # We reduce here the extent of the EuroPP() map
        # between 8°W and 18°E, and 40°N and 60°N
        ax0.set_extent((-8, 18, 40, 60))

        params = dict(fontname="Ubuntu", fontsize=18, pad=12)

        ax0.set_title("both()", **params)
        ax1.set_title("Expected()", **params)
        ax2.set_title("predicted()", **params)

        fig.tight_layout()
        plt.show()


def kalmanfilter(flight):

    t = 10
    std_Deviation_observation = 50
    kf = KalmanFilter(transition_matrices=[[1, 0, t, 0],
                                           [0, 1, 0, t],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]],
                      observation_matrices=[[1, 0, 0, 0],
                                            [0, 1, 0, 0]],
                      transition_covariance=[[10 ** 4 /4, 0, 10**3 /2, 0], #10 is delta t witch is 10s
                                             [0, 10**4 /4, 0, 10**3 /2],
                                             [0, 0, 10**2, 0],
                                             [0, 0, 0, 10**2]],
                      observation_covariance= [[50 ** 2, 0, t * (50**2), 0],
                                               [ 0, 50 ** 2, t * (50**2), 0]] )
    measurements = np.asarray([[flight.data['x'], flight.data['y'], 0, 0]])  # 3 observations
    kf = kf.em(measurements, n_iter=5)
    return kf


#input: flight (from filter), flight (from ground Data)
#output: [Rate of errors, avg distance of errors, maximum distance of one error]
def errorCalculations(predictedFllight, actualFlight):

    errorRate = 0
    errorDistance = 0
    maxErrorDistance = 0
    #for every flight
    for index in range(len(predictedFllight.data['longitude'])-1):
        #TO DO, USE GEOPY TO CALCULATE ACTUAL DISTANCE FOR ERRORDISTANCE

        #calculate the difference between the x and y of every flight
        error = abs(predictedFllight.data['longitude'][index] - actualFlight.data['longitude'][index]) + \
                abs(predictedFllight.data['latitude'][index] - actualFlight.data['latitude'][index])
        #if there was any error
        if(error != 0):
            errorRate += 1
            errorDistance += error
            #if the error is bigger then the last biggest error
            if(maxErrorDistance < error):
                maxErrorDistance = error

    #rate calculated by combining all and dividing by how many flights
    return [errorRate/len(predictedFllight), errorDistance/errorRate, maxErrorDistance]



def testing():
    flights = []  #predicted locations
    flights2 = [] #actual locations
    radarData = get_radar_data()
    true_data = get_ground_truth_data()
    #print(flight)
    #print(kalmanfilter(flight))
    t = 10
    standardDeviation_covariance = 50
    standardDeviation_transition = np.random.uniform(0.15, 0.3)
    kf = KalmanFilter(transition_matrices=[[1*standardDeviation_transition, 0, t*standardDeviation_transition, 0],
                                           [0, 1*standardDeviation_transition, 0, t*standardDeviation_transition],
                                           [0, 0, 1*standardDeviation_transition, 0],
                                           [0, 0, 0, 1*standardDeviation_transition]],
                      observation_matrices=[[1 *standardDeviation_covariance, 0, 0, 0],
                                            [0, 1 *standardDeviation_transition , 0, 0],
                                            [0,0,0,0],
                                            [0,0,0,0]],
                      transition_covariance=[[(t ** 4 / 4)*standardDeviation_transition , 0, (t ** 3 / 2)*standardDeviation_transition, 0],  # 10 is delta t witch is 10s
                                             [0, (t ** 4 / 4)*standardDeviation_transition, 0, (t ** 3 / 2)*standardDeviation_transition],
                                             [0, 0, (t ** 2)*standardDeviation_transition, 0],
                                             [0, 0, 0, (t ** 2)*standardDeviation_transition]],
                      observation_covariance=[  [standardDeviation_covariance ** 2, 0, t * (standardDeviation_covariance ** 2), 0],
                                                [0, standardDeviation_covariance ** 2, t * (standardDeviation_covariance ** 2), 0],
                                            [0,0,0,0],
                                            [0,0,0,0]]
                      )

    currIndex = -1
    for flight1 in radarData:
        flight = copy.copy(flight1)
        #print(flight.data.columns)
        currIndex += 1

        xv = (flight.data['x'][len(flight.data['x'])-1] - flight.data['x'][0])/ len(flight.data['x'])*10
        yv = (flight.data['y'][len(flight.data['y'])-1] - flight.data['y'][0])/ len(flight.data['y'])*10


        locations = np.stack((flight.data['x'], flight.data['y'], [xv]*flight.data['x'].size, [yv]*flight.data['y'].size ), axis=1)
        mesurements = kf.filter(locations) #returns [[x,y,velX,velY],[x,y,velX,velY]]
        xEs = []    #X's from the mesurements
        yEs =[]     #y's from the mesurements
        for i in mesurements:
            xEs.append(i[0])
            yEs.append(i[1])


        flight.data['x'].__setattr__('x', xEs)
        flight.data['y'].__setattr__('y', yEs)
        set_lat_lon_from_x_y(flight)

        flights.append(flight)


    for flight in true_data:
        flights2.append(flight)

    visualization2(flights, flights2)


    errorDistances = 0
    maxErrorDistance = 0
    errorRates = 0
    #error calculation for each flight
    for index in range(len(flights)):
        try:
            errorData = errorCalculations(flights[index], flights2[index])
            errorDistances += int(errorData[1])
            errorRates += errorData[0]
            if(errorData[2] > maxErrorDistance):
                maxErrorDistance = errorData[2]
            print(flight.callsign, ' has error rate ', errorData[0], ' max error ', errorData[2], ' avg error distance ', errorData[1])
            #print(errorCalculations(flights[index], flights2[index]))
        except:
            try:
                print(flight.data.columns)
                print(flight.callsign)
            except:
                print("no flight found ")
    print(errorDistances)
    print("mean error distance = ", (errorDistances/len(flights)))
    print("max distance error = ", maxErrorDistance)
    print("rate of errors over all = ", errorRates/len(flights))
    return "bla"


testing()

