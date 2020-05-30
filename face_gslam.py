import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import csv
import pandas as pd
import time
import math
import matplotlib
import icp
import random

#from read_data import read_world, read_sensor_data



map_cordinates = []
map_cordinates.append([])
map_cordinates.append([])


##correction step cordinates
cordinates2 = []
cordinates2.append([])
cordinates2.append([])

##prediction step cordinates
cordinates1 = []
cordinates1.append([])
cordinates1.append([])

lidar_data =[]
odom_data =[]
uwb_data =[]

now_lidar_data = []
now_uwb_data  = []
now_odom_data  = []

occ_val = []
xocc_val = []
occ_val_empty = []
res = 0.01

#initialize belief
mu = [0.0, 0.0, 0.0]

#not sigma degerleri degistirilebilir
sigma = np.array([[500.0, 0.0, 0.0],
                   [0.0, 500.0, 0.0],
                   [0.0, 0.0, 0.05]])

landmarks = [[600,750,0],[-600,1050,0],[1400,2000,0],[-1400,2000,0],[1400,-2000,0],[-1400,-2000,0]]
#landmarks = [[1400,2000,0],[-1400,2000,0],[1400,-2000,0],[-1400,-2000,0]]




##plot preferences, interactive plotting mode
#fig = plt.figure()
#plt.ion()
#plt.show()

def plot_state():
    yocc_val = occ_val

    for i in range(len(cordinates1[0])):
        yocc_val[int(cordinates1[0][i]/(res))][int(cordinates1[1][i]/(res))]=1

    #shift map
    yocc_val= np.roll(yocc_val, int(len(occ_val)/2), axis=0)
    yocc_val= np.roll(yocc_val, int(len(occ_val)/2), axis=1)


    plt.clf()
    plt.imshow(yocc_val)
    plt.pause(0.01)
    plt.clf()



def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion
    # model
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    #delta_vel = odometry['r1']     # redefine r1                   odom=>twist=>linear=>x
    #delta_vel_y = odometry['r1']   # redefine r1                   odom=>twist=>linear=>y
    #delta_w = odometry['t']        # redefine t                    odom=>twist=>angular=>z

    delta_vel = odometry[0]         # redefine r1                   odom=>twist=>linear=>x
    delta_w = odometry[1]           # redefine t                    odom=>twist=>angular=>z


    #motion noise                                             refine the value
    Q = np.array([[0.2, 0.0, 0.0],
                   [0.0, 0.2, 0.0],
                   [0.0, 0.0, 0.02]])

    noise = 0.1**2
    v_noise = delta_vel**2
    w_noise = delta_w**2

    sigma_u = np.array([[noise + v_noise, 0.0],[0.0, noise + w_noise]])
    B = np.array([[np.cos(theta), 0.0],[np.sin(theta), 0.0],[0.0, 1.0]])        #Q ile değiştir

    #noise free motion
    x_new = x + delta_vel*np.cos(theta)/30
    y_new = y + delta_vel*np.sin(theta)/30

    #we divided by 30 because delta w unit is rad/sec
    theta_new = theta + delta_w/30

    #delta_vel = math.sqrt(math.square(delta_vel_x)+math.square(delta_vel_y))
    #Jakobian of g with respect to the state
    G = np.array([[1.0, 0.0, -delta_vel * np.sin(theta)],
                    [0.0, 1.0, delta_vel * np.cos(theta)],
                    [0.0, 0.0, 1.0]])

    #new mu and sigma
    mu = [x_new, y_new, theta_new]
    #sigma = np.dot(np.dot(G,sigma),np.transpose(G)) + Q
    sigma = np.dot(np.dot(G, sigma), np.transpose(G)) + np.dot(np.dot(B, sigma_u), np.transpose(B))

    global cordinates1
    cordinates1[0].append(mu[0])
    cordinates1[1].append(mu[1])


    return mu, sigma


def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    #
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    #measured landmark ids and ranges
    ##!!!!!!! ros da kullanirken acilacak
    #ids = sensor_data['id']
    #ranges = sensor_data['range']
    ranges = sensor_data
    ids = len(landmarks) ## rosa tasirken kapat


    # Compute the expected range measurements for each landmark.
    # This corresponds to the function h
    H = []
    Z = []
    expected_ranges = []
    for i in range(ids):
        lm_id = i
        meas_range = ranges[i]
        lx = landmarks[lm_id][0]/1000
        ly = landmarks[lm_id][1]/1000
        #gercek de kullanmak icin lutfen z pozisyonu ekleyiniz !!!!
        #calculate expected range measurement
        range_exp = np.sqrt( (lx - x)**2 + (ly - y)**2 )
        #compute a row of H for each measurement
        H_i = [(x - lx)/range_exp, (y - ly)/range_exp, 0]
        H.append(H_i)
        Z.append(ranges[i]/1000)
        expected_ranges.append(range_exp)
    # noise covariance for the measurements
    R = 0.05 * np.eye(ids)
    # Kalman gain
    K_help = np.linalg.inv(np.dot(np.dot(H, sigma), np.transpose(H)) + R)
    K = np.dot(np.dot(sigma, np.transpose(H)), K_help)
    # Kalman correction of mean and covariance

    #print(np.dot(K, (np.array(Z) - np.array(expected_ranges))))

    mu = mu + np.dot(K, (np.array(Z) - np.array(expected_ranges)))
    sigma = np.dot(np.eye(len(sigma)) - np.dot(K, H), sigma)
    #mu[2] = theta
    print(mu)

    global cordinates2
    cordinates2[0].append(mu[0])
    cordinates2[1].append(mu[1])
    return mu, sigma


step = 0
def icp_process(lidar_scan):
    #global lidar_cordinates
    global mu,step

    lidar_cordinates = []
    lidar_cordinates.append([])
    lidar_cordinates.append([])

    s = 5
    step = step+ 1

    for i in range(len(lidar_scan[0])):
        if (math.isinf(lidar_scan[0][i]) == False):
            # rotate lidar_scan with robot pose mu
            G_scan = np.array([np.cos(mu[2]) * lidar_scan[0][i] - np.sin(mu[2]) * lidar_scan[1][i] + mu[0] ,
                               np.sin(mu[2]) * lidar_scan[0][i] + np.cos(mu[2]) * lidar_scan[1][i] + mu[1] ])

            # occ_value map and the gradient of occ_value map
            x_occ, y_occ = int(G_scan[0] / res), int(G_scan[1] / res)
            lidar_cordinates[0].append(x_occ)
            lidar_cordinates[1].append(y_occ)


    degers = []
    degers.append([])
    degers.append([])

    for i in range(len(map_cordinates[0])):
        degers[0].append(map_cordinates[0][i])
        degers[1].append(map_cordinates[1][i])




    dx,dy,rotation,error = icp.icp_start(degers,lidar_cordinates,step)
    print(np.deg2rad(rotation))

    #if np.abs(np.deg2rad(rotation))<3:
    mu[0] = mu[0] + (res*dx)
    mu[1] = mu[1] + (res*dy)
    mu[2] -= np.deg2rad(rotation)


    robot_x = mu[0]/res
    robot_y = mu[1]/res


    for i in range(len(lidar_scan[0])):
        if (math.isinf(lidar_scan[0][i]) == False):
            # rotate lidar_scan with robot pose mu
            G_scan = np.array([np.cos(mu[2]) * lidar_scan[0][i] - np.sin(mu[2]) * lidar_scan[1][i] + mu[0],
            np.sin(mu[2]) * lidar_scan[0][i] + np.cos(mu[2]) * lidar_scan[1][i] + mu[1]])

            # occ_value map and the gradient of occ_value map
            x_occ, y_occ = int(G_scan[0] / res), int(G_scan[1] / res)

            map_cordinates[0].append(x_occ)
            map_cordinates[1].append(y_occ)

            if occ_val[x_occ][y_occ] == 1:
                angle = np.rad2deg(np.arctan((x_occ - robot_y) / (y_occ - robot_x)))
                search_x = int(np.ceil(np.cos(np.deg2rad(angle)) + x_occ))
                search_y = int(np.ceil(np.sin(np.deg2rad(angle)) + y_occ))
                #occ_val[search_x][search_y]=0

                angle = angle - 180
                search_x = int(np.floor(np.cos(np.deg2rad(angle)) + x_occ))
                search_y = int(np.floor(np.sin(np.deg2rad(angle)) + y_occ))
                #occ_val[search_x][search_y]=0

            occ_val[x_occ][y_occ]=1

    global cordinates2
    cordinates2[0].append(mu[0])
    cordinates2[1].append(mu[1])


def map_matching(lidar_scan ):
    global mu,occ_val,occ_val_empty,xocc_val
    global map_cordinates

    global tra
    pos_x = mu[0]
    pos_y = mu[1]
    pos_theta = mu[2]
    H = np.zeros((3,3))
    G = np.zeros((3,1))


    xocc_val = occ_val_empty

    for i in range(len(lidar_scan[0])):
        if (math.isinf(lidar_scan[0][i])==False):

            # rotate lidar_scan with robot pose mu
            R_scan = np.array([np.cos(pos_theta)*lidar_scan[0][i] - np.sin(pos_theta)*lidar_scan[1][i] + pos_x,
                                    np.sin(pos_theta)*lidar_scan[0][i] + np.cos(pos_theta)*lidar_scan[1][i] + pos_y])


            # occ_value map and the gradient of occ_value map
            i_occ, j_occ = int(R_scan[0]/res), int(R_scan[1]/res)


            P_00 = [i_occ - 1, j_occ + 1]
            P_10 = [i_occ + 1, j_occ + 1]
            P_01 = [i_occ - 1, j_occ - 1]
            P_11 = [i_occ + 1, j_occ - 1]

            M_occ = ((R_scan[1]-P_00[1]*res)/(P_01[1]*res - P_00[1]*res)*\
                    (((R_scan[0] - P_00[0]*res)/(P_11[0]*res - P_00[0]*res))*occ_val[P_11[0]][P_11[1]] +\
                    ((P_11[0]*res - R_scan[0])/(P_11[0]*res - P_00[0]*res))*occ_val[P_01[0]][P_01[1]])) +\
                    ((P_01[1]*res-R_scan[1])/(P_01[1]*res - P_00[1]*res)*\
                    (((R_scan[0] - P_00[0]*res)/(P_11[0]*res - P_00[0]*res))*occ_val[P_10[0]][P_10[1]]+\
                    ((P_11[0]*res - R_scan[0])/(P_11[0]*res - P_00[0]*res))*occ_val[P_00[0]][P_00[1]]))


            if M_occ>0.01:
                M_occ = 1
            else:
                M_occ=0


            G_M = np.array([(((R_scan[1]-P_00[1]*res)/(P_01[1]*res - P_00[1]*res))*(occ_val[P_11[0]][P_11[1]] - occ_val[P_01[0]][P_01[1]]) +\
                  ((P_01[1]*res-R_scan[1])/(P_01[1]*res - P_00[1]*res))*(occ_val[P_10[0]][P_10[1]]-occ_val[P_00[0]][P_00[1]]),
                           ((R_scan[0]-P_00[0]*res)/(P_11[0]*res - P_00[0]*res))*(occ_val[P_11[0]][P_11[1]] - occ_val[P_10[0]][P_10[1]]) +\
                  ((P_11[0]*res-R_scan[0])/(P_11[0]*res - P_00[0]*res))*(occ_val[P_01[0]][P_01[1]]-occ_val[P_00[0]][P_00[1]]))])

            #gradient of R_scan with respect to pose
            G_S = np.array([[1, 0, -np.sin(pos_theta)*lidar_scan[0][i]-np.cos(pos_theta)*lidar_scan[1][i]], [0, 1, np.cos(pos_theta)*lidar_scan[0][i]-np.sin(pos_theta)*lidar_scan[1][i]]])


            H = H + np.dot(np.transpose(np.dot(G_M,G_S)),np.dot(G_M,G_S))

            G = G + np.transpose(np.dot(G_M,G_S)*(1-M_occ))


    delta_pos = np.dot(np.linalg.pinv(H),G)
    delta_pos = np.transpose(delta_pos)


    if ((delta_pos[0][0]*delta_pos[0][0] + delta_pos[0][1]*delta_pos[0][1]) <0.01): #0.16 dusurulebilir

        mu[0] += delta_pos[0][0]
        mu[1] += delta_pos[0][1]
        mu[2] += delta_pos[0][2]
        #mu[2] = pos_theta

        for i in range(len(lidar_scan[0])):
            if (math.isinf(lidar_scan[0][i]) == False  ):
                G_scan = np.array([np.cos(mu[2]) * lidar_scan[0][i] - np.sin(mu[2]) * lidar_scan[1][i] + mu[0],
                    np.sin(mu[2]) * lidar_scan[0][i] + np.cos(mu[2]) * lidar_scan[1][i] + mu[1]])

                # occ_value map and the gradient of occ_value map
                x_occ, y_occ = int(G_scan[0] / res), int(G_scan[1] / res)
                occ_val[x_occ][y_occ]=1

        '''
        print("")
        print("delta pos\t:",np.rad2deg(delta_pos[0][2]))
        print("thetha  \t:",pos_theta)
        print("mu[2]  \t\t:",mu[2])
        '''




def map_init(res,width,height):
    global occ_val,occ_val_empty
    occ_val = np.zeros((int(width/res), int(height/res)))
    occ_val_empty =np.zeros((int(width/res), int(height/res)))

def map_init_fill(res):
    global occ_val,now_lidar_data
    global map_cordinates
    global mu

    pos_x = mu[0]
    pos_y = mu[1]
    pos_theta = mu[2]



    for i in range(len(now_lidar_data[0])):
        if (math.isinf(now_lidar_data[0][i])==False):
            R_scan = np.array([np.cos(pos_theta) * now_lidar_data[0][i] - np.sin(pos_theta) * now_lidar_data[1][i] + pos_x,
                               np.sin(pos_theta) * now_lidar_data[0][i] + np.cos(pos_theta) * now_lidar_data[1][i] + pos_y])

            x_cor = int(np.floor(R_scan[0]/res))
            y_cor = int(np.floor(R_scan[1]/res))

            occ_val[x_cor][y_cor] = 1

            map_cordinates[0].append(x_cor)
            map_cordinates[1].append(y_cor)
            #occ_val[int(np.floor(occ_val.shape[0] / 2))][int(np.floor(occ_val.shape[1] / 2))] = 2



def time_step():
    global  mu ,sigma,occ_val
    global now_odom_data,now_lidar_data,now_uwb_data

    uwb_init_control = 0
    mu[2] = np.deg2rad(-106)
    i = 0
    while (i<len(odom_data)-7 ): #-2 sonradan silinecek
        now_odom_data=odom_cal(i)
        mu, sigma = prediction_step(now_odom_data, mu, sigma)
        if((i%6)==0):
            uwb_init_control = uwb_init_control + 1

            now_lidar_data = lidar_pos_cal(i/6)
            now_uwb_data = uwb_cal(i/6)
            #mu, sigma = correction_step(now_uwb_data, mu, sigma, landmarks)


            if uwb_init_control==10:
                map_init_fill(res)
            elif uwb_init_control>10:

                if uwb_init_control==25:
                    occ_val = np.zeros((int(10.0 / res), int(10.0/ res)))


                #map_matching(now_lidar_data)
                icp_process(now_lidar_data)
                plot_state()

            #sınırlandırmak için
            #if uwb_init_control==40:
            #     break

        i=i+1
        #time.sleep(0.01)


    #add trajectory
    for i in range(len(cordinates1[0])):
        occ_val[int(cordinates1[0][i]/(res))][int(cordinates1[1][i]/(res))]=1

    ##shift map
    occ_val = np.roll(occ_val, int(len(occ_val)/2), axis=0)
    occ_val = np.roll(occ_val, int(len(occ_val)/2), axis=1)

    for  x in range(len(occ_val[0])):
        for y in range(len(occ_val[0])):
            if  occ_val[x][y]>0:
                occ_val[x][y]=1

    plt.imshow(occ_val)
    #plt.plot(cordinates1[0],cordinates1[1], 'ro')
    plt.show()

    #plt.axes().set_aspect('equal', 'datalim')
    #plt.plot(cordinates1[0], cordinates1[1], 'ro')
    #plt.show()



def lidar_pos_cal(indexx):
    try:
        scan = lidar_data.ranges[indexx][1:-1].split(', ')
    except:
        print(indexx)

    map(float, scan)

    cordinates = []
    cordinates.append([])
    cordinates.append([])

    for i in range(len(scan)):
        cordinates[0].append(float(scan[i]) * math.cos(np.deg2rad(i)))
        cordinates[1].append(float(scan[i]) * math.sin(np.deg2rad(i)))

    return cordinates

def uwb_cal(indexx):
    uwb = uwb_data.distance[indexx][1:-1].split(', ')
    uwb = list(map(float, uwb))
    return uwb

def odom_cal(indexx):
    odom_linear_x = odom_data.twist_twist_linear_x[indexx]
    odom_angular_z= odom_data.twist_twist_angular_z[indexx]
    return [odom_linear_x,odom_angular_z]




def main():
    # implementation of an extended Kalman filter for robot pose estimation
    global lidar_data
    global odom_data
    global uwb_data
    folder = "samples/2/"
    print("Reading sensor data")
    lidar_data = pd.read_csv(folder+"scan.csv")
    odom_data = pd.read_csv(folder+"odom.csv")
    uwb_data = pd.read_csv(folder+"uwb.csv")



    #sensor_readings = read_sensor_data("../data/sensor_data.dat")

    map_init(res,10.0,10.0)

    #occ_value = [] # 0 (serbest) ve 1 (dolu) doldurulması gerekiyor (400 cmx 400 cm, 5cm resolution, 80 cells for 4 m)
    #res = 5 # cm
    # map_limits = [-1, 12, -1, 10]

    #run kalman filter


    #run simulatoion
    time_step()
    '''
    for timestep in range(len(sensor_readings)//2):

        # #plot the current state
        # plot_state(mu, sigma, landmarks, map_limits)

        #perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep,'odometry'], mu, sigma)

        #perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

        #perform scan matching and calculate the offset

        #mu, occ_value = map_matching(mu, lidar_scan, map_scan, occ_value, res)
    '''

if __name__ == "__main__":
    main()