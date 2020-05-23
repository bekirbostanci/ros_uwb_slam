import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
#from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse
import csv
import pandas as pd
import time
import math
import matplotlib

# #plot preferences, interactive plotting mode
# fig = plt.figure()
# plt.axis([-1, 12, 0, 10])
# plt.ion()
# plt.show()
#
# def plot_state(mu, sigma, landmarks, map_limits):
#     # Visualizes the state of the kalman filter.
#     #
#     # Displays the mean and standard deviation of the belief,
#     # the state covariance sigma and the position of the
#     # landmarks.
#
#     # landmark positions
#     lx = []
#     ly = []
#
#     for i in range (len(landmarks)):
#         lx.append(landmarks[i+1][0])
#         ly.append(landmarks[i+1][1])
#
#     # mean of belief as current estimate
#     estimated_pose = mu
#
#     #calculate and plot covariance ellipse
#     covariance = sigma[0:2, 0:2]
#     eigenvals, eigenvecs = np.linalg.eig(covariance)
#
#     #get largest eigenvalue and eigenvector
#     max_ind = np.argmax(eigenvals)
#     max_eigvec = eigenvecs[:, max_ind]
#     max_eigval = eigenvals[max_ind]
#
#     #get smallest eigenvalue and eigenvector
#     min_ind = 0
#     if max_ind == 0:
#         min_ind = 1
#
#     min_eigvec = eigenvecs[:, min_ind]
#     min_eigval = eigenvals[min_ind]
#
#     #chi-square value for sigma confidence interval
#     chisquare_scale = 2.2789
#
#     #calculate width and height of confidence ellipse
#     width = 2 * np.sqrt(chisquare_scale*max_eigval)
#     height = 2 * np.sqrt(chisquare_scale*min_eigval)
#     angle = np.arctan2(max_eigvec[1], max_eigvec[0])
#
#     #generate covariance ellipse
#     ell = Ellipse(xy = [estimated_pose[0], estimated_pose[1]], width=width, height=height, angle=angle/np.pi*180)
#     ell.set_alpha(0.25)
#
#     # plot filter state and covariance
#     plt.clf()
#     plt.gca().add_artist(ell)
#     plt.plot(lx, ly, 'bo', markersize=10)
#     plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy', scale_units='xy')
#     plt.axis(map_limits)
#
#     plt.pause(0.01)

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


    #delta_vel = odometry['r1'] # redefine r1                  odom=>twist=>linear=>x
    #delta_vel_y = odometry['r1'] # redefine r1               odom=>twist=>linear=>y
    #delta_w = odometry['t']    # redefine t                   odom=>twist=>angular=>z

    delta_vel = odometry[0] # redefine r1                  odom=>twist=>linear=>x
    delta_w = odometry[1]    # redefine t                   odom=>twist=>angular=>z


    #motion noise                                             refine the value
    Q = np.array([[0.2, 0.0, 0.0],  
                   [0.0, 0.2, 0.0],
                   [0.0, 0.0, 0.02]])

    noise = 0.1**2
    v_noise = delta_vel**2
    w_noise = delta_w**2
    #
    sigma_u = np.array([[noise + v_noise, 0.0],[0.0, noise + w_noise]])
    B = np.array([[np.cos(theta), 0.0],
                   [np.sin(theta), 0.0],
                   [0.0, 1.0]])

    #noise free motion
    x_new = x + delta_vel*np.cos(theta)
    y_new = y + delta_vel*np.sin(theta)
    theta_new = theta + delta_w
    
    #delta_vel = math.sqrt(math.square(delta_vel_x)+math.square(delta_vel_y))
    #Jakobian of g with respect to the state
    G = np.array([[1.0, 0.0, -delta_vel * np.sin(theta)],
                    [0.0, 1.0, delta_vel * np.cos(theta)],
                    [0.0, 0.0, 1.0]])
    
    #new mu and sigma
    mu = [x_new, y_new, theta_new]
    #sigma = np.dot(np.dot(G,sigma),np.transpose(G)) + Q
    sigma = np.dot(np.dot(G, sigma), np.transpose(G)) + np.dot(np.dot(B, sigma_u), np.transpose(B))
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
    ids = 4 ## rosa tasirken kapat


    # Compute the expected range measurements for each landmark.
    # This corresponds to the function h
    H = []
    Z = []
    expected_ranges = []
    for i in range(ids):
        lm_id = i
        meas_range = ranges[i]
        lx = landmarks[lm_id][0]
        ly = landmarks[lm_id][1]
        #gercek de kullanmak icin lutfen z pozisyonu ekleyiniz !!!!
        #calculate expected range measurement
        range_exp = np.sqrt( (lx - x)**2 + (ly - y)**2 )
        #compute a row of H for each measurement
        H_i = [(x - lx)/range_exp, (y - ly)/range_exp, 0]
        H.append(H_i)
        Z.append(ranges[i])
        expected_ranges.append(range_exp)
    # noise covariance for the measurements
    R = 0.5 * np.eye(ids)
    # Kalman gain
    K_help = np.linalg.inv(np.dot(np.dot(H, sigma), np.transpose(H)) + R)
    K = np.dot(np.dot(sigma, np.transpose(H)), K_help)
    # Kalman correction of mean and covariance
    mu = mu + np.dot(K, (np.array(Z) - np.array(expected_ranges)))
    sigma = np.dot(np.eye(len(sigma)) - np.dot(K, H), sigma)

    #print(mu)
    return mu, sigma


def map_matching(lidar_scan , occ_value, res):
    global mu,occ_val,occ_val_empty
    pos_x = mu[0]/1000
    pos_y = mu[1]/1000
    pos_theta = mu[2]
    H = np.zeros((3,3))
    G = np.zeros((3,1))


    occ_val_empty = np.zeros((200,200))
    plot1 = plt.figure(1)
    plt.imshow(occ_val)

    xocc_val = occ_val_empty


    for i in range(len(lidar_scan[0])):
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
                (((R_scan[0] - P_00[0]*res)/(P_11[0]*res - P_00[0]*res))*occ_value[P_11[0]][P_11[1]] +\
                ((P_11[0]*res - R_scan[0])/(P_11[0]*res - P_00[0]*res))*occ_value[P_01[0]][P_01[1]])) +\
                ((P_01[1]*res-R_scan[1])/(P_01[1]*res - P_00[1]*res)*\
                (((R_scan[0] - P_00[0]*res)/(P_11[0]*res - P_00[0]*res))*occ_value[P_10[0]][P_10[1]]+\
                ((P_11[0]*res - R_scan[0])/(P_11[0]*res - P_00[0]*res))*occ_value[P_00[0]][P_00[1]]))

        if M_occ>0.01:
            M_occ = 1
            xocc_val[i_occ][j_occ] = 1
        else:
            xocc_val[i_occ][j_occ] = 0



        G_M = np.array([(((R_scan[1]-P_00[1]*res)/(P_01[1]*res - P_00[1]*res))*(occ_value[P_11[0]][P_11[1]] - occ_value[P_01[0]][P_01[1]]) +\
              ((P_01[1]*res-R_scan[1])/(P_01[1]*res - P_00[1]*res))*(occ_value[P_10[0]][P_10[1]]-occ_value[P_00[0]][P_00[1]]),
                       ((R_scan[0]-P_00[0]*res)/(P_11[0]*res - P_00[0]*res))*(occ_value[P_11[0]][P_11[1]] - occ_value[P_10[0]][P_10[1]]) +\
              ((P_11[0]*res-R_scan[0])/(P_11[0]*res - P_00[0]*res))*(occ_value[P_01[0]][P_01[1]]-occ_value[P_00[0]][P_00[1]]))])

        #gradient of R_scan with respect to pose
        G_S = np.array([[1, 0, -np.sin(pos_theta)*lidar_scan[0][i]-np.cos(pos_theta)*lidar_scan[1][i]], [0, 1, np.cos(pos_theta)*lidar_scan[0][i]-np.sin(pos_theta)*lidar_scan[1][i]]])


        H = H + np.dot(np.transpose(np.dot(G_M,G_S)),np.dot(G_M,G_S))

        G = G + np.transpose(np.dot(G_M,G_S)*(1-M_occ))


    delta_pos = np.dot(np.linalg.pinv(H),G)
    delta_pos = np.transpose(delta_pos)
    print(delta_pos)


    mu[0] += delta_pos[0][0]*1000
    mu[1] += delta_pos[0][1]*1000
    mu[2] += delta_pos[0][2]

    #for i in range(len(M_occ_list)):
    #occ_value = ...

    plot2 = plt.figure(2)
    plt.imshow(xocc_val)
    plt.show()

    occ_val = xocc_val

    return mu, occ_value


lidar_data =[]
odom_data =[]
uwb_data =[]

now_lidar_data = []
now_uwb_data  = []
now_odom_data  = []

occ_val = []
occ_val_empty = []


#initialize belief
mu = [0.0, 0.0, 0.0]
sigma = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

landmarks = [[1400,2000,0],[-1400,2000,0],[1400,-2000,0],[-1400,-2000,0]]


def mapInitFill(res):
    global occ_val,now_lidar_data
    global mu

    pos_x = mu[0]/1000
    pos_y = mu[1]/1000
    pos_theta = mu[2]



    for i in range(len(now_lidar_data[0])):

        R_scan = np.array([np.cos(pos_theta) * now_lidar_data[0][i] - np.sin(pos_theta) * now_lidar_data[1][i] + pos_x,
                           np.sin(pos_theta) * now_lidar_data[0][i] + np.cos(pos_theta) * now_lidar_data[1][i] + pos_y])

        x_cor = int(np.floor(R_scan[0]/res))
        y_cor = int(np.floor(R_scan[1]/res))
        '''

        x_cor = int(np.floor(R_scan[0]/res)+np.floor(occ_val.shape[0]/2))
        y_cor = int(np.floor(R_scan[1]/res)+np.floor(occ_val.shape[1]/2))
        '''
        occ_val[x_cor][y_cor] = 1

        #occ_val[int(np.floor(occ_val.shape[0] / 2))][int(np.floor(occ_val.shape[1] / 2))] = 2



def timeStep():
    uwb_init_control = 0

    global  mu ,sigma,occ_val
    global now_odom_data,now_lidar_data,now_uwb_data

    i = 0
    while (i<len(odom_data)-2): #-2 sonradan silinecek
        now_odom_data=odomCal(i)
        mu, sigma = prediction_step(now_odom_data, mu, sigma)
        if((i%6)==0):
            uwb_init_control = uwb_init_control + 1

            now_lidar_data = lidarPosCal(i/6)
            now_uwb_data = uwbCal(i/6)
            mu, sigma = correction_step(now_uwb_data, mu, sigma, landmarks)

            if uwb_init_control==5:
                mapInitFill(0.05)
            elif uwb_init_control>5:
                mu , occ_val= map_matching(now_lidar_data,occ_val,0.05)

        i=i+1
        time.sleep(0.033)


def lidarPosCal(indexx):
    scan = lidar_data.ranges[indexx][1:-1].split(', ')

    map(float, scan)

    cordinates = []
    cordinates.append([])
    cordinates.append([])

    for i in range(len(scan)):
        cordinates[0].append(float(scan[i]) * math.cos(np.deg2rad(i)))
        cordinates[1].append(float(scan[i]) * math.sin(np.deg2rad(i)))

    return cordinates

def uwbCal(indexx):
    uwb = uwb_data.distance[indexx][1:-1].split(', ')
    uwb = list(map(float, uwb))
    return uwb

def odomCal(indexx):
    odom_linear_x = odom_data.twist_twist_linear_x[indexx]
    odom_angular_z= odom_data.twist_twist_angular_z[indexx]
    return [odom_linear_x,odom_angular_z]

def mapInit(res,width,height):
    global occ_val,occ_val_empty
    occ_val = np.zeros((int(width/res), int(height/res)))
    occ_val_empty =occ_val


def main():
    # implementation of an extended Kalman filter for robot pose estimation
    global lidar_data
    global odom_data
    global uwb_data

    print("Reading sensor data")
    lidar_data = pd.read_csv("scan.csv")
    odom_data = pd.read_csv("odom.csv")
    uwb_data = pd.read_csv("uwb.csv")



    #sensor_readings = read_sensor_data("../data/sensor_data.dat")

    mapInit(0.05,10.0,10.0)

    #occ_value = [] # 0 (serbest) ve 1 (dolu) doldurulması gerekiyor (400 cmx 400 cm, 5cm resolution, 80 cells for 4 m)
    res = 5 # cm
    # map_limits = [-1, 12, -1, 10]

    #run kalman filter

    #run simulatoion
    timeStep()
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
    plt.show('hold')

if __name__ == "__main__":
    main()