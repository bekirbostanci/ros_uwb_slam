import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

'''
df = pd.read_csv("scan.csv")

scan = df.ranges[0][1:-1].split(', ')


map(float, scan)

cordinates = []
cordinates.append([])
cordinates.append([])

for i in range(len(scan)):
    cordinates[0].append(float(scan[i]) * math.cos(np.deg2rad(i)))
    cordinates[1].append(float(scan[i]) * math.sin(np.deg2rad(i)))

plt.plot(cordinates[0],cordinates[1], 'ro')
plt.show()


dff = pd.read_csv("uwb.csv")
uwb = dff.distance[0][1:-1].split(', ')
uwb = list(map(float, uwb))
print(uwb)


print(np.zeros((2,3)))
'''

robot_x = 0
robot_y = 0

duvar_x = 5
duvar_y = 4

angle = np.rad2deg(np.arctan((duvar_y-robot_y)/(duvar_x-robot_x)))
angle = angle-180
print(angle)
search_x = int(np.cos(np.deg2rad(angle))+duvar_x)
search_y = int(np.sin(np.deg2rad(angle))+duvar_y)


print(search_x)
print(search_y)


