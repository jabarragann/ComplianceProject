# Compliance and Kinematic error repository

# Package installation

`pip install -e .`

# Commands
`rosservice list` --> List active services.

`rossrv list` --> List all srv files.
`rossrv show cisst_msgs/QueryForwardKinematics`

http://wiki.ros.org/rosservice
http://wiki.ros.org/action/show/rosmsg?action=show&redirect=rossrv



# To do 

* Multiple marker problem solutions. 


# Failure cases that I need to revise

## Failure in fiducial from Yaw calculation
With script 02_pitch_yaw_roll_analysis.py
Calibration data: d04-rec-06-traj01
Step: 1680
Problem: Outlier in roll2 data creates a wrong circle 

With script 02_pitch_yaw_roll_analysis.py
Calibration data: d04-rec-06-traj01
Step: 240
Problem: Outlier in roll2 data creates a wrong circle 

## Failure in rotation axis from Marker
## Todo: Inspect data for data points that seem very odd.