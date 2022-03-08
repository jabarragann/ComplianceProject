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
* ftk_uilts::identify_marker functions might be creating outlier data.

# Failure cases that I need to revise

## Failure in fiducial from Yaw calculation
## TODO make the circle least square estimation robust to outliers
With script 02_pitch_yaw_roll_analysis.py
Calibration data: d04-rec-06-traj01
Step: 1680
Problem: Outlier in roll2 data creates a wrong circle 

With script 02_pitch_yaw_roll_analysis.py
Calibration data: d04-rec-06-traj01
Step: 240
Problem: Outlier in roll2 data creates a wrong circle 

With script 02_pitch_yaw_roll_analysis.py
Calibration data: d04-rec-06-traj01
Step: 440
Problem: Outlier in pitch data creates a wrong circle 

Ransac might be a good solution.
https://scikit-image.org/docs/dev/auto_examples/transform/plot_ransac.html
https://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf

Outlier detection
https://nirpyresearch.com/detecting-outliers-using-mahalanobis-distance-pca-python/
https://www.statology.org/how-to-find-iqr-of-box-plot/

## Failure in rotation axis from Marker
## Todo: Inspect data for data points that seem very odd.