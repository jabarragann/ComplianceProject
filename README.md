# Compliance and Kinematic error repository

# Package installation

`pip install -e .`

# Commands
`rosservice list` --> List active services.

`rossrv list` --> List all srv files.
`rossrv show cisst_msgs/QueryForwardKinematics`

http://wiki.ros.org/rosservice
http://wiki.ros.org/action/show/rosmsg?action=show&redirect=rossrv


# Commands 
## Launching sensors
Forse sensor
```
rosrun atinetft_ros atinetft_xml -i 192.168.0.2 __ns:=force_sensor
```

## Creating ros bags
Recording a rosbag with a PSM trajectory
```
rosbag record -O test -e "/PSM2/(measured|setpoint).*"
```
Cropping rosbag to remove section of no movement at beginning and end
```
todo
```

# Scripts
## Data collection scripts 
Collecting calibration data
```
python3 scripts/01_calibration_exp/03_collect_calibration_data.py  -m calib -b data/psm2_trajectories/pitch_exp_traj_02_test_cropped.bag -r data/03_replay_trajectory/d04-rec-08-traj02 
```

Collecting test trajectories
```
python3 scripts/01_calibration_exp/03_collect_calibration_data.py  -m test -b data/psm2_trajectories/pitch_exp_traj_02_test_cropped.bag -r data/03_replay_trajectory/d04-rec-07-traj01 -t 02
```

Manual data collection script
```
python3 scripts/robot_experiments/04_collect_touch_registration_data_sensor.py  -r data/03_replay_trajectory/d04-rec-16-trajsoft/registration_exp/registration_with_teleop -s pedals --testid 20
```

## Data analysis scripts
Calculate robot-tracker registration
```
python3 scripts/01_calibration_exp/03_robot_tracker_registration.py -r data/03_replay_trajectory/d04-rec-18-trajsoft 
```

Calculating ground-truth joint values
```
python3 scripts/01_calibration_exp/04_calculate_ground_truth_jp.py -r data/03_replay_trajectory/d04-rec-07-traj01 -t --trajid 1 --reset
```

Calculate joints with network
```
python3 scripts/01_calibration_exp/07_plot_corrected_joints.py -r ./data/03_replay_trajectory/d04-rec-18-trajsoft/ --testid 5 20 21 -p -t -m best_model6_psm2
```

# To do 

* Multiple marker problem solutions. 
* ftk_utils::identify_marker functions might be creating outlier data.
* Rename the pitch frame to the roll frame. See the frame that is calculated in the registration script.
* Todo: make the circle least square estimation robust to outliers
* Todo: Inspect data for data points that seem very odd.   
* Pandas is deprecating the append method in version 1.4.0. You will need to adapt the code to Pandas.concat instead.
* Take a look to the following warning "Failed to load Python extension for LZ4 support." It started appearing when changing to the rospy simple packages.

# Failure cases 

## Failure in fiducial from Yaw calculation
* With script 02_pitch_yaw_roll_analysis.py
```
Calibration data: d04-rec-06-traj01
Step: 1680
Problem: Outlier in roll2 data creates a wrong circle 

Calibration data: d04-rec-06-traj01
Step: 240
Problem: Outlier in roll2 data creates a wrong circle 

Calibration data: d04-rec-06-traj01
Step: 440
Problem: Outlier in pitch data creates a wrong circle 

```

Ransac might be a good solution.
https://scikit-image.org/docs/dev/auto_examples/transform/plot_ransac.html
https://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf

Outlier detection
https://nirpyresearch.com/detecting-outliers-using-mahalanobis-distance-pca-python/
https://www.statology.org/how-to-find-iqr-of-box-plot/
