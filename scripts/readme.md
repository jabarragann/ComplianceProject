# Script lists


## 01_calibration_experiments

* 02_pitch_exp.py: Capture shaft's marker and wrist's fiducial. Move only the wrist. 
* 03_trajectory_exp.py: Replay trajectory. Capture only shaft's marker position. From the robot collected: measured_cp,  and measured_js.position.
* 04_trajectory_wrist.py: Replay trajectory + wrist motion in some points. 

## Utils

* get_dvrk_jp.py: Get the joint position of the specified arm
* crop_rosbag_with_vel.py: Crop rosbag to remove the parts of not motion in the beginning and end of the bag.