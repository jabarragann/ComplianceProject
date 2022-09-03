# Dataset descriptions


* d04-rec-00:
* d04-rec-01: Wrist fiducial attach with tape on robot. Midpoint error very high.
* d04-rec-02: Better attachment of wrist fiducial. Midpoint error reduced.
The next datasets where collected together with an additional jaw swing. They have no joint values from the robot.
* d04-rec-03: Little just to try.
* d04-rec-04: Rosbag trajectory 2
* d04-rec-05: Rosbag trajectory 1.

* d04-rec-11-traj01: First good dataset with random trajectories.
* d04-rec-11-traj01: Big but noisy dataset, sensor was too close.
* d04-rec-12-traj01: Big dataset 2 random datasets of 1000 points each and 1 of 400 points.

Recorded on same day.
* d04-rec-13-traj01: Big dataset 3. This dataset includes hystersis curves
* d04-rec-14-traj01: Collected with the same setup as collection 13 (Sensor was not moved). Calibration values should be the same as in 13.

Recorded on the same day.
* d04-rec-15-trajsoft: Calibration done with the soft random trajectory.
    * T04: Random soft traj (samples per step=18)
    * T05: Random soft traj (samples per step=20)
    * T06: Random soft traj (samples per step=20)
    * T07: Random soft traj (samples per step=30)

    * T08: Random soft traj (samples per step=30) with 50 grams
    * T09: Random soft traj (samples per step=30) with 50 grams
    * T10: Random soft traj (samples per step=30) with 100 grams
    * T11: Random soft traj (samples per step=30) with 200 grams
    * T12: Random soft traj (samples per step=30) with 300 grams - not done

    * T13: Random soft traj (samples per step=10)
* d04-rec-16-trajsoft: Calibration done with the soft random trajectory.
    * Registration/compliance experiments. See next section. 
        * exp1: registration_with_sensor
        * exp2: registration_with_teleop (might need need calibration values from rec 16)
            * 0-1: Plane
            * 2: triangle plane

RECORDED ON SAME DAY 17-18 (Soft random traj - Samples per step 20)
No tape attached to the marker :( Hopefully that didn't affect results.
wrist marker was setup to low :(. Make sure this never happen again.
And I put the sensor upside down.

* d04-rec-17-trajsoft: Failed
* d04-rec-18-trajsoft:  
    * testid 20: data with teleop 01
    * testid 21: data with teleop 02
Points in the phantom were touch with the PSM in order from 1-9 and A-B. Only Point D was not collected. This give us a total of 11.

RECORDED ONSAME DAY 19-20 (Soft random traj - Samples per step 20)
* d04-rec-19-trajsoft: Failed
* d04-rec-20-trajsoft: 34 steps collected 
	* test 1-3: prerecorded trajectories
	* test 4-7: Soft random trajectories (N=1200)

	* 20-23: teleop no force (N=120) 
	* 24-26: force applied (N=63)
    Trajectories 20-26 are saved in registration experiments and test trajectories directories.
	
	* reg experiment with phat 1-5: everything except D 

# Registration data collection with sensor

## Experiment 1:
* stored: registration_sensor_exp
Points in the phantom were touch with the PSM in order from 1-9 and A-B. Point 7-D were not collected. This give us a total of 10.

## Experiment 2
* stored: registration_with_teleop
Touch random points on a plane while teleoperating with the console
