import time
import rospy
import crtk


class AtiForceSensor:
    def __init__(self, device_namespace="force_sensor", expected_interval=0.01):
        # ROS initialization
        if not rospy.get_node_uri():
            rospy.init_node("ATI_node", anonymous=True, log_level=rospy.WARN)

        # populate this class
        self.crtk_utils = crtk.utils(self, device_namespace, expected_interval=expected_interval)
        self.crtk_utils.add_measured_cf()


if __name__ == "__main__":

    force_sensor = AtiForceSensor()
    time.sleep(1)

    for i in range(10):
        print(force_sensor.measured_cf())
        time.sleep(1)
