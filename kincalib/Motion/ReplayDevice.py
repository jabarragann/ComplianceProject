import crtk
from kincalib.utils.Logger import Logger

log = Logger(__name__).log


class ReplayDevice:
    """Simplified arm class to replay motion, better performance than
    dvrk.arm since we're only subscribing to topics we need.
    """

    class __jaw_device:
        """Simplified jaw class to control the jaws, will not be used without the -j option"""

        def __init__(self, jaw_namespace, expected_interval, operating_state_instance):
            self.__crtk_utils = crtk.utils(self, jaw_namespace, expected_interval, operating_state_instance)
            self.__crtk_utils.add_move_jp()
            self.__crtk_utils.add_servo_jp()
            self.__crtk_utils.add_measured_js()

    def __init__(self, device_namespace, expected_interval):
        # populate this class with all the ROS topics we need
        self.crtk_utils = crtk.utils(self, device_namespace, expected_interval)
        self.crtk_utils.add_operating_state()
        self.crtk_utils.add_servo_jp()
        self.crtk_utils.add_move_jp()
        self.crtk_utils.add_servo_cp()
        self.crtk_utils.add_move_cp()
        self.crtk_utils.add_measured_js()
        self.crtk_utils.add_measured_cp()
        self.crtk_utils.add_setpoint_js()
        self.crtk_utils.add_setpoint_cp()
        self.jaw = self.__jaw_device(device_namespace + "/jaw", expected_interval, operating_state_instance=self)

    def jaw_jp(self):
        try:
            jaw_pose = self.jaw.measured_jp()[0]
        except RuntimeWarning as e:
            log.error("Run time warning raised when reading jaw jp")
            return -505
