from typing import Any
import rospy
from sensor_msgs.msg import Joy


class PedalsIO:
    def __init__(self, namespace, action_list=[], kwargs={}) -> None:
        assert namespace in ["camera", "cam_minus"], "Incorrect namespace."

        # Function/callable object that will be executed when the suj clutch is pressed
        self.action_list = action_list
        self._suj_clutch_sub = rospy.Subscriber(f"/footpedals/{namespace}", Joy, self.__suj_clutch_cb, queue_size=1)
        self.kwargs = kwargs

    def __suj_clutch_cb(self, data):
        if data.buttons[0]:
            for action in self.action_list:
                action(**self.kwargs)


class PsmIO:
    def __init__(self, namespace, action_list=[], kwargs={}) -> None:

        # Function/callable object that will be executed when the suj clutch is pressed
        self.action_list = action_list
        self._suj_clutch_sub = rospy.Subscriber(f"/{namespace}/io/suj_clutch", Joy, self.__suj_clutch_cb, queue_size=1)
        self.kwargs = kwargs

    def __suj_clutch_cb(self, data):
        if data.buttons[0]:
            for action in self.action_list:
                action(**self.kwargs)


def action():
    print("button pressed!")


class Action2:
    def __init__(self):
        self.name = "CALLABLE_CLASS"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print(f"button pressed from {self.name}")


if __name__ == "__main__":
    rospy.init_node("psm_io_test")
    psm_io = PsmIO("PSM2", action_list=[action, Action2()])

    rospy.spin()
