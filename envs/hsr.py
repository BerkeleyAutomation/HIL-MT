import math

import numpy as np

import vision
from utils import DictTree

try:
    import ar_markers
    import control_msgs.msg as ctrl_msg
    import cv2
    import cv_bridge
    import rospy
    import sensor_msgs.msg as sens_msg
    import tmc_control_msgs.msg as tmc_msg
    import trajectory_msgs.msg as traj_msg
except ImportError, e:
    print(e)

import env

V = 0.07
OMEGA = math.pi / 5.


class MoveArm(env.Action):
    arg_in_len = 5
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll = arg[:5]
        point = traj_msg.JointTrajectoryPoint()
        point.positions = [arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll]
        point.velocities = [0] * 5
        point.time_from_start = rospy.Time(0.5)
        traj = ctrl_msg.FollowJointTrajectoryActionGoal()
        traj.goal.trajectory.joint_names = ['arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
        traj.goal.trajectory.points = [point]
        hsr.publishers['arm'].publish(traj)
        rospy.sleep(1.)
        while True:
            error = hsr.subscribers['arm_pose'].callback.data['error']
            if all(abs(x) <= 1e-2 for x in error):
                break
            rospy.sleep(0.1)


class MoveBaseAbs(env.Action):
    arg_in_len = 5
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        x, y, theta, v, omega = arg[:5]
        x0, y0, theta0 = hsr.subscribers['base_pose'].callback.data['actual']
        dx = ((x - x0) ** 2 + (y - y0) ** 2) ** .5
        dtheta = abs(theta - theta0)
        t = max(dx / v, dtheta / omega)
        acceleration = 0.05
        point = traj_msg.JointTrajectoryPoint()
        point.positions = [x, y, theta]
        point.accelerations = [acceleration] * 3
        point.effort = [1]
        point.time_from_start = rospy.Time(t)
        traj = ctrl_msg.FollowJointTrajectoryActionGoal()
        traj.goal.trajectory.joint_names = ['odom_x', 'odom_y', 'odom_t']
        traj.goal.trajectory.points = [point]
        hsr.publishers['base'].publish(traj)
        rospy.sleep(1.)
        while True:
            error = hsr.subscribers['base_pose'].callback.data['error']
            if all(abs(x) <= 1e-6 for x in error):
                break
            rospy.sleep(0.1)


class MoveBaseRel(env.Action):
    arg_in_len = 5
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        x, y, theta, v, omega = arg[:5]
        x0, y0, theta0 = hsr.subscribers['base_pose'].callback.data['actual']
        x, y, theta = (
            x * math.cos(theta0) - y * math.sin(theta0) + x0,
            x * math.sin(theta0) + y * math.cos(theta0) + y0,
            theta + theta0)
        MoveBaseAbs.apply(hsr, [x, y, theta, v, omega])


class MoveGripper(env.Action):
    arg_in_len = 1
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        [close] = arg[:1]
        if close:
            rospy.sleep(1.)
        grip = tmc_msg.GripperApplyEffortActionGoal()
        grip.goal.effort = -0.1 if close > 0.5 else 0.1
        hsr.publishers['grip'].publish(grip)
        rospy.sleep(1.)


class MoveHead(env.Action):
    arg_in_len = 2
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        tilt, pan = arg[:2]
        point = traj_msg.JointTrajectoryPoint()
        point.positions = [tilt, pan]
        point.velocities = [0] * 2
        point.time_from_start = rospy.Time(0.5)
        traj = ctrl_msg.FollowJointTrajectoryActionGoal()
        traj.goal.trajectory.joint_names = ['head_tilt_joint', 'head_pan_joint']
        traj.goal.trajectory.points = [point]
        hsr.publishers['head'].publish(traj)
        rospy.sleep(2.)


class LocateObject(env.Action):
    arg_in_len = 3
    ret_out_len = 4

    @staticmethod
    def apply(hsr, arg):
        motion_cnt, obj_class, obj_color = arg[:3]
        img = hsr.subscribers['head_cam'].callback.data
        objs = hsr.vision.get_objs(img, 0.1)
        objs = list(sorted((obj for obj in objs if obj['class'] in hsr.obj_classes_filter), key=lambda o: o['bottom'], reverse=True))
        detected_obj_class = hsr.obj_classify([[motion_cnt, obj['left'], obj['top'], obj['right'], obj['bottom']] for obj in objs])
        for i, obj in enumerate(objs):
            obj['class_idx'] = detected_obj_class[i]
            obj['mean_color'] = img[obj['top']:obj['bottom'], obj['left']:obj['right'], :].mean(0).mean(0) - img.mean(0).mean(0)
            obj['color_idx'] = 1 + np.argmax(obj['mean_color'])
        for i, obj in enumerate(objs):
            box_color = (255, 0, 0)
            cv2.rectangle(img, (obj['left'], obj['top']), (obj['right'], obj['bottom']), box_color, 4)
            cv2.putText(
                img, "{} {}".format(hsr.obj_colors[obj['color_idx']], hsr.obj_classes[obj['class_idx']]),
                (obj['left'], obj['top']), cv2.FONT_HERSHEY_PLAIN, 2, box_color, 1, 8)
        cv2.imshow('head', img)
        cv2.waitKey(3)
        if abs(obj_class - hsr.obj_classes.index(None)) > 0.5:
            objs = [obj for obj in objs if abs(obj['class_idx'] - obj_class) < 0.5]
        if abs(obj_color - hsr.obj_colors.index(None)) > 0.5:
            objs = [obj for obj in objs if abs(obj['color_idx'] - obj_color) < 0.5]
        if len(objs) > 0:
            obj = objs[0]
            box_color = (0, 255, 0)
            cv2.rectangle(img, (obj['left'], obj['top']), (obj['right'], obj['bottom']), box_color, 4)
            cv2.putText(
                img, "{} {}".format(hsr.obj_colors[obj['color_idx']], hsr.obj_classes[obj['class_idx']]),
                (obj['left'], obj['top']), cv2.FONT_HERSHEY_PLAIN, 2, box_color, 1, 8)
            return [obj['class_idx'], obj['color_idx'], (obj['left'] + obj['right']) / 2, (obj['top'] + obj['bottom']) / 2]
        else:
            return [0, 0, 0, 0]


class LocateMarkers(env.Action):
    arg_in_len = 0
    ret_out_len = 16

    @staticmethod
    def apply(hsr, arg):
        while True:
            rospy.sleep(5.)
            img = hsr.subscribers['head_cam'].callback.data
            _, img = cv2.threshold(img, 32, 255, cv2.THRESH_BINARY)
            markers = {marker.id: marker for marker in ar_markers.detect_markers(img)}.values()
            for marker in markers:
                marker.draw_contour(img)
            cv2.imshow('head', img)
            cv2.waitKey(3)
            if len(markers) == 2:
                return sort_markers(sum([list(marker.contours.flat) for marker in markers], []))
            else:
                print('found {} markers'.format(len(markers)))
                rospy.sleep(1.)


def sort_markers(markers, sort_between_markers=True):
    assert len(markers) == 16
    markers = [int(x) for x in markers]
    res = []
    for marker in [markers[i:i + 8] for i in range(0, 16, 8)]:
        # sort 4 points from left to right
        points_xy = sorted(marker[i:i + 2] for i in range(0, 8, 2))
        # sort each pair of points from top to bottom, and flatten
        res.append(sum(sum((sorted(points_xy[i:i + 2], key=lambda xy: xy[1]) for i in range(0, 4, 2)), []), []))
    if sort_between_markers:
        return sum(sorted(res), [])
    else:
        return sum(res, [])


class HSREnv(env.Env):
    actions = [
        MoveArm,
        MoveBaseAbs,
        MoveBaseRel,
        MoveGripper,
        MoveHead,
        LocateObject,
        LocateMarkers,
    ]
    action_lookup = {action.__name__: action for action in actions}

    def __init__(self):
        rospy.init_node('main', anonymous=True)
        self.subscribers = {
            'head_cam': rospy.Subscriber('/hsrb/head_r_stereo_camera/image_raw', sens_msg.Image, ImageCallback(), queue_size=1),
            'arm_pose': rospy.Subscriber('/hsrb/arm_trajectory_controller/state', ctrl_msg.JointTrajectoryControllerState, PoseCallback(), queue_size=1),
            'base_pose': rospy.Subscriber('/hsrb/omni_base_controller/state', ctrl_msg.JointTrajectoryControllerState, PoseCallback(), queue_size=1),
        }
        self.publishers = {
            'arm': rospy.Publisher('/hsrb/arm_trajectory_controller/follow_joint_trajectory/goal', ctrl_msg.FollowJointTrajectoryActionGoal, queue_size=1),
            'base': rospy.Publisher('/hsrb/omni_base_controller/follow_joint_trajectory/goal', ctrl_msg.FollowJointTrajectoryActionGoal, queue_size=1),
            'grip': rospy.Publisher('/hsrb/gripper_controller/grasp/goal', tmc_msg.GripperApplyEffortActionGoal, queue_size=1),
            'head': rospy.Publisher('/hsrb/head_trajectory_controller/follow_joint_trajectory/goal', ctrl_msg.FollowJointTrajectoryActionGoal, queue_size=1),
        }
        try:
            while (any(publisher.get_num_connections() == 0 for publisher in self.publishers.values())
                   or any(subscriber.get_num_connections() == 0 or subscriber.callback.data is None for subscriber in self.subscribers.values())):
                if rospy.is_shutdown():
                    raise ValueError()
                print('Waiting for: {}'.format(
                    [name for name, publisher in self.publishers.items() if publisher.get_num_connections() == 0] +
                    [name for name, subscriber in self.subscribers.items() if subscriber.get_num_connections() == 0 or subscriber.callback.data is None]))
                rospy.sleep(0.1)
        except KeyboardInterrupt:
            rospy.loginfo(KeyboardInterrupt)
            raise
        self.vision = vision.Yolo()
        self.obs = None

    def observe(self):
        return self.obs

    def step(self, act_name, act_arg):
        self.obs = self.action_lookup[act_name].apply(self, act_arg)

    def record(self):
        return [
            self.subscribers['arm_pose'].callback.data,
            self.subscribers['base_pose'].callback.data,
        ]


class DataCallback(object):
    def __init__(self, sleep=0.1):
        self.data = None
        self.sleep = sleep

    def __call__(self, data):
        self.data = data
        rospy.sleep(self.sleep)


class ImageCallback(DataCallback):
    def __init__(self, sleep=0.1):
        super(ImageCallback, self).__init__(sleep)
        self.bridge = cv_bridge.CvBridge()

    def __call__(self, img):
        img = self.bridge.imgmsg_to_cv2(img, 'bgr8')
        img = cv2.resize(img, (920, 690), interpolation=cv2.INTER_CUBIC)
        super(ImageCallback, self).__call__(img)


class PoseCallback(DataCallback):
    def __call__(self, data):
        pose = DictTree(
            actual=data.actual.positions,
            error=data.error.positions,
        )
        super(PoseCallback, self).__call__(pose)
