import math
import pickle

import hierarchical
import utils
from envs.dishes import DishesEnv

PRETRAINED = False

V = 0.07
OMEGA = math.pi / 5.


class MoveObjects(hierarchical.Skill):
    arg_in_len = 1
    sub_skill_names = ['MoveHome', 'MoveObject']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [task_id]
        ret_val: after MoveObject: [success]
        """
        [task_id] = arg
        obj_cnt = cnt - 1
        if ret_name is None:
            return 'MoveHome', None
        elif ret_name == 'MoveHome' or (ret_name == 'MoveObject' and DishesEnv.obj_classes[ret_val[0]] is not None):
            return 'MoveObject', [task_id, obj_cnt]
        else:
            return None, None


class MoveHome(hierarchical.Skill):
    arg_in_len = 0
    sub_skill_names = ['MoveGripper', 'MoveArm', 'MoveBaseAbs']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: None
        ret_val: None
        """
        return [
            ('MoveGripper', [0]),
            ('MoveArm', [0.45, 0., 0., -math.pi / 2., -math.pi / 2.]),
            ('MoveBaseAbs', [0., 0., 0., V, OMEGA]),
            (None, None)
        ][cnt]


class MoveObject(hierarchical.Skill):
    model_name = 'table|log_poly2'
    arg_in_len = 2
    max_cnt = 3
    sub_skill_names = ['PickObject', 'PlaceObject']
    ret_out_len = 1

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [task_id, obj_cnt]
        ret_val: after PickObject: [obj_class, obj_color]
        """
        [task_id, obj_cnt] = arg
        if task_id == 0:
            obj_class = DishesEnv.obj_classes.index(None)
            obj_color = DishesEnv.obj_colors.index(None)
        elif task_id == 1:
            obj_class = [DishesEnv.obj_classes.index('plate'), DishesEnv.obj_classes.index('cup')][obj_cnt % 2]
            obj_color = [
                DishesEnv.obj_colors.index('blue'),
                DishesEnv.obj_colors.index('green'),
                DishesEnv.obj_colors.index('red'),
                DishesEnv.obj_colors.index('done'),
            ][obj_cnt / 2]
        else:
            raise NotImplementedError
        if cnt == 0:
            return 'PickObject', [obj_class, obj_color]
        elif cnt == 1:
            if DishesEnv.obj_classes[ret_val[0]] is None:
                return None, [False]
            else:
                return 'PlaceObject', [task_id] + ret_val
        else:
            return None, [True]


class PickObject(hierarchical.Skill):
    arg_in_len = 2
    sub_skill_names = ['MoveToObject', 'GraspObject']
    ret_out_len = 2

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [obj_class, obj_color]
        ret_val: after MoveToObject: [obj_class, obj_color]; after GraspObject: [obj_class, obj_color]
        """
        if cnt == 0:
            return 'MoveToObject', arg + [cnt]
        elif cnt == 1:
            # TODO: keep same class and color for second motion
            if DishesEnv.obj_classes[ret_val[0]] is None:
                return None, ret_val
            else:
                return 'MoveToObject', arg + [cnt]
        elif cnt == 2:
            return 'GraspObject', ret_val
        else:
            return None, ret_val


class PlaceObject(hierarchical.Skill):
    arg_in_len = 3
    sub_skill_names = ['MoveBaseAbs', 'MoveArm', 'MoveHome']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [task_id, obj_class, obj_color]
        ret_val: None
        """
        [task_id, obj_class, obj_color] = arg
        if task_id == 0:
            [x, y, z, theta] = [0.2, 0., 0.55, math.pi / 2.]
        elif task_id == 1:
            [x, y] = [
                None,
                [0., 0.2],
                [0.3, 0.2],
                [0.15, 0.],
            ][obj_color]
            y += {
                'plate': 0.,
                'cup': 0.06,
            }[DishesEnv.obj_classes[obj_class]]
            z = {
                'plate': 0.36,
                'cup': 0.41,
            }[DishesEnv.obj_classes[obj_class]]
            theta = math.pi / 2.
        else:
            raise NotImplementedError
        return [
            ('MoveBaseAbs', [x, y, theta, V, OMEGA]),
            ('MoveArm', [z, -math.pi / 2., 0., -math.pi / 2., -math.pi / 2.]),
            ('MoveHome', None),
            (None, None)
        ][cnt]


class MoveToObject(hierarchical.Skill):
    model_name = 't_log_poly2'
    arg_in_len = 3
    max_cnt = 4
    sub_skill_names = ['MoveHead', 'LocateObject', 'MoveToLocation']
    ret_out_len = 2

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [obj_class, obj_color, motion_cnt]
        ret_val: after LocateObject: [obj_class, obj_color, obj_pixel_x, obj_pixel_y]; after MoveToLocation: [obj_class, obj_color]
        """
        [obj_class, obj_color, motion_cnt] = arg
        if cnt == 0:
            return 'MoveHead', [-math.pi / 4., 0.]
        elif cnt == 1:
            return 'LocateObject', [motion_cnt, obj_class, obj_color]
        elif cnt == 2:
            [found_obj_class, found_obj_color, obj_pixel_x, obj_pixel_y] = ret_val
            if DishesEnv.obj_classes[found_obj_class] is None:
                return None, [found_obj_class, found_obj_color]
            else:
                return 'MoveToLocation', [found_obj_class, found_obj_color, obj_pixel_x, obj_pixel_y]
        else:
            return None, ret_val


class GraspObject(hierarchical.Skill):
    arg_in_len = 2
    sub_skill_names = ['MoveArm', 'MoveGripper']
    ret_out_len = 2

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [obj_class, obj_color]
        ret_val: None
        """
        [obj_class, obj_color] = arg
        z = {
            'plate': 0.35,
            'cup': 0.4,
        }[DishesEnv.obj_classes[obj_class]]
        return [
            ('MoveArm', [z, -math.pi / 2., 0., -math.pi / 2., -math.pi / 2.]),
            ('MoveGripper', [1]),
            ('MoveArm', [0.65, -math.pi / 2., 0., -math.pi / 2., -math.pi / 2.]),
            (None, [obj_class, obj_color])
        ][cnt]


class MoveToLocation(hierarchical.Skill):
    arg_in_len = 4
    sub_skill_names = ['MoveBaseRel']
    ret_out_len = 2

    if PRETRAINED:
        teacher_model = pickle.load(open("model/dishes/teacher/move_to_location.pkl", 'rb'))

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [obj_class, obj_color, obj_pixel_x, obj_pixel_y]
        ret_val: None
        """
        [obj_class, obj_color, obj_pixel_x, obj_pixel_y] = arg
        if cnt == 0:
            if PRETRAINED:
                [x, y, theta] = MoveToLocation.teacher_model.predict(
                    [utils.one_hot({1: 1, 2: 3}[obj_class], len(DishesEnv.obj_classes) + 1) + [obj_pixel_x, obj_pixel_y]])[0]
                return 'MoveBaseRel', [x, y, theta, V, OMEGA]
            else:
                return 'Record_MoveBaseRel', None  # this will be demonstrated by teleoperation
        else:
            return None, [obj_class, obj_color]


class DishesAgent(hierarchical.HierarchicalAgent):
    root_skill_name = 'MoveObjects'
    skills = [
        MoveObjects,
        MoveHome,
        MoveObject,
        PickObject,
        PlaceObject,
        MoveToObject,
        GraspObject,
        MoveToLocation,
    ]
    actions = DishesEnv.actions
    default_model_name = 'log_poly2'

    def __init__(self, config):
        super(DishesAgent, self).__init__(config)
