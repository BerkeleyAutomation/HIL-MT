import math
import time

import numpy as np

import hierarchical
from envs.pyramid import PyramidEnv

V = 0.07
OMEGA = math.pi / 5.

CUP_SPACING = 0.095
TABLE_HEIGHT = 0.375
CUP_HEIGHT = 0.06

EPSILON = 1e-6


class Pyramid(hierarchical.Skill):
    arg_in_len = 2
    max_cnt = 4
    sub_skill_names = ['MoveHome', 'BuildPyramid', 'EndTask']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [height, pos]
        ret_val: None
        """
        [height, pos] = arg
        return [
            ('MoveHome', None),
            ('BuildPyramid', [height, pos]),
            ('EndTask', None),
            (None, None)
        ][cnt]


class MoveHome(hierarchical.Skill):
    arg_in_len = 0
    max_cnt = 4
    sub_skill_names = ['MoveGripper', 'MoveArm', 'MoveHead']
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
            ('MoveHead', [-math.pi / 8., 0.]),
            (None, None)
        ][cnt]


class BuildPyramid(hierarchical.Skill):
    model_name = 'log_lin'
    arg_in_len = 2
    sub_skill_names = ['BuildLevel']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [height, pos]
        ret_val: None
        """
        [height, pos] = arg
        if cnt < height:
            return 'BuildLevel', [cnt, pos, height - cnt]
        else:
            return None, None


class EndTask(hierarchical.Skill):
    arg_in_len = 0
    max_cnt = 2
    sub_skill_names = ['MoveArm']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: None
        ret_val: None
        """
        return [
            ('MoveArm', [0., 0., 0., -math.pi / 2., 0.]),
            (None, None)
        ][cnt]


class BuildLevel(hierarchical.Skill):
    model_name = 'log_lin'
    arg_in_len = 3
    sub_skill_names = ['MoveCup']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [level, pos, num_cups]
        ret_val: None
        """
        [level, pos, num_cups] = arg
        if cnt < num_cups:
            return 'MoveCup', [pos + cnt, level]
        else:
            return None, None


class MoveCup(hierarchical.Skill):
    arg_in_len = 2
    max_cnt = 3
    sub_skill_names = ['PickCup', 'PlaceCup']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [pos, level]
        ret_val: None
        """
        [pos, level] = arg
        return [
            ('PickCup', None),
            ('PlaceCup', [pos, level]),
            (None, None)
        ][cnt]


class PickCup(hierarchical.Skill):
    arg_in_len = 0
    max_cnt = 6
    sub_skill_names = ['MoveBaseRel', 'MoveArm', 'MoveGripper']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: None
        ret_val: None
        """
        if cnt == 2:
            time.sleep(3)
        return [
            ('MoveBaseRel', [0., 0., -math.pi / 2., V, OMEGA]),
            ('MoveArm', [0.37, -math.pi / 2., 0., -math.pi / 2., -math.pi / 2.]),
            ('MoveGripper', [1]),
            ('MoveArm', [0.45, 0., 0., -math.pi / 2., -math.pi / 2.]),
            ('MoveBaseRel', [0., 0., math.pi / 2., V, OMEGA]),
            (None, None)
        ][cnt]


class PlaceCup(hierarchical.Skill):
    arg_in_len = 2
    max_cnt = 5
    sub_skill_names = ['MoveToPosition', 'PutCup', 'MoveBaseRel']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [pos, level]
        ret_val: None
        """
        [pos, level] = arg
        if cnt < 2:
            return 'MoveToPosition', [pos, level, cnt, 0]
        elif cnt == 2:
            return 'PutCup', [level]
        elif cnt == 3:
            return 'MoveToPosition', [0, 0, 0, 1]
        else:
            return None, None


class MoveToPosition(hierarchical.Skill):
    arg_in_len = 4
    max_cnt = 3
    sub_skill_names = ['LocateMarkers', 'MoveBaseRel']
    ret_out_len = 0
    sub_arg_accuracy = [1e-2, 1e-2, math.radians(1), EPSILON, EPSILON]

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [pos, level, motion_cnt, away]
        ret_val: None
        """
        [pos, level, motion_cnt, away] = arg
        if cnt == 0:
            return 'LocateMarkers', None
        elif cnt == 1:  # this will be demonstrated by teleoperation
            if away == 0 and motion_cnt == 0:
                return 'Record_MoveBaseRel', [np.random.normal(0., 0.02), np.random.normal(0., 0.02), np.random.normal(0., math.radians(2)), V, OMEGA]
            else:
                return 'Record_MoveBaseRel', [0., 0., 0., V, OMEGA]
        else:
            return None, None


class PutCup(hierarchical.Skill):
    arg_in_len = 1
    max_cnt = 4
    sub_skill_names = ['MoveArm', 'MoveGripper']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [pos, level, motion_cnt]
        ret_val: None
        """
        [level] = arg
        return [
            ('MoveArm', [TABLE_HEIGHT + CUP_HEIGHT * level] + [-math.pi / 2., 0., -math.pi / 2., -math.pi / 2.]),
            ('MoveGripper', [0]),
            ('MoveArm', [0.45, 0., 0., -math.pi / 2., -math.pi / 2.]),
            (None, None)
        ][cnt]


class PyramidAgent(hierarchical.HierarchicalAgent):
    root_skill_name = 'Pyramid'
    skills = [
        Pyramid,
        MoveHome,
        BuildPyramid,
        EndTask,
        BuildLevel,
        MoveCup,
        PickCup,
        PlaceCup,
        MoveToPosition,
        PutCup,
    ]
    actions = PyramidEnv.actions
    default_model_name = 't_log_lin'

    def __init__(self, config):
        super(PyramidAgent, self).__init__(config)
