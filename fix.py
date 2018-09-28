import glob
import math
import pickle

from utils import DictTree

V = 0.07
OMEGA = math.pi / 5.


def fix_record(traces_dirname_out, traces_dirname_in):
    for filename_in in glob.iglob("{}/*.pkl".format(traces_dirname_in)):
        print(filename_in)
        trace = pickle.load(open(filename_in, 'rb'))
        t = 0
        while t < len(trace):
            time_step = trace[t]
            i = 0
            while i < len(time_step.info.steps):
                skill_step = time_step.info.steps[i]
                print('-->', skill_step)
                if skill_step.ret_name is not None and skill_step.ret_name.startswith('Record_'):
                    skill_step.ret_name = skill_step.ret_name.replace('Record_', '')
                if skill_step.ret_val is not None:
                    skill_step.ret_val = [float(x) for x in skill_step.ret_val]
                if skill_step.sub_name is not None and skill_step.sub_name.startswith('Record_'):
                    if skill_step.sub_name == 'Record_MoveBaseRel':
                        skill_step.sub_name = 'MoveBaseRel'
                        skill_step.sub_arg = rel_pos(time_step.info.before, time_step.info.after)
                        del time_step.info.before
                        del time_step.info.after
                    else:
                        raise ValueError
                print('<--', skill_step)
                i += 1
            if t == len(trace) - 1:
                skill_step = DictTree(
                    name='EndTask',
                    arg=None,
                    cnt=1,
                    ret_name='MoveArm',
                    ret_val=None,
                    sub_name=None,
                    sub_arg=None,
                )
                time_step.info.steps.append(skill_step)
                print('<--', skill_step)
            t += 1
        if trace is not None:
            filename_out = filename_in.replace(traces_dirname_in, traces_dirname_out)
            pickle.dump(trace, open(filename_out, 'wb'), protocol=2)


def rel_pos(before, after):
    x0, y0, theta0 = before[1].actual
    x1, y1, theta1 = after[1].actual
    dx = x1 - x0
    dy = y1 - y0
    dtheta = theta1 - theta0
    return [
        dx * math.cos(theta0) + dy * math.sin(theta0),
        -dx * math.sin(theta0) + dy * math.cos(theta0),
        dtheta, V, OMEGA]
