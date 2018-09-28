import time

from utils import DictTree


class Agent(object):
    def __init__(self, config):
        self.domain_name = config.domain_name
        self.task_name = config.task_name

    def __repr__(self):
        return self.task_name

    def reset(self, init_arg):
        raise NotImplementedError

    def rollout(self, env):
        trace = []
        while True:
            obs = env.observe()
            act_name, act_arg, info = self.step(obs)
            if act_name is not None:
                if act_name.startswith('Record_'):
                    act_name = act_name.replace('Record_', '')
                    info.before = env.record()
                    raw_input("Press Enter to continue...")
                    info.after = env.record()
                    env.step(act_name, act_arg)
                else:
                    env.step(act_name, act_arg)
            trace.append(DictTree(
                timestamp=time.time(),
                act_name=act_name,
                act_arg=act_arg,
                info=info,
            ))
            if act_name is None:
                return trace

    def step(self, obs):
        raise NotImplementedError
