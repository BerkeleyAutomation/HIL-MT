import pickle

import agent
import utils
from utils import DictTree


class HierarchicalAgent(agent.Agent):
    def __init__(self, config):
        super(HierarchicalAgent, self).__init__(config)
        self.skillset = DictTree({skill.__name__: DictTree(
            step=getattr(skill, 'step', None) if config.rollable else None,
            model_name=getattr(skill, 'model_name', self.default_model_name),
            arg_in_len=skill.arg_in_len,
            max_cnt=getattr(skill, 'max_cnt', None),
            sub_skill_names=getattr(skill, 'sub_skill_names', []),
            ret_out_len=skill.ret_out_len,
            min_valid_data=getattr(skill, 'min_valid_data', None),
            sub_arg_accuracy=getattr(skill, 'sub_arg_accuracy', None),
        ) for skill in self.skills + self.actions})
        for skill in self.skillset.values():
            if skill.sub_skill_names:
                skill.ret_in_len = max(self.skillset[sub_skill_name].ret_out_len for sub_skill_name in skill.sub_skill_names)
                skill.arg_out_len = max(skill.ret_out_len, max(self.skillset[sub_skill_name].arg_in_len for sub_skill_name in skill.sub_skill_names))
        if config.rollable and not config.teacher:
            for skill_name, skill in self.skillset.items():
                if skill.sub_skill_names:
                    skill.step = load_skill(config.model_dirname, skill_name, skill)
        self.stack = None
        self.last_act_name = None

    @property
    def root_skill_name(self):
        raise NotImplementedError

    @property
    def skills(self):
        raise NotImplementedError

    @property
    def actions(self):
        raise NotImplementedError

    @property
    def default_model_name(self):
        raise NotImplementedError

    def reset(self, init_arg):
        self.stack = [DictTree(name=self.root_skill_name, arg=init_arg, cnt=0)]
        self.last_act_name = None

    def step(self, obs):
        ret_name = self.last_act_name
        ret_val = obs
        steps = []
        while self.stack:
            top = self.stack[-1]
            sub_name, sub_arg = self.skillset[top.name].step(top.arg, top.cnt, ret_name, ret_val, obs)
            steps.append(DictTree(
                name=top.name,
                arg=top.arg,
                cnt=top.cnt,
                ret_name=ret_name,
                ret_val=ret_val,
                sub_name=sub_name,
                sub_arg=sub_arg,
            ))
            print("{}({}, {}, {}, {}) -> {}({})".format(top.name, top.arg, top.cnt, ret_name, ret_val, sub_name, sub_arg))
            if sub_name is None:
                self.stack.pop()
                ret_name = top.name
                ret_val = sub_arg
            elif self.skillset[sub_name.replace('Record_', '')].sub_skill_names:
                top.cnt += 1
                self.stack.append(DictTree(name=sub_name, arg=sub_arg, cnt=0))
                ret_name = None
                ret_val = None
            else:
                top.cnt += 1
                self.last_act_name = sub_name
                return sub_name, sub_arg, DictTree(steps=steps)
        self.last_act_name = None
        return None, None, DictTree(steps=steps)


class Skill(object):
    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        raise NotImplementedError


def load_skill(model_dirname, skill_name, skill):
    model = pickle.load(open("{}/{}.pkl".format(model_dirname, skill_name), 'rb'))

    def step(arg, cnt, ret_name, ret_val, obs):
        if arg is not None:
            assert not any(arg[skill.arg_in_len:])
            arg = arg[:skill.arg_in_len]
        if ret_val is not None:
            assert not any(ret_val[skill.ret_in_len:])
            ret_val = ret_val[:skill.ret_in_len]
        sub_skill_names = [None] + skill.sub_skill_names
        iput = (utils.pad(arg, skill.arg_in_len) + [cnt]
                + utils.one_hot(sub_skill_names.index(ret_name), len(sub_skill_names))
                + utils.pad(ret_val, skill.ret_in_len)
                + obs)
        oput = model.predict([iput])
        sub_name = sub_skill_names[oput.sub[0]]
        sub_arg = list(oput.arg[0])
        if sub_name is None:
            return None, sub_arg
        else:
            assert not any(sub_arg[skill.arg_out_len:])
            sub_arg = sub_arg[:skill.arg_out_len]
            return sub_name, sub_arg

    return step
