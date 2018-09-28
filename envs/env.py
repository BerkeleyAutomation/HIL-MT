class Action(object):
    pass


class Env(object):
    def init_arg(self, task_name):
        raise NotImplementedError

    def reset(self, task_name):
        return self.init_arg(task_name)

    def observe(self):
        raise NotImplementedError

    def step(self, act_name, act_arg):
        raise NotImplementedError
