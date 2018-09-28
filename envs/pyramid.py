import hsr


class PyramidEnv(hsr.HSREnv):
    def init_arg(self, task_name):
        return [int(task_name.replace('Pyramid', '')), 0]
