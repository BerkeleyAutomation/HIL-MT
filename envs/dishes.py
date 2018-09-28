import pickle

import hsr


class DishesEnv(hsr.HSREnv):
    init_args = {
        'ClearTable': [0],
        'SetTable': [1],
    }

    obj_classes = [None, 'plate', 'cup']
    obj_colors = [None, 'blue', 'green', 'red', 'done']
    obj_classes_filter = {'bowl', 'cup', 'frisbee', 'orange'}
    _obj_classifier = pickle.load(open("model/vision/obj_class.pkl", 'rb'))

    def init_arg(self, task_name):
        return self.init_args[task_name]

    @staticmethod
    def obj_classify(iput):
        if iput:
            return [{1: 1, 3: 2}.get(x, 0) for x in DishesEnv._obj_classifier.predict(iput).argmax(1)]
        else:
            return []
