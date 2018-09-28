import numpy as np

try:
    import pyyolo
except ImportError:
    print("Please follow these instructions to install pyyolo: https://github.com/digitalbrain79/pyyolo#building")
    print("then update the paths in vision.py to your installation path")
    raise


class Yolo(object):
    def __init__(self):
        if pyyolo is not None:
            darknet_path = './darknet'
            datacfg = 'cfg/coco.data'
            cfgfile = 'cfg/yolo.cfg'
            weightfile = 'yolo.weights'
            pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)
            self.hier_thresh = 0.1

    def __del__(self):
        if pyyolo is not None:
            pyyolo.cleanup()

    def get_objs(self, image, thresh):
        if pyyolo is None:
            return None
        else:
            img = image.transpose(2, 0, 1)
            c, h, w = img.shape[0], img.shape[1], img.shape[2]
            data = img.ravel() / 255.0
            data = np.ascontiguousarray(data, dtype=np.float32)
            objs = pyyolo.detect(w, h, c, data, thresh, self.hier_thresh)
            return objs
