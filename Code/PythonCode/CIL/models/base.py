class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task: int = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
