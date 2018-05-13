class Epoch(object):
    def __init__(self, e, **kwargs):
        self.number = e
        self.data = dict()
        self.input_output = dict()
