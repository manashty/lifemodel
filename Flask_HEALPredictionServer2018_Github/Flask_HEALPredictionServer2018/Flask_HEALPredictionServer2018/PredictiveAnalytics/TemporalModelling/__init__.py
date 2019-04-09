
class TemporalAbstraction:
    def __init__(self, f, v, s, e, **kwargs):
        self.f = f#kwargs["f"]
        self.v = v#kwargs["v"]
        self.s = s
        self.e = e
        self.diagnosis = 0
    @property
    def Length(self):
        return self.e - self.s
    
    def __repr__(self, **kwargs):
        return "({0}, {1}, {2}, {3})".format(self.f,self.v,self.s,self.e)

