


FEATURE_MAX_D = 128

class Feature:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.scl = 0.0
        self.ori = 0.0
        self.d = 0
        self.descr=[None] * FEATURE_MAX_D
        self.type = 0
        self.categlory = 0
        self.fwd_match = Feature()
        self.bck_match = Feature()
        self.mdl_match = Feature()
        self.img_pt = None
        self.mdl_pt = None
        self.feature_data = None



