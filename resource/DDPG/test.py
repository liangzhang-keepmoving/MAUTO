import numpy as np
import math
class Test:
    def __init__(self):
        self.reference_path_loss = 1e-5
        self.noise_power = 1e-10
        self.transmission_power = 1
        self.bandwidth = 1
        self.distance_3d = math.sqrt(5000)
        
    def test(self):
        snr_db = self.transmission_power*self.reference_path_loss/self.distance_3d/self.distance_3d/self.noise_power
        print(snr_db)
        print(self.bandwidth * np.log2(1 + snr_db))
        
test = Test()
test.test()