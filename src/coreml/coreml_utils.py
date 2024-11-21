
class AffineQuantParams:
    def __init__(self, quantized_data=None, zero_point=None, scale=None, axis=None):
        self.quantized_data = quantized_data
        self.zero_point = zero_point
        self.scale = scale
        self.axis = axis
