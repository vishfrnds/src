from tinygrad.engine.lazy import LazyBuffer
from tinygrad.tensor import Function, Tensor


class NonLinear(Function):
    # def __init__(self, middle_slope=0.2, outer_slope=0.8, transition_points=(-0.5, 0.5), smoothing=0.1):
    #     """
    #     Parameters:
    #     - middle_slope: slope in the middle region
    #     - outer_slope: slope in the outer regions
    #     - transition_points: tuple of (lower, upper) transition points
    #     - smoothing: controls the smoothness of transition (higher = smoother)
    #     """
    #     self.middle_slope = middle_slope
    #     self.outer_slope = outer_slope
    #     self.lower_point, self.upper_point = transition_points
    #     self.smoothing = smoothing
    
    def _smooth_step(self, x: LazyBuffer, edge0: float, edge1: float) -> LazyBuffer:
        """
        Smooth transition between 0 and 1 using cubic interpolation
        """
        t = (x - edge0) / (edge1 - edge0)
        # t = t.clamp(0.0, 1.0)
        # t = (t < 0).where(0.0, t)
        # t = (t > 1).where(1.0, t)
        return t * t * (3.0 - 2.0 * t)
    
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        
        self.middle_slope = 0.2
        self.outer_slope = 0.8
        self.lower_point, self.upper_point = (-0.5, 0.5)
        self.smoothing = 0.1
        # Calculate smooth transitions
        lower_transition = self._smooth_step(
            x, 
            self.lower_point - self.smoothing, 
            self.lower_point + self.smoothing
        )
        upper_transition = self._smooth_step(
            x, 
            self.upper_point - self.smoothing, 
            self.upper_point + self.smoothing
        )
        
        # Calculate the three regions
        middle_region = self.middle_slope * x
        upper_offset = (self.upper_point * (self.middle_slope - self.outer_slope))
        lower_offset = (self.lower_point * (self.middle_slope - self.outer_slope))
        
        upper_region = self.outer_slope * x - upper_offset
        lower_region = self.outer_slope * x - lower_offset
        
        # Blend regions using smooth transitions
        result = middle_region * (1 - lower_transition) * (1 - upper_transition)
        result += lower_region * lower_transition
        result += upper_region * upper_transition
        
        return result
    
    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # Calculate smooth transitions for gradient
        lower_transition = self._smooth_step(
            self.x, 
            self.lower_point - self.smoothing, 
            self.lower_point + self.smoothing
        )
        upper_transition = self._smooth_step(
            self.x, 
            self.upper_point - self.smoothing, 
            self.upper_point + self.smoothing
        )
        
        # Blend slopes using smooth transitions
        grad = self.middle_slope * (1 - lower_transition) * (1 - upper_transition)
        grad += self.outer_slope * (lower_transition + upper_transition)
        
        return grad_output * grad

# Example usage
# def smooth_nonlinear(x: Tensor) -> Tensor:
#     return ImprovedNonLinear(
#         middle_slope=0.2,
#         outer_slope=0.8,
#         transition_points=(-0.5, 0.5),
#         smoothing=0.1
#     ).apply(x)


# class NonLinear(Function):
#   def forward(self, x: LazyBuffer) -> LazyBuffer:
#     self.x = x

#     # Create masks for each region
#     middle_mask = (x >= -0.5) & (x <= 0.5)
#     upper_mask = x > 0.5
#     lower_mask = x < -0.5

#     # Calculate each piece
#     middle = 0.2 * x * middle_mask.cast(x.dtype)
#     upper = (0.8 * x - 0.3) * upper_mask.cast(x.dtype)
#     lower = (0.8 * x + 0.3) * lower_mask.cast(x.dtype)

#     # Combine the pieces
#     return middle + upper + lower

#   def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
#     # Derivative is 1 for |x| <= 0.5, and 0.9 otherwise
#     middle_mask = (self.x >= -0.5) & (self.x <= 0.5)

#     # Gradient is 1 in middle region and 0.9 in outer regions
#     # grad = middle_mask.cast(self.x.dtype) + (0.9 * (1 - middle_mask).cast(self.x.dtype))
#     grad = middle_mask.cast(self.x.dtype) * 0.2 + (0.8 * (1 - middle_mask).cast(self.x.dtype))

#     return grad_output * grad
