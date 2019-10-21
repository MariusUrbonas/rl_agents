import numpy as np

class EpsilonScheduler:

    def __init__(self, decay_func, **kwargs):
        self.decay_func = decay_func
        self.kwargs = kwargs
        self.step_c = 0

    def step(self):
        self.step_c += 1

    def val(self):
        return self.decay_func(self.step_c, self.kwargs)

    @classmethod
    def LinearlyDecaying(cls, total_steps, final_eps, decay_fract, warmup_steps):

        def func(step, kwargs):
            total_steps = kwargs.get('total_steps', None)
            decay_fract = kwargs.get('decay_fract', None)
            warmup_steps = kwargs.get('warmup_steps', None)
            final_eps = kwargs.get('final_eps', None)

            decay_period = total_steps*decay_fract

            steps_left = decay_period + warmup_steps - step
            bonus = (1.0 - final_eps) * steps_left / decay_period
            bonus = np.clip(bonus, 0., 1. - final_eps)
            return final_eps + bonus


        return cls(decay_func=func, total_steps=total_steps, final_eps=final_eps, decay_fract=decay_fract, warmup_steps=warmup_steps)