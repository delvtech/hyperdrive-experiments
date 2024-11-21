import math
import numpy as np

class AdaptiveIRM:

    _u_target: float
    _rate_at_target: float
    _c: float

    def __init__(
        self,
        u_target: float | None = 0.90,
        rate_at_target: float | None = 0,
        c: float | None = 4.0
    ):
        _u_target = u_target
        _rate_at_target = rate_at_target
        _c = c

    @property
    def u_target(self) -> float:
        return self._u_target
    
    @u_target.setter
    def u_target(self, u: float):
        self._u_target = u
    
    @property
    def rate_at_target(self) -> float:
        return self._rate_at_target

    @rate_at_target.setter
    def rate_at_target(self, rate: float):
        self._rate_at_target = rate

    @property
    def c(self) -> float:
        return self._c
    
    @c.setter
    def c(self, c: float):
        self._c = c

    def e(self, u: float) -> float:
        if u > u_target:
            result = (u - self.u_target) / (1 - self.u_target)
        else:
            result = (u - self.u_target) / self.u_target
        return result

    def calc_borrow_rate(self, u: float) -> float:
        if u > u_target:
            borrow_rate = self.rate_at_target * ((self.c - 1) * self.e(u) + 1)
        else:
            borrow_rate = self.rate_at_target * ((1 - 1 / self.c) * self.e(u) + 1)

    def calc_supply_rate(self, u: float) -> float:
        return self.calc_borrow_rate(u) * u

    def gap(self, u):
        return self.calc_borrow_rate(u) - self.calc_supply_rate(u)

    def calc_fixed_cost_from_hd_rate(hd_fixed_rate: float) -> float:
        # if loan = 100
        # short 100 bonds
        # hd_fixed_rate = 5%
        # basePaid = 1 - 1 / (1 + fixedRate)
        #          = 1 - 1 / 1.05 
        #          = 0.0476 -> fixed_cost
        return 1 - (1 / (1 + hd_fixed_rate))

    def calc_effective_rate_from_rate(self, u: float, hd_fixed_rate: float) -> float:
        return self.calc_effective_rate_from_cost(u, self.calc_fixed_cost_from_hd_rate(hd_fixed_rate))
    
    def calc_effective_rate_from_cost(self, u: float, fixed_cost: float) -> float:
        return fixed_cost + self.gap(u)
    
    def calc_new_rate_at_target_given_utilization(self, u: float, time_delta_in_years: float) -> float:
        return self.rate_at_target * math.exp(50 * self.e(u)*time_delta_in_years)
    
    def calc_utilization_given_new_rate_at_target(self, new_rate_at_target: float, time_delta_in_years: float) -> float:
        return ((1 - self.u_target) * math.log(new_rate_at_target / self.rate_at_target)) / (50 * time_delta_in_years) + self.u_target

    def calc_quoted_rate_from_hd_rate(self, worst_u: float, hd_fixed_rate: float) -> float:
        return self.calc_quoted_rate_from_cost(worst_u, self.calc_fixed_cost_from_hd_rate(hd_fixed_rate))
    
    def calc_quoted_rate_from_cost(self, worst_u: float, fixed_cost: float) -> float:
        return self.calc_effective_rate_from_cost(worst_u, fixed_cost)

    def calc_hpr_given_apy(apy: float, holding_period_in_years):
        return math.pow(1 + apy, holding_period_in_years) - 1
    
