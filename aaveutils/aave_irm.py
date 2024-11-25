import math
import numpy as np

@dataclass
class AaveIRM:
    u_target: float = 0.92
    slope_1: float = 0 #rate_at_target when u <= u_target
    slope_2: float = 0 #rate_at_target when u > u_target
    reserve_factor: float = 0.10
    base_rate: float = 0

    def __post_init__(self):
        if self.u_target is None:
            raise ValueError("u_target cannot be None")
        if self.slope_1 is None:
            raise ValueError("slope_1 cannot be None")
        if self.slope_2 is None:
            raise ValueError("slope_2 cannot be None")
        if self.reserve_factor is None:
            raise ValueError("reserve_factor cannot be None")
        if self.base_rate is None:
            raise ValueError("base_rate cannot be None")

    def calc_borrow_rate(self, utilization: float) -> float:
        if utilization > self.u_target:
            borrow_rate = self.base_rate + self.slope_1 + self.slope_2 * ((utilization - self.u_target) / (1 - self.u_target))
        else:
            borrow_rate = self.base_rate + self.slope_1 * (utilization / self.u_target)
        return borrow_rate

    def calc_supply_rate(self, utilization: float) -> float:
        return self.calc_borrow_rate(utilization) * utilization * (1 - self.reserve_factor)

    def gap(self, utilization: float):
        return self.calc_borrow_rate(utilization) - self.calc_supply_rate(utilization)

    def calc_fixed_cost_from_hd_rate(self, hd_fixed_rate: float) -> float:
        # if loan = 100
        # short 100 bonds
        # hd_fixed_rate = 5%
        # basePaid = 1 - 1 / (1 + fixedRate)
        #          = 1 - 1 / 1.05 
        #          = 0.0476 -> fixed_cost
        return 1 - (1 / (1 + hd_fixed_rate))

    def calc_effective_rate_from_hd_rate(self, utilization: float, hd_fixed_rate: float) -> float:
        return self.calc_effective_rate_from_cost(utilization, self.calc_fixed_cost_from_hd_rate(hd_fixed_rate))
    
    def calc_effective_rate_from_cost(self, utilization: float, fixed_cost: float) -> float:
        return fixed_cost + self.gap(utilization)
    
    def calc_new_rate_at_target_given_utilization(self, utilization: float, time_delta_in_years: float) -> float:
        return self.rate_at_target * math.exp(50 * self.e(utilization)*time_delta_in_years)
    
    def calc_utilization_given_new_rate_at_target(self, new_rate_at_target: float, time_delta_in_years: float) -> float:
        return ((1 - self.u_target) * math.log(new_rate_at_target / self.rate_at_target)) / (50 * time_delta_in_years) + self.u_target

    def update_rate_at_target(self, utilization: float, time_delta_in_years: float, new_rate_at_target: float = None):
        """Update the rate at target based on the utilization and time delta."""
        if new_rate_at_target is None:
            self.rate_at_target = self.calc_new_rate_at_target_given_utilization(utilization, time_delta_in_years)
        else:
            self.rate_at_target = new_rate_at_target

    def calc_quoted_rate_from_hd_rate(self, worst_u: float, hd_fixed_rate: float) -> float:
        return self.calc_quoted_rate_from_cost(worst_u, self.calc_fixed_cost_from_hd_rate(hd_fixed_rate))
    
    def calc_quoted_rate_from_cost(self, worst_u: float, fixed_cost: float) -> float:
        return self.calc_effective_rate_from_cost(worst_u, fixed_cost)

    def calc_hd_rate_given_quoted_rate(self, quoted_rate: float, u: float=0.35):
        return 1 / (1 - (quoted_rate - self.gap(u))) - 1

    def calc_hpr_given_apy(self, apy: float, holding_period_in_years: float) -> float:
        return math.pow(1 + apy, holding_period_in_years) - 1
