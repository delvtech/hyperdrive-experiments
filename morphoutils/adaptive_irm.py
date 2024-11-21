from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AdaptiveIRM:
    u_target: float = 0.90
    rate_at_target: float = 0
    c: float = 4.0

    def __post_init__(self):
        if self.u_target is None:
            raise ValueError("u_target cannot be None")
        if self.rate_at_target is None:
            raise ValueError("rate_at_target cannot be None")
        if self.c is None:
            raise ValueError("c cannot be None")

    def e(self, utilization: float) -> float:
        # print(f"{utilization=}")
        # print(f"{self.u_target=}")
        # print(f"{(utilization - self.u_target)=}")
        # print(f"{(utilization - self.u_target) / self.u_target=}")
        if utilization > self.u_target:
            result = (utilization - self.u_target) / (1 - self.u_target)
        else:
            result = (utilization - self.u_target) / self.u_target
        return result

    def calc_borrow_rate(self, utilization: float) -> float:
        # print(f"{self.c=}")
        # print(f"{self.rate_at_target=}")
        # print(f"{self.e(utilization)=}")
        # print(f"{(1 - 1 / self.c)=}")
        # print(f"{(1 - 1 / self.c) * self.e(utilization)=}")
        # print(f"{((1 - 1 / self.c) * self.e(utilization) + 1)=}")
        if utilization > self.u_target:
            borrow_rate = self.rate_at_target * ((self.c - 1) * self.e(utilization) + 1)
        else:
            borrow_rate = self.rate_at_target * ((1 - 1 / self.c) * self.e(utilization) + 1)
        return borrow_rate

    def calc_supply_rate(self, utilization: float) -> float:
        return self.calc_borrow_rate(utilization) * utilization

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

    def update_rate_at_target(self, utilization: float, time_delta_in_years: float):
        """Update the rate at target based on the utilization and time delta."""
        self.rate_at_target = self.calc_new_rate_at_target_given_utilization(utilization, time_delta_in_years)

    def calc_quoted_rate_from_hd_rate(self, worst_u: float, hd_fixed_rate: float) -> float:
        return self.calc_quoted_rate_from_cost(worst_u, self.calc_fixed_cost_from_hd_rate(hd_fixed_rate))
    
    def calc_quoted_rate_from_cost(self, worst_u: float, fixed_cost: float) -> float:
        return self.calc_effective_rate_from_cost(worst_u, fixed_cost)

    def calc_hd_rate_given_quoted_rate(self, quoted_rate: float, u: float=0.35):
        return 1 / (1 - (quoted_rate - self.gap(u))) - 1

    def calc_hpr_given_apy(self, apy: float, holding_period_in_years: float) -> float:
        return math.pow(1 + apy, holding_period_in_years) - 1
