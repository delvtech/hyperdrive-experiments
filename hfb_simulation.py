"""Calculate an LP's PNL based on a certain set of parameters."""

# %%
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from morphoutils import AdaptiveIRM

# avoid unnecessary warning from using fixtures defined in outer scope
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring,too-many-statements,logging-fstring-interpolation,missing-return-type-doc
# pylint: disable=missing-return-doc,too-many-function-args

# %%
# define functions
def generate_brownian_bridge(n_steps, sigma=1.0, start_point=0.0, end_point=0.0):
    """Generate a Brownian Bridge time series that starts and ends at specified points.

    Arguments:
    n_steps (int): Number of time steps
    sigma (float): Volatility parameter
    start_point (float): Starting value
    end_point (float): Ending value

    Returns:
    numpy array: Time series following Brownian Bridge
    """
    # Generate standard Brownian motion
    dt = 1.0/n_steps
    t = np.linspace(0, 1, n_steps)
    dW = np.random.normal(0, sigma * np.sqrt(dt), n_steps)
    W = np.cumsum(dW)

    # Convert to bridge by conditioning on endpoint
    bridge = start_point + W - t * W[-1] + t * (end_point - start_point)

    return bridge

# %%
def generate_uniform_distribution(size=20, high=1., low=0.45, midpoint=0.91, lower_probability=0.5):
    """Sample from a uniform distribution, adjusting to new bounds and midpoint.
    Observations below the midpoint have lower_probability chance of occurring.

    The distribution is split into two parts:
    1. [low, midpoint] with probability lower_probability
    2. [midpoint, high] with probability (1-lower_probability)

    Within each range, values are uniformly distributed.
    """
    # Generate uniform values between 0 and 1
    values = np.random.uniform(0, 1, size=size)

    # For values that should be in lower range (with probability lower_probability)
    mask_lower = values <= lower_probability
    values[mask_lower] = low + (midpoint - low) * (values[mask_lower] / lower_probability)

    # For values that should be in upper range (with probability 1-lower_probability)
    mask_upper = ~mask_lower
    values[mask_upper] = midpoint + (high - midpoint) * ((values[mask_upper] - lower_probability) / (1 - lower_probability))

    return values

def calculate_expected_mean(low, high, midpoint, lower_probability):
    """Calculate the expected mean of the skewed uniform distribution.

    The mean is: P(lower) * E[lower_range] + P(upper) * E[upper_range]
    where E[lower_range] = (low + midpoint)/2
    and E[upper_range] = (midpoint + high)/2
    """
    lower_range_mean = (low + midpoint) / 2
    upper_range_mean = (midpoint + high) / 2
    return lower_probability * lower_range_mean + (1 - lower_probability) * upper_range_mean

def calc_summary_stats(utilization, rates):
    # Calculate some summary statistics
    avg_rate = np.mean(rates)
    max_rate = np.max(rates)
    min_rate = np.min(rates)
    rate_volatility = np.std(rates)
    utilization_volatility = np.std(utilization)

    print("\nVolatility Comparison:")
    print(f"Input sigma parameter: {0.1:.2%}")
    print(f"Realized utilization volatility: {utilization_volatility:.2%}")
    print(f"Realized rate volatility: {rate_volatility:.2%}")

    print("\nIRM Statistics:")
    print(f"Average Borrow Rate: {avg_rate:.2%}")
    print(f"Maximum Borrow Rate: {max_rate:.2%}")
    print(f"Minimum Borrow Rate: {min_rate:.2%}")

def plot_utilization_and_rates(_irm: AdaptiveIRM, utilization, borrow_rates, supply_rates=None, effective_rates=None, rate_at_target_list=None):
    # Plot both utilization and rates
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot utilization
    ax1.step(range(1, len(utilization)+1), utilization, label='Utilization')
    ax1.set_ylabel('Utilization')
    ax1.set_title('Pool Utilization Over Time')
    ax1.grid(True)
    ax1.legend()

    # Plot rates
    lines = []
    labels = []
    
    # Borrow Rate
    line = ax2.step(range(1, len(borrow_rates)+1), borrow_rates, color='orange', where='post')[0]
    lines.append(line)
    labels.append(f'Borrow Rate (avg={np.mean(borrow_rates):.2%})')
    
    if supply_rates is not None:
        # Supply Rate
        line = ax2.step(range(1, len(supply_rates)+1), supply_rates, color='blue', where='post')[0]
        lines.append(line)
        labels.append(f'Supply Rate (avg={np.mean(supply_rates):.2%})')
        
        # Gap
        gap = np.array(borrow_rates) - np.array(supply_rates)
        line = ax2.step(range(1, len(gap)+1), gap, color='green', linestyle='dashed', where='post')[0]
        lines.append(line)
        labels.append(f'Gap (avg={np.mean(gap):.2%})')
    
    if effective_rates is not None:
        line = ax2.step(range(1, len(effective_rates)+1), effective_rates, color='red', where='post')[0]
        lines.append(line)
        labels.append(f'Effective Rate (avg={np.mean(effective_rates):.2%})')
    
    if rate_at_target_list is not None:
        line = ax2.step(range(1, len(rate_at_target_list)+1), rate_at_target_list, color='purple', where='post')[0]
        lines.append(line)
        labels.append(f'Rate at Target (avg={np.mean(rate_at_target_list):.2%})')
    
    ax2.set_ylabel('Rate')
    ax2.set_title('Borrow Rate Over Time')
    ax2.grid(True)
    ax2.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_transformation_function(_irm: AdaptiveIRM, utilization):
    # Plot the rate transformation function
    u_range = np.linspace(0.0, 1.0, 100)
    transformed_rates = [_irm.calc_borrow_rate(u) for u in u_range]

    plt.figure(figsize=(10, 6))
    plt.step(u_range, transformed_rates, label='Rate Function')
    plt.scatter(utilization, borrow_rates, color='red', alpha=0.5, label='Observed Points')
    plt.axvline(x=_irm.u_target, color='gray', linestyle='--', label='Target Utilization')
    plt.xlabel('Utilization')
    plt.ylabel('Borrow Rate')
    plt.title('IRM Rate Transformation Function')
    plt.grid(True)
    plt.legend()
    plt.show()

# %%
def calculate_irm_stats(_irm: AdaptiveIRM, utilization_list, utilization_gap_point = 0.35) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Calculate rates at each utilization level
    start_time = time.time()
    # we know this scenario is not going to happen
    # but it is one indicative scenario of a possible bad scenario (not even the worst)
    hd_fixed_rate = _irm.calc_hd_rate_given_quoted_rate(QUOTED_RATE, utilization_gap_point)
    borrow_rates = []
    supply_rates = []
    rate_at_target_list = []
    effective_rates = []
    # effective_rate_80_list = []
    for utilization_value in utilization_list:
        # Get the borrow rate from our adaptive IRM
        borrow_rate = _irm.calc_borrow_rate(utilization_value)
        supply_rate = _irm.calc_supply_rate(utilization_value)
        effective_rate = _irm.calc_effective_rate_from_hd_rate(utilization_value, hd_fixed_rate)
        rate_at_target = _irm.rate_at_target
        borrow_rates.append(borrow_rate)
        supply_rates.append(supply_rate)
        rate_at_target_list.append(rate_at_target)
        effective_rates.append(effective_rate)

        _irm.update_rate_at_target(utilization_value, 1/365)

    borrow_rates = np.array(borrow_rates)
    supply_rates = np.array(supply_rates)
    effective_rates = np.array(effective_rates)
    rate_at_target_list = np.array(rate_at_target_list)
    # print(f"rates simulated in {time.time() - start_time:.2f} seconds")
    return borrow_rates, supply_rates, effective_rates, rate_at_target_list

# %%
# create one trial
QUOTED_RATE = 0.07
CURRENT_RATE_AT_TARGET = 0.0673
HIGH_UTILIZATION_RATE = 0.95
LOW_UTILIZATION_RATE = 0.45
MIDPOINT = 0.90
LOW_PROBABILITY = 0.50
NUM_DAYS = 180
utilization = generate_uniform_distribution(size=NUM_DAYS, high=HIGH_UTILIZATION_RATE, low=LOW_UTILIZATION_RATE, midpoint=MIDPOINT, lower_probability=LOW_PROBABILITY)
UTILIZATION_POINT = 0.77  # lower is more conservative, creating more of a buffer
# HD fixed rate
irm = AdaptiveIRM(u_target=0.9, rate_at_target=CURRENT_RATE_AT_TARGET)
hd_fixed_rate = irm.calc_hd_rate_given_quoted_rate(quoted_rate=QUOTED_RATE, u=UTILIZATION_POINT)
print(f"{hd_fixed_rate=:.2%}")
borrow_rates, supply_rates, effective_rates, rates_at_target = calculate_irm_stats(irm, utilization, UTILIZATION_POINT)
# plot_utilization_and_rates(irm, utilization, borrow_rates, supply_rates, effective_rates, rates_at_target)

# %%
# create histogram of above
# == Supply Rates ==
# Sky Savings Rate = 8.5%
# Aave USDC Supply Rate = 14.9%
# Morpho USDC GTUSDCCORE Supply Rate = 15.7%
# Morpho USDC USUALUSDC+ Supply Rate = 18.3%

# == Borrow Rates ==
# Aave USDC Borrow Rate = 19.6%
# Morpho USDC Borrow Rate against wBTC = 12.5%
# Morpho USDC Borrow Rate against cbBTC = 12.04%
hist_utilization = []
hist_effective_avg = []
hist_borrow = []
hist_borrow_avg = []
NUM_TRIALS = 1_000
for _ in range(NUM_TRIALS):
    utilization = generate_uniform_distribution(size=NUM_DAYS, high=HIGH_UTILIZATION_RATE, low=LOW_UTILIZATION_RATE, midpoint=MIDPOINT, lower_probability=LOW_PROBABILITY)
    irm = AdaptiveIRM(u_target=0.9, rate_at_target=CURRENT_RATE_AT_TARGET)
    borrow_rates, supply_rates, effective_rates, rates_at_target = calculate_irm_stats(irm, utilization, UTILIZATION_POINT)
    hist_borrow.append(borrow_rates)
    hist_borrow_avg.append(np.mean(borrow_rates))
    hist_utilization.append(utilization)
    hist_effective_avg.append(np.mean(effective_rates))
hist_effective_avg = np.array(hist_effective_avg)

print(f"average effective rate: {np.mean(hist_effective_avg):.3%}")
print(f"average morpho borrow rate: {np.mean(hist_borrow_avg):.3%}")
plt.figure(figsize=(10, 6))
p = plt.hist(hist_effective_avg, bins=30, label=f"Effective Fixed Borrow Rate (avg={np.mean(hist_effective_avg):.3%})", alpha=0.5, color='orange')
plt.xlabel('Average Rate')
plt.ylabel('Frequency')
plt.title('Histogram of Average Rate')
plt.grid(True)
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(hist_effective_avg, bins=30, label=f"Effective Fixed Borrow Rate (avg={np.mean(hist_effective_avg):.3%})", alpha=0.5, color='orange')
plt.hist(hist_borrow_avg, bins=30, alpha=0.5, color='blue', label=f"Variable Borrow Rate (avg={np.mean(hist_borrow_avg):.3%})")

# Add vertical lines for extra scenarios
colors = ['red', 'green', 'purple']
extra_effective_rates = [
    ("14 days increase then 6 decrease", 0.0811),
    ("12 days increase then 8 decrease", 0.0805),
    ("10 days increase then 10 decrease", 0.0792),
]
for (scenario, rate), color in zip(extra_effective_rates, colors):
    plt.axvline(x=rate, color=color, linestyle='--', label=f"{scenario} ({rate:.3%})")

plt.xlabel('Average Rate')
plt.ylabel('Frequency')
plt.title('Histogram of Average Rate')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# plot historical rate paths
fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(hist_borrow)
# transpose the array
hist_borrow_transposed = np.array(hist_borrow).T
ax.plot(hist_borrow_transposed, alpha=0.05)
# plot percentiles of the data
prctiles = [10, 50, 90]
for percentile in prctiles:
    ax.plot(np.percentile(hist_borrow_transposed, percentile, axis=1), label=f'Percentile {percentile}')

ax.set_xlabel('Time')
ax.set_ylabel('Borrow Rate')
ax.set_title('Historical Borrow Rate Paths')
ax.grid(True)
ax.legend()
plt.show()

# %%
# plot a specific historical trial
idx = np.where(hist_effective_avg == np.max(hist_effective_avg))[0][0]
# find the trial closest to the 50th percentile
ptile = np.percentile(hist_effective_avg, 50)
print(f"finding trial closest to {ptile}")
idx = np.argmin(np.abs(hist_effective_avg - ptile))
utilization = hist_utilization[idx]
borrow_rates, supply_rates, effective_rates, rates_at_target = calculate_irm_stats(irm, utilization, UTILIZATION_POINT)
plot_utilization_and_rates(irm, utilization, borrow_rates, supply_rates, effective_rates, rates_at_target)

# %%
# manually set utilization schedule
irm = AdaptiveIRM(u_target=0.9, rate_at_target=CURRENT_RATE_AT_TARGET)
utilization = np.array([0.95] * 10 + [0.35] * 10)
borrow_rates, supply_rates, effective_rates, rate_at_target = calculate_irm_stats(irm, utilization)
plot_utilization_and_rates(irm, utilization, borrow_rates, supply_rates, effective_rates, rate_at_target)
# the average gap between borrow and supply is your extra cost of the fixed borrow
gap = np.array(borrow_rates) - np.array(supply_rates)
print(f"Average gap: {np.mean(gap):.2%}")

# %%
results = []
HIGH_UTILIZATION_RATE = 0.95
LOW_UTILIZATION_RATE = 0.45
worst_n_days_high = 0
worst_gap = 0
NUM_DAYS = 20
for n_days_high in range(1, NUM_DAYS + 1):
    nday_low = NUM_DAYS - n_days_high
    irm = AdaptiveIRM(u_target=0.9, rate_at_target=CURRENT_RATE_AT_TARGET)
    utilization = np.array([HIGH_UTILIZATION_RATE] * n_days_high + [LOW_UTILIZATION_RATE] * nday_low)
    # utilization = np.tile(utilization, 9)  # repeat the array to match 180 days
    borrow_rates, supply_rates, effective_rates, rates_at_target = calculate_irm_stats(irm, utilization, UTILIZATION_POINT)
    gap = np.array(borrow_rates) - np.array(supply_rates)
    avg_gp = np.mean(gap)
    if avg_gp > worst_gap:
        worst_gap = avg_gp
        worst_n_days_high = 10
    print(f"N days {HIGH_UTILIZATION_RATE}: {n_days_high}, N days {LOW_UTILIZATION_RATE}: {nday_low}, Average gap: {avg_gp:.4%}, Average effective rate: {np.mean(effective_rates):.4%}")
print(f"quoting hyperdrive fixed rate at {UTILIZATION_POINT=:.2%}")
print(f"worst scenario is at {HIGH_UTILIZATION_RATE} for {worst_n_days_high} days and {LOW_UTILIZATION_RATE} for {NUM_DAYS-worst_n_days_high} days with an average gap of {worst_gap:.4%}")
n_days_high = worst_n_days_high
nday_low = NUM_DAYS - n_days_high
irm = AdaptiveIRM(u_target=0.9, rate_at_target=CURRENT_RATE_AT_TARGET)
utilization = np.array([HIGH_UTILIZATION_RATE] * n_days_high + [LOW_UTILIZATION_RATE] * nday_low)
# utilization = np.tile(utilization, 9)  # repeat the array to match 180 days
borrow_rates, supply_rates, effective_rates, rates_at_target = calculate_irm_stats(irm, utilization, UTILIZATION_POINT)
avg_effective_rate = np.mean(effective_rates)
print(f"Average effective rate: {avg_effective_rate:.2%}")
plot_utilization_and_rates(irm, utilization, borrow_rates, supply_rates, effective_rates, rates_at_target)

# %%

# Test the distribution
np.random.seed(42)  # For reproducibility
size = 10000

# Test 1: Symmetric case
v = generate_uniform_distribution(size=size, high=1, low=0, midpoint=0.5, lower_probability=0.5)
expected_mean = calculate_expected_mean(0, 1, 0.5, 0.5)
print(f"Symmetric case:")
print(f"Expected mean: {expected_mean:.2%}")
print(f"Actual mean: {np.mean(v):.2%}")

# Test 2: Skewed case
lower_probability = 0.1
v = generate_uniform_distribution(size=size, high=1, low=0, midpoint=0.5, lower_probability=lower_probability)
expected_mean = calculate_expected_mean(0, 1, 0.5, lower_probability)
print(f"\nSkewed case (lower_probability={lower_probability}):")
print(f"Expected mean: {expected_mean:.2%}")
print(f"Actual mean: {np.mean(v):.2%}")
