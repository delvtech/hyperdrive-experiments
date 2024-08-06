# Points Distribution

## Summary

## Details

Hyperdrive has an amount of base tokens delegated to the underlying yield source.
We currently attribute that yield to LPs and shorts, based on the short exposure at a given point in time.
How do we do this? Coding magic.

## Options
Thesis: points and rewards should go to variable yield positions, since asset delegation is tthe source of both variable yield and points. Points are just another form of variable yield.

### Shorts Only

This is the easiest option, though calls into question the 'fairness' of the approach. The argument for this approach is that it is simpler to implement and easier to understand and maintain. It could be argued that providing maxiumum incentive to the shorts will ultimately benefit the LPs and the protocol in general the most as it will generate the most activity. Doing so could be seen as a competitive advantage over Pendle, if you get more points all else equal.
However, it may not be the most 'fair' approach because there could be a large amount of LP capital that is invested in the underlying yield source and therefore has rights to the rewards/points emissions of the yield source. There are also edge cases where there aren't that many short positions relative to the LP positions which 'over incentive' the short positions and the LPs would then lose out.

### Shorts and LP

This is the more fair approach and possibly also the best approach since attracting LPs to the protocol with rewards/points incentives will likely result in a larger amount of activity. However, it is more complex to implement and maintain.  In this approach, LPs and shorts are given points/rewards, and longs are not.

In Hyperdrive, LP funds back any trade by opening an opposite position. When the market is not perfectly balanced, the LPs will either have long exposure or short exposure.  The LP's short exposure plus the idle capital reflects how much variable yield LPs are earning, and as a result how many points they should be allocated (lp_short_positions + lp_idle_capital).

Shorts are much simplier, the total amount of active short positions is the total capital allocation for shorts that should be earning points and rewards emissions (short_allocation = shorts_outstanding).

To figure out the amount of points each user gets, we can break down the problem into two parts:
1. Find the total capital for LPs and the total capital for user short positions that should be earning points and rewards emissions.  
2. For each group, LPs and shorts, determine each user's share of the group at each timestep to get their time-weighted shares of the rewards/points emissions.

#### Finding the total capital allocation for LPs and shorts
There are a couple ways we can do this.  Since we know that the total amount of shorts outstanding is the total capital allocation for shorts, we could just say that total LP allocation is the TVL of the pool minus the shorts outstanding:

The entire amount of money in the pool is earning points and we know the short allocation is just the total amount of shorts outstanding.  So we could find the LP allocation by simple subraction:

short_allocation = shorts_outstanding
FORMULA ONE: lp_allocation = total_pool_tvl - short_allocation 

Alternatively, we could try to calculate the total capital allocation for the LPs directly.  There are two parts to this: the part that is backing longs (therefore taking short positions) and the idle capital simply earning the variable yield:

lp_short_positions = long_exposure
lp_idle_capital = share_reserves - long_exposure

Adding these two together,
lp_short_positions + lp_idle_capital = long_exposure + share_reserves - long_exposure

therefore:

lp_short_positions + lp_idle_capital = share_reserves

Now we can check the result and hopefully assert that

FORMULA TWO: lp_allocation = share_reserves 
FORMULA THREE: lp_short_positions + lp_idle_capital

#### Finding the time-weighted shares of the rewards/points emissions


# Distribution





### Dscussion
Pro-ration of points can be done similarly for LPs and shorts.

for attributing fees, we do:

```
share_of_pool = your_lp_tokens / total_lp_tokens
your_fees = share_of_pool * total_fees
```

for attributing points to LPs, we do:

```
share_of_pool = your_lp_tokens / total_lp_tokens
share_of_capital_that_is_idle = idle_capital / total_capital
your_points = share_of_pool * share_of_capital_that_is_idle * total_points
```

for attributing points to Shorts, we do:

```
share_of_shorts = your_short_tokens / total_short_tokens
your_points = share_of_shorts * total_points
```

this is complicated because:

1. you have to account for every block (changes w/ every txn)

LP PIE =

- X% idle
- Y% backing shorts (LP effectively holds long position)
- Z% backing longs (LP effectively holds short position) - tracked by the variable long_exposure (that's the number of longs open in the system) - this is user_long_exposure from the perspective of a user - effectively equivalent to lp_short_exposure
  only X and Z earn points

how does long_exposure work?

- longs increase long_exposure
- long_exposure is increased by the fixed rate that will be paid to the user
- shorts are tracked by decreases in share_reserves
- share_reserves measures lp capital that isn't backing shorts

// non-netted longs in the market
long_exposure

// idle captial in the maarket
idle_capital = share_reserves - long_exposure

// amount of longs users have
longs_outstanding

For example:

user buys 10 longs for 9 base when price is 0.9

long_exposure += 10 (equal to amount of capital LPs back longs)
longs_outstanding += 10 (equal to the # of longs)
share_reserves += 9
∆_idle_captial = ∆_share_reserves - ∆_long_expsure = 9 - 10 = -1

which makes sense, because the LPs back the long buy guaranteeing the yield for the fixed rate.

So at every timestep the LPs earn points on idle_captial + long_exposure, which just equals share_reserves (in base).

Each individual LP earns their pro-rata share based on their percent share of total lp tokens at each timestep.

Similarly, shorts are paid by the size of their short position(s) and the total amount of shorts outstanding at any given timestep.

### Examples
Let's say we start with a market of 1 LP of 100 base tokens.

```
The total LP supply is 100.
The exposure is 0.
The idle capital is 100.
The LP is given all of the yield source rewards/points.
pool_share_reserves = 100
idle_capital_calc = 100 - 0 = 100
```

If a user opens a short position, the LP would open a long position to back the trade.

```
A user shorts 10 base tokens for 1 base token.
The LP longs 10 base tokens.
The long exposure is 10.
The LP's idle capital is 90.
pool_share_reserves = 100 - 10 = 90
idle_capital_calc = 90 - 0 = 90
```

So,

```
The LP would earn points on 9/10 of their LP position.
The LP would earn a fixed yield on 10 base tokens.
The LP would earn fees from short trade.
The short position would earn points 10 base tokens, a 10x multiplier.
```

Now what happens if the short position is netted out by another user that opens a long position?

```
A user longs for 10 base tokens.
Now the market is netted.
The long position earns fixed yield on 10 base tokens.
The LP would earn points on their entire position.
The short position is still earning 10x multiplier on points.
pool_share_reserves = 100 (because short and long net out)
idle_capital_calc = 100 - 0 = 100
```

Now what happens if another long is opened?

```
Another user longs for 10 base tokens.
The LP shorts 10 base tokens for 1 base token to back the trade.

The short exposure is 10.
The LP's idle capital is 99.

pool_share_reserves = 110
idle_capital_calc = 110 - 1 = 109
```

So,

```
Both long positions earn fixed yield on 10 base tokens.

The LP would earn 10x multiplier on the short exposure.
The LP would earn 1x points on the idle capital.
Total multiplier for the LP is 109 / 100 = 1.09.

The short position is still earning  10x multiplier on points.
```