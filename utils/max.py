from agent0.hyperdrive.interactive import InteractiveHyperdrive
from fixedpointmath import FixedPoint

from typing import NamedTuple

Max = NamedTuple("Max", [("base", FixedPoint), ("bonds", FixedPoint)])
GetMax = NamedTuple("GetMax", [("long", Max), ("short", Max)])


def get_max(
    _interactive_hyperdrive: InteractiveHyperdrive,
    _share_price: FixedPoint,
    _current_base: FixedPoint,
) -> GetMax:
    """Get max trade sizes.

    Returns
    -------
    GetMax
        A NamedTuple containing the max long in base, max long in bonds, max short in bonds, and max short in base.
    """
    max_long_base = _interactive_hyperdrive.interface.calc_max_long(budget=_current_base)
    max_long_bonds = _interactive_hyperdrive.interface.calc_bonds_out_given_shares_in_down(max_long_base / _share_price)
    max_short_bonds = _interactive_hyperdrive.interface.calc_max_short(budget=_current_base)
    max_short_shares = _interactive_hyperdrive.interface.calc_shares_out_given_bonds_in_down(max_short_bonds)
    max_short_base = max_short_shares * _share_price
    return GetMax(
        Max(max_long_base, max_long_bonds),
        Max(max_short_base, max_short_bonds),
    )
