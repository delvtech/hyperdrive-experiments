"""Calculate an LP's PNL based on a certain set of parameters."""

from __future__ import annotations

from morphoutils import AdaptiveIRM

# avoid unnecessary warning from using fixtures defined in outer scope
# pylint: disable=redefined-outer-name

# pylint: disable=missing-function-docstring,too-many-statements,logging-fstring-interpolation,missing-return-type-doc
# pylint: disable=missing-return-doc,too-many-function-args

QUOTED_RATE = 0.07

irm = AdaptiveIRM(u_target=0.9, rate_at_target=0.06)