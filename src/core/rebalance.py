"""
Core sleeve rebalance logic.
Skill reference: .claude/skills/position-sizing/SKILL.md
"""


def core_rebalance_orders(
    current_weights: dict,
    target_weights: dict,
    sleeve_nav: float,
    prices: dict,
    rebalance_band: float = 0.05,
) -> list:
    """
    Generate rebalance orders for Core sleeve.
    Only trades positions that drifted beyond the band.
    """
    orders = []

    for symbol, target_wt in target_weights.items():
        actual_wt = current_weights.get(symbol, 0.0)
        drift = actual_wt - target_wt

        if abs(drift) > rebalance_band:
            trade_notional = abs(drift) * sleeve_nav
            side = "SELL" if drift > 0 else "BUY"
            orders.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "notional": trade_notional,
                    "order_type": "MOC",
                    "sleeve": "core",
                    "reason": f"drift={drift:.2%}",
                }
            )

    return orders
