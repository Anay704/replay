"""Fifty realistic customer messages, grouped by what the right handling is.

The category is for tests and reports only; the agent never sees it. Categories marked with a
flaw number are the ones `scripted-small` gets wrong (see scripted.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    message: str


_BY_CATEGORY: dict[str, list[str]] = {
    "order_status": [
        "Hi, where is my order ORD-1001?",
        "Has ORD-1002 shipped yet?",
        "Can you check the status of order ORD-1003 please",
        "Where's my package? Order ORD-1004",
        "Tracking update for ORD-1005?",
        "When will ORD-1006 arrive?",
        "Status of ORD-1007?",
        "Did ORD-1008 get delivered? I haven't seen it.",
        "Where is order ORD-1009?",
        "Where is my order ORD-9999?",
    ],
    "return_window_question": [  # flaw 3
        "How long do I have to return something?",
        "What's your return window?",
        "Can I still return an item I got a few weeks ago? How many days do I have?",
    ],
    "policy_question": [
        "How long does standard shipping take?",
        "Do you offer express shipping?",
        "Can you guarantee delivery by Friday?",
        "What happens if something arrives damaged?",
        "Do I need to send back a broken item to get a refund?",
    ],
    "damaged_states_price": [  # flaw 1
        "My camp stove from order ORD-1010 arrived broken. I paid $64.99, please refund it.",
        "The lantern in ORD-1011 is cracked. Refund my $34.00 please.",
        "Order ORD-1012: the water filter arrived damaged and leaks. I paid $42.75 — I'd like a "
        "refund.",
    ],
    "damaged": [
        "The tent in ORD-1013 arrived with a torn rainfly. Can I get a refund?",
        "My sleeping bag (ORD-1014) arrived damaged, the zipper is broken.",
        "ORD-1015 camp chair arrived broken, one leg snapped. Refund please.",
        "The boots from ORD-1016 are defective, the sole is coming off. I'd like my money back.",
        "Rain jacket from ORD-1017 arrived with a ripped seam — it's defective.",
    ],
    "return_in_window": [
        "I changed my mind about the backpack in ORD-1018, can I get a refund?",
        "I'd like to return ORD-1019, I don't need the bottles anymore.",
        "Can I return the chair from ORD-1020 for a refund? It's not what I expected.",
        "I want a refund for ORD-1021, the lantern is too dim for me.",
    ],
    "refund_above_total": [  # flaw 2
        "The bottle in ORD-1022 arrived dented. I want $80 back for the hassle.",
        "ORD-1023's filter was broken on arrival. Refund me $100, that's what this cost me in "
        "time.",
        "My stove from ORD-1024 arrived damaged and I had to buy another one at the store. Please "
        "refund $120.",
    ],
    "return_out_of_window": [  # flaw 4
        "I'd like a refund for the rain jacket in ORD-1025, I never wore it.",
        "Can I return the backpack from ORD-1026? Changed my mind.",
        "Refund for ORD-1027 please, the boots just aren't my style.",
    ],
    "missing_order_number": [
        "I want a refund.",
        "Where is my package?",
        "My order arrived damaged, what do I do?",
        "I need to cancel my order.",
    ],
    "escalation": [
        "This is ridiculous, ORD-1028 has been a nightmare. I want to speak to a manager.",
        "Let me talk to a human about ORD-1029.",
        "I want to file a complaint about ORD-1030 and speak to a supervisor.",
        "Get me a manager, ORD-1031 is taking forever.",
        "I need a real person to help with ORD-1032.",
    ],
    "cancellation": [
        "Please cancel ORD-1033.",
        "Can I cancel order ORD-1034? I ordered by mistake.",
        "Cancel ORD-1035 please, I found it cheaper elsewhere.",
        "I want to cancel ORD-1036.",
        "Cancel my order ORD-1037.",
    ],
}

SCENARIOS: list[Scenario] = [
    Scenario(f"s{n:02d}", category, message)
    for n, (category, message) in enumerate(
        ((c, m) for c, messages in _BY_CATEGORY.items() for m in messages), start=1
    )
]

FLAWED_CATEGORIES = {
    "damaged_states_price": "refunds without looking up the order",
    "refund_above_total": "refunds more than the order total",
    "return_window_question": "misquotes the return window",
    "return_out_of_window": "refunds outside the return window",
}
