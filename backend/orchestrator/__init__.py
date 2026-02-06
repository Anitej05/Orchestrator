"""
Orchestrator - The OMNI-DISPATCHER System

A simple 2-node cycle: Brain <-> Hands
- Brain: Reasoning engine that selects resources based on state
- Hands: Stateless dispatcher that executes selected actions

Integration includes:
- Content Management System (CMS) hooks
- Telemetry and monitoring
- Dynamic replanning on failure
"""

from .brain import Brain, BrainDecision
from .hands import Hands
from .omni_dispatcher import (
    brain,
    hands,
    omni_dispatch,
    omni_brain_node,
    omni_hands_node,
    omni_route_condition,
    should_continue,
)

__all__ = [
    "Brain",
    "BrainDecision",
    "Hands",
    "brain",
    "hands",
    "omni_dispatch",
    "omni_brain_node",
    "omni_hands_node",
    "omni_route_condition",
    "should_continue",
]
