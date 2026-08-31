"""Planners.

``fake``  a deterministic scripted planner, for tests

The real planner is :class:`agent.planner.LLMPlanner`, which lives one level
up because it is the production path. What is here is the test double that
makes the whole agent suite runnable with no credential, no network and no
non-determinism.
"""

from agent.planners.fake import PLANS, FakePlanner

__all__ = ["PLANS", "FakePlanner"]
