#! /usr/bin/env python
"""ObjectProvider abstraction for the L4 object graph (OpenGoalNav T1.2).

A provider only emits raw per-keyframe observations; cross-frame association and
merging are done by ObjectGraph, so different providers (msgnav, boxer, ...) share
identical merge behavior. Concrete providers (e.g. msgnav) are implemented in T1.3
after the schema is frozen.
"""
from abc import ABC, abstractmethod
from typing import Any, List

from object_node import ObjectObservation


class ObjectProvider(ABC):
    """Emits raw object observations for a keyframe (no association)."""

    provider_name: str = "base"

    @abstractmethod
    def on_keyframe(self, kf: Any, obs: Any) -> List[ObjectObservation]:
        """Return raw ObjectObservation list detected in this keyframe/observation."""
        raise NotImplementedError


class MockObjectProvider(ObjectProvider):
    """Test provider: replays a pre-scripted list of observations per keyframe."""

    provider_name = "mock"

    def __init__(self, scripted: List[List[ObjectObservation]]) -> None:
        self._scripted = scripted
        self._call = 0

    def on_keyframe(self, kf: Any, obs: Any) -> List[ObjectObservation]:
        result = self._scripted[self._call] if self._call < len(self._scripted) else []
        self._call += 1
        return result
