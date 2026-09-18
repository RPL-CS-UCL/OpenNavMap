"""In-process pub/sub used by the runner to feed WebSocket subscribers."""
import asyncio
from typing import Any, Dict, Set

from ..models import now_iso


class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, Set["asyncio.Queue[Dict[str, Any]]"]] = {}
        self._seq: Dict[str, int] = {}

    def subscribe(self, topic: str) -> "asyncio.Queue[Dict[str, Any]]":
        q: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self._subs.setdefault(topic, set()).add(q)
        return q

    def unsubscribe(self, topic: str, q: "asyncio.Queue[Dict[str, Any]]") -> None:
        subs = self._subs.get(topic)
        if subs:
            subs.discard(q)
            if not subs:
                del self._subs[topic]

    def publish(self, topic: str, type_: str, data: Any) -> Dict[str, Any]:
        seq = self._seq.get(topic, 0) + 1
        self._seq[topic] = seq
        msg = {"topic": topic, "type": type_, "seq": seq, "ts": now_iso(), "data": data}
        for q in list(self._subs.get(topic, ())):
            q.put_nowait(msg)
        return msg
