"""One multiplexed WebSocket per browser tab: {op: sub|unsub, topic} in, bus messages out."""
import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..jobs.progress import read_log_lines
from ..models import now_iso

router = APIRouter()
SNAPSHOT_LINES = 2000


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    bus = websocket.app.state.bus
    runner = websocket.app.state.runner
    out: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
    forwarders: Dict[str, "asyncio.Task[None]"] = {}
    queues: Dict[str, "asyncio.Queue[Dict[str, Any]]"] = {}

    async def forward(topic: str) -> None:
        q = queues[topic]
        while True:
            out.put_nowait(await q.get())

    def snapshot(topic: str) -> None:
        """Recent log lines so a late subscriber catches up before live job.log chunks arrive."""
        jid = topic[len("job:"):]
        try:
            job = runner.store.get(jid)
        except KeyError:
            return
        lines = read_log_lines(Path(job.log_path))
        out.put_nowait({"topic": topic, "type": "job.snapshot", "seq": 0, "ts": now_iso(),
                        "data": {"job": job.model_dump(), "lines": lines[-SNAPSHOT_LINES:], "next_seq": len(lines)}})

    def handle(msg: Dict[str, Any]) -> None:
        op, topic = msg.get("op"), msg.get("topic")
        if op == "sub" and isinstance(topic, str):
            if topic not in queues:
                queues[topic] = bus.subscribe(topic)
                forwarders[topic] = asyncio.ensure_future(forward(topic))
            if topic.startswith("job:"):
                snapshot(topic)
        elif op == "unsub" and isinstance(topic, str):
            task = forwarders.pop(topic, None)
            if task:
                task.cancel()
            q = queues.pop(topic, None)
            if q is not None:
                bus.unsubscribe(topic, q)
        else:
            out.put_nowait({"topic": "", "type": "error", "seq": 0, "ts": now_iso(),
                            "data": {"detail": f"unknown op {op!r}"}})

    recv = asyncio.ensure_future(websocket.receive_json())
    send = asyncio.ensure_future(out.get())
    try:
        while True:
            done, _ = await asyncio.wait({recv, send}, return_when=asyncio.FIRST_COMPLETED)
            if recv in done:
                handle(recv.result())
                recv = asyncio.ensure_future(websocket.receive_json())
            if send in done:
                await websocket.send_json(send.result())
                send = asyncio.ensure_future(out.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        for t in list(forwarders.values()) + [recv, send]:
            t.cancel()
        for topic, q in queues.items():
            bus.unsubscribe(topic, q)
