import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SocketClient, type WsMessage } from "./socket-client";

class FakeWs {
  static instances: FakeWs[] = [];
  readyState = 0;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  url: string;
  constructor(url: string) {
    this.url = url;
    FakeWs.instances.push(this);
  }
  send(s: string) {
    this.sent.push(s);
  }
  close() {
    this.readyState = 3;
    this.onclose?.();
  }
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  push(m: WsMessage) {
    this.onmessage?.({ data: JSON.stringify(m) });
  }
}

describe("SocketClient", () => {
  beforeEach(() => {
    FakeWs.instances = [];
    vi.useFakeTimers();
  });
  afterEach(() => vi.useRealTimers());

  it("sends sub on open, dispatches by topic, unsubscribes", () => {
    const c = new SocketClient("ws://x/ws", FakeWs as unknown as typeof WebSocket);
    const got: WsMessage[] = [];
    const off = c.subscribe("jobs", (m) => got.push(m));
    expect(c.getStatus()).toBe("connecting");
    const ws = FakeWs.instances[0];
    ws.open();
    expect(c.getStatus()).toBe("open");
    expect(ws.sent).toEqual([JSON.stringify({ op: "sub", topic: "jobs" })]);
    ws.push({ topic: "jobs", type: "job.state", seq: 1, ts: "t", data: { id: "j" } });
    ws.push({ topic: "other", type: "x", seq: 1, ts: "t", data: null });
    expect(got.map((m) => m.topic)).toEqual(["jobs"]);
    off();
    expect(ws.sent.at(-1)).toBe(JSON.stringify({ op: "unsub", topic: "jobs" }));
  });

  it("reconnects with backoff and re-subscribes", () => {
    const c = new SocketClient("ws://x/ws", FakeWs as unknown as typeof WebSocket);
    const statuses: string[] = [];
    c.onStatus((s) => statuses.push(s));
    c.subscribe("job:1", () => {});
    c.subscribe("run:2", () => {});
    FakeWs.instances[0].open();
    FakeWs.instances[0].close();
    expect(c.getStatus()).toBe("closed");
    vi.advanceTimersByTime(499);
    expect(FakeWs.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeWs.instances).toHaveLength(2);
    FakeWs.instances[1].open();
    expect(FakeWs.instances[1].sent).toEqual([
      JSON.stringify({ op: "sub", topic: "job:1" }),
      JSON.stringify({ op: "sub", topic: "run:2" }),
    ]);
    expect(statuses).toEqual(["connecting", "open", "closed", "connecting", "open"]);
    // a successful open resets the backoff to 500 ms
    FakeWs.instances[1].close();
    vi.advanceTimersByTime(500);
    expect(FakeWs.instances).toHaveLength(3);
    // without an open in between, the next wait doubles to 1000 ms
    FakeWs.instances[2].close();
    vi.advanceTimersByTime(999);
    expect(FakeWs.instances).toHaveLength(3);
    vi.advanceTimersByTime(1);
    expect(FakeWs.instances).toHaveLength(4);
  });

  it("close() stops reconnecting", () => {
    const c = new SocketClient("ws://x/ws", FakeWs as unknown as typeof WebSocket);
    c.subscribe("jobs", () => {});
    c.close();
    vi.advanceTimersByTime(20_000);
    expect(FakeWs.instances).toHaveLength(1);
    expect(c.getStatus()).toBe("closed");
  });
});
