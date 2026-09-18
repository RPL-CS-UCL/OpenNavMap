export type SocketStatus = "connecting" | "open" | "closed";
export interface WsMessage<T = unknown> {
  topic: string;
  type: string;
  seq: number;
  ts: string;
  data: T;
}
type Handler = (m: WsMessage) => void;

const BACKOFF_MIN = 500;
const BACKOFF_MAX = 10_000;

/** One multiplexed connection per tab; topics are re-subscribed after every reconnect. */
export class SocketClient {
  private url: string;
  private Impl: typeof WebSocket;
  private ws: WebSocket | null = null;
  private status: SocketStatus = "closed";
  private handlers = new Map<string, Set<Handler>>();
  private statusListeners = new Set<(s: SocketStatus) => void>();
  private backoff = BACKOFF_MIN;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private closed = false;

  constructor(url: string, Impl: typeof WebSocket = WebSocket) {
    this.url = url;
    this.Impl = Impl;
  }

  getStatus(): SocketStatus {
    return this.status;
  }

  onStatus(cb: (s: SocketStatus) => void): () => void {
    this.statusListeners.add(cb);
    return () => {
      this.statusListeners.delete(cb);
    };
  }

  subscribe(topic: string, handler: Handler): () => void {
    let set = this.handlers.get(topic);
    const fresh = !set;
    if (!set) {
      set = new Set();
      this.handlers.set(topic, set);
    }
    set.add(handler);
    if (fresh) this.send({ op: "sub", topic });
    this.connect();
    const owned = set;
    return () => {
      owned.delete(handler);
      if (owned.size === 0) {
        this.handlers.delete(topic);
        this.send({ op: "unsub", topic });
      }
    };
  }

  connect(): void {
    if (this.ws || this.closed) return;
    this.setStatus("connecting");
    const ws = new this.Impl(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this.backoff = BACKOFF_MIN;
      this.setStatus("open");
      for (const topic of this.handlers.keys()) ws.send(JSON.stringify({ op: "sub", topic }));
    };
    ws.onmessage = (ev: MessageEvent) => {
      let msg: WsMessage;
      try {
        msg = JSON.parse(String(ev.data)) as WsMessage;
      } catch {
        return;
      }
      const set = this.handlers.get(msg.topic);
      if (set) for (const h of set) h(msg);
    };
    ws.onerror = () => ws.close();
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.setStatus("closed");
      if (this.closed || this.handlers.size === 0) return;
      const wait = this.backoff;
      this.backoff = Math.min(this.backoff * 2, BACKOFF_MAX);
      this.timer = setTimeout(() => {
        this.timer = null;
        this.connect();
      }, wait);
    };
  }

  close(): void {
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const ws = this.ws;
    this.ws = null;
    ws?.close();
    this.setStatus("closed");
  }

  private send(payload: { op: "sub" | "unsub"; topic: string }): void {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(payload));
  }

  private setStatus(s: SocketStatus): void {
    if (this.status === s) return;
    this.status = s;
    for (const cb of this.statusListeners) cb(s);
  }
}

function defaultUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}

export const socket = new SocketClient(defaultUrl());
