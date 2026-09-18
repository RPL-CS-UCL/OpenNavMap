import { useEffect, useRef, useSyncExternalStore } from "react";
import { type SocketStatus, type WsMessage, socket } from "./socket-client";

export function useSocketStatus(): SocketStatus {
  return useSyncExternalStore(
    (cb) => socket.onStatus(() => cb()),
    () => socket.getStatus(),
    () => "closed" as const,
  );
}

/** Subscribe to one topic for the component's lifetime; the handler ref is kept fresh without resubscribing. */
export function useTopic(topic: string | null, handler: (m: WsMessage) => void): void {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    if (!topic) return;
    return socket.subscribe(topic, (m) => ref.current(m));
  }, [topic]);
}
