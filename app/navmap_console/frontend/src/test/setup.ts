import "@testing-library/jest-dom/vitest";
import { afterAll, afterEach, beforeAll } from "vitest";
import { resetState } from "./handlers";
import { server } from "./server";

// A WebSocket that never opens: the singleton socket client must not reach out from jsdom.
class IdleWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  readyState = 0;
  url: string;
  onopen: null | (() => void) = null;
  onclose: null | (() => void) = null;
  onmessage: null | ((ev: { data: string }) => void) = null;
  onerror: null | (() => void) = null;
  constructor(url: string) {
    this.url = url;
  }
  send(): void {}
  close(): void {
    this.readyState = 3;
  }
}
Object.defineProperty(globalThis, "WebSocket", { value: IdleWebSocket, writable: true });
if (!("ResizeObserver" in globalThis)) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}
// Radix (pointer capture) and Virtuoso (scrollIntoView) call these; jsdom lacks them.
Element.prototype.hasPointerCapture ??= () => false;
Element.prototype.setPointerCapture ??= () => {};
Element.prototype.releasePointerCapture ??= () => {};
Element.prototype.scrollIntoView ??= () => {};

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => {
  server.resetHandlers();
  resetState();
});
afterAll(() => server.close());
