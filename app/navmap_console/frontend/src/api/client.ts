import type { ZodType } from "zod";

// Explicit fields instead of parameter properties: tsconfig has erasableSyntaxOnly.
export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
    this.name = "ApiError";
  }
}

// Node's fetch (used by vitest/jsdom) rejects relative URLs; browsers accept both.
function absolute(path: string): string {
  return /^https?:\/\//.test(path) ? path : new URL(path, window.location.origin).toString();
}

async function parse<T>(res: Response, schema?: ZodType<T>): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  const data = await res.json();
  return schema && import.meta.env.DEV ? schema.parse(data) : (data as T);
}

export function apiGet<T>(path: string, schema?: ZodType<T>): Promise<T> {
  return fetch(absolute(path)).then((r) => parse(r, schema));
}

export function apiSend<T>(
  method: "POST" | "PATCH" | "DELETE",
  path: string,
  body?: unknown,
  schema?: ZodType<T>,
): Promise<T> {
  return fetch(absolute(path), {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((r) => parse(r, schema));
}

export function apiUpload<T>(path: string, form: FormData, schema?: ZodType<T>): Promise<T> {
  return fetch(absolute(path), { method: "POST", body: form }).then((r) => parse(r, schema));
}

export async function apiGetBinary(path: string): Promise<ArrayBuffer> {
  const res = await fetch(absolute(path));
  if (!res.ok) throw new ApiError(res.status, res.statusText || `HTTP ${res.status}`);
  return res.arrayBuffer();
}

export function errorDetail(err: unknown): string {
  if (err instanceof ApiError) return err.detail;
  if (err instanceof Error) return err.message;
  return String(err);
}
