/**
 * Talking to the pipeline.
 *
 * Thin wrappers, no client library. A handful of endpoints do not need one, and a fetch
 * that throws a readable error is more useful here than an abstraction.
 */

import type { DocumentInfo, Health, Preview, RunRecord, Summary } from "./types";

export const API =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`, { cache: "no-store" });

  if (!response.ok) {
    // The API's error detail is written for a person. Passing it through means the
    // interface can say "no such folder: /nowhere" instead of "request failed".
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }

  return response.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `${response.status} ${response.statusText}`);
  }

  return response.json();
}

export const api = {
  health: () => get<Health>("/api/health"),

  documents: () =>
    get<{ path: string; documents: DocumentInfo[] }>("/api/documents"),

  /**
   * Run the agents. One document if given a path, everything otherwise.
   *
   * `reset` is off for a single document — clearing the ledger to process one invoice
   * would throw away every decision already made. A full run resets, because that is
   * a fresh start; a single review is an addition to what is there.
   *
   * `asOf` is the date the due-date checks are measured against. It is the CLI's
   * `--as-of`, and it is here for the same reason: an invoice is only late relative to
   * some particular day, and the day the reviewer means is not always today.
   */
  start: (path?: string, asOf?: string) =>
    post<{ batch_id: string; total: number }>("/api/runs", {
      ...(path ? { invoice_path: path, reset: false } : { reset: true }),
      ...(asOf ? { as_of: asOf } : {}),
    }),

  preview: (path: string) =>
    get<Preview>(`/api/documents/preview?path=${encodeURIComponent(path)}`),

  /** Record a decision somebody made themselves, without running the agents. */
  decideManually: (path: string, verdict: "approve" | "deny", note = "") =>
    post<RunRecord>("/api/documents/decide", { path, verdict, note }),

  runs: () => get<RunRecord[]>("/api/runs"),
  run: (id: number) => get<RunRecord>(`/api/runs/${id}`),
  summary: () => get<Summary>("/api/summary"),

  review: (id: number, verdict: "approve" | "deny") =>
    post<RunRecord>(`/api/runs/${id}/review`, { verdict }),
};

/** "12500.0000" → "$12,500.00". Nothing in a finance tool should render raw. */
export function money(value: string | null | undefined): string {
  if (!value) return "—";
  const n = Number(value);
  if (Number.isNaN(n)) return value;
  return n.toLocaleString("en-US", { style: "currency", currency: "USD" });
}

export function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

/** The filename, which is what a person recognises. */
export function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

/** "Atlas Industrial Supply" -> "AI". A vendor is recognised by its name, not its id. */
export function initials(name: string): string {
  const words = name
    .replace(/[^a-zA-Z0-9 ]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  if (words.length === 0) return "??";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

export function kb(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
}

/**
 * The one-line reason, for a card.
 *
 * The approver records its concerns as a list, so the first one is the headline.
 * Failing that, the first finding. Failing that, the rationale's opening sentence — a
 * card should never be blank where a reason belongs.
 */
export function reasonFrom(record: {
  rationale: string;
  concerns?: string[];
  findings: { message: string }[];
}): string {
  if (record.concerns?.length) return record.concerns[0];
  if (record.findings.length > 0) return record.findings[0].message;

  const sentence = record.rationale.split(/(?<=\.)\s/)[0];
  return sentence || "No findings";
}
