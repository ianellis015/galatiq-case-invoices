"use client";

/**
 * The dashboard's state.
 *
 * Two sources, and the split between them is the whole model:
 *
 *   **The inbox** is what is on disk with no decision against it. Derived, not stored —
 *   a document leaves the inbox by acquiring a run record, which means the state survives
 *   a refresh, a restart, and closing the laptop, without anything having to remember it.
 *
 *   **The queue** is what has been decided, read back from the database.
 *
 * While a batch runs, live events overlay the inbox so a document shows which agent is
 * holding it. That layer is transient; the two above are not.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { API, api } from "./api";
import type { DocumentInfo, RunRecord, StepKind, StreamEvent } from "./types";

export interface Processing {
  step: string;
  kind: StepKind;
  parallel: boolean;
}

export interface Handoff {
  id: number;
  from: string;
  to: string;
  reason: string;
  invoice: string | null;
}

/** Today, in the browser's own timezone. `toISOString` would use UTC and be a day out. */
function today(): string {
  const now = new Date();
  return new Date(now.getTime() - now.getTimezoneOffset() * 60_000)
    .toISOString()
    .slice(0, 10);
}

export function useDashboard() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  // The date the due-date checks are measured against — the CLI's `--as-of`. Today by
  // default, because that is what a person running this on a Tuesday means.
  const [asOf, setAsOf] = useState(today);
  const [processing, setProcessing] = useState<Map<string, Processing>>(new Map());
  const [handoff, setHandoff] = useState<Handoff | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const source = useRef<EventSource | null>(null);
  const handoffId = useRef(0);

  // Every finished document triggers a refresh, and eight run at once, so several are
  // always in flight together. They do not come back in the order they were sent: a
  // request issued at document eighteen can resolve after the one issued at twenty and
  // overwrite twenty rows with eighteen. That is how the last two invoices of a batch
  // went missing until the page was reloaded by hand.
  //
  // So each request takes a number, and a response is only allowed to land if nothing
  // newer has already landed.
  const issued = useRef(0);
  const applied = useRef(0);

  const refresh = useCallback(async () => {
    const ticket = ++issued.current;

    try {
      const [{ documents: docs }, records] = await Promise.all([
        api.documents(),
        api.runs(),
      ]);

      if (ticket < applied.current) return;
      applied.current = ticket;

      setDocuments(docs);
      setRuns(records);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Strict mode mounts effects twice in development, which would open two EventSource
  // connections and double every event. Closing on unmount is what prevents it.
  useEffect(() => {
    return () => {
      source.current?.close();
      source.current = null;
    };
  }, []);

  const watch = useCallback(
    (batchId: string) => {
      source.current?.close();

      const stream = new EventSource(`${API}/api/stream/${batchId}`);
      source.current = stream;

      stream.onmessage = (message) => {
        const event: StreamEvent = JSON.parse(message.data);

        switch (event.type) {
          case "step.start":
            setProcessing((prev) =>
              new Map(prev).set(event.doc, {
                step: event.label,
                kind: event.kind,
                parallel: event.parallel,
              }),
            );
            break;

          case "handoff":
            setHandoff({
              id: handoffId.current++,
              from: event.from,
              to: event.to,
              reason: event.reason,
              invoice: event.invoice,
            });
            break;

          case "document.end":
            setProcessing((prev) => {
              const next = new Map(prev);
              next.delete(event.doc);
              return next;
            });
            // Read the finished record back rather than assembling it from the event.
            // The database is what actually happened, and the detail page reads the
            // same place — two sources would eventually disagree.
            refresh();
            break;

          case "batch.end":
            setProcessing(new Map());
            setHandoff(null);
            stream.close();
            source.current = null;
            refresh();
            break;

          case "batch.error":
            setError(event.message);
            stream.close();
            source.current = null;
            break;
        }
      };

      stream.onerror = () => {
        stream.close();
        source.current = null;
        setProcessing(new Map());
        refresh();
      };
    },
    [refresh],
  );

  /** Run the agents over one document, or over everything still in the inbox. */
  const review = useCallback(
    async (path?: string) => {
      setError(null);

      // Mark it immediately. The first real event is a second or two away, and a button
      // that appears to do nothing gets pressed again.
      if (path) {
        setProcessing((prev) =>
          new Map(prev).set(path, {
            step: "Starting",
            kind: "deterministic",
            parallel: false,
          }),
        );
      }

      try {
        const { batch_id } = await api.start(path, asOf);
        watch(batch_id);
      } catch (e) {
        setError((e as Error).message);
        setProcessing(new Map());
      }
    },
    [watch, asOf],
  );

  // A document leaves the inbox by acquiring a *decision*, not merely a record. One that
  // failed on the way — an unreadable file, a model that could not be reached — has not
  // been dealt with, and quietly removing it from the inbox would lose it: no verdict, no
  // payment, and nowhere left to try again from.
  const decided = new Set(
    runs.filter((r) => r.outcome).map((r) => r.source_path),
  );
  const inbox = documents.filter((d) => !decided.has(d.path));

  // Failures still surface, as their own row in the inbox rather than as a card in a
  // queue of decisions. "Could not read this" is not a verdict on an invoice.
  const failures = new Map(
    runs.filter((r) => !r.outcome && r.error).map((r) => [r.source_path, r.error!]),
  );

  return {
    inbox,
    failures,
    runs: runs.filter((r) => r.outcome),
    processing,
    handoff,
    loading,
    error,
    review,
    refresh,
    asOf,
    setAsOf,
    busy: processing.size > 0,
  };
}
