"use client";

import { useMemo, useState } from "react";

import { useDashboard } from "@/lib/useDashboard";
import { InboxItem } from "@/components/InboxItem";
import { QueueCard } from "@/components/QueueCard";
import styles from "./page.module.css";

/**
 * Invoice operations.
 *
 * Two columns, and the split between them is the state of the work: what has not been
 * dealt with, and what has. A document leaves the inbox by acquiring a decision, so the
 * board survives a refresh without anything having to remember it.
 *
 * Three ways to make progress, none privileged over the others:
 *
 *   read one yourself,
 *   hand one to the agents,
 *   hand them all over.
 *
 * A tool that only offered the last would be a demo. A finance team will want the first
 * on the invoice from the vendor they have used for nine years.
 */

type Filter = "all" | "review" | "approved" | "denied";

export default function Dashboard() {
  const {
    inbox,
    failures,
    runs,
    processing,
    handoff,
    loading,
    error,
    review,
    busy,
    asOf,
    setAsOf,
  } = useDashboard();
  const [filter, setFilter] = useState<Filter>("all");

  const counts = useMemo(
    () => ({
      all: runs.length,
      review: runs.filter((r) => r.awaiting_review).length,
      approved: runs.filter((r) => r.outcome === "APPROVED").length,
      denied: runs.filter((r) => r.outcome === "REJECTED").length,
    }),
    [runs],
  );

  const visible = useMemo(() => {
    switch (filter) {
      case "review":
        return runs.filter((r) => r.awaiting_review);
      case "approved":
        return runs.filter((r) => r.outcome === "APPROVED");
      case "denied":
        return runs.filter((r) => r.outcome === "REJECTED");
      default:
        return runs;
    }
  }, [runs, filter]);

  // Anything needing a person comes first. A queue where the one thing being asked of
  // you is nineteenth is a queue you stop reading.
  const ordered = useMemo(
    () =>
      [...visible].sort(
        (a, b) => Number(b.awaiting_review) - Number(a.awaiting_review),
      ),
    [visible],
  );

  // There is one record per *document*, and a single invoice number can arrive on more
  // than one of them — the same bill as a PDF and a text file, or a revision of one
  // already received. The card says which copy it is looking at, so two entries with the
  // same number read as two documents rather than as a double-processing bug.
  const copies = useMemo(() => {
    const total = new Map<string, number>();
    const index = new Map<number, number>();

    for (const r of runs) {
      if (!r.invoice_number) continue;
      const n = (total.get(r.invoice_number) ?? 0) + 1;
      total.set(r.invoice_number, n);
      index.set(r.id, n);
    }

    return { total, index };
  }, [runs]);

  return (
    <main className="shell">
      <div className={styles.intro}>
        <div>
          <h1 className={styles.title}>Invoice processing</h1>
          <p className={styles.subtitle}>
            Triage incoming invoices yourself, or hand them to the review agents to
            approve, deny, or escalate back to you.
          </p>
        </div>

        {/* An invoice is only late relative to some particular day, and the day the
            reviewer means is not always today — a batch pulled from an archive is read
            against the date it was received. Same knob as the CLI's `--as-of`. */}
        <label className={styles.asOf}>
          <span className={styles.asOfLabel}>Reviewing as of</span>
          <input
            type="date"
            className={styles.asOfInput}
            value={asOf}
            onChange={(e) => setAsOf(e.target.value)}
            disabled={busy}
          />
        </label>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      {handoff && (
        <div className={styles.activity}>
          <span className={styles.activityRoute}>
            {handoff.from} → {handoff.to}
          </span>
          <span className={styles.activityReason}>{handoff.reason}</span>
        </div>
      )}

      <div className={styles.columns}>
        <section className={styles.panel}>
          <div className={styles.panelHead}>
            <span className={styles.panelTitle}>
              Inbox
              <span className={styles.count}>{inbox.length}</span>
            </span>
            <button
              className="btn btn-primary"
              onClick={() => review()}
              disabled={busy || inbox.length === 0}
            >
              ✦ Review all
            </button>
          </div>

          {loading ? (
            <p className={styles.empty}>Looking for invoices…</p>
          ) : inbox.length === 0 ? (
            <p className={styles.empty}>
              <span className={styles.emptyStrong}>Inbox clear</span>
              Every invoice has been dealt with.
            </p>
          ) : (
            <ul className={`${styles.inboxList} scroll-slim`}>
              {inbox.map((doc) => (
                <InboxItem
                  key={doc.path}
                  document={doc}
                  processing={processing.get(doc.path)}
                  failure={failures.get(doc.path)}
                  onReview={() => review(doc.path)}
                  disabled={busy}
                />
              ))}
            </ul>
          )}
        </section>

        <section>
          <div className={styles.queueHead}>
            <span className={styles.queueTitle}>Review queue</span>

            <div className={styles.filters}>
              {(
                [
                  ["all", "All"],
                  ["review", "Needs review"],
                  ["approved", "Approved"],
                  ["denied", "Denied"],
                ] as [Filter, string][]
              ).map(([key, label]) => (
                <button
                  key={key}
                  className={styles.filter}
                  data-active={filter === key}
                  onClick={() => setFilter(key)}
                >
                  {label}
                  <span className={styles.filterCount}>{counts[key]}</span>
                </button>
              ))}
            </div>
          </div>

          {ordered.length === 0 ? (
            <div className={`${styles.panel} ${styles.empty}`}>
              <span className={styles.emptyStrong}>Nothing here yet</span>
              {counts.all === 0
                ? "Review an invoice from the inbox and it will appear here."
                : "No invoices match this filter."}
            </div>
          ) : (
            <div className={styles.cards}>
              {ordered.map((record) => (
                <QueueCard
                  key={record.id}
                  record={record}
                  copies={
                    record.invoice_number
                      ? copies.total.get(record.invoice_number) ?? 1
                      : 1
                  }
                  copy={copies.index.get(record.id) ?? 1}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
