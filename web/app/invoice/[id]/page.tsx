"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useEffect, useState } from "react";

import { api, basename, initials, money, seconds } from "@/lib/api";
import type { RunRecord } from "@/lib/types";
import { FindingsList } from "@/components/FindingsList";
import styles from "./page.module.css";

/**
 * One invoice, and why it went the way it did.
 *
 * The same page for a settled invoice and one still waiting. That is deliberate: the
 * information a person needs to *make* the decision is the same information they would
 * want to *understand* it later, and maintaining two versions of that would mean one of
 * them drifting.
 *
 * What differs is the right-hand column. A held invoice puts two buttons there; a decided
 * one reports what happened.
 */

const BADGE: Record<string, { className: string; label: string }> = {
  APPROVED: { className: "badge badge-approved", label: "Approved" },
  REJECTED: { className: "badge badge-denied", label: "Denied" },
  HELD_FOR_REVIEW: { className: "badge badge-review", label: "Needs your review" },
  FAILED: { className: "badge badge-failed", label: "Could not process" },
};

export default function InvoiceDetail({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();

  const [record, setRecord] = useState<RunRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .run(Number(id))
      .then(setRecord)
      .catch((e: Error) => setError(e.message));
  }, [id]);

  async function decide(verdict: "approve" | "deny") {
    if (!record) return;

    setBusy(true);
    setError(null);

    try {
      setRecord(await api.review(record.id, verdict));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (error && !record) {
    return (
      <main className="shell">
        <Back />
        <p className={styles.state}>{error}</p>
      </main>
    );
  }

  if (!record) {
    return (
      <main className="shell">
        <Back />
        <p className={styles.state}>Loading…</p>
      </main>
    );
  }

  const failed = !record.outcome && Boolean(record.error);
  const badge = BADGE[failed ? "FAILED" : record.outcome ?? "HELD_FOR_REVIEW"];

  return (
    <main className="shell">
      <Back />

      <div className={styles.head}>
        <div className={styles.identity}>
          <span className={styles.avatar}>
            {initials(record.vendor ?? record.invoice_number ?? "??")}
          </span>
          <div>
            <h1 className={styles.vendor}>
              {record.vendor ?? "Vendor not named"}
            </h1>
            <p className={styles.number}>
              {record.invoice_number ?? "No invoice number"}
            </p>
          </div>
        </div>

        <div className={styles.headRight}>
          <span className={badge.className}>{badge.label}</span>
          <div className={`${styles.amount} tabular`}>
            {money(record.usd_total)}
          </div>
        </div>
      </div>

      <div className={styles.columns}>
        <div>
          {failed && (
            <section className={styles.card}>
              <h2 className={styles.cardHead}>This document was not processed</h2>
              <p className={styles.rationale}>
                <span>
                  Nothing was extracted and no decision was reached, so there is no
                  verdict on this invoice — it is still waiting to be dealt with.
                  Sending it back through the agents is safe: no payment was made.
                </span>
              </p>
              <pre className={styles.failure}>{record.error}</pre>
            </section>
          )}

          <section className={styles.card}>
            <h2 className={styles.cardHead}>What the agents concluded</h2>

            {/* The short reasons, listed. They used to be square-bracketed prefixes on
                the front of the narrative, which meant the first line of every
                explanation was a run of brackets nobody could parse at a glance. */}
            {record.concerns.length > 0 && (
              <ul className={styles.concerns}>
                {record.concerns.map((concern, i) => (
                  <li key={i} className={styles.concern}>
                    {concern}
                  </li>
                ))}
              </ul>
            )}

            <div className={styles.rationale}>
              {paragraphs(record.rationale).map((para, i) => (
                <p key={i}>{para}</p>
              ))}
            </div>

            {record.policy_refs.length > 0 && (
              <div className={styles.rules}>
                {record.policy_refs.map((rule) => (
                  <span key={rule} className={styles.rule}>
                    Rule {rule}
                  </span>
                ))}
              </div>
            )}
          </section>

          <section className={styles.card}>
            <h2 className={styles.cardHead}>
              Findings ({record.findings.length})
            </h2>
            <FindingsList findings={record.findings} />
          </section>

          {record.invoice && record.invoice.line_items.length > 0 && (
            <section className={styles.card}>
              <h2 className={styles.cardHead}>What was on the invoice</h2>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Item</th>
                    <th>Qty</th>
                    <th>Unit price</th>
                  </tr>
                </thead>
                <tbody>
                  {record.invoice.line_items.map((line, i) => (
                    <tr key={i}>
                      <td>
                        <span className={line.item ? "" : styles.unknown}>
                          {line.item ?? line.raw_name}
                          {!line.item && " · not in catalog"}
                        </span>
                        {/* Shown whenever the catalog name differs from what the
                            document said, so a reviewer can check against the paper. */}
                        {line.item && line.item !== line.raw_name && (
                          <span className={styles.rawName}>
                            document says “{line.raw_name}”
                          </span>
                        )}
                      </td>
                      <td>{line.quantity ?? line.quantity_raw ?? "—"}</td>
                      <td className="tabular">
                        {line.unit_price_raw ?? line.unit_price ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </div>

        <aside>
          {record.awaiting_review ? (
            <div className={styles.decide}>
              <h2 className={styles.decideTitle}>Your decision</h2>
              <p className={styles.decideText}>
                The agents would not settle this one alone. Approving pays{" "}
                {money(record.usd_total)} to {record.vendor ?? "the vendor"}.
              </p>

              {error && (
                <p className={styles.decideText} style={{ color: "var(--denied)" }}>
                  {error}
                </p>
              )}

              <div className={styles.decideActions}>
                <button
                  className="btn btn-approve"
                  onClick={() => decide("approve")}
                  disabled={busy}
                >
                  {busy ? "Working…" : "Approve payment"}
                </button>
                <button
                  className="btn btn-deny"
                  onClick={() => decide("deny")}
                  disabled={busy}
                >
                  Deny
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => router.push("/")}
                  disabled={busy}
                >
                  Decide later
                </button>
              </div>
            </div>
          ) : (
            <div className={styles.settled}>
              <h2 className={styles.cardHead}>Record</h2>
              <div className={styles.facts}>
                <Fact label="Risk" value={`${record.risk_score} / 100`} />
                <Fact label="Payment" value={record.payment_status ?? "—"} />
                <Fact label="Source" value={basename(record.source_path)} />
                <Fact label="Format" value={record.source_format ?? "—"} />
                <Fact
                  label="Decided by"
                  value={record.provider ? `${record.model}` : "—"}
                />
                <Fact label="Took" value={seconds(record.latency_ms)} />
              </div>
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}

/**
 * Break the model's reasoning into paragraphs.
 *
 * It answers in one block: what the invoice claims, then what it checked, then what it
 * recommends. Three or four sentences of that is a wall, and a reviewer skims a wall
 * instead of reading it. If the model gave us its own breaks I keep them; otherwise I
 * group sentences in twos, which is close enough to where the thought turns.
 */
function paragraphs(text: string): string[] {
  const trimmed = text.trim();
  if (!trimmed) return ["No rationale recorded."];

  const explicit = trimmed.split(/\n{2,}/).filter(Boolean);
  if (explicit.length > 1) return explicit;

  const sentences = trimmed.match(/[^.!?]+[.!?]+(\s|$)|[^.!?]+$/g);
  if (!sentences || sentences.length < 4) return [trimmed];

  const grouped: string[] = [];
  for (let i = 0; i < sentences.length; i += 2) {
    grouped.push(sentences.slice(i, i + 2).join("").trim());
  }
  return grouped;
}

function Back() {
  return (
    <Link href="/" className={styles.back}>
      ← Invoice processing
    </Link>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.fact}>
      <span className={styles.factLabel}>{label}</span>
      <span className={styles.factValue}>{value}</span>
    </div>
  );
}
