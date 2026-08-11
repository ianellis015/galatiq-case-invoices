import Link from "next/link";

import { basename, initials, money, reasonFrom } from "@/lib/api";
import type { RunRecord } from "@/lib/types";
import styles from "./QueueCard.module.css";

/**
 * One invoice the system has finished with.
 *
 * Everything here is clickable through to the full decision. What differs is what the
 * card is asking: one that needs a person says so prominently, and one already decided
 * simply reports what happened and why.
 */

const BADGE: Record<string, { className: string; label: string }> = {
  APPROVED: { className: "badge badge-approved", label: "Approved" },
  REJECTED: { className: "badge badge-denied", label: "Denied" },
  HELD_FOR_REVIEW: { className: "badge badge-review", label: "Needs your review" },
  // A document that never reached a decision. Distinct from the three real outcomes,
  // because "we could not process this" is not a verdict on the invoice, and showing it
  // as one would put a failure in the review queue disguised as a judgement.
  FAILED: { className: "badge badge-failed", label: "Could not process" },
};

export function QueueCard({
  record,
  copies,
  copy,
}: {
  record: RunRecord;
  /** How many documents in this batch carry the same invoice number. */
  copies: number;
  /** Which of them this one is, in the order they were processed. */
  copy: number;
}) {
  const failed = !record.outcome && Boolean(record.error);
  const badge = BADGE[failed ? "FAILED" : record.outcome ?? "HELD_FOR_REVIEW"];
  const flagged = record.awaiting_review;

  return (
    <Link
      href={`/invoice/${record.id}`}
      className={styles.card}
      data-flagged={flagged}
    >
      <div className={styles.head}>
        <span className={styles.avatar}>
          {initials(record.vendor ?? record.invoice_number ?? "??")}
        </span>

        <div className={styles.who}>
          {/* Nothing was extracted from a failed document, so the filename is the only
              thing that identifies it. */}
          <div className={styles.vendor}>
            {failed
              ? basename(record.source_path)
              : record.vendor ?? "Vendor not named"}
          </div>
          <div className={styles.number}>
            {failed
              ? "Not read"
              : record.invoice_number ?? "No invoice number"}
            {/* One record per *document*, so the same invoice number can appear more
                than once — a PDF and a text copy of the same bill, or a revision of one
                already sent. Without saying so, that reads as the system having
                processed something twice by mistake. */}
            {copies > 1 && (
              <span className={styles.copy}>
                · document {copy} of {copies} with this number
              </span>
            )}
          </div>
        </div>

        <span className={badge.className}>{badge.label}</span>
      </div>

      <div className={styles.amountRow}>
        <div>
          <div className={styles.amountLabel}>Invoice total</div>
          <div className={`${styles.amount} tabular`}>
            {money(record.usd_total)}
          </div>
        </div>
        <div className={styles.source}>{record.source_format?.toUpperCase()}</div>
      </div>

      {failed ? (
        // The error itself, not a risk score. There is no risk score for a document
        // nothing managed to read, and showing "risk 0" would read as "this is fine".
        <div className={styles.foot}>
          <span className={styles.why}>{record.error}</span>
        </div>
      ) : flagged ? (
        <div className={styles.flag}>⚑ Flagged for your review</div>
      ) : (
        <div className={styles.foot}>
          {/* The leading reason, not the whole rationale. A queue is for scanning; the
              detail page is for reading. */}
          <span className={styles.why}>{reasonFrom(record)}</span>
          <Risk score={record.risk_score} />
        </div>
      )}
    </Link>
  );
}

function Risk({ score }: { score: number }) {
  const colour =
    score >= 60 ? "var(--denied)" : score >= 30 ? "var(--review)" : "var(--approved)";

  return (
    <span className={styles.risk}>
      <span className={styles.riskBar}>
        <span
          className={styles.riskFill}
          style={{ width: `${score}%`, background: colour }}
        />
      </span>
      risk {score}
    </span>
  );
}
