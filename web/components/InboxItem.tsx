"use client";

import { useRouter } from "next/navigation";

import { initials, kb } from "@/lib/api";
import type { DocumentInfo } from "@/lib/types";
import type { Processing } from "@/lib/useDashboard";
import styles from "./InboxItem.module.css";

/**
 * One document waiting to be dealt with.
 *
 * Two ways forward on every row. Handing everything to the agents is the fast path, and
 * reading it yourself is the one a finance team will actually want on the invoice from
 * the vendor they have used for nine years. Neither is the "correct" one, so neither is
 * hidden behind the other.
 *
 * Filenames are shown rather than invoice numbers, because until something has read the
 * document the filename is all anyone honestly knows about it.
 */

export function InboxItem({
  document,
  processing,
  failure,
  onReview,
  disabled,
}: {
  document: DocumentInfo;
  processing?: Processing;
  /** Why the last attempt on this document did not reach a decision, if it did not. */
  failure?: string;
  onReview: () => void;
  disabled?: boolean;
}) {
  const router = useRouter();

  return (
    <li
      className={styles.item}
      data-busy={Boolean(processing)}
      data-failed={Boolean(failure) && !processing}
    >
      <div className={styles.top}>
        <span className={styles.avatar}>{initials(document.name)}</span>

        <div className={styles.body}>
          <div className={styles.name}>{document.name}</div>
          <div className={styles.sub}>
            {failure && !processing ? "Last attempt failed" : "Not yet read"}
          </div>
        </div>

        <div className={styles.right}>
          <div className={styles.format}>{document.format}</div>
          <div className={styles.size}>{kb(document.bytes)}</div>
        </div>
      </div>

      {/* The reason, so a retry is an informed decision rather than a guess. Nothing was
          paid and nothing was decided, so trying again is always safe. */}
      {failure && !processing && <div className={styles.failure}>{failure}</div>}

      {processing ? (
        <Working processing={processing} />
      ) : (
        <div className={styles.actions}>
          <button className="btn btn-primary" onClick={onReview} disabled={disabled}>
            {failure ? "✦ Try again" : "✦ Review with AI"}
          </button>
          <button
            className="btn btn-secondary"
            onClick={() =>
              router.push(`/document?path=${encodeURIComponent(document.path)}`)
            }
          >
            Review manually
          </button>
        </div>
      )}
    </li>
  );
}

function Working({ processing }: { processing: Processing }) {
  if (processing.parallel) {
    return (
      <div className={`${styles.working} ${styles.machine}`}>
        <span className={styles.parallelDots} aria-hidden>
          {Array.from({ length: 8 }, (_, i) => (
            <span key={i} />
          ))}
        </span>
        Running eight checks at once
      </div>
    );
  }

  const agent = processing.kind === "agent";

  return (
    <div className={`${styles.working} ${agent ? styles.agent : styles.machine}`}>
      {agent ? (
        <span className={styles.dot} aria-hidden />
      ) : (
        <span className={styles.line} aria-hidden />
      )}
      {processing.step}
    </div>
  );
}
