import type { Finding } from "@/lib/types";
import styles from "./FindingsList.module.css";

/**
 * Everything that was noticed, worst first.
 *
 * Evidence is collapsed by default. The message answers "what is wrong"; the evidence
 * answers "prove it", and only some readers want the second. Expanding is one click and
 * costs nothing — showing everything at once turns twelve findings into a wall.
 *
 * `<details>` rather than state, so it works before hydration and keyboard navigation is
 * free.
 */

export function FindingsList({ findings }: { findings: Finding[] }) {
  if (findings.length === 0) {
    return <p className={styles.empty}>Nothing was found wrong with this invoice.</p>;
  }

  return (
    <div className={styles.list}>
      {findings.map((finding, i) => (
        <details key={`${finding.code}-${i}`} className={styles.item}>
          <summary className={styles.summary}>
            <span className={`${styles.severity} ${styles[finding.severity]}`}>
              {finding.severity}
            </span>

            <span className={styles.body}>
              <span className={styles.message}>{finding.message}</span>
              <span className={styles.code}>{finding.code}</span>
            </span>

            <span className={styles.chevron} aria-hidden>
              ›
            </span>
          </summary>

          <div className={styles.evidence}>
            {finding.evidence || "No evidence recorded."}
          </div>
        </details>
      ))}
    </div>
  );
}
