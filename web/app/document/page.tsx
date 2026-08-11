"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { Preview } from "@/lib/types";
import styles from "./page.module.css";

/**
 * Reading an invoice yourself.
 *
 * The third path, and the one a finance team will use most on invoices they already know.
 * Somebody who has paid this vendor monthly for nine years should not have to wait two
 * minutes for an agent to tell them it looks fine.
 *
 * Approving from here skips every check. That is a legitimate thing to want and a
 * dangerous thing to do unknowingly, so the panel says so once, plainly, and offers the
 * agents as the alternative right beside it.
 */

export default function DocumentPage() {
  return (
    <Suspense fallback={<main className="shell" />}>
      <DocumentReview />
    </Suspense>
  );
}

function DocumentReview() {
  const params = useSearchParams();
  const router = useRouter();
  const path = params.get("path") ?? "";

  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!path) return;

    api
      .preview(path)
      .then(setPreview)
      .catch((e: Error) => setError(e.message));
  }, [path]);

  async function decide(verdict: "approve" | "deny") {
    setBusy(true);
    setError(null);

    try {
      await api.decideManually(path, verdict);
      router.push("/");
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  async function handOver() {
    setBusy(true);
    try {
      await api.start(path);
      router.push("/");
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  if (!preview) {
    return (
      <main className="shell">
        <Back />
        <p className={styles.state}>{error ?? "Loading…"}</p>
      </main>
    );
  }

  return (
    <main className="shell">
      <Back />

      <div className={styles.head}>
        <h1 className={styles.name}>{preview.name}</h1>
        <p className={styles.sub}>
          {preview.format?.toUpperCase()} · nothing has read this yet
        </p>
      </div>

      <div className={styles.columns}>
        <section className={styles.document}>
          <div className={styles.documentHead}>The document, as it arrived</div>

          {preview.text.trim() ? (
            <pre className={styles.text}>{preview.text}</pre>
          ) : (
            <p className={styles.unreadable}>
              This file has no readable text. It cannot be reviewed by eye.
            </p>
          )}
        </section>

        <aside className={styles.panel}>
          <h2 className={styles.panelTitle}>Decide yourself</h2>
          <p className={styles.panelText}>
            You are looking at the raw document. Nothing has been extracted, checked
            against stock, or reconciled.
          </p>

          <div className={styles.caution}>
            No checks have run. Approving here pays without verifying stock, arithmetic,
            duplicates or fraud signals.
          </div>

          {error && <p className={styles.error}>{error}</p>}

          <div className={styles.actions}>
            <button
              className="btn btn-approve"
              onClick={() => decide("approve")}
              disabled={busy || !preview.text.trim()}
            >
              Approve anyway
            </button>
            <button
              className="btn btn-deny"
              onClick={() => decide("deny")}
              disabled={busy}
            >
              Deny
            </button>
          </div>

          <hr className={styles.divider} />

          <p className={styles.panelText}>
            Or let the agents read it first — about a minute, and you decide afterwards
            with everything they found in front of you.
          </p>

          <button
            className="btn btn-primary"
            style={{ width: "100%", padding: "11px" }}
            onClick={handOver}
            disabled={busy}
          >
            ✦ Review with AI instead
          </button>
        </aside>
      </div>
    </main>
  );
}

function Back() {
  return (
    <Link href="/" className={styles.back}>
      ← Invoice processing
    </Link>
  );
}
