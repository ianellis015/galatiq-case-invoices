"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { api } from "@/lib/api";
import type { Health } from "@/lib/types";
import styles from "./Header.module.css";

/**
 * ACME's tool, not ours.
 *
 * The model name sits on the right because it is operational information — somebody
 * looking at a decision months later needs to know what made it, and that starts with it
 * being visible while it works. It is deliberately small: which model is running matters
 * less than whether one is.
 */

export function Header() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ ok: false }));
  }, []);

  return (
    <header className={styles.bar}>
      <div className={styles.inner}>
        <Link href="/" className={styles.brand}>
          <span className={styles.mark}>A</span>
          <span className={styles.company}>ACME Corp</span>
          <span className={styles.divider}>|</span>
          <span className={styles.section}>Invoice Operations</span>
        </Link>

        <div className={styles.right}>
          <span
            className={`${styles.status} ${health?.ok ? "" : styles.statusOffline}`}
          >
            {health?.ok ? health.model : health?.detail ?? "Connecting…"}
          </span>
        </div>
      </div>
    </header>
  );
}
