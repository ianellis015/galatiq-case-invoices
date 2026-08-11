/**
 * Mirrors the Python models.
 *
 * Hand-written rather than generated. The API's shapes are small and stable, and a
 * codegen step for this many endpoints would be more machinery than it saves.
 */

export type Outcome = "APPROVED" | "REJECTED" | "HELD_FOR_REVIEW";
export type Severity = "CRITICAL" | "WARN" | "INFO";
export type StepKind = "agent" | "deterministic";

export interface DocumentInfo {
  path: string;
  name: string;
  format: string;
  bytes: number;
}

export interface Finding {
  code: string;
  severity: Severity;
  message: string;
  evidence: string;
}

export interface LineItem {
  raw_name: string;
  item: string | null;
  quantity: number | null;
  quantity_raw: string | null;
  unit_price: string | null;
  unit_price_raw: string | null;
  stated_amount: string | null;
  note: string | null;
}

export interface Invoice {
  invoice_number: string | null;
  vendor: string;
  vendor_address: string | null;
  issue_date_raw: string | null;
  due_date_raw: string | null;
  line_items: LineItem[];
  subtotal_raw: string | null;
  tax_amount_raw: string | null;
  shipping_raw: string | null;
  total_raw: string | null;
  currency: string | null;
  payment_terms: string | null;
  notes: string | null;
  source_path: string | null;
  source_format: string | null;
}

export interface RunRecord {
  id: number;
  source_path: string;
  source_format: string | null;
  invoice_number: string | null;
  vendor: string | null;
  usd_total: string | null;
  outcome: Outcome | null;
  rationale: string;
  policy_refs: string[];
  concerns: string[];
  risk_score: number;
  findings: Finding[];
  payment_status: string | null;
  invoice: Invoice | null;
  awaiting_review: boolean;
  latency_ms: number;
  provider: string | null;
  model: string | null;
  error: string | null;
}

export interface Summary {
  documents: number;
  unique_invoices: number;
  approved: number;
  held: number;
  rejected: number;
  awaiting_review: number;
  prevented_usd: string;
  elapsed_ms: number;
}

export interface Health {
  ok: boolean;
  provider?: string;
  model?: string;
  detail?: string;
}

/**
 * What arrives on the stream.
 *
 * `kind` on a step event is the field that matters most: five nodes call a model and the
 * rest are arithmetic, and rendering them identically would misrepresent the system.
 */
export type StreamEvent =
  | { type: "batch.start"; batch: string; total: number; documents: string[] }
  | { type: "document.start"; doc: string }
  | {
      type: "step.start";
      doc: string;
      invoice: string | null;
      step: string;
      label: string;
      kind: StepKind;
      parallel: boolean;
    }
  | { type: "step.end"; doc: string; invoice: string | null; step: string; label: string }
  | {
      type: "handoff";
      doc: string;
      invoice: string | null;
      from: string;
      to: string;
      reason: string;
    }
  | {
      type: "document.end";
      doc: string;
      invoice: string | null;
      outcome: Outcome | null;
      risk: number;
      awaiting_review: boolean;
      vendor: string | null;
      usd_total: string | null;
      latency_ms: number;
    }
  | { type: "batch.end"; batch: string }
  | { type: "batch.error"; message: string };

export interface Preview {
  path: string;
  name: string;
  format: string | null;
  text: string;
  structured: Record<string, unknown> | null;
  findings: Finding[];
}
