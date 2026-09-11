const view = document.getElementById("view");
const title = document.getElementById("page-title");
const lede = document.getElementById("page-lede");
const meta = document.getElementById("model-meta");
const drawer = document.getElementById("drawer");

const titles = {
  dashboard: ["Overview", "Merchant loss desk for costly returns. Scores are advisory."],
  queue: ["Review queue", "Flagged orders wait for a human. The model cannot punish a customer."],
  score: ["Score an order", "Decision-time features only. No post-delivery or payment-credential fields."],
  lab: ["Threshold lab", "Move one cutoff and watch precision, recall, workload, and rupee cost move together."],
  health: ["Model health", "Held-out temporal test versus validation. Champion selected without peeking at test."],
  quality: ["Data quality", "Synthetic demo data with leakage checks and a locked chronological split."],
  audit: ["Audit trail", "Every score and operator action is append-only and reconstructable."],
  safety: ["Safety & model card", "Intended use, prohibited use, and known failure modes."],
  order: ["Order detail", "Why the score fired, which actions are allowed, and that the score is not proof."],
};

const inr = (n) =>
  "₹" + Math.round(Number(n) || 0).toLocaleString("en-IN");
const pct = (n) => (Number(n) * 100).toFixed(1) + "%";
const bandHtml = (b) =>
  `<span class="band ${b}"><span class="dot"></span>${b}</span>`;

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.json();
}

function nav() {
  const route = location.hash.replace("#/", "") || "dashboard";
  document.querySelectorAll(".rail a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === route.split("/")[0]);
  });
  return route;
}

function kpi(label, value, hint) {
  return `<article class="card kpi"><div class="label">${label}</div><div class="value">${value}</div><div class="hint">${hint || ""}</div></article>`;
}

async function renderDashboard() {
  const d = await api("/api/v1/dashboard");
  const k = d.kpis;
  meta.textContent = `${k.model_version} · test set locked`;
  const maxCat = Math.max(...d.category_risk.map((x) => x.mean_risk), 1);
  view.innerHTML = `
    <div class="grid kpis">
      ${kpi("Orders scored", k.orders_scored.toLocaleString("en-IN"), "Demo scored slice")}
      ${kpi("High risk", k.high_risk.toLocaleString("en-IN"), "At cost-optimal threshold")}
      ${kpi("Precision / recall", `${pct(k.precision)} · ${pct(k.recall)}`, "Held-out test")}
      ${kpi("FP cost (test)", inr(k.fp_cost_total), `${k.fp} false positives · ${inr(k.cost_per_1000_orders)} / 1k orders`)}
    </div>
    <div class="grid cards-2" style="margin-top:16px">
      <article class="card">
        <h2>Risk by category</h2>
        <div class="bars">
          ${d.category_risk
            .map(
              (c) => `<div class="bar-row"><span>${c.category}</span><div class="track"><div class="fill" style="width:${(c.mean_risk / maxCat) * 100}%"></div></div><span>${c.mean_risk}%</span></div>`
            )
            .join("")}
        </div>
      </article>
      <article class="card">
        <h2>Drift sentinel</h2>
        <p class="muted">PSI train→test. Alert if a feature moves materially; do not silently retrain.</p>
        <p>Return-rate PSI: <strong>${d.drift.historical_return_rate_psi?.toFixed(3)}</strong></p>
        <p>Order-value PSI: <strong>${d.drift.order_value_psi?.toFixed(3)}</strong></p>
        <p>Train vs test positive rate: ${(d.drift.train_positive_rate * 100).toFixed(1)}% → ${(d.drift.test_positive_rate * 100).toFixed(1)}%</p>
        <p class="notice">PR-AUC ${Number(k.pr_auc).toFixed(3)} on locked test. Expected decision cost ${inr(k.expected_decision_cost)}.</p>
      </article>
    </div>
    <article class="card" style="margin-top:16px">
      <h2>Recent scored orders</h2>
      ${orderTable(d.recent)}
    </article>
  `;
}

function orderTable(items, extra = "") {
  return `<table><thead><tr><th>Order</th><th>Risk</th><th>Value</th><th>Category</th><th>Top reasons</th><th></th></tr></thead><tbody>
    ${items
      .map(
        (o) => `<tr>
          <td><a href="#/order/${o.order_id}">${o.order_id}</a></td>
          <td>${bandHtml(o.risk_band)} ${o.risk_score}</td>
          <td>${inr(o.order_value)}</td>
          <td>${o.product_category}</td>
          <td>${(o.reason_codes || []).slice(0, 2).join(", ")}</td>
          <td><button class="secondary" data-open="${o.order_id}">Open</button></td>
        </tr>`
      )
      .join("")}
  </tbody></table>${extra}`;
}

async function renderQueue() {
  const q = await api("/api/v1/queue");
  view.innerHTML = `
    <article class="card">
      <h2>Human-in-the-loop queue</h2>
      <p class="muted">Threshold ${q.threshold.toFixed(2)}. Allowed responses: confirm delivery, fit reminder, address check, or manual review. No auto-cancel.</p>
      ${orderTable(q.items)}
    </article>`;
}

async function renderScore() {
  view.innerHTML = `
    <article class="card">
      <h2>New order intake</h2>
      <form id="score-form" class="grid-form">
        ${field("order_id", "text", "ORD-DEMO-01")}
        ${field("product_category", "select", "electronics", ["apparel", "electronics", "home", "beauty", "sports", "accessories"])}
        ${field("product_price", "number", "18999")}
        ${field("quantity", "number", "1")}
        ${field("discount_pct", "number", "18")}
        ${field("shipping_zone", "select", "Z3", ["Z1", "Z2", "Z3", "Z4"])}
        ${field("promised_delivery_days", "number", "5")}
        ${field("payment_method", "select", "upi", ["upi", "card", "cod", "wallet", "netbanking"])}
        ${field("account_age_days", "number", "12")}
        ${field("historical_orders", "number", "2")}
        ${field("historical_returns", "number", "2")}
        ${field("prior_delivery_delay_rate", "number", "0.12")}
        ${field("address_changes_90d", "number", "2")}
        ${field("orders_30d", "number", "3")}
        ${field("refunds_30d", "number", "2")}
        <div style="grid-column:1/-1"><button type="submit">Analyze risk</button></div>
      </form>
      <div id="score-out"></div>
    </article>`;
  document.getElementById("score-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const body = Object.fromEntries(fd.entries());
    ["product_price", "quantity", "discount_pct", "promised_delivery_days", "account_age_days", "historical_orders", "historical_returns", "prior_delivery_delay_rate", "address_changes_90d", "orders_30d", "refunds_30d"].forEach(
      (k) => (body[k] = Number(body[k]))
    );
    const out = document.getElementById("score-out");
    try {
      const r = await api("/api/v1/score", { method: "POST", body: JSON.stringify(body) });
      out.innerHTML = renderScoreCard(r);
    } catch (err) {
      out.innerHTML = `<p class="notice">${err.message}</p>`;
    }
  });
}

function field(name, type, value, opts) {
  if (type === "select") {
    return `<label>${name}<select name="${name}">${opts
      .map((o) => `<option ${o === value ? "selected" : ""}>${o}</option>`)
      .join("")}</select></label>`;
  }
  return `<label>${name}<input name="${name}" type="${type}" value="${value}" step="any" required /></label>`;
}

function renderScoreCard(r) {
  return `<div class="score-hero" style="margin-top:18px">
      <div>
        <div class="muted">Risk score</div>
        <div class="score-num">${r.risk_score}</div>
        <div>${bandHtml(r.risk_band)} · p=${r.risk_probability}</div>
      </div>
      <div>
        <p><strong>Recommended action:</strong> ${r.recommended_action.replaceAll("_", " ")}</p>
        <p class="muted">${r.disclaimer}</p>
        <p>Window: return within ${r.prediction_window_days} days · ${r.model_version} / ${r.policy_version}</p>
        <h3>Why flagged?</h3>
        <ul class="reasons">${r.reason_codes.map((x) => `<li><strong>${x.code}</strong> — ${x.detail}</li>`).join("")}</ul>
        <p class="notice">${r.model_note}</p>
      </div>
    </div>`;
}

async function renderLab() {
  const lab = await api("/api/v1/threshold-lab");
  const grid = lab.test_grid;
  view.innerHTML = `
    <article class="card">
      <h2>Cost-sensitive cutoff</h2>
      <p class="muted">Threshold was chosen on validation expected cost, then reported once on locked test. Slider below inspects the test grid for the demo.</p>
      <div class="slider-wrap">
        <input id="thr" type="range" min="0" max="${grid.length - 1}" value="0" />
        <strong id="thr-label"></strong>
      </div>
      <div class="grid lab-kpis" id="lab-kpis" style="margin-top:16px"></div>
      <p class="notice" style="margin-top:16px">FP cost ${lab.cost_assumptions.fp_cost_inr} INR · FN = ${lab.cost_assumptions.fn_fixed_inr} + ${lab.cost_assumptions.fn_margin_share * 100}% of order value · TP saves ${lab.cost_assumptions.tp_save_rate * 100}% minus intervention ${lab.cost_assumptions.intervention_cost_inr}.</p>
      <table style="margin-top:16px"><thead><tr><th>Thr</th><th>Precision</th><th>Recall</th><th>Review rate</th><th>FP cost</th><th>Decision cost</th></tr></thead>
      <tbody>${grid
        .map(
          (r) =>
            `<tr><td>${r.threshold.toFixed(2)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.review_rate)}</td><td>${inr(r.fp_cost_total)}</td><td>${inr(r.expected_decision_cost)}</td></tr>`
        )
        .join("")}</tbody></table>
    </article>`;
  const slider = document.getElementById("thr");
  const paint = () => {
    const r = grid[Number(slider.value)];
    document.getElementById("thr-label").textContent = "threshold " + r.threshold.toFixed(2);
    document.getElementById("lab-kpis").innerHTML =
      kpi("Precision", pct(r.precision)) +
      kpi("Recall", pct(r.recall)) +
      kpi("Review rate", pct(r.review_rate), `${r.tp + r.fp} flags`) +
      kpi("FP cost", inr(r.fp_cost_total), `${r.fp} false positives`) +
      kpi("Net decision cost", inr(r.expected_decision_cost), `${r.fn} misses`);
  };
  const target = lab.validation_choice?.threshold;
  const idx = grid.findIndex((x) => Math.abs(x.threshold - target) < 0.001);
  slider.value = idx >= 0 ? idx : 4;
  slider.addEventListener("input", paint);
  paint();
}

async function renderHealth() {
  const m = await api("/api/v1/metrics");
  const fail = await api("/api/v1/failures");
  const t = m.held_out_test;
  const rows = Object.entries(m.model_comparison_test)
    .map(
      ([name, r]) =>
        `<tr><td>${name}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${Number(r.pr_auc).toFixed(3)}</td><td>${Number(r.brier).toFixed(3)}</td><td>${inr(r.expected_decision_cost)}</td></tr>`
    )
    .join("");
  view.innerHTML = `
    <div class="grid kpis">
      ${kpi("Test PR-AUC", Number(t.pr_auc).toFixed(3), "Primary ranking metric")}
      ${kpi("Brier", Number(t.brier).toFixed(3), "Calibration (lower is better)")}
      ${kpi("Confusion", `TP ${t.tp} / FP ${t.fp}`, `FN ${t.fn} · TN ${t.tn}`)}
      ${kpi("Threshold", t.threshold, "Chosen on validation cost")}
    </div>
    <article class="card" style="margin-top:16px">
      <h2>Locked test comparison</h2>
      <table><thead><tr><th>Approach</th><th>Precision</th><th>Recall</th><th>PR-AUC</th><th>Brier</th><th>Cost</th></tr></thead><tbody>${rows}</tbody></table>
    </article>
    <article class="card" style="margin-top:16px">
      <h2>Known mistakes (required for an honest demo)</h2>
      <p>False positive example: <code>${JSON.stringify(fail.false_positive_example)}</code></p>
      <p>False negative example: <code>${JSON.stringify(fail.false_negative_example)}</code></p>
      <p class="muted">The model can inconvenience a legitimate customer or miss a costly return. That is why operators remain in the loop.</p>
    </article>`;
}

async function renderQuality() {
  const q = await api("/api/v1/quality");
  const m = await api("/api/v1/metrics");
  view.innerHTML = `
    <article class="card">
      <h2>Dataset contract</h2>
      <p class="notice">This dataset is <strong>synthetic</strong>. It is not real merchant data.</p>
      <p>Rows: ${q.n_rows} · duplicate order ids: ${q.n_duplicates_order_id}</p>
      <p>Target balance: ${JSON.stringify(q.target_balance)}</p>
      <p>Leakage columns present: ${q.leakage_columns_present.length ? q.leakage_columns_present.join(", ") : "none"}</p>
      <p>Warnings: ${q.warnings.join(" ") || "none"}</p>
      <p>Split protocol: ${m.splits.protocol}</p>
      <p>Train ends ${m.splits.train.end} · valid ends ${m.splits.valid.end} · test ${m.splits.test.start} → ${m.splits.test.end}</p>
    </article>`;
}

async function renderAudit() {
  const a = await api("/api/v1/audit");
  view.innerHTML = `<article class="card"><h2>Append-only events</h2>
    <table><thead><tr><th>Time</th><th>Type</th><th>Order</th><th>Payload</th></tr></thead>
    <tbody>${a.events
      .map(
        (e) =>
          `<tr><td>${e.ts}</td><td>${e.event_type}</td><td>${e.order_id || ""}</td><td><code>${JSON.stringify(e.payload).slice(0, 180)}</code></td></tr>`
      )
      .join("") || `<tr><td colspan="4">No events yet. Score an order first.</td></tr>`}
    </tbody></table></article>`;
}

function renderSafety() {
  view.innerHTML = `
    <article class="card">
      <h2>Intended use</h2>
      <p>Prioritize benign interventions before fulfillment: address confirmation, fit/size reminders, delivery preference checks, or manual review of likely costly returns.</p>
      <h2>Prohibited use</h2>
      <ul>
        <li>Generating or testing stolen payment credentials</li>
        <li>Bypassing payment, refund, or identity systems</li>
        <li>Automated cancellation, blacklisting, or punitive customer treatment</li>
        <li>Using protected attributes as risk signals</li>
        <li>Claiming the score is proof of fraud or abuse</li>
      </ul>
      <h2>Limitations</h2>
      <p>Synthetic labels, delayed real-world outcomes, concept drift, and category mix shifts will degrade performance. Fail closed to manual review if the model is down.</p>
    </article>`;
}

async function renderOrder(id) {
  const o = await api("/api/v1/orders/" + id);
  view.innerHTML = `
    <article class="card">
      ${renderScoreCard(o)}
      <h3>Allowed actions</h3>
      <p>${(o.allowed_actions || []).map((a) => `<span class="pill">${a}</span>`).join(" ")}</p>
      <form id="rev">
        <label>Operator<input name="operator" value="demo.operator" /></label>
        <label>Decision
          <select name="decision">
            ${(o.allowed_actions || ["manual_review"]).map((a) => `<option value="${a}">${a}</option>`).join("")}
          </select>
        </label>
        <label>Note<textarea name="note" placeholder="Why this action?"></textarea></label>
        <button type="submit">Record operator decision</button>
      </form>
      <p class="muted">Matured label in demo set: ${o.problematic_return === 1 ? "costly return" : "not a costly return"} — used only for evaluation, never as a live feature.</p>
    </article>`;
  document.getElementById("rev").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    await api("/api/v1/review", {
      method: "POST",
      body: JSON.stringify({
        order_id: id,
        operator: fd.get("operator"),
        decision: fd.get("decision"),
        note: fd.get("note"),
      }),
    });
    alert("Decision stored in the audit log.");
    location.hash = "#/audit";
  });
}

async function route() {
  const r = nav();
  const [name, id] = r.split("/");
  const t = titles[name] || titles.dashboard;
  title.textContent = t[0];
  lede.textContent = t[1];
  try {
    if (name === "queue") return renderQueue();
    if (name === "score") return renderScore();
    if (name === "lab") return renderLab();
    if (name === "health") return renderHealth();
    if (name === "quality") return renderQuality();
    if (name === "audit") return renderAudit();
    if (name === "safety") return renderSafety();
    if (name === "order" && id) return renderOrder(id);
    return renderDashboard();
  } catch (err) {
    view.innerHTML = `<article class="card"><p class="notice">${err.message}</p></article>`;
  }
}

document.body.addEventListener("click", (e) => {
  const id = e.target?.dataset?.open;
  if (id) location.hash = "#/order/" + id;
});

window.addEventListener("hashchange", route);
route();
