// ChurnGuard in-browser scorer.
// Re-implements the production sklearn pipeline (FeatureEngineer -> median imputer ->
// one-hot -> LightGBM -> Platt calibration) over the exported champion trees.
// Verified against Python predictions by site/parity_test.js.
(function (root) {
  "use strict";

  const ENGINEERED = (c) => {
    const tenureClip = Math.max(c.tenure_months, 1);
    const total = c.total_charges == null ? c.monthly_charges * c.tenure_months : c.total_charges;
    return {
      avg_charge_per_month: total / tenureClip,
      tickets_per_tenure_year: c.support_tickets_90d / (tenureClip / 12 + 1),
      is_new_customer: c.tenure_months < 6 ? 1 : 0,
      charge_per_product: c.monthly_charges / Math.max(c.num_products, 1),
      nps_missing: c.nps_score == null ? 1 : 0,
      engagement_drop: c.usage_change_pct < -15 ? 1 : 0,
    };
  };

  function originalFeature(col, categorical) {
    const i = col.indexOf("__");
    const prefix = col.slice(0, i);
    const rest = col.slice(i + 2);
    if (prefix === "cat") {
      for (const c of categorical) if (rest.startsWith(c + "_")) return c;
    }
    return rest;
  }

  function createScorer(model) {
    const groups = model.columns.map((c) => originalFeature(c, model.categorical));

    function vectorize(customer) {
      const eng = ENGINEERED(customer);
      const full = Object.assign({}, customer, eng);
      const x = [];
      model.numeric.forEach((name, i) => {
        const v = full[name];
        x.push(v == null || Number.isNaN(v) ? model.medians[i] : Number(v));
      });
      model.categorical.forEach((name, j) => {
        for (const cat of model.categories[j]) x.push(String(customer[name]) === cat ? 1 : 0);
      });
      return { x, engineered: full };
    }

    function score(customer, explain = false) {
      const { x, engineered } = vectorize(customer);
      let raw = 0;
      const contrib = explain ? new Float64Array(model.columns.length) : null;
      let bias = 0;
      for (const tree of model.trees) {
        let n = tree[0];
        if (explain) bias += n[4];
        while (n[0] !== -1) {
          const next = tree[x[n[0]] <= n[1] ? n[2] : n[3]];
          if (explain) contrib[n[0]] += next[4] - n[4];
          n = next;
        }
        raw += n[4];
      }
      const pBase = 1 / (1 + Math.exp(-raw));
      const { a, b } = model.calibration;
      // sklearn >= 1.8 applies Platt scaling to the log-odds of the base model.
      const p = 1 / (1 + Math.exp(a * raw + b));
      const out = { p, raw, pBase };
      if (explain) {
        const byFeature = {};
        contrib.forEach((v, i) => { byFeature[groups[i]] = (byFeature[groups[i]] || 0) + v; });
        out.bias = bias;
        out.contributions = byFeature;
        out.values = engineered;
      }
      return out;
    }

    return { score, vectorize };
  }

  const api = { createScorer, originalFeature };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ChurnGuardScorer = api;
})(typeof window !== "undefined" ? window : globalThis);
