// Verifies the in-browser scorer reproduces the Python champion's probabilities.
//   node site/parity_test.js
const fs = require("fs");
const path = require("path");
const { createScorer } = require("./scorer.js");

const html = fs.readFileSync(path.join(__dirname, "..", "docs", "index.html"), "utf8");
const marker = "window.CHURNGUARD = ";
const start = html.indexOf(marker) + marker.length;
const end = html.indexOf(";\n", start);
const data = JSON.parse(html.slice(start, end));
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "parity_fixture.json"), "utf8"));

const scorer = createScorer(data.model);
let maxErr = 0;
let explainErr = 0;
fixture.rows.forEach((row, i) => {
  const r = scorer.score(row, true);
  maxErr = Math.max(maxErr, Math.abs(r.p - fixture.p[i]));
  const sum = r.bias + Object.values(r.contributions).reduce((s, v) => s + v, 0);
  explainErr = Math.max(explainErr, Math.abs(sum - r.raw));
});
console.log(`rows=${fixture.rows.length} max|p_js - p_python|=${maxErr.toExponential(2)} explain_residual=${explainErr.toExponential(2)}`);
if (maxErr > 1e-4 || explainErr > 1e-6) {
  console.error("PARITY FAILED");
  process.exit(1);
}
console.log("PARITY OK");
