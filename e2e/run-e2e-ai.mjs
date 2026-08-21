// Optional e2e for the REAL AI depth path: downloads Depth Anything V2 small
// through transformers.js (network required; ~25 MB on first run) and runs
// wasm inference in headless Chromium. Slow — not part of the default suite.
// Run: node e2e/run-e2e-ai.mjs
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
function loadPlaywright() {
  for (const c of ['playwright', '/opt/node22/lib/node_modules/playwright']) {
    try { return require(c); } catch { /* next */ }
  }
  throw new Error('playwright not found');
}
const { chromium } = loadPlaywright();

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png',
};
const server = http.createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = path.normalize(path.join(root, p === '/' ? 'index.html' : p));
  if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

let failures = 0;
const check = (cond, name, extra = '') => {
  if (cond) console.log(`  ok  ${name}`);
  else { failures++; console.error(`FAIL  ${name} ${extra}`); }
};

const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
page.on('console', (m) => {
  const t = m.text();
  if (m.type() === 'error' || /model|depth|wasm|onnx/i.test(t)) console.log('  [page]', t.slice(0, 200));
});

try {
  await page.goto(`${base}/?nowiggle=1&quality=low&maxpx=120000`);
  await page.waitForFunction(() => window.__gp, null, { timeout: 30000 });

  console.log('  loading photo through the AI path (model download + wasm inference)…');
  await page.evaluate(async () => {
    const blob = await (await fetch('/assets/sample.png')).blob();
    window.__gp.openBlob(blob); // no await: resolves after full build
  });
  await page.waitForFunction(
    () => window.__gp.app.disparityKind !== undefined &&
      window.__gp.app.cloud && window.__gp.app.disparityKind !== 'gt',
    null, { timeout: 600000 });

  const kind = await page.evaluate(() => window.__gp.app.disparityKind);
  check(kind === 'ai', `depth came from the AI model (kind=${kind})`,
    kind === 'none' ? '(model unreachable — heuristic fallback took over; network?)' : '');

  if (kind === 'ai') {
    const backend = await page.evaluate(() => window.__gp.app.estimator && window.__gp.app.estimator.backend);
    console.log(`  backend: ${backend}`);
    // disparity sanity: near (bottom) should be closer than sky (top)
    const sane = await page.evaluate(() => {
      const { imageData, disparity } = window.__gp.app;
      const w = imageData.width, h = imageData.height;
      const avg = (y0, y1) => {
        let s = 0, n = 0;
        for (let y = y0; y < y1; y++) for (let x = 0; x < w; x += 3) { s += disparity[y * w + x]; n++; }
        return s / n;
      };
      return { top: avg(0, Math.floor(h * 0.15)), bottom: avg(Math.floor(h * 0.85), h) };
    });
    check(sane.bottom > sane.top, `AI disparity: bottom nearer than sky (top ${sane.top.toFixed(2)}, bottom ${sane.bottom.toFixed(2)})`);

    const stats = await page.evaluate(() => {
      const cap = window.__gp.captureNow(1);
      let sum = 0;
      const n = cap.width * cap.height;
      for (let i = 0; i < n; i++) sum += (cap.pixels[i * 4] + cap.pixels[i * 4 + 1] + cap.pixels[i * 4 + 2]) / 3;
      return sum / n;
    });
    check(stats > 15, `AI-depth splat renders (mean luma ${stats.toFixed(1)})`);
  }
} finally {
  await browser.close();
  server.close();
}
console.log(failures === 0 ? '\ne2e-ai: all checks passed' : `\ne2e-ai: ${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
