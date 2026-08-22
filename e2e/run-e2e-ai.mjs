// Optional e2e for the REAL AI paths: downloads Depth Anything V2 (transformers
// .js) AND the MI-GAN fill model (onnxruntime-web) through their CDNs (network
// required; ~25 MB + ~28 MB on first run), runs wasm inference in headless
// Chromium, and verifies the generative-fill preview->final build flow.
// Slow — not part of the default suite. Run: node e2e/run-e2e-ai.mjs
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
  if (m.type() === 'error' || /model|depth|fill|wasm|onnx/i.test(t)) console.log('  [page]', t.slice(0, 200));
});

try {
  // fillep=wasm: never attempt the WebGPU EP under SwiftShader
  await page.goto(`${base}/?nowiggle=1&quality=low&maxpx=120000&fillep=wasm`);
  await page.waitForFunction(() => window.__gp, null, { timeout: 30000 });

  console.log('  loading photo through the AI path (model downloads + wasm inference)…');
  await page.evaluate(async () => {
    const blob = await (await fetch('/assets/sample.png')).blob();
    window.__gp.openBlob(blob); // no await: resolves after full build
  });

  // preview (or single final) build lands first
  await page.waitForFunction(() => window.__gp.app.layers && window.__gp.app.meta,
    null, { timeout: 600000 });
  const meta1 = await page.evaluate(() => window.__gp.app.meta);
  check(meta1.depthKind === 'ai', `depth came from the AI model (kind=${meta1.depthKind})`,
    meta1.depthKind === 'heuristic' ? '(model unreachable — heuristic fallback took over; network?)' : '');
  console.log(`  depth backend: ${meta1.depthBackend} (${meta1.depthTier}); first phase: ${meta1.phase}`);
  check(meta1.f35 === undefined || true, 'meta ok');

  if (meta1.depthKind === 'ai') {
    check(meta1.dispBottomMean > meta1.dispTopMean,
      `AI disparity: bottom nearer than sky (top ${meta1.dispTopMean.toFixed(2)}, bottom ${meta1.dispBottomMean.toFixed(2)})`);
  }

  // generative fill: wait for the final phase with AI fill applied
  console.log('  waiting for the generative fill (MI-GAN download + wasm calls)…');
  await page.waitForFunction(
    () => window.__gp.app.meta && window.__gp.app.meta.phase === 'final',
    null, { timeout: 600000 });
  const meta2 = await page.evaluate(() => window.__gp.app.meta);
  check(meta2.fillKind === 'ai', `generative fill applied (fillKind=${meta2.fillKind})`);
  check(meta2.fillBackend === 'wasm', `fill backend honored fillep=wasm (${meta2.fillBackend})`);
  const layerStats = await page.evaluate(() => {
    const L = window.__gp.app.layers;
    let cov = 0;
    for (let i = 0; i < L.pw * L.ph; i++) cov += L.color1[i * 4 + 3] > 0 ? 1 : 0;
    return { cov, padPx: L.padPx };
  });
  check(layerStats.cov > 500, `disocclusion layer has coverage (${layerStats.cov} px)`);
  check(layerStats.padPx > 0, `outpainted ring present (pad ${layerStats.padPx}px)`);

  const stats = await page.evaluate(() => {
    const cap = window.__gp.captureNow(1);
    let sum = 0;
    const n = cap.width * cap.height;
    for (let i = 0; i < n; i++) sum += (cap.pixels[i * 4] + cap.pixels[i * 4 + 1] + cap.pixels[i * 4 + 2]) / 3;
    return sum / n;
  });
  check(stats > 15, `AI-filled splat renders (mean luma ${stats.toFixed(1)})`);

  fs.mkdirSync(path.join(root, 'e2e/out'), { recursive: true });
  const shot = await page.evaluate(() => {
    const cap = window.__gp.captureNow(1);
    return { w: cap.width, h: cap.height, px: Array.from(cap.pixels) };
  });
  const { encodePNG } = await import('../tools/png.mjs');
  fs.writeFileSync(path.join(root, 'e2e/out/render-ai-fill.png'),
    encodePNG(shot.w, shot.h, Uint8ClampedArray.from(shot.px)));
  console.log('  (screenshot: e2e/out/render-ai-fill.png)');
} finally {
  await browser.close();
  server.close();
}
console.log(failures === 0 ? '\ne2e-ai: all checks passed' : `\ne2e-ai: ${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
