// Headless end-to-end test: boots the app in Chromium (SwiftShader WebGL2),
// builds the sample splat offline (?nomodel), and verifies rendering,
// interaction, refocus, and PNG capture. Run: node e2e/run-e2e.mjs
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
function loadPlaywright() {
  const candidates = [
    'playwright',
    '/opt/node22/lib/node_modules/playwright',
    '/usr/lib/node_modules/playwright',
  ];
  for (const c of candidates) {
    try { return require(c); } catch { /* next */ }
  }
  throw new Error('playwright not found — npm i -g playwright');
}
const { chromium } = loadPlaywright();

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.png': 'image/png', '.json': 'application/json',
  '.svg': 'image/svg+xml',
};

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  let file = path.normalize(path.join(root, urlPath === '/' ? 'index.html' : urlPath));
  if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});

await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const base = `http://127.0.0.1:${port}`;

let failures = 0;
const check = (cond, name, extra = '') => {
  if (cond) console.log(`  ok  ${name}`);
  else { failures++; console.error(`FAIL  ${name} ${extra}`); }
};

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 900, height: 640 } });
page.on('console', (m) => {
  if (m.type() === 'error') console.error('  [page error]', m.text());
});
page.on('pageerror', (e) => { failures++; console.error('FAIL  pageerror:', e.message); });

try {
  await page.goto(`${base}/?demo=1&nomodel=1&nowiggle=1&quality=low&maxpx=140000`);
  await page.waitForFunction(() => window.__gp && window.__gp.app.cloud, null, { timeout: 90000 });
  const count = await page.evaluate(() => window.__gp.app.cloud.count);
  check(count > 100000, `cloud built (${count} splats)`);

  const meta = await page.evaluate(() => window.__gp.app.meta);
  check(meta.bgCount > 0, `background layer synthesized (${meta.bgCount})`);
  check(meta.skirtCount > 0, `skirt present (${meta.skirtCount})`);
  check(meta.underCount > 0, `underlayer present (${meta.underCount})`);

  // wait for first sort to land, then a couple frames
  await page.waitForFunction(() => window.__gp.app.lastSortView, null, { timeout: 20000 });
  await page.waitForTimeout(400);

  const stats = await page.evaluate(() => {
    const cap = window.__gp.captureNow(1);
    let sum = 0, sum2 = 0, black = 0;
    const n = cap.width * cap.height;
    for (let i = 0; i < n; i++) {
      const l = (cap.pixels[i * 4] + cap.pixels[i * 4 + 1] + cap.pixels[i * 4 + 2]) / 3;
      sum += l; sum2 += l * l;
      if (l < 4) black++;
    }
    const mean = sum / n;
    return { mean, std: Math.sqrt(Math.max(sum2 / n - mean * mean, 0)), blackFrac: black / n, w: cap.width, h: cap.height };
  });
  check(stats.mean > 15, `render not empty (mean luma ${stats.mean.toFixed(1)})`);
  check(stats.std > 12, `render has structure (std ${stats.std.toFixed(1)})`);
  check(stats.blackFrac < 0.6, `not mostly black (${(stats.blackFrac * 100).toFixed(1)}%)`);

  // capture a reference, drag-orbit, verify the view (and pixels) changed
  const before = await page.evaluate(() => {
    const cap = window.__gp.captureNow(1);
    return Array.from(cap.pixels.filter((_, i) => i % 997 === 0));
  });
  await page.mouse.move(450, 320);
  await page.mouse.down();
  for (let i = 1; i <= 8; i++) await page.mouse.move(450 - i * 18, 320 + i * 6);
  await page.mouse.up();
  await page.waitForTimeout(500);
  const yaw = await page.evaluate(() => window.__gp.controls.yaw);
  check(Math.abs(yaw) > 0.02, `drag orbits the camera (yaw ${yaw.toFixed(3)})`);
  const after = await page.evaluate(() => {
    const cap = window.__gp.captureNow(1);
    return Array.from(cap.pixels.filter((_, i) => i % 997 === 0));
  });
  let diff = 0;
  for (let i = 0; i < before.length; i++) diff += Math.abs(before[i] - after[i]);
  check(diff / before.length > 1.5, `pixels changed after orbit (avg diff ${(diff / before.length).toFixed(2)})`);

  // pinch-style dolly via wheel
  const distBefore = await page.evaluate(() => window.__gp.controls.distance);
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(120);
  const distAfter = await page.evaluate(() => window.__gp.controls.distance);
  check(distAfter < distBefore, `wheel dollies in (${distBefore.toFixed(2)} -> ${distAfter.toFixed(2)})`);

  // double-click refocus: near the big foreground stone (left-low) vs sky (top)
  await page.evaluate(() => window.__gp.controls.reset());
  await page.waitForTimeout(150);
  await page.mouse.dblclick(280, 500);
  const focusNear = await page.evaluate(() => window.__gp.app.targetFocus);
  await page.mouse.dblclick(450, 80);
  const focusFar = await page.evaluate(() => window.__gp.app.targetFocus);
  check(focusFar > focusNear * 1.3, `refocus near->far (${focusNear.toFixed(2)} -> ${focusFar.toFixed(2)})`);

  // DoF changes pixels
  const sharp = await page.evaluate(() => {
    window.__gp.settings.aperture = 0;
    const cap = window.__gp.captureNow(1);
    return Array.from(cap.pixels.filter((_, i) => i % 997 === 0));
  });
  const blurred = await page.evaluate(() => {
    window.__gp.settings.aperture = 1;
    const cap = window.__gp.captureNow(1);
    return Array.from(cap.pixels.filter((_, i) => i % 997 === 0));
  });
  let dofDiff = 0;
  for (let i = 0; i < sharp.length; i++) dofDiff += Math.abs(sharp[i] - blurred[i]);
  check(dofDiff / sharp.length > 0.5, `DoF affects the image (avg diff ${(dofDiff / sharp.length).toFixed(2)})`);

  // save flow: capture -> PNG blob
  const pngSize = await page.evaluate(async () => {
    const cap = window.__gp.captureNow(2);
    const canvas = document.createElement('canvas');
    canvas.width = cap.width; canvas.height = cap.height;
    canvas.getContext('2d').putImageData(new ImageData(cap.pixels, cap.width, cap.height), 0, 0);
    const blob = await new Promise((r) => canvas.toBlob(r, 'image/png'));
    return blob ? blob.size : 0;
  });
  check(pngSize > 50000, `PNG capture encodes (${(pngSize / 1024).toFixed(0)} KB)`);

  // .splat export via worker round-trip
  const splatBytes = await page.evaluate(() => new Promise((resolve) => {
    const { app } = window.__gp;
    const w = new Worker(new URL('./src/pipeline/pipeline-worker.js', location.href), { type: 'module' });
    w.onmessage = (e) => { if (e.data.type === 'exported') resolve(e.data.bytes.length); };
    const pos = app.cloud.positions.slice();
    const cov = app.cloud.cov.slice();
    const col = app.cloud.colors.slice();
    w.postMessage({ type: 'export', id: 'export', count: app.cloud.count,
      positions: pos.buffer, cov: cov.buffer, colors: col.buffer },
      [pos.buffer, cov.buffer, col.buffer]);
    setTimeout(() => resolve(0), 60000);
  }));
  const expected = await page.evaluate(() => window.__gp.app.cloud.count * 32);
  check(splatBytes === expected, `.splat export size (${splatBytes} bytes)`);

  // record shots: sharp home view, orbited parallax view, shallow DoF close-up
  fs.mkdirSync(path.join(root, 'e2e', 'out'), { recursive: true });
  const snap = async (name, setup) => {
    const shot = await page.evaluate(async (setupSrc) => {
      // eslint-disable-next-line no-new-func
      new Function('gp', `(${setupSrc})(gp)`)(window.__gp);
      await new Promise((r) => setTimeout(r, 700)); // let sort settle
      const cap = window.__gp.captureNow(1);
      const canvas = document.createElement('canvas');
      canvas.width = cap.width; canvas.height = cap.height;
      canvas.getContext('2d').putImageData(new ImageData(cap.pixels, cap.width, cap.height), 0, 0);
      return canvas.toDataURL('image/png');
    }, setup.toString());
    fs.writeFileSync(path.join(root, 'e2e', 'out', name),
      Buffer.from(shot.split(',')[1], 'base64'));
    console.log(`  (screenshot: e2e/out/${name})`);
  };
  await snap('render-home.png', (gp) => {
    gp.controls.reset(); gp.settings.aperture = 0;
  });
  await snap('render-orbit.png', (gp) => {
    gp.controls.reset(); gp.settings.aperture = 0;
    gp.controls.yaw = 0.45; gp.controls.pitch = 0.12;
  });
  await snap('render-dof.png', (gp) => {
    gp.controls.reset(); gp.settings.aperture = 0.7;
    gp.app.targetFocus = gp.app.focusDist = gp.app.meta.nearZ * 1.6;
  });
} finally {
  await browser.close();
  server.close();
}

console.log(failures === 0 ? '\ne2e: all checks passed' : `\ne2e: ${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
