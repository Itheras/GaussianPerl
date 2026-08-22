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
  await page.waitForFunction(() => window.__gp && window.__gp.app.meta && window.__gp.app.meta.phase === 'final',
    null, { timeout: 90000 });
  const meta = await page.evaluate(() => window.__gp.app.meta);
  check(meta.fillKind === 'classical', `classical fill under nomodel (${meta.fillKind})`);
  check(meta.pw === meta.w + 2 * meta.padPx, `layer dims consistent (${meta.pw})`);
  check(meta.dSub > 0 && meta.dMax > meta.dMin, `depth anchors sane (dSub ${meta.dSub.toFixed(3)})`);
  const cov1 = await page.evaluate(() => {
    const L = window.__gp.app.layers;
    let n = 0;
    for (let i = 0; i < L.pw * L.ph; i++) n += L.color1[i * 4 + 3] > 0 ? 1 : 0;
    return n;
  });
  check(cov1 > 1000, `background layer has coverage (${cov1} px)`);

  await page.waitForTimeout(300);
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
    return { mean, std: Math.sqrt(Math.max(sum2 / n - mean * mean, 0)), blackFrac: black / n };
  });
  check(stats.mean > 15, `render not empty (mean luma ${stats.mean.toFixed(1)})`);
  check(stats.std > 12, `render has structure (std ${stats.std.toFixed(1)})`);
  check(stats.blackFrac < 0.6, `not mostly black (${(stats.blackFrac * 100).toFixed(1)}%)`);

  // drag translates the window camera and changes pixels
  const before = await page.evaluate(() => {
    const cap = window.__gp.captureNow(1);
    return Array.from(cap.pixels.filter((_, i) => i % 997 === 0));
  });
  await page.mouse.move(450, 320);
  await page.mouse.down();
  for (let i = 1; i <= 8; i++) await page.mouse.move(450 - i * 18, 320 + i * 6);
  await page.mouse.up();
  await page.waitForTimeout(400);
  const ex = await page.evaluate(() => window.__gp.cam.ex);
  check(Math.abs(ex) > 0.002, `drag translates the camera (ex ${ex.toFixed(4)})`);
  const after = await page.evaluate(() => {
    const cap = window.__gp.captureNow(1);
    return Array.from(cap.pixels.filter((_, i) => i % 997 === 0));
  });
  let diff = 0;
  for (let i = 0; i < before.length; i++) diff += Math.abs(before[i] - after[i]);
  check(diff / before.length > 0.5, `pixels changed after pan (avg diff ${(diff / before.length).toFixed(2)})`);

  // wheel dollies (ez > 0)
  await page.evaluate(() => window.__gp.cam.reset());
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(120);
  const ez = await page.evaluate(() => window.__gp.cam.ez);
  check(ez > 0.005, `wheel dollies in (ez ${ez.toFixed(3)})`);

  // double-click re-pivots + refocuses: foreground stone vs sky
  await page.evaluate(() => window.__gp.cam.reset());
  await page.waitForTimeout(150);
  await page.mouse.dblclick(280, 500);
  const focusNear = await page.evaluate(() => window.__gp.app.targetFocus);
  await page.mouse.dblclick(450, 80);
  const focusFar = await page.evaluate(() => window.__gp.app.targetFocus);
  check(focusFar > focusNear * 1.2, `refocus near->far (${focusNear.toFixed(2)} -> ${focusFar.toFixed(2)})`);

  // subject-plane lock: at the pivot plane, a full-envelope pan moves content <1.5px
  const lock = await page.evaluate(async () => {
    const gp = window.__gp;
    gp.cam.reset(); gp.settings.aperture = 0;
    gp.app.dConv = gp.app.meta.dSub;
    gp.app.needsRender = true;
    await new Promise((r) => setTimeout(r, 100));
    const capA = gp.captureNow(1);
    gp.cam.ex = gp.cam.exyMax; gp.app.needsRender = true;
    await new Promise((r) => setTimeout(r, 100));
    const capB = gp.captureNow(1);
    // measure shift of the pivot-plane content: find disparity==dSub pixels is
    // hard from pixels alone; instead assert the CENTER ROW cross-correlation
    // peak is at a sub-2px shift (subject dominates the center)
    const w = capA.width, h = capA.height;
    const row = (cap, y) => {
      const out = new Float32Array(w);
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        out[x] = (cap.pixels[i] + cap.pixels[i + 1] + cap.pixels[i + 2]) / 3;
      }
      return out;
    };
    const a = row(capA, Math.floor(h * 0.55)), b = row(capB, Math.floor(h * 0.55));
    let bestShift = 0, bestScore = Infinity;
    for (let s = -12; s <= 12; s++) {
      let sc = 0, n = 0;
      for (let x = Math.max(0, -s); x < Math.min(w, w - s); x += 2) {
        sc += Math.abs(a[x] - b[x + s]); n++;
      }
      sc /= n;
      if (sc < bestScore) { bestScore = sc; bestShift = s; }
    }
    return bestShift;
  });
  check(Math.abs(lock) <= 2, `pivot-plane content stays locked under full pan (shift ${lock}px)`);

  // DoF changes pixels (focus pulled off-subject so blur has something to do)
  const sharp = await page.evaluate(() => {
    const gp = window.__gp;
    gp.cam.reset();
    gp.settings.aperture = 0;
    gp.app.targetFocus = gp.app.focusDist = 0.6;
    const cap = gp.captureNow(1);
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
    window.__gp.settings.aperture = 0;
    const cap = window.__gp.captureNow(2);
    const canvas = document.createElement('canvas');
    canvas.width = cap.width; canvas.height = cap.height;
    canvas.getContext('2d').putImageData(new ImageData(cap.pixels, cap.width, cap.height), 0, 0);
    const blob = await new Promise((r) => canvas.toBlob(r, 'image/png'));
    return blob ? blob.size : 0;
  });
  check(pngSize > 50000, `PNG capture encodes (${(pngSize / 1024).toFixed(0)} KB)`);

  // record shots
  fs.mkdirSync(path.join(root, 'e2e', 'out'), { recursive: true });
  const snap = async (name, setup) => {
    const shot = await page.evaluate(async (setupSrc) => {
      // eslint-disable-next-line no-new-func
      new Function('gp', `(${setupSrc})(gp)`)(window.__gp);
      await new Promise((r) => setTimeout(r, 300));
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
    gp.cam.reset(); gp.settings.aperture = 0;
    gp.app.targetFocus = gp.app.focusDist = 1;
  });
  await snap('render-pan.png', (gp) => {
    gp.cam.reset(); gp.settings.aperture = 0;
    gp.cam.ex = gp.cam.exyMax; gp.cam.ey = gp.cam.exyMax * 0.6;
  });
  await snap('render-dof.png', (gp) => {
    gp.cam.reset(); gp.settings.aperture = 0.7;
    gp.app.targetFocus = gp.app.focusDist = 0.9;
  });
} finally {
  await browser.close();
  server.close();
}

console.log(failures === 0 ? '\ne2e: all checks passed' : `\ne2e: ${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
