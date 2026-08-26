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
  const orb = await page.evaluate(() => ({
    yaw: window.__gp.cam.yaw, pitch: window.__gp.cam.pitch,
    C: Array.from(window.__gp.cam.center()),
  }));
  check(Math.abs(orb.yaw) > 0.01, `drag orbits the camera (yaw ${orb.yaw.toFixed(3)})`);
  check(Math.hypot(orb.C[0], orb.C[1], orb.C[2]) > 0.02,
    `orbiting actually moves the eye (|C| ${Math.hypot(orb.C[0], orb.C[1], orb.C[2]).toFixed(3)})`);
  const after = await page.evaluate(() => {
    const cap = window.__gp.captureNow(1);
    return Array.from(cap.pixels.filter((_, i) => i % 997 === 0));
  });
  let diff = 0;
  for (let i = 0; i < before.length; i++) diff += Math.abs(before[i] - after[i]);
  check(diff / before.length > 0.5, `pixels changed after pan (avg diff ${(diff / before.length).toFixed(2)})`);

  // wheel dollies: the orbit radius shrinks
  await page.evaluate(() => window.__gp.cam.reset());
  const dist0 = await page.evaluate(() => window.__gp.cam.dist);
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(120);
  const dist1 = await page.evaluate(() => window.__gp.cam.dist);
  check(dist1 < dist0 - 0.005, `wheel dollies in (${dist0.toFixed(3)} -> ${dist1.toFixed(3)})`);

  // keyboard fly moves the eye
  await page.evaluate(() => window.__gp.cam.reset());
  await page.keyboard.down('a');
  await page.waitForTimeout(350);
  await page.keyboard.up('a');
  const flew = await page.evaluate(() => Array.from(window.__gp.cam.center()));
  check(flew[0] < -0.005, `A key flies left (x ${flew[0].toFixed(4)})`);

  // double-click re-pivots + refocuses: foreground stone vs sky
  await page.evaluate(() => window.__gp.cam.reset());
  await page.waitForTimeout(150);
  await page.mouse.dblclick(280, 500);
  const focusNear = await page.evaluate(() => window.__gp.app.targetFocus);
  await page.mouse.dblclick(450, 80);
  const focusFar = await page.evaluate(() => window.__gp.app.targetFocus);
  check(focusFar > focusNear * 1.2, `refocus near->far (${focusNear.toFixed(2)} -> ${focusFar.toFixed(2)})`);

  // Rest fidelity: at the home pose the render must BE the photograph. This is
  // the invariant the whole free camera is built around — every anchor, every
  // confidence term and the whole generation loop must leave it untouched.
  const restStats = await page.evaluate(async () => {
    const gp = window.__gp;
    gp.cam.reset(); gp.settings.aperture = 0;
    gp.app.needsRender = true;
    await new Promise((r) => setTimeout(r, 120));
    const st = gp.renderState();
    const cap = gp.renderer.captureAnchorFrame({
      ...st, cam: { R: st.cam.R, C: st.cam.C, K: st.cam.K },
      trustBase: false, dofStrength: 0,
    }, 160, 120);
    let holes = 0, minConf = 1, minBase = 1;
    for (let i = 0; i < cap.conf.length; i++) {
      if (cap.conf[i] < 0.5) holes++;
      if (cap.conf[i] < minConf) minConf = cap.conf[i];
      if (cap.baseShare[i] < minBase) minBase = cap.baseShare[i];
    }
    return { holeFrac: holes / cap.conf.length, minConf, minBase };
  });
  check(restStats.holeFrac === 0,
    `at rest nothing is missing (hole fraction ${restStats.holeFrac})`);
  // confidence and provenance both saturated is what makes the screen-space
  // grain weight exactly zero at home — i.e. the photograph is returned
  // untouched, which is the property the whole design is built around
  check(restStats.minConf > 0.99 && restStats.minBase > 0.99,
    `at rest every pixel is pure photograph (conf ${restStats.minConf.toFixed(3)}, base ${restStats.minBase.toFixed(3)})`);

  // THE GATE. At the home pose the render must be the photograph, byte for
  // byte — not "close", not "no holes". Every later stage (native shell,
  // semantic inpainting, reconstructed people) plugs into the same anchor
  // loop, and each one is a chance to silently perturb this. Asserted here
  // three times: fresh, after an anchor is committed, and after a GPU context
  // loss, which is the path that would otherwise lose generated anchors and
  // re-upload something subtly different.
  const pixelExact = await page.evaluate(async () => {
    const gp = window.__gp;
    const m = gp.app.meta;
    const src = gp.app.imageData.data;

    const measure = () => {
      gp.cam.reset();
      gp.settings.aperture = 0;
      const st = gp.renderState();
      const cap = gp.renderer.captureAnchorFrame({
        ...st, cam: { R: st.cam.R, C: st.cam.C, K: st.cam.K },
        trustBase: false, dofStrength: 0,
      }, m.w, m.h);
      let maxDelta = 0, holes = 0;
      for (let i = 0; i < m.w * m.h; i++) {
        if (cap.conf[i] < 0.5) holes++;
        for (let c = 0; c < 3; c++) {
          const d = Math.abs(cap.rgba[i * 4 + c] - src[i * 4 + c]);
          if (d > maxDelta) maxDelta = d;
        }
      }
      return { maxDelta, holeFrac: holes / (m.w * m.h) };
    };

    const fresh = measure();

    // ...still exact once a generated anchor exists in the scene
    gp.cam.yaw = 0.16; gp.cam._clamp();
    gp.app.needsRender = true;
    gp.requestExpand(true);
    for (let i = 0; i < 200 && gp.app.expandInFlight; i++) {
      await new Promise((r) => setTimeout(r, 100));
    }
    gp.renderer.anchors.forEach((a) => { a.weight = 1; });
    const anchors = gp.app.anchors.length;
    const withAnchor = measure();

    // ...and still exact after the GPU drops the context out from under us
    let restored = false;
    const ext = gp.renderer.gl.getExtension('WEBGL_lose_context');
    if (ext) {
      const done = new Promise((r) => {
        gp.renderer.canvas.addEventListener('webglcontextrestored', () => r(), { once: true });
      });
      ext.loseContext();
      await new Promise((r) => setTimeout(r, 60));
      ext.restoreContext();
      await Promise.race([done, new Promise((r) => setTimeout(r, 4000))]);
      await new Promise((r) => setTimeout(r, 250));
      restored = !gp.renderer.contextLost;
    }
    const afterLoss = restored ? measure() : null;
    return { fresh, withAnchor, anchors, restored, afterLoss, keptAnchors: gp.app.anchors.length };
  });

  check(pixelExact.fresh.maxDelta === 0 && pixelExact.fresh.holeFrac === 0,
    `home pose IS the photograph (max channel delta ${pixelExact.fresh.maxDelta})`);
  check(pixelExact.anchors > 0, `a generated anchor exists (${pixelExact.anchors})`);
  check(pixelExact.withAnchor.maxDelta === 0 && pixelExact.withAnchor.holeFrac === 0,
    `still exact with a generated anchor committed (delta ${pixelExact.withAnchor.maxDelta})`);
  if (pixelExact.restored) {
    check(pixelExact.afterLoss.maxDelta === 0 && pixelExact.afterLoss.holeFrac === 0,
      `still exact after a GPU context loss (delta ${pixelExact.afterLoss.maxDelta})`);
    check(pixelExact.keptAnchors === pixelExact.anchors,
      `generated anchors survive a context loss (${pixelExact.keptAnchors}/${pixelExact.anchors})`);
  } else {
    console.log('  (skipped: WEBGL_lose_context unavailable)');
  }
  await page.evaluate(() => {
    window.__gp.renderer.clearAnchors();
    window.__gp.app.anchors = [];
    window.__gp.cam.reset();
  });

  // Orbiting opens real disocclusions, and they are DETECTED as such — the
  // whole point of M9 is that a stretched silhouette wall reads as a hole
  // rather than as confident geometry.
  const orbitHoles = await page.evaluate(async () => {
    const gp = window.__gp;
    gp.cam.reset();
    gp.cam.yaw = 0.22; gp.cam.pitch = 0.05; gp.cam._clamp();
    gp.app.needsRender = true;
    await new Promise((r) => setTimeout(r, 120));
    return gp.probeHoles();
  });
  check(orbitHoles > 0.02, `orbit opens detectable holes (${orbitHoles.toFixed(3)})`);

  // ...and generating an anchor closes them
  const closed = await page.evaluate(async () => {
    const gp = window.__gp;
    const before = gp.probeHoles();
    gp.requestExpand(true);
    for (let i = 0; i < 200 && gp.app.expandInFlight; i++) {
      await new Promise((r) => setTimeout(r, 100));
    }
    gp.renderer.anchors.forEach((a) => { a.weight = 1; });
    gp.renderer.render(gp.renderState());
    return { before, after: gp.probeHoles(), anchors: gp.app.anchors.length };
  });
  check(closed.anchors === 1, `an anchor was committed (${closed.anchors})`);
  check(closed.after < closed.before * 0.4,
    `generated anchor closes the holes (${closed.before.toFixed(3)} -> ${closed.after.toFixed(3)})`);
  await page.evaluate(() => {
    window.__gp.renderer.clearAnchors();
    window.__gp.app.anchors = [];
    window.__gp.cam.reset();
  });

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

  // Regression: a photo whose padded depth width is not a multiple of 4 bytes
  // per half-float row (e.g. 431 px -> pad 43 -> 517 texels -> 1034 bytes).
  // Before UNPACK_ALIGNMENT=1 the disparity upload failed silently and the
  // whole frame read as far shell at the home pose.
  const oddWidth = await page.evaluate(async () => {
    const gp = window.__gp;
    const c = document.createElement('canvas');
    c.width = 431; c.height = 323;
    const ctx = c.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, 323);
    grad.addColorStop(0, '#4a7fb5'); grad.addColorStop(1, '#6b8e23');
    ctx.fillStyle = grad; ctx.fillRect(0, 0, 431, 323);
    ctx.fillStyle = '#aaa'; ctx.fillRect(150, 120, 120, 203);
    const blob = await new Promise((r) => c.toBlob(r, 'image/png'));
    const id0 = gp.app.buildId;
    gp.openBlob(blob);
    for (let i = 0; i < 600; i++) {
      await new Promise((r) => setTimeout(r, 100));
      if (gp.app.buildId > id0 && gp.app.meta && gp.app.meta.phase === 'final' && gp.app.setupId === gp.app.buildId) break;
    }
    const m = gp.app.meta;
    gp.cam.reset(); gp.app.needsRender = true;
    await new Promise((r) => setTimeout(r, 100));
    const st = gp.renderState();
    const cap = gp.renderer.captureAnchorFrame({
      ...st, cam: { R: st.cam.R, C: st.cam.C, K: st.cam.K }, trustBase: false, dofStrength: 0,
    }, m.w, m.h);
    let holes = 0;
    for (let i = 0; i < cap.conf.length; i++) if (cap.conf[i] < 0.5) holes++;
    return { pdw: m.pdw, holeFrac: holes / cap.conf.length };
  });
  check(oddWidth.pdw % 2 === 1 || (oddWidth.pdw * 2) % 4 !== 0,
    `odd-width photo exercises the alignment path (pdw ${oddWidth.pdw})`);
  check(oddWidth.holeFrac === 0,
    `odd-width photo is fully confident at home (hole fraction ${oddWidth.holeFrac})`);

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
  await snap('render-orbit-wide.png', (gp) => {
    gp.cam.reset(); gp.settings.aperture = 0;
    gp.cam.yaw = 0.3; gp.cam.pitch = 0.14; gp.cam._clamp();
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
