// Read-only photo evaluation harness.
//
// Boots the real browser app with semantic models disabled, loads an arbitrary
// local image, and measures the geometry/rendering baseline at fixed yaw angles.
// It never writes screenshots or other artifacts.
//
// Usage:
//   node e2e/run-photo-eval.mjs /absolute/path/to/photo.avif
//   node e2e/run-photo-eval.mjs /absolute/path/to/photo.avif --force-fill=10
//   node e2e/run-photo-eval.mjs /absolute/path/to/photo.avif --maxpx=200000 --quality=low
//
// --force-fill[=DEGREES] optionally exercises the classical anchor-commit path
// after all baseline views have been measured. Confidence closure from that
// check is explicitly non-semantic: it does not prove the invented pixels are
// correct. DEGREES must be one of 0, -5, 5, -10, 10, -15, 15 (default: 10).

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);

function loadPlaywright() {
  for (const candidate of [
    'playwright',
    '/opt/node22/lib/node_modules/playwright',
    '/usr/lib/node_modules/playwright',
  ]) {
    try { return require(candidate); } catch { /* try the next location */ }
  }
  throw new Error('playwright not found — install it before running photo evaluation');
}

const YAW_DEGREES = [0, -5, 5, -10, 10, -15, 15];
const CONFIDENCE_THRESHOLD = 0.5;
const ANCHOR_PAD = 0.12;
const VISIBLE_INSET = ANCHOR_PAD / (1 + 2 * ANCHOR_PAD);

function usage() {
  return [
    'Usage: node e2e/run-photo-eval.mjs /absolute/path/to/image [options]',
    'Options:',
    '  --quality=low|medium|high|ultra  Browser working-quality tier (default: low)',
    '  --maxpx=N                       Override working pixels (default: 200000)',
    '  --force-fill[=DEGREES]          Non-semantic classical fill check (default yaw: 10)',
  ].join('\n');
}

function parseArgs(argv) {
  if (!argv.length || argv.includes('--help') || argv.includes('-h')) {
    console.log(usage());
    process.exit(argv.length ? 0 : 2);
  }
  const imageArg = argv[0];
  if (!path.isAbsolute(imageArg)) throw new Error('image path must be absolute');
  const imagePath = fs.realpathSync(imageArg);
  const st = fs.statSync(imagePath);
  if (!st.isFile()) throw new Error('image path is not a file');

  let quality = 'low';
  let maxPixels = 200_000;
  let forceFillYaw = null;
  for (const arg of argv.slice(1)) {
    if (arg.startsWith('--quality=')) {
      quality = arg.slice('--quality='.length);
      if (!['low', 'medium', 'high', 'ultra'].includes(quality)) {
        throw new Error(`invalid quality: ${quality}`);
      }
    } else if (arg.startsWith('--maxpx=')) {
      maxPixels = Number(arg.slice('--maxpx='.length));
      if (!Number.isInteger(maxPixels) || maxPixels < 10_000) {
        throw new Error('--maxpx must be an integer >= 10000');
      }
    } else if (arg === '--force-fill') {
      forceFillYaw = 10;
    } else if (arg.startsWith('--force-fill=')) {
      forceFillYaw = Number(arg.slice('--force-fill='.length));
      if (!YAW_DEGREES.includes(forceFillYaw)) {
        throw new Error(`--force-fill yaw must be one of ${YAW_DEGREES.join(', ')}`);
      }
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return { imagePath, imageBytes: st.size, quality, maxPixels, forceFillYaw };
}

function safePath(root, urlPath) {
  const rel = urlPath === '/' ? 'index.html' : urlPath.replace(/^\/+/, '');
  const file = path.resolve(root, rel);
  const back = path.relative(root, file);
  return (!back.startsWith('..') && !path.isAbsolute(back)) ? file : null;
}

const opts = parseArgs(process.argv.slice(2));
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.avif': 'image/avif',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
};

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const file = safePath(root, urlPath);
  if (!file) { res.writeHead(403); res.end('forbidden'); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store, must-revalidate',
    });
    res.end(data);
  });
});

let browser = null;
try {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const { chromium } = loadPlaywright();
  browser = await chromium.launch({
    headless: true,
    args: ['--enable-unsafe-swiftshader'],
  });
  const page = await browser.newPage({ viewport: { width: 900, height: 640 } });
  const pageErrors = [];
  const consoleWarnings = [];
  const blockedExternalOrigins = new Set();
  page.on('pageerror', (err) => pageErrors.push(err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'warning') consoleWarnings.push(msg.text());
    const expectedOfflineBlock = msg.text().includes('net::ERR_FAILED')
      || msg.text().includes('net::ERR_BLOCKED_BY_CLIENT');
    if (msg.type() === 'error' && !expectedOfflineBlock) {
      pageErrors.push(msg.text());
    }
  });

  // `nomodel=1` disables depth/fill models. Blocking every non-loopback request
  // additionally makes EXIF's optional CDN import take its documented fallback,
  // so this run is deterministic and genuinely offline.
  await page.route('**/*', (route) => {
    const target = new URL(route.request().url());
    if (target.origin === origin) route.continue();
    else {
      blockedExternalOrigins.add(target.origin);
      route.abort('blockedbyclient');
    }
  });

  const query = new URLSearchParams({
    nomodel: '1',
    noauto: '1',
    nowiggle: '1',
    quality: opts.quality,
    maxpx: String(opts.maxPixels),
  });
  await page.goto(`${origin}/?${query}`);
  await page.waitForFunction(() => !!window.__gp, null, { timeout: 30_000 });
  await page.locator('#file').setInputFiles(opts.imagePath);
  await page.waitForFunction(
    () => window.__gp.app.meta && window.__gp.app.meta.phase === 'final',
    null,
    { timeout: 120_000 },
  );

  const metrics = await page.evaluate(async ({ yawDegrees, threshold, visibleInset, forceFillYaw }) => {
    const gp = window.__gp;
    const m = gp.app.meta;

    const regionStats = (cap, inset = 0) => {
      const x0 = Math.max(0, Math.round(cap.w * inset));
      const x1 = Math.min(cap.w, Math.round(cap.w * (1 - inset)));
      const y0 = Math.max(0, Math.round(cap.h * inset));
      const y1 = Math.min(cap.h, Math.round(cap.h * (1 - inset)));
      let holes = 0, pixels = 0, confSum = 0, baseSum = 0;
      let minConfidence = 1, minBaseShare = 1;
      for (let y = y0; y < y1; y++) {
        const row = y * cap.w;
        for (let x = x0; x < x1; x++) {
          const i = row + x;
          const conf = cap.conf[i];
          const base = cap.baseShare[i];
          pixels++;
          if (conf < threshold) holes++;
          confSum += conf;
          baseSum += base;
          if (conf < minConfidence) minConfidence = conf;
          if (base < minBaseShare) minBaseShare = base;
        }
      }
      return {
        pixels,
        holes,
        holeFraction: pixels ? holes / pixels : 0,
        confidence: {
          min: pixels ? minConfidence : 0,
          mean: pixels ? confSum / pixels : 0,
        },
        baseShare: {
          min: pixels ? minBaseShare : 0,
          mean: pixels ? baseSum / pixels : 0,
        },
      };
    };

    const setYaw = (degrees) => {
      gp.cam.reset();
      gp.cam.yaw = degrees * Math.PI / 180;
      gp.cam.pitch = 0;
      gp.cam._clamp();
      gp.app.needsRender = true;
    };

    const measureView = (degrees) => {
      setYaw(degrees);
      const dims = gp.renderer.genDims;
      if (!dims) throw new Error('anchor store dimensions unavailable');
      const cap = gp.renderer.captureAnchorFrame(gp.anchorState(), dims.cw, dims.ch);
      if (!cap) throw new Error(`capture failed at yaw ${degrees}`);
      return {
        yawDegrees: degrees,
        yawRadians: degrees * Math.PI / 180,
        cameraCenter: Array.from(gp.cam.center()),
        visible: regionStats(cap, visibleInset),
        paddedAnchor: {
          width: cap.w,
          height: cap.h,
          ...regionStats(cap, 0),
        },
      };
    };

    // Pixel-exact home check uses the original photo frame, not the wider anchor
    // capture, so corresponding texels can be compared directly.
    setYaw(0);
    const st = gp.renderState();
    const homeCap = gp.renderer.captureAnchorFrame({
      ...st,
      cam: { R: st.cam.R, C: st.cam.C, K: st.cam.K },
      trustBase: false,
      dofStrength: 0,
    }, m.w, m.h);
    if (!homeCap) throw new Error('home capture failed');
    const source = gp.app.imageData.data;
    let maxChannelDelta = 0, sumAbsChannelDelta = 0, differingChannels = 0;
    for (let i = 0; i < m.w * m.h; i++) {
      for (let c = 0; c < 3; c++) {
        const delta = Math.abs(homeCap.rgba[i * 4 + c] - source[i * 4 + c]);
        if (delta > maxChannelDelta) maxChannelDelta = delta;
        sumAbsChannelDelta += delta;
        if (delta !== 0) differingChannels++;
      }
    }
    const homeRegion = regionStats(homeCap, 0);
    const home = {
      width: m.w,
      height: m.h,
      pixelExact: maxChannelDelta === 0,
      maxChannelDelta,
      meanAbsChannelDelta: sumAbsChannelDelta / (m.w * m.h * 3),
      differingChannels,
      holeFraction: homeRegion.holeFraction,
      confidence: homeRegion.confidence,
      baseShare: homeRegion.baseShare,
    };

    const views = yawDegrees.map(measureView);

    const memorySummary = () => {
      const prompt = gp.pointPrompt({ maxAnchors: 4, maxPoints: 1000, stride: 4 });
      let observedPoints = 0, uncertainPoints = 0;
      for (let i = 0; i < prompt.count; i++) {
        observedPoints += prompt.observed[i] ? 1 : 0;
        uncertainPoints += prompt.uncertainty[i] > 0 ? 1 : 0;
      }
      return {
        committedEntries: gp.sceneMemory.size,
        promptSchema: prompt.schema,
        promptPoints: prompt.count,
        observedPoints,
        speculativePoints: prompt.count - observedPoints,
        uncertainPoints,
        anchors: prompt.anchors,
      };
    };

    const memoryBefore = memorySummary();

    let forcedFill = null;
    if (forceFillYaw !== null) {
      const before = measureView(forceFillYaw);
      const startedAt = performance.now();
      const started = gp.requestExpand(true);
      let timedOut = false;
      if (started) {
        for (let i = 0; i < 600 && gp.app.expandInFlight; i++) {
          await new Promise((resolve) => setTimeout(resolve, 50));
        }
        timedOut = gp.app.expandInFlight;
      }
      gp.renderer.anchors.forEach((anchor) => { anchor.weight = 1; });
      const after = measureView(forceFillYaw);
      forcedFill = {
        classification: 'non-semantic plumbing check',
        caveat: 'confidence closure does not validate the correctness of invented pixels or geometry',
        yawDegrees: forceFillYaw,
        started,
        timedOut,
        elapsedMs: performance.now() - startedAt,
        committedAnchors: gp.app.anchors.length,
        anchorStats: gp.app.anchors.map((anchor) => anchor.stats),
        memory: memorySummary(),
        before,
        after,
      };
    }

    return {
      image: {
        nativeWidth: gp.app.natW,
        nativeHeight: gp.app.natH,
        workingWidth: m.w,
        workingHeight: m.h,
        intrinsics: gp.app.intrinsics,
      },
      pipeline: {
        depthKind: m.depthKind,
        fillKind: m.fillKind,
        dSub: m.dSub,
        dMin: m.dMin,
        dMax: m.dMax,
        dFloor: m.dFloor,
      },
      home,
      views,
      memory: memoryBefore,
      forcedFill,
    };
  }, {
    yawDegrees: YAW_DEGREES,
    threshold: CONFIDENCE_THRESHOLD,
    visibleInset: VISIBLE_INSET,
    forceFillYaw: opts.forceFillYaw,
  });

  const report = {
    schemaVersion: 1,
    classification: 'offline geometry/rendering baseline; semantic models disabled',
    caveat: 'hole closure and confidence are not evidence that unseen content is correct',
    image: {
      path: opts.imagePath,
      bytes: opts.imageBytes,
      ...metrics.image,
    },
    mode: {
      offline: true,
      semanticModels: false,
      autoFill: false,
      quality: opts.quality,
      maxPixels: opts.maxPixels,
      confidenceThreshold: CONFIDENCE_THRESHOLD,
      anchorPad: ANCHOR_PAD,
      visibleInset: VISIBLE_INSET,
    },
    pipeline: metrics.pipeline,
    home: metrics.home,
    views: metrics.views,
    memory: metrics.memory,
    forcedFill: metrics.forcedFill,
    diagnostics: {
      blockedExternalOrigins: [...blockedExternalOrigins].sort(),
      pageErrors,
      consoleWarnings,
    },
  };
  console.log(JSON.stringify(report, null, 2));
  if (pageErrors.length) process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
