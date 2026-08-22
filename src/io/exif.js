// Camera intrinsics from the photo's own EXIF — rendering with the CAPTURE
// field of view is a realism requirement, not a nicety: a mismatched FoV
// distorts perceived depth (Cooper/Banks), and under rotation it shears
// faces. Parsing via exifr (CDN, ~26KB gz, JPEG+HEIC); everything degrades to
// a sane default when EXIF is absent or the CDN is unreachable.

const EXIFR_CDN = 'https://cdn.jsdelivr.net/npm/exifr@7.1.3/dist/full.esm.mjs';

// iOS writes garbage FocalLengthIn35mmFormat (177/311…) on some third-party
// reduced-res captures — clamp to the physically plausible photo range.
const F35_MIN = 10, F35_MAX = 250;

// DigitalZoomRatio only multiplies f35 when f35 matches a KNOWN base lens —
// many phones bake the zoom into f35 already; applying it twice halves the FoV.
const BASE_LENSES_35MM = [13, 24, 26, 28, 48, 52, 65, 77, 100, 120];

/**
 * Pure math: intrinsics from a 35mm-equivalent focal length and the DISPLAYED
 * (orientation-applied) pixel dimensions. 43.267mm = full-frame diagonal.
 */
export function intrinsicsFrom35mm(f35, w, h) {
  const fPx = f35 * Math.hypot(w, h) / 43.267;
  return {
    fPx,
    fovXDeg: 2 * Math.atan(w / (2 * fPx)) * 180 / Math.PI,
    fovYDeg: 2 * Math.atan(h / (2 * fPx)) * 180 / Math.PI,
    f35,
    source: 'exif',
  };
}

/**
 * No-EXIF fallback: fPx = max(W, H) — 53.13° across the long side, the
 * production default (Shih et al.). Never assume a fixed vertical FoV: that
 * breaks between portrait and landscape.
 */
export function defaultIntrinsics(w, h) {
  const fPx = Math.max(w, h);
  return {
    fPx,
    fovXDeg: 2 * Math.atan(w / (2 * fPx)) * 180 / Math.PI,
    fovYDeg: 2 * Math.atan(h / (2 * fPx)) * 180 / Math.PI,
    f35: null,
    source: 'default',
  };
}

/** Interpret parsed EXIF tags (pure, unit-testable). */
export function intrinsicsFromTags(tags, w, h) {
  let f35 = tags && Number(tags.FocalLengthIn35mmFormat);
  if (!Number.isFinite(f35) || f35 < F35_MIN || f35 > F35_MAX) f35 = null;
  if (f35 !== null) {
    const dzr = Number(tags.DigitalZoomRatio);
    if (Number.isFinite(dzr) && dzr > 1 &&
        BASE_LENSES_35MM.includes(Math.round(f35))) {
      f35 = Math.min(f35 * dzr, F35_MAX);
    }
    return intrinsicsFrom35mm(f35, w, h);
  }
  return defaultIntrinsics(w, h);
}

let _exifrPromise = null;
function loadExifr() {
  if (!_exifrPromise) {
    _exifrPromise = import(/* @vite-ignore */ EXIFR_CDN)
      .then((m) => m.default ?? m)
      .catch((err) => {
        console.warn('exifr unavailable (offline?):', err);
        return null;
      });
  }
  return _exifrPromise;
}

/**
 * blob + DISPLAYED dims -> intrinsics. Works on JPEG and HEIC (exifr parses
 * metadata without decoding pixels). Never throws.
 */
export async function readIntrinsics(blob, w, h) {
  try {
    const exifr = await loadExifr();
    if (!exifr) return defaultIntrinsics(w, h);
    const tags = await exifr.parse(blob, {
      pick: ['FocalLengthIn35mmFormat', 'FocalLength', 'DigitalZoomRatio', 'Model'],
    });
    return intrinsicsFromTags(tags, w, h);
  } catch (err) {
    console.warn('EXIF parse failed:', err);
    return defaultIntrinsics(w, h);
  }
}
