// One place that answers "what can this environment actually do?" (M10).
//
// Three separate sites used to probe `navigator.gpu` independently — config.js
// for the quality tier, depth-ai.js for the transformers.js device, and
// inpaint-ai.js for the onnxruntime execution provider. That third one is the
// easy one to miss, and missing it in a native build leaves an
// onnxruntime-web WebGPU path alive inside a webview that has no WebGPU.
//
// This matters now because the app is becoming a native Mac app, and WKWebView
// does NOT expose WebGPU — `navigator.gpu` is simply absent there, and the
// only lever is a private WebKit call that is App-Store-rejectable. That is
// fine, because in the native build inference leaves the webview entirely, but
// it has to be a decision made once, in one place, not rediscovered three
// times.
//
// Pure and dependency-free so both the worker and the main thread can use it.

/** Is a real, usable WebGPU adapter available? Cached; never throws. */
let _adapterProbe = null;
export function hasWebGpuAdapter() {
  if (_adapterProbe) return _adapterProbe;
  _adapterProbe = (async () => {
    // presence is NOT enough: headless and blocklisted environments expose the
    // object with no adapter behind it, and a FAILED webgpu attempt can poison
    // the same-context wasm fallback (see SCRATCHPAD "platform gotchas")
    if (!hasWebGpuObject()) return false;
    try { return !!(await navigator.gpu.requestAdapter()); } catch { return false; }
  })();
  return _adapterProbe;
}

/** Cheap synchronous presence check — for choosing a default, never a backend. */
export function hasWebGpuObject() {
  return !!(globalThis.navigator && globalThis.navigator.gpu);
}

/**
 * Where the heavy models run. 'web' = in this process (transformers.js +
 * onnxruntime-web). 'native' = a local sidecar owns them, injected by the
 * desktop shell before any module loads.
 */
export function inferenceBackend() {
  return globalThis.__GP_NATIVE__ ? 'native' : 'web';
}

/** True when this build must not attempt any in-process WebGPU inference. */
export function webGpuInferenceBlocked() {
  return inferenceBackend() === 'native' || !hasWebGpuObject();
}
