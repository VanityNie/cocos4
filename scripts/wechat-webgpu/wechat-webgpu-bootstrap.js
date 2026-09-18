// WebGPU bootstrap v2: staged diagnostics, no additional GPU/device creation.
var window = GameGlobal;
const gpuDiagnostic = GameGlobal.__webgpuDiagnostic = { revision: 'wx-gpu-diag-2', stage: 'entry' };
const wxEntry = typeof wx === 'undefined' ? undefined : wx;
const readWXInfo = (name) => {
    try {
        return wxEntry && typeof wxEntry[name] === 'function' ? wxEntry[name]() : {};
    } catch (error) {
        console.log('[WebGPU][environment] ' + name + ' failed: ' + String(error));
        return {};
    }
};
const baseInfo = readWXInfo('getAppBaseInfo');
const deviceInfo = readWXInfo('getDeviceInfo');
const legacyInfo = !baseInfo.version || !deviceInfo.platform ? readWXInfo('getSystemInfoSync') : {};
console.log('[WebGPU][entry] ' + JSON.stringify({
    revision: gpuDiagnostic.revision,
    platform: deviceInfo.platform || legacyInfo.platform,
    system: deviceInfo.system || legacyInfo.system,
    model: deviceInfo.model || legacyInfo.model,
    wechatVersion: baseInfo.version || legacyInfo.version,
    SDKVersion: baseInfo.SDKVersion || legacyInfo.SDKVersion,
    wxType: typeof wxEntry,
    getGPUType: wxEntry ? typeof wxEntry.getGPU : 'no wx',
    gameGlobalWXType: typeof GameGlobal.wx,
    gameGlobalGetGPUType: GameGlobal.wx ? typeof GameGlobal.wx.getGPU : 'no wx',
    globalGetGPUType: globalThis.wx ? typeof globalThis.wx.getGPU : 'no wx',
    sameGameGlobalWX: wxEntry === GameGlobal.wx,
    sameGlobalWX: wxEntry === globalThis.wx,
}));
const directGPU = GameGlobal.gpu;
const navigatorGPU = GameGlobal.navigator && GameGlobal.navigator.gpu;
const hasWXGPU = typeof wx !== 'undefined' && typeof wx.getGPU === 'function';
const hostGPU = directGPU && typeof directGPU.requestAdapter === 'function' ? directGPU : navigatorGPU;
console.log('[WebGPU] entries before adapter: wx.getGPU=' + hasWXGPU + ', GameGlobal.gpu=' + !!directGPU + ', navigator.gpu=' + !!navigatorGPU);
gpuDiagnostic.stage = 'adapter-injection';
try {
    require('./web-adapter');
} catch (error) {
    console.log('[WebGPU][adapter-injection] failed: ' + String(error));
    throw error;
}
if (hostGPU && GameGlobal.navigator && !GameGlobal.navigator.gpu) {
    GameGlobal.navigator.gpu = hostGPU;
}
const runtimeGPU = GameGlobal.navigator && GameGlobal.navigator.gpu;
console.log('[WebGPU] after adapter: GPU=' + !!runtimeGPU);
console.log('[WebGPU][adapter-result] ' + JSON.stringify({
    adapterProbe: gpuDiagnostic.probe || 'not-observed',
    source: gpuDiagnostic.source || 'none',
    requestAdapterType: runtimeGPU ? typeof runtimeGPU.requestAdapter : 'no GPU',
    globalNavigatorMatches: globalThis.navigator === GameGlobal.navigator,
}));
if (!runtimeGPU || typeof runtimeGPU.requestAdapter !== 'function') {
    const reason = gpuDiagnostic.probe === 'not-callable' ? 'WX_GET_GPU_NOT_EXPOSED' :
        gpuDiagnostic.probe === 'returned-gpu' ? 'GPU_LOST_DURING_ADAPTER' :
        gpuDiagnostic.probe === 'threw' ? 'WX_GET_GPU_THREW' :
        gpuDiagnostic.probe === 'no-usable-gpu' ? 'WX_GET_GPU_NO_USABLE_GPU' : 'ADAPTER_PROBE_NOT_OBSERVED';
    gpuDiagnostic.stage = reason;
    throw new Error('[WebGPU][' + reason + '] No requestAdapter entry after wx.getGPU/adapter: wx.getGPU=' + hasWXGPU + ', GameGlobal.gpu=' + !!directGPU + ', original navigator.gpu=' + !!navigatorGPU + ', adapted navigator.gpu=' + !!runtimeGPU + '. See [entry], [getGPU], [adapter-result] logs.');
}
gpuDiagnostic.stage = 'entry-ready';
GameGlobal.GPUTextureUsage = GameGlobal.GPUTextureUsage || {COPY_SRC:1,COPY_DST:2,TEXTURE_BINDING:4,STORAGE_BINDING:8,RENDER_ATTACHMENT:16};
GameGlobal.GPUBufferUsage = GameGlobal.GPUBufferUsage || {MAP_READ:1,MAP_WRITE:2,COPY_SRC:4,COPY_DST:8,INDEX:16,VERTEX:32,UNIFORM:64,STORAGE:128,INDIRECT:256,QUERY_RESOLVE:512};
GameGlobal.GPUShaderStage = GameGlobal.GPUShaderStage || {VERTEX:1,FRAGMENT:2,COMPUTE:4};
GameGlobal.GPUMapMode = GameGlobal.GPUMapMode || {READ:1,WRITE:2};
GameGlobal.GPUColorWrite = GameGlobal.GPUColorWrite || {RED:1,GREEN:2,BLUE:4,ALPHA:8,ALL:15};
// No context is created on the main canvas before the WebGPU device owns it.
const firstScreen = {
    start: () => Promise.resolve(),
    setProgress: p => { console.log('[WebGPU] boot progress=' + p); return Promise.resolve(); },
    end: () => { console.log('[WebGPU] application.init complete'); return Promise.resolve(); },
};
