// Android WeChat exposes the GPU through wx.getGPU(), not necessarily a global.
// Preserve existing host entries as a fallback for the simulator.

/**
 * 探测微信运行时提供的 WebGPU 入口（navigator.gpu 形态，含 requestAdapter 等）。
 * @returns {object | undefined}
 */
export function getGPU () {
    const diagnostic = typeof GameGlobal !== 'undefined'
        ? (GameGlobal.__webgpuDiagnostic || (GameGlobal.__webgpuDiagnostic = {})) : {};
    diagnostic.probe = 'not-callable';
    console.log('[WebGPU][getGPU] wx.getGPU type=' + (typeof wx === 'undefined' ? 'no wx' : typeof wx.getGPU));
    if (typeof wx !== 'undefined' && typeof wx.getGPU === 'function') {
        try {
            diagnostic.probe = 'calling';
            console.warn('[WebGPU][getGPU] CALL wx.getGPU()', new Error('wx.getGPU caller').stack);
            const gpu = wx.getGPU();
            console.warn('[WebGPU][getGPU] RETURN wx.getGPU(): ' + (gpu ? typeof gpu.requestAdapter : 'no GPU'));
            diagnostic.probe = 'no-usable-gpu';
            console.log('[WebGPU][getGPU] ' + JSON.stringify({
                returnedType: typeof gpu, exists: !!gpu,
                requestAdapterType: gpu ? typeof gpu.requestAdapter : 'no GPU',
            }));
            if (gpu && typeof gpu.requestAdapter === 'function') {
                diagnostic.probe = 'returned-gpu';
                diagnostic.source = 'wx.getGPU';
                console.log('[WebGPU] GPU source=wx.getGPU()');
                return gpu;
            }
            console.warn('[WebGPU] wx.getGPU() returned no usable GPU');
        } catch (error) {
            diagnostic.probe = 'threw';
            console.warn('[WebGPU] wx.getGPU() failed', error);
        }
    }
    if (typeof GameGlobal !== 'undefined' && GameGlobal.gpu && typeof GameGlobal.gpu.requestAdapter === 'function') {
        diagnostic.source = 'GameGlobal.gpu';
        return GameGlobal.gpu;
    }

    if (typeof GameGlobal !== 'undefined' && GameGlobal.navigator && GameGlobal.navigator.gpu
        && typeof GameGlobal.navigator.gpu.requestAdapter === 'function') {
        diagnostic.source = 'GameGlobal.navigator.gpu';
        return GameGlobal.navigator.gpu;
    }

    return undefined;
}
