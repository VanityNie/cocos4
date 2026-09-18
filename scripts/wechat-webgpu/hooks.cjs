'use strict';
const path = require('node:path');
const { prepare } = require('./prepare.cjs');
exports.throwError = true;
function enabled(options) { return options.packages?.wechatgame?.useWebGPU === true; }
exports.onBeforeBuild = function (options) {
    if (!enabled(options)) return;
    if (options.md5Cache) throw Error('[WebGPU] Disable md5Cache: shader restoration runs after asset serialization.');
    if (options.packages?.wechatgame?.separateEngine) throw Error('[WebGPU] Separate engine plugin is unsupported.');
    if (options.wasmCompressionMode) throw Error('[WebGPU] Disable WASM compression for this experimental integration.');
};
exports.onAfterBuild = function (options, result) {
    if (!enabled(options)) return;
    exports.onBeforeBuild(options);
    const engineRoot = path.resolve(__dirname, '../..');
    const selectedEngine = options.engineInfo?.typescript?.path;
    if (selectedEngine && path.resolve(selectedEngine).toLowerCase() !== engineRoot.toLowerCase()) {
        throw Error('[WebGPU] Build is using a different engine: ' + selectedEngine);
    }
    const project = Editor.Project.path;
    if (!result?.dest) throw Error('[WebGPU] Build output directory unavailable');
    console.log('[WebGPU] prepared:', prepare(project, result.dest));
};
