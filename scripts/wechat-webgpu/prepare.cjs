'use strict';
const fs = require('node:fs');
const path = require('node:path');
function prepare(root, build) {
const backup = path.join(root, '.codex-backup/wechat-before-webgpu');
function write(file, text) {
    const previous = fs.readFileSync(file, 'utf8');
    if (previous === text) return;
    const dest = path.join(backup, path.relative(build, file));
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    if (!fs.existsSync(dest)) fs.writeFileSync(dest, previous);
    fs.writeFileSync(file, text);
}
function files(dir) {
    return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e =>
        e.isDirectory() ? files(path.join(dir, e.name)) : [path.join(dir, e.name)]);
}
function visit(value, fn) {
    if (!value || typeof value !== 'object') return;
    if (!Array.isArray(value)) fn(value);
    Object.values(value).forEach(v => visit(v, fn));
}
const config = JSON.parse(fs.readFileSync(path.join(build, 'project.config.json')));
if (config.compileType !== 'game') throw Error('Expected WeChat game build');
const entries = files(path.join(build, 'cocos-js'));
const engineCode = entries.filter(f => f.endsWith('.js')).map(f => fs.readFileSync(f, 'utf8')).join('\n');
if (!engineCode.includes('requestAdapter') || !engineCode.includes('WebGPUDevice')) throw Error('Enable gfx-webgpu in engine feature settings before building');
for (const name of ['glslang', 'twgsl']) {
    if (!entries.some(f => path.basename(f).startsWith(name) && f.endsWith('.wasm'))) throw Error(name + ' WASM missing; compressed/remote WASM is not supported by this integration yet');
}
if (config.plugins && Object.keys(config.plugins).length) throw Error('Separate engine plugins are not supported');
const changes = [];
let shaders = 0;
for (const f of files(build).filter(f => f.endsWith('.json'))) {
    const data = JSON.parse(fs.readFileSync(f, 'utf8'));
    visit(data, s => {
        if (typeof s.name !== 'string' || typeof s.hash !== 'number' || !Array.isArray(s.blocks)) return;
        const glsl4 = s.glsl4;
        if (!glsl4?.vert || !glsl4?.frag) throw Error(`Missing GLSL4 for ${s.name}; enable Use WebGPU in the patched WeChat platform plugin and rebuild`);
        shaders++;
    });
}
if (!shaders) throw Error('No JSON shaders found');
const gameFile = path.join(build, 'game.js');
let game = fs.readFileSync(gameFile, 'utf8');
// WeChat require resolves JS modules, not arbitrary JSON assets. Bake this flag
// at build time; settings.json is still loaded through the engine asset loader.
if (!game.includes('const webgpuEnabled = true; // ENGINE_WEBGPU_BUILD_FLAG')) {
    throw Error('WeChat template did not enable WebGPU; restart Creator with the patched platform plugin and enable useWebGPU');
}
if (!game.includes('[WebGPU] active backend=')) {
    const start = 'firstScreen.end().then(() => application.start())';
    if (!game.includes(start)) throw Error('Unrecognized application start chain');
    game = game.replace(start, `firstScreen.end().then(() => application.start()).then(() => System.import('cc')).then(cc => {
        const device = cc.director.root.device;
        console.log('[WebGPU] active backend=' + device.gfxAPI + ', expected=' + cc.gfx.API.WEBGPU);
        if (device.gfxAPI !== cc.gfx.API.WEBGPU) throw new Error('[WebGPU] Unexpected rendering backend');
        cc.director.once(cc.Director.EVENT_AFTER_SCENE_LAUNCH, () => console.log('[WebGPU] scene started: ' + cc.director.getScene()?.name));
    })`);
}
game = game.replace("console.log('[WebGPU] scene started: ' + cc.director.getScene().name);", "cc.director.once(cc.Director.EVENT_AFTER_SCENE_LAUNCH, () => console.log('[WebGPU] scene started: ' + cc.director.getScene()?.name));");
changes.push([gameFile, game]);
const settingsFile = path.join(build, 'src/settings.json');
const settings = JSON.parse(fs.readFileSync(settingsFile));
settings.rendering.renderMode = 4;
changes.push([settingsFile, JSON.stringify(settings)]);
// Mini Game CommonJS modules may shadow browser names with undefined arguments.
const aliases = 'var window=GameGlobal, self=GameGlobal, global=GameGlobal, document=GameGlobal.document, navigator=GameGlobal.navigator, DOMParser=GameGlobal.DOMParser;\n';
const flags = 'var GPUTextureUsage=GameGlobal.GPUTextureUsage, GPUBufferUsage=GameGlobal.GPUBufferUsage, GPUShaderStage=GameGlobal.GPUShaderStage, GPUMapMode=GameGlobal.GPUMapMode, GPUColorWrite=GameGlobal.GPUColorWrite;\n';
for (const f of files(path.join(build, 'cocos-js')).filter(f => f.endsWith('.js')).concat([path.join(build, 'engine-adapter.js')])) {
    let text = fs.readFileSync(f, 'utf8');
    if (!text.startsWith('// WebGPU module aliases v1')) {
        text = '// WebGPU module aliases v1\n' + aliases + flags + text;
    }
    changes.push([f, text]);
}
const adapterFile = path.join(build, 'web-adapter.js');
let adapter = fs.readFileSync(adapterFile, 'utf8');
if (!adapter.startsWith('var window=GameGlobal;')) adapter = 'var window=GameGlobal;\n' + adapter;
changes.push([adapterFile, adapter]);
for (const [f, text] of changes) write(f, text);
console.log(`WebGPU enabled: ${shaders} shaders checked; original files backed up at ${backup}`);
return { shaders };
}
module.exports = { prepare };
