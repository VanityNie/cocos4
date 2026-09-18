'use strict';
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const ts = require('typescript');
const root = path.resolve(__dirname, '../..');
const decoderSource = fs.readFileSync(path.join(__dirname, 'platform-source/platforms/wechat/wrapper/builtin/TextDecoder.js'), 'utf8');
const sandbox = { GameGlobal: {} };
vm.createContext(sandbox);
vm.runInContext(decoderSource, sandbox);
const Decoder = sandbox.TextDecoder;
assert.equal(Decoder, sandbox.GameGlobal.TextDecoder);
const nativeSandbox = { TextDecoder, GameGlobal: {} };
vm.runInNewContext(decoderSource, nativeSandbox);
assert.equal(nativeSandbox.TextDecoder, TextDecoder);
assert.equal(nativeSandbox.GameGlobal.TextDecoder, TextDecoder);
assert.throws(() => new Decoder('utf-16'), /UTF-8/);
assert.throws(() => new Decoder().decode('text'), /BufferSource/);
let seed = 73;
function random() { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; }
const vectors = [
  new TextEncoder().encode('Shader 中文 😀\n@fragment fn main() {}'),
  Uint8Array.from([0xef, 0xbb, 0xbf, 65, 0xef, 0xbb, 0xbf]),
  Uint8Array.from([0xed, 0xa0, 0x80, 0xf4, 0x90, 0x80, 0x80, 0xe2, 0x82]),
];
for (let i = 0; i < 1000; i++) vectors.push(Uint8Array.from({ length: random() % 48 }, () => random() >>> 24));
for (const bytes of vectors) {
  for (const ignoreBOM of [false, true]) {
    assert.equal(new Decoder('utf8', { ignoreBOM }).decode(bytes), new TextDecoder('utf8', { ignoreBOM }).decode(bytes));
    const ours = new Decoder('utf8', { ignoreBOM });
    const native = new TextDecoder('utf8', { ignoreBOM });
    for (let i = 0; i < bytes.length; i++) {
      const piece = bytes.subarray(i, i + 1);
      assert.equal(ours.decode(piece, { stream: true }), native.decode(piece, { stream: true }));
    }
    assert.equal(ours.decode(), native.decode());
  }
}
const backing = Uint8Array.from([0, 0xe4, 0xb8, 0xad, 0]);
assert.equal(new Decoder().decode(new DataView(backing.buffer, 1, 3)), '中');
assert.throws(() => new Decoder('utf8', { fatal: true }).decode(Uint8Array.from([0xff])), /Invalid UTF-8/);

const file = path.join(root, 'cocos/gfx/webgpu/webgpu-device.ts');
const source = fs.readFileSync(file, 'utf8');
const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true);
const cls = ast.statements.find(n => ts.isClassDeclaration(n) && n.name.text === 'WebGPUDevice');
const method = cls.members.find(n => n.name && n.name.getText(ast) === 'initDevice');
const harnessSource = ts.transpileModule('class Harness { ' + method.getText(ast) + ' }; this.Harness=Harness;', {
  compilerOptions: { target: ts.ScriptTarget.ES2020 },
}).outputText;

async function checkDevice({ configureFails = false, noFormatMethod = false, wasmFails = false } = {}) {
  const events = [];
  let configured = false;
  const nativeDevice = {
    limits: {}, lost: new Promise(() => {}),
    addEventListener(name) { assert.equal(name, 'uncapturederror'); events.push('listener'); },
  };
  const adapter = { limits: { maxVertexAttributes: 16, maxSampledTexturesPerShaderStage: 16 },
    features: new Set(), requestDevice: async () => { events.push('device'); return nativeDevice; } };
  const gpu = { requestAdapter: async () => { events.push('adapter'); return adapter; } };
  if (!noFormatMethod) gpu.getPreferredCanvasFormat = () => 'rgba8unorm';
  const canvas = { getContext(type) {
    assert.equal(type, 'webgpu'); events.push('context');
    return { configure(config) {
      assert.equal(config.device, nativeDevice);
      assert.equal(config.format, noFormatMethod ? 'bgra8unorm' : 'rgba8unorm');
      events.push('configure');
      if (configureFails) throw Error('configure failed');
      configured = true;
    } };
  } };
  const constants = new Proxy({}, { get: () => 1 });
  const context = { navigator: { gpu }, Device: { canvas }, console: { log(){},warn(){},error(){} },
    warn(){}, debug(){}, API: constants, Feature: constants, QueueType: constants,
    TextureType: constants, TextureUsageBit: constants, Format: constants, TextureFlagBit: constants,
    SampleCount: constants, BufferUsageBit: constants, MemoryUsageBit: constants, BufferFlagBit: constants,
    WGPUFormatToGFXFormat: f => f, webGPU: {},
    loadWebGPUWasmModule: async () => { assert.equal(configured, true); events.push('wasm'); if (wasmFails) throw Error('wasm failed'); },
  };
  for (const name of ['Size','QueueInfo','CommandBufferInfo','TextureInfo','BufferInfo','SamplerInfo']) context[name] = class {};
  vm.runInNewContext(harnessSource, context);
  const instance = new context.Harness();
  Object.assign(instance, { _caps: {}, _features: [], defaultResource: {}, initFormatFeatures(){}, getFormatFeatures(){return false;} });
  for (const name of ['createQueue','createCommandBuffer','createTexture','createBuffer','getSampler','_createDefaultDescSet']) {
    instance[name] = () => { assert.equal(configured, true); events.push(name); return {}; };
  }
  const info = { bindingMappingInfo: { setIndices: [0], maxBlockCounts: [1], maxSamplerTextureCounts: [1] } };
  if (configureFails || wasmFails) {
    await assert.rejects(instance.initDevice(info), configureFails ? /configure failed/ : /wasm failed/);
    assert.equal(events.includes('createTexture'), false);
    if (configureFails) assert.equal(events.includes('wasm'), false);
  } else {
    assert.equal(await instance.initDevice(info), true);
    assert.deepEqual(events.slice(0, 6), ['adapter', 'device', 'listener', 'context', 'configure', 'wasm']);
    assert.ok(events.includes('createTexture'));
  }
}

(async () => {
  await checkDevice();
  await checkDevice({ noFormatMethod: true });
  await checkDevice({ configureFails: true });
  await checkDevice({ wasmFails: true });
  console.log('PASS: UTF-8/native parity, streaming/BOM/offset/fatal, native preservation, real initDevice ordering and failure boundaries');
})().catch(error => { console.error(error); process.exitCode = 1; });
