/* Tests the actual TS compute implementation with a thin GFX allocation adapter.
 * node --test scripts/test-webgpu-morph.cjs
 * Optional real GPU checks: set MORPH_WEBGPU_MODULE to a node-webgpu index.js path.
 * No package installation or source generation is performed by this test.
 */
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { pathToFileURL } = require('node:url');
const ts = require('typescript');
// Keep a single Dawn instance alive across tests; collecting/recreating it can unload native state.
let testGPU;

function loadEngine(globals) {
    const filename = path.join(__dirname, '../cocos/gfx/webgpu/webgpu-morph-compute.ts');
    const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
        compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS },
    }).outputText;
    const exports = {};
    // Only the GFX construction surface is adapted; shader generation, packing,
    // dirty tracking, resource ownership and encoding run from the engine source.
    const define = {
        API: { WEBGPU: 1 }, Format: { RGBA32F: 1 }, TextureType: { TEX2D: 1 },
        TextureUsageBit: { STORAGE: 1, SAMPLED: 2 },
        TextureInfo: class {
            constructor(type, usage, format, width, height) {
                Object.assign(this, { type, usage, format, width, height });
            }
        },
    };
    vm.runInNewContext(code, {
        exports, ...globals, require: (name) => {
            assert.equal(name, '../base/define');
            return define;
        },
    }, { filename });
    return exports;
}

function layersFor(mask, vertices, targets) {
    return [0, 1, 2].map(attribute => mask & (1 << attribute)
        ? Array.from({ length: targets }, (_, t) => Float32Array.from({ length: vertices * 3 },
            (_, scalar) => Math.sin(scalar * 0.1 + t * 0.7 + attribute) * 0.03)) : null);
}

function fakeDevice() {
    const writes = [], textures = [], shaders = [], passes = [], buffers = [];
    let submits = 0;
    const device = {
        limits: { maxTextureDimension2D: 8192, maxStorageBufferBindingSize: 1 << 27,
            maxBufferSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
        queue: {
            submit: () => { ++submits; },
            writeTexture: () => {},
            writeBuffer: (buffer, offset, data, dataOffset = 0, size) => {
                const bytes = ArrayBuffer.isView(data)
                    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
                writes.push({ buffer, bytes: bytes.slice(dataOffset, size === undefined ? undefined : dataOffset + size) });
            },
        },
        createBuffer: info => { const buffer = { ...info, destroy() { this.destroyed = true; } }; buffers.push(buffer); return buffer; },
        createTexture: info => {
            const texture = { ...info, createView: () => ({}), destroy() { this.destroyed = true; } };
            textures.push(texture);
            return texture;
        },
        createBindGroupLayout: info => info,
        createPipelineLayout: info => info,
        createShaderModule: info => { shaders.push(info.code); return info; },
        createComputePipeline: info => ({ ...info, getBindGroupLayout: () => ({}) }),
        createBindGroup: info => info,
    };
    const gfx = {
        gfxAPI: 1, nativeDevice: device, memoryStatus: { bufferSize: 0, textureSize: 0 },
        createTexture: info => {
            const gpuTexture = device.createTexture(info);
            gfx.memoryStatus.textureSize += info.width * info.height * 16;
            return { gpuTexture: { gpuTexture }, destroy: () => { gpuTexture.destroy(); gfx.memoryStatus.textureSize -= info.width * info.height * 16; } };
        },
    };
    const encoder = { beginComputePass: () => {
        const pass = { pipelineBinds: 0, dispatches: 0, setPipeline() { ++this.pipelineBinds; },
            groups: [], setBindGroup() {}, dispatchWorkgroups(...groups) { ++this.dispatches; this.groups.push(groups); }, end() {} };
        passes.push(pass);
        return pass;
    } };
    return { device, gfx, encoder, writes, textures, shaders, passes, buffers, submits: () => submits };
}

const fakeGlobals = {
    GPUShaderStage: { COMPUTE: 4 }, GPUTextureUsage: { TEXTURE_BINDING: 4, COPY_DST: 2 },
    GPUBufferUsage: { STORAGE: 128, COPY_DST: 8, UNIFORM: 64 },
};

test('real GPU: full pages, multidimensional dispatch, dirty subsets and recycled atlas slots',
    { skip: !process.env.MORPH_WEBGPU_MODULE, timeout: 120000 }, async () => {
        const { create, globals } = await import(pathToFileURL(process.env.MORPH_WEBGPU_MODULE).href);
        const gpu = testGPU || (testGPU = create([]));
        const adapter = await gpu.requestAdapter();
        const device = await adapter.requestDevice();
        const errors = [];
        device.addEventListener('uncapturederror', e => errors.push(e.error.message));
        const engine = loadEngine(globals);
        let checked = 0;
        try {
            for (const limited of [false, true]) {
                const native = limited ? new Proxy(device, { get(target, key) {
                    if (key === 'limits') return { maxTextureDimension2D: 128, maxComputeWorkgroupsPerDimension: 4,
                        maxBufferSize: device.limits.maxBufferSize, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize };
                    const value = Reflect.get(target, key, target);
                    return typeof value === 'function' ? value.bind(target) : value;
                } }) : device;
                const gfx = { gfxAPI: 1, nativeDevice: native, memoryStatus: { bufferSize: 0, textureSize: 0 },
                    createTexture(info) {
                        const texture = device.createTexture({ size: [info.width, info.height], format: 'rgba32float',
                            usage: globals.GPUTextureUsage.STORAGE_BINDING | globals.GPUTextureUsage.TEXTURE_BINDING | globals.GPUTextureUsage.COPY_SRC });
                        const bytes = info.width * info.height * 16;
                        gfx.memoryStatus.textureSize += bytes;
                        return { gpuTexture: { gpuTexture: texture }, destroy() { texture.destroy(); gfx.memoryStatus.textureSize -= bytes; } };
                    } };
                const vertices = limited ? 257 : 129, targets = 8;
                const layers = layersFor(limited ? 7 : 1, vertices, targets);
                const owner = new engine.WebGPUMorphCompute(gfx, vertices, targets, layers);
                const count = limited ? 9 : 65;
                const entries = Array.from({ length: count }, (_, i) => {
                    const instance = owner.createInstance();
                    const weights = Array.from({ length: targets }, (_, t) => Math.sin(i + t) * 0.2);
                    instance.setWeights(weights);
                    return { instance, weights };
                });
                try {
                    for (let iteration = 0; iteration < 3; ++iteration) {
                        if (iteration === 1) {
                            const oldOffset = entries[0].instance.outputRowOffset;
                            entries[0].instance.destroy();
                            entries[0] = { instance: owner.createInstance(), weights: Array(targets).fill(0) };
                            assert.equal(entries[0].instance.outputRowOffset, oldOffset);
                            entries[2].weights = Array(targets).fill(0);
                            entries[2].instance.setWeights(entries[2].weights);
                            entries[count - 1].weights = [0, -0.5, 0, 0, 0, 0, 0, 0];
                            entries[count - 1].instance.setWeights(entries[count - 1].weights);
                        }
                        const encoder = device.createCommandEncoder();
                        const dispatches = engine.flushWebGPUMorphComputes(native, encoder);
                        assert.equal(dispatches, iteration === 0 ? Math.ceil(count / owner.batchCapacity) : iteration === 1 ? 2 : 0);
                        const rowBytes = Math.ceil(owner.outputWidth * 16 / 256) * 256;
                        const reads = [];
                        for (const { instance, weights } of entries) for (const attribute of owner.attributes) {
                            const readback = device.createBuffer({ size: rowBytes * owner.outputHeight,
                                usage: globals.GPUBufferUsage.COPY_DST | globals.GPUBufferUsage.MAP_READ });
                            encoder.copyTextureToBuffer({ texture: instance.outputs[attribute].gpuTexture.gpuTexture,
                                origin: [0, instance.outputRowOffset] }, { buffer: readback, bytesPerRow: rowBytes },
                            [owner.outputWidth, owner.outputHeight]);
                            reads.push({ readback, weights, attribute });
                        }
                        device.queue.submit([encoder.finish()]);
                        for (const { readback, weights, attribute } of reads) {
                            await readback.mapAsync(globals.GPUMapMode.READ);
                            const actual = new Float32Array(readback.getMappedRange());
                            for (let v = 0; v < vertices; ++v) for (let c = 0; c < 3; ++c) {
                                let expected = 0;
                                for (let t = 0; t < targets; ++t) expected += layers[attribute][t][v * 3 + c] * Math.fround(weights[t]);
                                const index = Math.floor(v / owner.outputWidth) * rowBytes / 4 + v % owner.outputWidth * 4 + c;
                                assert.ok(Math.abs(actual[index] - expected) < 2e-6,
                                    `limited=${limited} iteration=${iteration} vertex=${v} attribute=${attribute}`);
                                ++checked;
                            }
                            readback.unmap(); readback.destroy();
                        }
                    }
                } finally { owner.destroy(); }
                assert.deepEqual(gfx.memoryStatus, { bufferSize: 0, textureSize: 0 });
            }
            await device.queue.onSubmittedWorkDone();
            assert.deepEqual(errors, []);
            console.log(JSON.stringify({ scope: 'real-GPU-batch-paging-and-slot-reuse', checked, errors }));
        } finally { device.destroy(); }
    });

test('sparse/dense updates preserve weights, deduplicate dirty work and never submit independently', () => {
    const engine = loadEngine(fakeGlobals), f = fakeDevice();
    const owner = new engine.WebGPUMorphCompute(f.gfx, 129, 8, layersFor(1, 129, 8));
    const a = owner.createInstance(), b = owner.createInstance();
    assert.equal(f.textures.length, 2); // one input, one shared output atlas
    assert.equal(a.outputs[0], b.outputs[0]);
    assert.notEqual(a.outputRowOffset, b.outputRowOffset);
    assert.equal(owner.texture.size[2], 1);
    a.setWeights([0, 0.5, 0, 0, -0.25, 0, 0, 0]);
    a.setWeights([0, 0.5, 0, 0, -0.25, 0, 0, 0]);
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 1);
    assert.deepEqual(f.passes[0].groups, [[3, 1, 2]]);
    assert.equal(f.passes[0].pipelineBinds, 1);
    assert.equal(f.submits(), 0);
    const packed = f.writes[1].bytes;
    const u32 = new Uint32Array(packed.buffer), f32 = new Float32Array(packed.buffer);
    assert.deepEqual([u32[0], u32[1], u32[2], u32[3], u32[5]], [0, 2, 1, 1, 4]);
    assert.deepEqual([f32[4], f32[6]], [0.5, -0.25]);
    assert.equal(packed.byteLength, owner.weightsSize * 2);
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 0);
    a.setWeights(Array(8).fill(1));
    engine.flushWebGPUMorphComputes(f.device, f.encoder);
    const dense = f.writes.at(-1).bytes;
    assert.deepEqual(Array.from(new Uint32Array(dense.buffer).slice(0, 3)), [0, 8, 0]);
    assert.equal(dense.byteLength, 44);
    a.setWeights(Array(8).fill(0));
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 1);
    assert.equal(new Uint32Array(f.writes.at(-1).bytes.buffer)[1], 0); // zero count clears old results
    a.setWeights(Array(8).fill(0));
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 0);
    b.setWeights(Array(8).fill(1));
    b.destroy();
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 0);
    owner.destroy();
    assert.ok(f.textures.every(texture => texture.destroyed));
    assert.ok(f.buffers.every(buffer => buffer.destroyed));
    assert.deepEqual(f.gfx.memoryStatus, { bufferSize: 0, textureSize: 0 });
});

test('all attribute masks allocate only present semantics and share compatible pipelines', () => {
    const engine = loadEngine(fakeGlobals), f = fakeDevice();
    for (let mask = 1; mask < 8; ++mask) {
        const layers = layersFor(mask, 3, 1);
        const owner = new engine.WebGPUMorphCompute(f.gfx, 3, 1, layers);
        const other = new engine.WebGPUMorphCompute(f.gfx, 3, 1, layers);
        assert.equal(owner.pipeline, other.pipeline);
        const instance = owner.createInstance();
        assert.equal(instance.outputs.filter(Boolean).length, layers.filter(Boolean).length);
        assert.equal(owner.texture.size[2], layers.filter(Boolean).length);
        for (let attribute = 0; attribute < 3; ++attribute) {
            assert.equal(instance.outputs[attribute] !== null, Boolean(mask & (1 << attribute)));
        }
        owner.destroy(); other.destroy();
    }
    assert.equal(f.shaders.length, 7);
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 0);
});

test('storage size limit includes weight header; unchanged f32 values do not dispatch', () => {
    const engine = loadEngine(fakeGlobals), f = fakeDevice();
    f.device.limits.maxStorageBufferBindingSize = 16;
    assert.throws(() => new engine.WebGPUMorphCompute(f.gfx, 3, 3, layersFor(1, 3, 3)), /limits/);
    const owner = new engine.WebGPUMorphCompute(f.gfx, 3, 1, layersFor(1, 3, 1));
    const instance = owner.createInstance();
    instance.setWeights([1]);
    engine.flushWebGPUMorphComputes(f.device, f.encoder);
    instance.setWeights([1 + Number.EPSILON]);
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 0);
    owner.destroy();
});

test('pages compact only dirty slots, reuse freed slots, and release empty pages', () => {
    const engine = loadEngine(fakeGlobals), f = fakeDevice();
    const owner = new engine.WebGPUMorphCompute(f.gfx, 129, 8, layersFor(1, 129, 8));
    const instances = Array.from({ length: 130 }, () => owner.createInstance());
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 3);
    assert.deepEqual(f.passes[0].groups, [[3, 1, 64], [3, 1, 64], [3, 1, 2]]);
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 0);
    const shared = instances[0].outputs[0];
    instances[0].setWeights(Array(8).fill(0.25));
    instances[63].setWeights(Array(8).fill(0.5));
    instances[64].setWeights(Array(8).fill(0.75));
    instances[63].destroy(); // pending destruction must not dispatch into a freed slot
    const replacement = owner.createInstance();
    assert.equal(replacement.slot, 63);
    assert.equal(replacement.outputs[0], shared);
    const before = f.writes.length;
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 2);
    assert.equal(f.writes.length - before, 2); // one upload per dirty page
    const records = new Uint32Array(f.writes[before].bytes.buffer);
    assert.equal(records[0], 0);
    assert.equal(records[owner.weightsSize / 4], 63);
    assert.equal(records[owner.weightsSize / 4 + 1], 0); // clear recycled slot
    assert.deepEqual(f.passes[1].groups, [[3, 1, 2], [3, 1, 1]]);
    for (let i = 64; i < 128; ++i) instances[i].destroy();
    assert.ok(instances[64].outputs[0].gpuTexture.gpuTexture.destroyed);
    assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 0);
    replacement.destroy(); owner.destroy(); owner.destroy();
    assert.ok(f.buffers.every(buffer => buffer.destroyed));
    assert.ok(f.textures.every(texture => texture.destroyed));
    assert.deepEqual(f.gfx.memoryStatus, { bufferSize: 0, textureSize: 0 });
    assert.throws(() => replacement.setWeights(Array(8).fill(1)), /destroyed/);
    assert.throws(() => owner.createInstance(), /destroyed/);
});

test('capacity obeys atlas, storage, dispatch and reserved-memory limits; single is an A/B baseline', () => {
    const engine = loadEngine(fakeGlobals);
    for (const limit of ['texture', 'storage', 'dispatch', 'memory', 'single']) {
        const f = fakeDevice();
        let vertices = 129, targets = 1;
        if (limit === 'texture') f.device.limits.maxTextureDimension2D = 16;
        if (limit === 'storage') f.device.limits.maxStorageBufferBindingSize = 32;
        if (limit === 'dispatch') f.device.limits.maxComputeWorkgroupsPerDimension = 2;
        if (limit === 'memory') vertices = 20000;
        const owner = new engine.WebGPUMorphCompute(f.gfx, vertices, targets, layersFor(7, vertices, targets), limit === 'single' ? 1 : 64);
        const expected = { texture: 1, storage: 2, dispatch: 2, memory: 7, single: 1 }[limit];
        assert.equal(owner.batchCapacity, expected, limit);
        const instances = Array.from({ length: expected + 1 }, () => owner.createInstance());
        assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), 2);
        assert.equal(instances[expected].outputRowOffset, 0);
        assert.notEqual(instances[0].outputs[0], instances[expected].outputs[0]);
        assert.ok(instances[0].outputHeight <= f.device.limits.maxTextureDimension2D);
        owner.destroy();
        assert.deepEqual(f.gfx.memoryStatus, { bufferSize: 0, textureSize: 0 });
    }
});

test('demo mesh distribution: 192 actors produce 228 single dispatches or 7 batches', () => {
    const engine = loadEngine(fakeGlobals);
    const vertices = [656, 244, 912, 781, 3929, 5634, 301];
    const counts = [37, 38, 37, 36, 8, 36, 36];
    for (const capacity of [1, 64]) {
        const f = fakeDevice();
        const owners = vertices.map((v, i) => {
            const owner = new engine.WebGPUMorphCompute(f.gfx, v, 8, layersFor(1, v, 8), capacity);
            for (let n = 0; n < counts[i]; ++n) owner.createInstance().setWeights(Array(8).fill(0.5));
            return owner;
        });
        assert.equal(engine.flushWebGPUMorphComputes(f.device, f.encoder), capacity === 1 ? 228 : 7);
        for (const owner of owners) owner.destroy();
        assert.deepEqual(f.gfx.memoryStatus, { bufferSize: 0, textureSize: 0 });
    }
});

test('real GPU: every attribute mask, CS-to-VS same-submit reads, sparse/dense/zero and independent instances',
    { skip: !process.env.MORPH_WEBGPU_MODULE, timeout: 120000 }, async () => {
        const { create, globals } = await import(pathToFileURL(process.env.MORPH_WEBGPU_MODULE).href);
        const gpu = testGPU || (testGPU = create([]));
        const adapter = await gpu.requestAdapter();
        assert.ok(adapter, 'No real WebGPU adapter');
        const device = await adapter.requestDevice();
        const errors = [];
        device.addEventListener('uncapturederror', event => errors.push(event.error.message));
        const engine = loadEngine(globals);
        const gfx = {
            gfxAPI: 1, nativeDevice: device, memoryStatus: { bufferSize: 0, textureSize: 0 },
            createTexture: info => {
                const texture = device.createTexture({ size: [info.width, info.height], format: 'rgba32float',
                    usage: globals.GPUTextureUsage.STORAGE_BINDING | globals.GPUTextureUsage.TEXTURE_BINDING });
                return { gpuTexture: { gpuTexture: texture }, destroy: () => texture.destroy() };
            },
        };
        const vertices = 129, targets = 65;
        const module = device.createShaderModule({ code: `
            @group(0) @binding(0) var result: texture_2d<f32>;
            struct Vertex { @builtin(position) p: vec4<f32>, @location(0) @interpolate(flat) d: vec4<f32> }
            @vertex fn vs(@builtin(vertex_index) id: u32, @builtin(instance_index) rowOffset: u32) -> Vertex {
                let width = textureDimensions(result).x;
                var v: Vertex;
                v.p = vec4<f32>((f32(id) + 0.5) / ${vertices}.0 * 2.0 - 1.0, 0.0, 0.0, 1.0);
                v.d = textureLoad(result, vec2<i32>(i32(id % width), i32(id / width + rowOffset)), 0);
                return v;
            }
            @fragment fn fs(v: Vertex) -> @location(0) vec4<f32> { return v.d; }
        ` });
        const layout = device.createBindGroupLayout({ entries: [{ binding: 0,
            visibility: globals.GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } }] });
        const pipeline = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba32float' }] },
            primitive: { topology: 'point-list' } });
        const target = device.createTexture({ size: [vertices, 1], format: 'rgba32float',
            usage: globals.GPUTextureUsage.RENDER_ATTACHMENT | globals.GPUTextureUsage.COPY_SRC });
        const rowBytes = Math.ceil(vertices * 16 / 256) * 256;
        let maxError = 0, checked = 0;
        try {
            for (const capacity of [1, 2, 64]) for (let mask = 1; mask < 8; ++mask) {
                const layers = layersFor(mask, vertices, targets);
                const owner = new engine.WebGPUMorphCompute(gfx, vertices, targets, layers, capacity);
                const a = owner.createInstance(), b = owner.createInstance();
                const stableWeights = Array(targets).fill(0); stableWeights[3] = -0.75;
                b.setWeights(stableWeights);
                const sparse = Array(targets).fill(0); sparse[1] = 0.5; sparse[64] = -0.25;
                const dense = Array.from({ length: targets }, (_, i) => Math.cos(i) * 0.1);
                const allZero = Array(targets).fill(0);
                for (const [iteration, weights] of [sparse, dense, allZero, allZero, sparse].entries()) {
                    a.setWeights(weights);
                    const encoder = device.createCommandEncoder();
                    const count = engine.flushWebGPUMorphComputes(device, encoder);
                    assert.equal(count, iteration === 3 ? 0 : iteration === 0 && capacity === 1 ? 2 : 1);
                    const readbacks = [];
                    for (const [instance, values] of [[a, weights], [b, stableWeights]]) {
                        for (const attribute of owner.attributes) {
                            const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(),
                                loadOp: 'clear', storeOp: 'store', clearValue: [9, 9, 9, 9] }] });
                            pass.setPipeline(pipeline);
                            pass.setBindGroup(0, device.createBindGroup({ layout, entries: [{ binding: 0,
                                resource: instance.outputs[attribute].gpuTexture.gpuTexture.createView() }] }));
                            pass.draw(vertices, 1, 0, instance.outputRowOffset); pass.end();
                            const readback = device.createBuffer({ size: rowBytes,
                                usage: globals.GPUBufferUsage.COPY_DST | globals.GPUBufferUsage.MAP_READ });
                            encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: rowBytes }, [vertices, 1]);
                            readbacks.push({ readback, attribute, values });
                        }
                    }
                    device.queue.submit([encoder.finish()]);
                    for (const { readback, attribute, values } of readbacks) {
                        await readback.mapAsync(globals.GPUMapMode.READ);
                        const actual = new Float32Array(readback.getMappedRange());
                        for (let v = 0; v < vertices; ++v) {
                            for (let component = 0; component < 3; ++component) {
                                let expected = 0;
                                for (let t = 0; t < targets; ++t) {
                                    expected += layers[attribute][t][v * 3 + component] * Math.fround(values[t]);
                                }
                                const error = Math.abs(actual[v * 4 + component] - expected);
                                maxError = Math.max(maxError, error);
                                assert.ok(error < 2e-6, `mask=${mask} iteration=${iteration} vertex=${v} error=${error}`);
                                ++checked;
                            }
                        }
                        readback.unmap(); readback.destroy();
                    }
                }
                a.setWeights(dense); a.destroy();
                assert.equal(engine.flushWebGPUMorphComputes(device, device.createCommandEncoder()), 0);
                owner.destroy();
            }
            await device.queue.onSubmittedWorkDone();
            assert.deepEqual(errors, []);
            console.log(JSON.stringify({ scope: 'actual-engine-kernel-and-CS-to-VS; thin-GFX-adapter; not-mobile-performance',
                vendor: adapter.info.vendor, device: adapter.info.description, checked, maxError, errors }));
        } finally {
            target.destroy(); device.destroy();
        }
    });
