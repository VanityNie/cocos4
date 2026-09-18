/*
 Copyright (c) 2026 Xiamen Yaji Software Co., Ltd.
 SPDX-License-Identifier: MIT
*/

import { API, Format, TextureInfo, TextureType, TextureUsageBit } from '../base/define';
import type { Texture } from '../base/texture';
import type { Device } from '../base/device';
import type { WebGPUTexture } from './webgpu-texture';
import type { WebGPUDevice } from './webgpu-device';

// Compute is a WebGPU implementation detail. Its outputs use the existing
// CC_MORPH_PRECOMPUTED texture contract, shared with CPU morph rendering.
// The input texture is target-major within each position/normal/tangent array layer.
function morphComputeSource (attributes: readonly number[]): string {
    const accumulate = `
        let pixel = targetIndex * info.vertexCount + vertex;
        let uv = vec2<i32>(i32(pixel % info.width), i32(pixel / info.width));
        ${attributes.map((attribute, layer) => `sum${attribute} += textureLoad(targets, uv, ${layer}, 0).xyz * weight;`).join('\n        ')}`;
    return `
struct MorphInfo { vertexCount: u32, targetCount: u32, width: u32, outputRows: u32 }
@group(0) @binding(0) var targets: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<uniform> info: MorphInfo;
${attributes.map((attribute) => `@group(0) @binding(${outputBindings[attribute]}) var output${attribute}: texture_storage_2d<rgba32float, write>;`).join('\n')}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let vertex = id.x + id.y * groups.x * 64u;
    if (vertex >= info.vertexCount) { return; }
    // Z indexes compact dirty records; the first word maps back to a stable atlas slot.
    let record = id.z * max(4u, 3u + info.targetCount);
    let slot = weights[record];
    let count = weights[record + 1u];
    let sparse = weights[record + 2u];
    let data = record + 3u;
    ${attributes.map((attribute) => `var sum${attribute} = vec3<f32>(0.0);`).join('\n    ')}
    if (sparse != 0u) {
        for (var entry = 0u; entry < count; entry++) {
            let targetIndex = weights[data + entry * 2u];
            let weight = bitcast<f32>(weights[data + entry * 2u + 1u]);
            ${accumulate}
        }
    } else {
        for (var targetIndex = 0u; targetIndex < count; targetIndex++) {
            let weight = bitcast<f32>(weights[data + targetIndex]);
            if (weight == 0.0) { continue; }
            ${accumulate}
        }
    }
    let outputWidth = textureDimensions(output${attributes[0]}).x;
    let outputUV = vec2<i32>(i32(vertex % outputWidth), i32(vertex / outputWidth + slot * info.outputRows));
    ${attributes.map((attribute) => `textureStore(output${attribute}, outputUV, vec4<f32>(sum${attribute}, 0.0));`).join('\n    ')}
}`;
}

const outputBindings = [2, 4, 5];

const pending = new WeakMap<GPUDevice, Set<WebGPUMorphComputeBatch>>();
const pipelines = new WeakMap<GPUDevice, Map<number, GPUComputePipeline>>();

/** Record before the consuming render pass in the same encoder. The caller submits
 * immediately after recording rendering; weight uploads are queue operations. */
export function flushWebGPUMorphComputes (device: GPUDevice, encoder: GPUCommandEncoder): number {
    const batches = pending.get(device);
    if (!batches?.size) return 0;
    const dispatchCount = batches.size;
    const pass = encoder.beginComputePass({ label: 'Morph displacement blend' });
    let pipeline: GPUComputePipeline | undefined;
    for (const batch of batches) {
        if (pipeline !== batch.owner.pipeline) {
            pipeline = batch.owner.pipeline;
            pass.setPipeline(pipeline);
        }
        batch.dispatch(pass);
    }
    pass.end();
    batches.clear();
    return dispatchCount;
}

/** Mesh-owned input and fixed atlas pages. Renderers keep stable slots until destroyed. */
export class WebGPUMorphCompute {
    public readonly device: GPUDevice;
    public readonly pipeline: GPUComputePipeline;
    public readonly texture: GPUTexture;
    public readonly info: GPUBuffer;
    public readonly groupsX: number;
    public readonly groupsY: number;
    public readonly outputWidth: number;
    public readonly outputHeight: number;
    /** Semantic indices (position=0, normal=1, tangent=2), packed into input layers. */
    public readonly attributes: readonly number[];
    public readonly weightsSize: number;
    public readonly batchCapacity: number;
    private readonly _textureBytes: number;
    private readonly _batches: WebGPUMorphComputeBatch[] = [];
    private readonly _instances = new Set<WebGPUMorphComputeInstance>();
    private _destroyed = false;

    constructor (
        public readonly gfxDevice: Device,
        public readonly vertexCount: number,
        public readonly targetCount: number,
        // A null layer denotes an absent morph attribute; it is never read.
        layers: readonly (readonly Float32Array[] | null)[],
        maxBatchSize = 64,
    ) {
        if (gfxDevice.gfxAPI !== API.WEBGPU) throw new Error('Compute morph rendering requires WebGPU.');
        this.device = (gfxDevice as WebGPUDevice).nativeDevice!;
        const device = this.device;
        const limits = device.limits;
        this.attributes = [0, 1, 2].filter((attribute) => layers[attribute] !== null && layers[attribute] !== undefined);
        if (!this.attributes.length) throw new Error('Compute morph requires at least one displacement attribute.');
        const attributeMask = this.attributes.reduce((mask, attribute) => mask | (1 << attribute), 0);
        if (!Number.isInteger(maxBatchSize) || maxBatchSize < 1) throw new Error('Morph batch size must be a positive integer.');
        // Slot, count, sparse flag, then either T dense weights or K (target, weight) pairs.
        // Sparse encoding is selected only when 2K < T, so it never enlarges the payload.
        this.weightsSize = Math.max(16, 12 + targetCount * 4);
        const pixels = Math.max(1, vertexCount * targetCount);
        const width = Math.min(pixels, limits.maxTextureDimension2D);
        const height = Math.ceil(pixels / width);
        this.outputWidth = Math.min(Math.max(1, vertexCount), limits.maxTextureDimension2D);
        this.outputHeight = Math.max(1, Math.ceil(vertexCount / this.outputWidth));
        if (height > limits.maxTextureDimension2D
            || this.outputHeight > limits.maxTextureDimension2D
            || this.weightsSize > limits.maxStorageBufferBindingSize
            || this.weightsSize > limits.maxBufferSize) {
            throw new Error('Morph data exceeds the WebGPU device texture or storage buffer limits.');
        }
        const groups = Math.max(1, Math.ceil(vertexCount / 64));
        this.groupsX = Math.min(groups, limits.maxComputeWorkgroupsPerDimension);
        this.groupsY = Math.ceil(groups / this.groupsX);
        if (this.groupsY > limits.maxComputeWorkgroupsPerDimension) throw new Error('Morph vertex count exceeds WebGPU dispatch limits.');
        const instanceOutputBytes = this.outputWidth * this.outputHeight * 16 * this.attributes.length;
        // Bound reserved output memory to 8 MiB/page, except when one instance needs more.
        this.batchCapacity = Math.min(
            maxBatchSize,
            limits.maxComputeWorkgroupsPerDimension,
            Math.floor(limits.maxTextureDimension2D / this.outputHeight),
            Math.floor(Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) / this.weightsSize),
            Math.max(1, Math.floor(8 * 1024 * 1024 / instanceOutputBytes)),
        );
        let devicePipelines = pipelines.get(device);
        if (!devicePipelines) {
            devicePipelines = new Map();
            pipelines.set(device, devicePipelines);
        }
        let pipeline = devicePipelines.get(attributeMask);
        if (!pipeline) {
            const layout = device.createBindGroupLayout({ entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                ...this.attributes.map((attribute): GPUBindGroupLayoutEntry => ({
                    binding: outputBindings[attribute],
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: 'write-only', format: 'rgba32float' },
                })),
            ] });
            pipeline = device.createComputePipeline({
                label: 'Morph displacement blend',
                layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module: device.createShaderModule({ code: morphComputeSource(this.attributes) }), entryPoint: 'main' },
            });
            devicePipelines.set(attributeMask, pipeline);
        }
        this.pipeline = pipeline;
        this.texture = device.createTexture({
            label: 'Shared morph targets',
            size: [width, height, this.attributes.length],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        const uploadPixels = new Float32Array(width * height * 4);
        for (let layer = 0; layer < this.attributes.length; ++layer) {
            const targets = layers[this.attributes[layer]]!;
            const pixels = uploadPixels;
            for (let target = 0; target < targetCount; ++target) {
                const source = targets[target];
                for (let vertex = 0; vertex < vertexCount; ++vertex) {
                    const offset = (target * vertexCount + vertex) * 4;
                    pixels[offset] = source[vertex * 3];
                    pixels[offset + 1] = source[vertex * 3 + 1];
                    pixels[offset + 2] = source[vertex * 3 + 2];
                }
            }
            device.queue.writeTexture(
                { texture: this.texture, origin: [0, 0, layer] },
                pixels,
                { bytesPerRow: width * 16, rowsPerImage: height },
                [width, height, 1],
            );
        }
        this.info = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.info, 0, new Uint32Array([vertexCount, targetCount, width, this.outputHeight]));
        this._textureBytes = width * height * this.attributes.length * 16;
        gfxDevice.memoryStatus.textureSize += this._textureBytes;
        gfxDevice.memoryStatus.bufferSize += 16;
    }

    public createInstance (): WebGPUMorphComputeInstance {
        if (this._destroyed) throw new Error('Morph compute has been destroyed.');
        let batch = this._batches.find((candidate) => candidate.hasSpace);
        if (!batch) {
            batch = new WebGPUMorphComputeBatch(this);
            this._batches.push(batch);
        }
        const instance = batch.createInstance();
        this._instances.add(instance);
        return instance;
    }

    public release (instance: WebGPUMorphComputeInstance): void {
        this._instances.delete(instance);
    }

    public releaseBatch (batch: WebGPUMorphComputeBatch): void {
        const index = this._batches.indexOf(batch);
        if (index >= 0) this._batches.splice(index, 1);
    }

    public destroy (): void {
        if (this._destroyed) return;
        this._destroyed = true;
        for (const instance of this._instances) instance.destroy();
        this.texture.destroy();
        this.info.destroy();
        this.gfxDevice.memoryStatus.textureSize -= this._textureBytes;
        this.gfxDevice.memoryStatus.bufferSize -= 16;
    }
}

/** One stable output atlas and one compact dirty-weight upload per dispatch. */
class WebGPUMorphComputeBatch {
    public readonly outputs: readonly (Texture | null)[];
    private readonly _weights: GPUBuffer;
    private readonly _packedWeights: Uint32Array;
    private readonly _packedFloats: Float32Array;
    private readonly _bindings: GPUBindGroup;
    private readonly _freeSlots: number[];
    private readonly _dirty = new Set<WebGPUMorphComputeInstance>();
    private _liveCount = 0;

    constructor (public readonly owner: WebGPUMorphCompute) {
        const device = owner.device;
        this.outputs = [0, 1, 2].map((attribute) => (owner.attributes.includes(attribute) ? owner.gfxDevice.createTexture(new TextureInfo(
            TextureType.TEX2D,
            TextureUsageBit.STORAGE | TextureUsageBit.SAMPLED,
            Format.RGBA32F,
            owner.outputWidth,
            owner.outputHeight * owner.batchCapacity,
        )) : null));
        const bytes = owner.weightsSize * owner.batchCapacity;
        this._weights = device.createBuffer({ label: 'Morph batch weights',
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        owner.gfxDevice.memoryStatus.bufferSize += bytes;
        this._packedWeights = new Uint32Array(bytes / 4);
        this._packedFloats = new Float32Array(this._packedWeights.buffer);
        this._freeSlots = Array.from({ length: owner.batchCapacity }, (_, i) => owner.batchCapacity - 1 - i);
        this._bindings = device.createBindGroup({ layout: owner.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: owner.texture.createView({ dimension: '2d-array' }) },
                { binding: 1, resource: { buffer: this._weights } },
                { binding: 3, resource: { buffer: owner.info } },
                ...owner.attributes.map((attribute) => ({
                    binding: outputBindings[attribute],
                    resource: (this.outputs[attribute] as WebGPUTexture).gpuTexture.gpuTexture!.createView(),
                })),
            ] });
    }

    public get hasSpace (): boolean { return this._freeSlots.length > 0; }

    public createInstance (): WebGPUMorphComputeInstance {
        const slot = this._freeSlots.pop();
        if (slot === undefined) throw new Error('Morph batch is full.');
        ++this._liveCount;
        const instance = new WebGPUMorphComputeInstance(this, slot);
        this.markDirty(instance); // Clear a new or recycled output slot, including zero weights.
        return instance;
    }

    public markDirty (instance: WebGPUMorphComputeInstance): void {
        this._dirty.add(instance);
        let batches = pending.get(this.owner.device);
        if (!batches) {
            batches = new Set();
            pending.set(this.owner.device, batches);
        }
        batches.add(this);
    }

    public dispatch (pass: GPUComputePassEncoder): void {
        const stride = this.owner.weightsSize / 4;
        let record = 0;
        for (const instance of this._dirty) {
            instance.pack(this._packedWeights, this._packedFloats, record++ * stride);
        }
        this.owner.device.queue.writeBuffer(this._weights, 0, this._packedWeights.buffer, 0, record * this.owner.weightsSize);
        pass.setBindGroup(0, this._bindings);
        // X/Y tile vertices, Z selects a dirty instance; unchanged slots are not touched.
        pass.dispatchWorkgroups(this.owner.groupsX, this.owner.groupsY, record);
        this._dirty.clear();
    }

    public release (instance: WebGPUMorphComputeInstance): void {
        this._dirty.delete(instance);
        if (!this._dirty.size) pending.get(this.owner.device)?.delete(this);
        this._freeSlots.push(instance.slot);
        this.owner.release(instance);
        if (--this._liveCount === 0) {
            this._weights.destroy();
            this.owner.gfxDevice.memoryStatus.bufferSize -= this.owner.weightsSize * this.owner.batchCapacity;
            for (const output of this.outputs) output?.destroy();
            this.owner.releaseBatch(this);
        }
    }
}

export class WebGPUMorphComputeInstance {
    private readonly _values: Float32Array;
    private _destroyed = false;

    constructor (private readonly _batch: WebGPUMorphComputeBatch, public readonly slot: number) {
        this._values = new Float32Array(_batch.owner.targetCount);
    }

    public get outputs (): readonly (Texture | null)[] { return this._batch.outputs; }
    public get outputWidth (): number { return this._batch.owner.outputWidth; }
    public get outputHeight (): number { return this._batch.owner.outputHeight * this._batch.owner.batchCapacity; }
    public get outputRowOffset (): number { return this.slot * this._batch.owner.outputHeight; }

    public setWeights (weights: readonly number[]): void {
        if (this._destroyed) throw new Error('Morph compute instance has been destroyed.');
        if (weights.length !== this._values.length) throw new Error('Morph weight count must match the number of targets.');
        let changed = false;
        for (let i = 0; i < weights.length; ++i) {
            const value = Math.fround(weights[i]);
            if (this._values[i] !== value) {
                this._values[i] = value;
                changed = true;
            }
        }
        if (changed) this._batch.markDirty(this);
    }

    public pack (words: Uint32Array, floats: Float32Array, offset: number): void {
        let activeCount = 0;
        for (const weight of this._values) if (weight !== 0) ++activeCount;
        const sparse = activeCount * 2 < this._values.length;
        words[offset++] = this.slot;
        words[offset++] = sparse ? activeCount : this._values.length;
        words[offset++] = sparse ? 1 : 0;
        if (sparse) {
            // Keep target order to preserve the original accumulation order.
            for (let target = 0; target < this._values.length; ++target) {
                const weight = this._values[target];
                if (weight === 0) continue;
                words[offset++] = target;
                floats[offset++] = weight;
            }
        } else {
            floats.set(this._values, offset);
        }
    }

    public destroy (): void {
        if (this._destroyed) return;
        this._destroyed = true;
        this._batch.release(this);
    }
}
