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
const morphComputeSource = `
struct MorphInfo { vertexCount: u32, targetCount: u32, width: u32, attributes: u32 }
@group(0) @binding(0) var targets: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var positions: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> info: MorphInfo;
@group(0) @binding(4) var normals: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var tangents: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let vertex = id.x + id.y * groups.x * 64u;
    if (vertex >= info.vertexCount) { return; }
    var position = vec3<f32>(0.0);
    var normal = vec3<f32>(0.0);
    var tangent = vec3<f32>(0.0);
    for (var targetIndex = 0u; targetIndex < info.targetCount; targetIndex++) {
        let weight = weights[targetIndex];
        if (weight == 0.0) { continue; }
        let pixel = targetIndex * info.vertexCount + vertex;
        let uv = vec2<i32>(i32(pixel % info.width), i32(pixel / info.width));
        if ((info.attributes & 1u) != 0u) { position += textureLoad(targets, uv, 0, 0).xyz * weight; }
        if ((info.attributes & 2u) != 0u) { normal += textureLoad(targets, uv, 1, 0).xyz * weight; }
        if ((info.attributes & 4u) != 0u) { tangent += textureLoad(targets, uv, 2, 0).xyz * weight; }
    }
    let outputWidth = textureDimensions(positions).x;
    let outputUV = vec2<i32>(i32(vertex % outputWidth), i32(vertex / outputWidth));
    textureStore(positions, outputUV, vec4<f32>(position, 0.0));
    textureStore(normals, outputUV, vec4<f32>(normal, 0.0));
    textureStore(tangents, outputUV, vec4<f32>(tangent, 0.0));
}`;

const pending = new WeakMap<GPUDevice, Set<WebGPUMorphComputeInstance>>();
const pipelines = new WeakMap<GPUDevice, GPUComputePipeline>();

/** Called before every render pass, including shadow passes. Queue ordering makes
 * these writes visible to all subsequently submitted render command buffers. */
export function flushWebGPUMorphComputes (device: GPUDevice): number {
    const instances = pending.get(device);
    if (!instances?.size) return 0;
    const dispatchCount = instances.size;
    const encoder = device.createCommandEncoder({ label: 'Morph compute before rendering' });
    const pass = encoder.beginComputePass({ label: 'Morph displacement blend' });
    for (const instance of instances) instance.dispatch(pass);
    pass.end();
    device.queue.submit([encoder.finish()]);
    instances.clear();
    return dispatchCount;
}

/** Mesh-owned static input. Only weights and output are allocated per renderer. */
export class WebGPUMorphCompute {
    public readonly device: GPUDevice;
    public readonly pipeline: GPUComputePipeline;
    public readonly texture: GPUTexture;
    public readonly info: GPUBuffer;
    public readonly groupsX: number;
    public readonly groupsY: number;
    public readonly outputWidth: number;
    public readonly outputHeight: number;
    private readonly _instances = new Set<WebGPUMorphComputeInstance>();

    constructor (
        public readonly gfxDevice: Device,
        public readonly vertexCount: number,
        public readonly targetCount: number,
        // A null layer denotes an absent morph attribute; it is never read.
        layers: readonly (readonly Float32Array[] | null)[],
    ) {
        if (gfxDevice.gfxAPI !== API.WEBGPU) throw new Error('Compute morph rendering requires WebGPU.');
        this.device = (gfxDevice as WebGPUDevice).nativeDevice!;
        const device = this.device;
        const limits = device.limits;
        const pixels = Math.max(1, vertexCount * targetCount);
        const width = Math.min(pixels, limits.maxTextureDimension2D);
        const height = Math.ceil(pixels / width);
        this.outputWidth = Math.min(Math.max(1, vertexCount), limits.maxTextureDimension2D);
        this.outputHeight = Math.max(1, Math.ceil(vertexCount / this.outputWidth));
        if (height > limits.maxTextureDimension2D
            || this.outputHeight > limits.maxTextureDimension2D
            || Math.max(4, targetCount * 4) > limits.maxStorageBufferBindingSize
            || Math.max(4, targetCount * 4) > limits.maxBufferSize) {
            throw new Error('Morph data exceeds the WebGPU device texture or storage buffer limits.');
        }
        const groups = Math.max(1, Math.ceil(vertexCount / 64));
        this.groupsX = Math.min(groups, limits.maxComputeWorkgroupsPerDimension);
        this.groupsY = Math.ceil(groups / this.groupsX);
        if (this.groupsY > limits.maxComputeWorkgroupsPerDimension) throw new Error('Morph vertex count exceeds WebGPU dispatch limits.');
        let pipeline = pipelines.get(device);
        if (!pipeline) {
            const layout = device.createBindGroupLayout({ entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },
            ] });
            pipeline = device.createComputePipeline({
                label: 'Morph displacement blend',
                layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module: device.createShaderModule({ code: morphComputeSource }), entryPoint: 'main' },
            });
            pipelines.set(device, pipeline);
        }
        this.pipeline = pipeline;
        this.texture = device.createTexture({
            label: 'Shared morph targets',
            size: [width, height, 3],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        let attributes = 0;
        for (let layer = 0; layer < 3; ++layer) {
            const targets = layers[layer];
            if (!targets) continue;
            attributes |= 1 << layer;
            const pixels = new Float32Array(width * height * 4);
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
        device.queue.writeBuffer(this.info, 0, new Uint32Array([vertexCount, targetCount, width, attributes]));
    }

    public createInstance (): WebGPUMorphComputeInstance {
        const instance = new WebGPUMorphComputeInstance(this);
        this._instances.add(instance);
        return instance;
    }

    public release (instance: WebGPUMorphComputeInstance): void {
        this._instances.delete(instance);
    }

    public destroy (): void {
        for (const instance of this._instances) instance.destroy();
        this.texture.destroy();
        this.info.destroy();
    }
}

export class WebGPUMorphComputeInstance {
    public readonly outputs: readonly Texture[];
    private readonly _weights: GPUBuffer;
    private readonly _values: Float32Array;
    private readonly _bindings: GPUBindGroup;
    private _destroyed = false;

    constructor (private readonly _owner: WebGPUMorphCompute) {
        const device = _owner.device;
        this.outputs = [0, 1, 2].map(() => _owner.gfxDevice.createTexture(new TextureInfo(
            TextureType.TEX2D,
            TextureUsageBit.STORAGE | TextureUsageBit.SAMPLED,
            Format.RGBA32F,
            _owner.outputWidth,
            _owner.outputHeight,
        )));
        this._weights = device.createBuffer({ size: Math.max(4, _owner.targetCount * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this._values = new Float32Array(_owner.targetCount);
        this._bindings = device.createBindGroup({ layout: _owner.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: _owner.texture.createView({ dimension: '2d-array' }) },
                { binding: 1, resource: { buffer: this._weights } },
                { binding: 2, resource: (this.outputs[0] as WebGPUTexture).gpuTexture.gpuTexture!.createView() },
                { binding: 3, resource: { buffer: _owner.info } },
                { binding: 4, resource: (this.outputs[1] as WebGPUTexture).gpuTexture.gpuTexture!.createView() },
                { binding: 5, resource: (this.outputs[2] as WebGPUTexture).gpuTexture.gpuTexture!.createView() },
            ] });
        this._markDirty();
    }

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
        if (changed) this._markDirty();
    }

    public dispatch (pass: GPUComputePassEncoder): void {
        const owner = this._owner;
        if (this._values.length) owner.device.queue.writeBuffer(this._weights, 0, this._values);
        pass.setPipeline(owner.pipeline);
        pass.setBindGroup(0, this._bindings);
        pass.dispatchWorkgroups(owner.groupsX, owner.groupsY);
    }

    public destroy (): void {
        if (this._destroyed) return;
        this._destroyed = true;
        pending.get(this._owner.device)?.delete(this);
        this._owner.release(this);
        this._weights.destroy();
        for (const output of this.outputs) output.destroy();
    }

    private _markDirty (): void {
        let instances = pending.get(this._owner.device);
        if (!instances) {
            instances = new Set();
            pending.set(this._owner.device, instances);
        }
        instances.add(this);
    }
}
