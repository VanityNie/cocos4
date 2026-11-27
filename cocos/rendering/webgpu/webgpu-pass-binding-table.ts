/*
 Copyright (c) 2026 Xiamen Yaji Software Co., Ltd.

 https://www.cocos.com/

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

import { Type, ViewDimension } from '../../gfx';
import { DescriptorTypeOrder } from '../custom/layout-graph';
import type { LayoutGraphData } from '../custom/layout-graph';
import type { ComputeView, RenderData } from '../custom/render-graph';
import type { UpdateFrequency } from '../custom/types';

export const DEFAULT_PASS_BINDING_IDLE_FRAMES = 20;
export const DEFAULT_SAMPLER_BINDING_OFFSET = 16;
const DEFAULT_RING_CHUNK_BYTES = 65536;
const EMPTY_DYNAMIC_OFFSETS = new Uint32Array(0);
const UNIFORM_BUFFER_USAGE = 0x0040 | 0x0008;

export interface IWebGPUBindingHost {
    readonly device: GPUDevice;
    uidOf (obj: object): number;
    getBindGroupLayout (layoutID: number, frequency: UpdateFrequency): GPUBindGroupLayout | null;
    getTextureViewByName (name: string): GPUTextureView | null;
    getBufferByName (name: string): GPUBuffer | null;
    getSamplerByName (name: string): GPUSampler | null;
    getExternalTextureByName? (name: string): GPUExternalTexture | null;
    unwrapTextureView (tex: object): GPUTextureView | null;
    unwrapBuffer (buf: object): GPUBuffer | null;
    unwrapSampler (sampler: object, tex?: object | null): GPUSampler | null;
    getDefaultTextureView (dimension: GPUTextureViewDimension): GPUTextureView;
    getDefaultSampler (): GPUSampler;
    getDefaultStorageBuffer (): GPUBuffer;
}

export interface WebGPUPassBindResult {
    bindGroup: GPUBindGroup;
    dynamicOffsets: Uint32Array;
    dynamicOffsetStart: number;
    dynamicOffsetCount: number;
}

const enum WebGPUBindingKind {
    TEXTURE = 0,
    SAMPLER = 1,
    SAMPLER_TEXTURE = 2,
    STORAGE_BUFFER = 3,
    STORAGE_TEXTURE = 4,
}

const enum ConstantScalarKind {
    FLOAT = 0,
    INT = 1,
    UINT = 2,
}

interface UniformBlockSchema {
    readonly descriptorID: number;
    readonly binding: number;
    readonly byteSize: number;
    readonly floatCount: number;
    readonly isDynamic: boolean;
    slotIndex: number;
}

interface ConstantFieldLoc {
    readonly blockIndex: number;
    readonly floatOffset: number;
    readonly floatCount: number;
    readonly chunkFloats: number;
    readonly chunkStride: number;
    readonly scalarKind: ConstantScalarKind;
}

interface ResourceBindingDesc {
    readonly descriptorID: number;
    readonly binding: number;
    readonly kind: WebGPUBindingKind;
    readonly isDynamic: boolean;
    readonly samplerBinding: number;
    readonly viewDimension: GPUTextureViewDimension;
    index: number;
}

interface DynamicOffsetEntry {
    readonly binding: number;
    readonly uboSlotIndex: number; // >= 0 for dynamic UBO, -1 for dynamic storage buffer (offset 0)
    readonly blockIndex: number;   // >= 0 for dynamic UBO in schema.uboBlocks, -1 for dynamic storage buffer
}

interface PassLayoutSchema {
    readonly schemaKey: number;
    readonly bgl: GPUBindGroupLayout;
    readonly bglUID: number;
    readonly uboBlocks: UniformBlockSchema[];
    readonly uboByID: Map<number, number>;
    readonly staticUboCount: number;
    readonly dynamicUboCount: number;
    readonly dynamicEntries: DynamicOffsetEntry[];
    readonly resources: ResourceBindingDesc[];
    readonly constants: Map<number, ConstantFieldLoc>;
    readonly resourceByID: Map<number, ResourceBindingDesc>;
    readonly slotSigLen: number;
}

interface StaticUniformBlock {
    readonly binding: number;
    readonly byteSize: number;
    readonly gpuBuffer: GPUBuffer;
    readonly bufferUID: number;
    readonly cpuBuffer: ArrayBuffer;
    readonly cpuU32: Uint32Array;
    dirty: boolean;
    queued: boolean;
}

interface StaticUniformSlot {
    readonly blocks: StaticUniformBlock[];
    lastSerial: number;
}

interface UniformRingChunk {
    readonly gpuBuffer: GPUBuffer;
    readonly bufferUID: number;
    readonly capacity: number;
    readonly stagingBuffer: ArrayBuffer;
    readonly stagingBytes: Uint8Array;
    readonly stagingF32: Float32Array;
    readonly stagingI32: Int32Array;
    readonly stagingU32: Uint32Array;
    usedBytes: number;
    lastSerial: number;
}

interface BindGroupEntry {
    readonly bindGroup: GPUBindGroup;
    readonly sig: Int32Array;
    readonly boundUIDs: number[];
    lastSerial: number;
}

function alignUp (bytes: number, alignment: number): number {
    if (alignment <= 1) return bytes;
    if ((alignment & (alignment - 1)) !== 0) return Math.ceil(bytes / alignment) * alignment;
    return ((bytes + alignment - 1) & ~(alignment - 1)) >>> 0;
}

function nextPow2 (n: number): number {
    return n <= 16 ? 16 : 2 ** Math.ceil(Math.log2(n));
}

function hashSignature (sig: Int32Array, len: number): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < len; ++i) {
        h ^= sig[i];
        h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
}

function signatureEquals (a: Int32Array, b: Int32Array, len: number): boolean {
    if (a.length !== len) return false;
    for (let i = 0; i < len; ++i) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function toWGPUViewDimension (dim: ViewDimension): GPUTextureViewDimension {
    switch (dim) {
    case ViewDimension.TEX1D: return '1d';
    case ViewDimension.TEX2D_ARRAY:
    case ViewDimension.TEX2DMS_ARRAY: return '2d-array';
    case ViewDimension.TEXCUBE: return 'cube';
    case ViewDimension.TEXCUBE_ARRAY: return 'cube-array';
    case ViewDimension.TEX3D: return '3d';
    default: return '2d';
    }
}

interface UboMemberLayout {
    readonly alignFloats: number;
    readonly sizeFloats: number;
    readonly elemFloats: number;
    readonly chunkFloats: number;
    readonly chunkStride: number;
}

function getUboMemberLayout (type: Type, rawCount: number): UboMemberLayout {
    const count = rawCount > 0 ? rawCount : 1;
    let alignFloats = 1;
    let elemFloats = 0;
    let sizeFloats = 0;
    let chunkFloats = 0;
    let chunkStride = 0;

    switch (type) {
    case Type.BOOL:
    case Type.INT:
    case Type.UINT:
    case Type.FLOAT:
        alignFloats = 1; elemFloats = 1; sizeFloats = 1; chunkFloats = 1; chunkStride = 1; break;
    case Type.BOOL2:
    case Type.INT2:
    case Type.UINT2:
    case Type.FLOAT2:
        alignFloats = 2; elemFloats = 2; sizeFloats = 2; chunkFloats = 2; chunkStride = 2; break;
    case Type.BOOL3:
    case Type.INT3:
    case Type.UINT3:
    case Type.FLOAT3:
        alignFloats = 4; elemFloats = 3; sizeFloats = 3; chunkFloats = 3; chunkStride = 3; break;
    case Type.BOOL4:
    case Type.INT4:
    case Type.UINT4:
    case Type.FLOAT4:
        alignFloats = 4; elemFloats = 4; sizeFloats = 4; chunkFloats = 4; chunkStride = 4; break;
    case Type.MAT2:
        alignFloats = 2; elemFloats = 4; sizeFloats = 4; chunkFloats = 4; chunkStride = 4; break;
    case Type.MAT2X3:
        alignFloats = 4; elemFloats = 6; sizeFloats = 8; chunkFloats = 3; chunkStride = 4; break;
    case Type.MAT3X2:
        alignFloats = 2; elemFloats = 6; sizeFloats = 6; chunkFloats = 6; chunkStride = 6; break;
    case Type.MAT2X4:
        alignFloats = 4; elemFloats = 8; sizeFloats = 8; chunkFloats = 8; chunkStride = 8; break;
    case Type.MAT4X2:
        alignFloats = 2; elemFloats = 8; sizeFloats = 8; chunkFloats = 8; chunkStride = 8; break;
    case Type.MAT3:
        alignFloats = 4; elemFloats = 9; sizeFloats = 12; chunkFloats = 3; chunkStride = 4; break;
    case Type.MAT3X4:
        alignFloats = 4; elemFloats = 12; sizeFloats = 12; chunkFloats = 12; chunkStride = 12; break;
    case Type.MAT4X3:
        alignFloats = 4; elemFloats = 12; sizeFloats = 16; chunkFloats = 3; chunkStride = 4; break;
    case Type.MAT4:
        alignFloats = 4; elemFloats = 16; sizeFloats = 16; chunkFloats = 16; chunkStride = 16; break;
    default:
        return { alignFloats: 1, sizeFloats: 0, elemFloats: 0, chunkFloats: 0, chunkStride: 0 };
    }

    if (count > 1) {
        alignFloats = 4;
        const arrayStride = alignUp(sizeFloats, 4);
        if (chunkFloats === chunkStride && arrayStride !== sizeFloats) {
            chunkFloats = elemFloats;
            chunkStride = arrayStride;
        }
        return {
            alignFloats,
            sizeFloats: arrayStride * count,
            elemFloats: elemFloats * count,
            chunkFloats,
            chunkStride,
        };
    }

    return { alignFloats, sizeFloats, elemFloats, chunkFloats, chunkStride };
}

function constantScalarKind (type: Type): ConstantScalarKind {
    switch (type) {
    case Type.INT:
    case Type.INT2:
    case Type.INT3:
    case Type.INT4: return ConstantScalarKind.INT;
    case Type.BOOL:
    case Type.BOOL2:
    case Type.BOOL3:
    case Type.BOOL4:
    case Type.UINT:
    case Type.UINT2:
    case Type.UINT3:
    case Type.UINT4: return ConstantScalarKind.UINT;
    default: return ConstantScalarKind.FLOAT;
    }
}

function getResourceBindingKind (type: DescriptorTypeOrder): WebGPUBindingKind {
    switch (type) {
    case DescriptorTypeOrder.TEXTURE:
    case DescriptorTypeOrder.INPUT_ATTACHMENT: return WebGPUBindingKind.TEXTURE;
    case DescriptorTypeOrder.SAMPLER: return WebGPUBindingKind.SAMPLER;
    case DescriptorTypeOrder.SAMPLER_TEXTURE: return WebGPUBindingKind.SAMPLER_TEXTURE;
    case DescriptorTypeOrder.STORAGE_BUFFER:
    case DescriptorTypeOrder.DYNAMIC_STORAGE_BUFFER: return WebGPUBindingKind.STORAGE_BUFFER;
    case DescriptorTypeOrder.STORAGE_IMAGE: return WebGPUBindingKind.STORAGE_TEXTURE;
    default: throw new Error(`Unsupported WebGPU resource descriptor type: ${type}`);
    }
}

class WebGPUUniformRing {
    public totalUsedBytes = 0;

    private readonly _align: number;
    private readonly _maxChunkBytes: number;
    private readonly _chunks: UniformRingChunk[] = [];
    private _activeChunkIdx = 0;

    constructor (
        private readonly _host: IWebGPUBindingHost,
        private readonly _initialCapacity = DEFAULT_RING_CHUNK_BYTES,
    ) {
        this._align = Math.max(16, this._host.device.limits.minUniformBufferOffsetAlignment || 256);
        this._maxChunkBytes = this._host.device.limits.maxBufferSize || 268435456;
    }

    public beginFrame (): void {
        this.totalUsedBytes = 0;
        this._activeChunkIdx = 0;
        for (const chunk of this._chunks) chunk.usedBytes = 0;
    }

    public allocDynamicBlocks (uboBlocks: readonly UniformBlockSchema[], outByteOffsets: Uint32Array, serial: number): UniformRingChunk {
        let totalStride = 0;
        for (const block of uboBlocks) {
            if (block.isDynamic) totalStride += alignUp(Math.max(block.byteSize, this._align), this._align);
        }

        const chunks = this._chunks;
        if (!chunks.length) {
            const cap = Math.min(this._maxChunkBytes, Math.max(this._initialCapacity, nextPow2(totalStride)));
            chunks.push(this._createChunk(Math.max(cap, totalStride)));
        }
        while (this._activeChunkIdx < chunks.length) {
            const chunk = chunks[this._activeChunkIdx];
            if (chunk.capacity - chunk.usedBytes >= totalStride) break;
            ++this._activeChunkIdx;
        }
        if (this._activeChunkIdx === chunks.length) {
            const doubled = Math.min(this._maxChunkBytes, chunks[chunks.length - 1].capacity * 2);
            const target = Math.min(this._maxChunkBytes, nextPow2(totalStride));
            chunks.push(this._createChunk(Math.max(doubled, target, totalStride)));
            this._activeChunkIdx = chunks.length - 1;
        }

        const chunk = chunks[this._activeChunkIdx];
        chunk.lastSerial = serial;
        let index = 0;
        for (const block of uboBlocks) {
            if (!block.isDynamic) continue;
            const offset = chunk.usedBytes;
            chunk.stagingU32.fill(0, offset >>> 2, (offset + block.byteSize) >>> 2);
            outByteOffsets[index++] = offset;
            chunk.usedBytes += alignUp(Math.max(block.byteSize, this._align), this._align);
        }
        this.totalUsedBytes += totalStride;
        return chunk;
    }

    public flush (): void {
        for (const chunk of this._chunks) {
            if (!chunk.usedBytes) continue;
            this._host.device.queue.writeBuffer(chunk.gpuBuffer, 0, chunk.stagingBytes, 0, chunk.usedBytes);
        }
    }

    public endFrameReconcile (onChunkEvict: (bufferUID: number) => void): void {
        if (this._chunks.length <= 1 || this.totalUsedBytes > this._maxChunkBytes) return;
        const capacity = Math.min(this._maxChunkBytes, Math.max(this._initialCapacity, nextPow2(this.totalUsedBytes)));
        for (const chunk of this._chunks) {
            onChunkEvict(chunk.bufferUID);
            chunk.gpuBuffer.destroy();
        }
        this._chunks.length = 0;
        this._chunks.push(this._createChunk(capacity));
        this._activeChunkIdx = 0;
    }

    public trimUnused (currentSerial: number, onChunkEvict: (bufferUID: number) => void): void {
        for (let i = this._chunks.length - 1; i >= 1; --i) {
            const chunk = this._chunks[i];
            if (chunk.lastSerial >= currentSerial) continue;
            onChunkEvict(chunk.bufferUID);
            chunk.gpuBuffer.destroy();
            this._chunks.splice(i, 1);
        }
        if (this._activeChunkIdx >= this._chunks.length) {
            this._activeChunkIdx = Math.max(0, this._chunks.length - 1);
        }
    }

    public release (onChunkEvict?: (bufferUID: number) => void): void {
        for (const chunk of this._chunks) {
            if (onChunkEvict) onChunkEvict(chunk.bufferUID);
            chunk.gpuBuffer.destroy();
        }
        this._chunks.length = 0;
        this._activeChunkIdx = 0;
        this.totalUsedBytes = 0;
    }

    private _createChunk (capacity: number): UniformRingChunk {
        const gpuBuffer = this._host.device.createBuffer({ size: capacity, usage: UNIFORM_BUFFER_USAGE });
        const bufferUID = this._host.uidOf(gpuBuffer);
        const stagingBuffer = new ArrayBuffer(capacity);
        return {
            gpuBuffer,
            bufferUID,
            capacity,
            stagingBuffer,
            stagingBytes: new Uint8Array(stagingBuffer),
            stagingF32: new Float32Array(stagingBuffer),
            stagingI32: new Int32Array(stagingBuffer),
            stagingU32: new Uint32Array(stagingBuffer),
            usedBytes: 0,
            lastSerial: -1,
        };
    }
}

export class WebGPUPassBindingTable {
    private _serial = 0;
    private readonly _schemas = new Map<number, PassLayoutSchema | null>();
    private readonly _staticSlots = new Map<number, Map<string, StaticUniformSlot>>();
    private readonly _dirtyStaticBlocks: StaticUniformBlock[] = [];
    private readonly _ring: WebGPUUniformRing;
    private readonly _cache = new Map<number, BindGroupEntry[]>();

    private _sig = new Int32Array(32);
    private _dynamicOffsets = new Uint32Array(64);
    private _dynamicOffsetCursor = 0;
    private _dynBlockOffsets = new Uint32Array(16);

    private readonly _entriesScratch: GPUBindGroupEntry[] = [];
    private readonly _boundUIDsScratch: number[] = [];
    private readonly _explicitSamplers = new Set<number>();
    private readonly _resolvedUboBufs: (GPUBuffer | null)[] = [];
    private readonly _resolvedRawTexs: (object | null)[] = [];
    private readonly _resolvedTexViews: (GPUTextureView | null)[] = [];
    private readonly _resolvedBufs: (GPUBuffer | null)[] = [];
    private readonly _resolvedSamplers: (GPUSampler | null)[] = [];
    private readonly _resolvedExtTexs: (GPUExternalTexture | null)[] = [];
    private readonly _scalarScratchBuf = new ArrayBuffer(4);
    private readonly _scalarF32 = new Float32Array(this._scalarScratchBuf);
    private readonly _scalarI32 = new Int32Array(this._scalarScratchBuf);
    private readonly _scalarU32 = new Uint32Array(this._scalarScratchBuf);
    private readonly _bindResult: WebGPUPassBindResult = {
        bindGroup: null!,
        dynamicOffsets: EMPTY_DYNAMIC_OFFSETS,
        dynamicOffsetStart: 0,
        dynamicOffsetCount: 0,
    };

    constructor (
        private readonly _host: IWebGPUBindingHost,
        private readonly _layoutGraph: LayoutGraphData,
        private readonly _samplerBindingOffset = DEFAULT_SAMPLER_BINDING_OFFSET,
    ) {
        this._ring = new WebGPUUniformRing(this._host);
    }

    public beginFrame (serial: number): void {
        this._serial = serial;
        this._dynamicOffsetCursor = 0;
        this._ring.beginFrame();
    }

    public prepareBindings (
        layoutID: number,
        frequency: UpdateFrequency,
        ownerKey: string,
        dataChain: readonly (RenderData | null | undefined)[],
        computeViews?: ReadonlyMap<string, readonly ComputeView[]> | null,
    ): WebGPUPassBindResult | null {
        const schema = this._getOrCompileSchema(layoutID, frequency);
        if (!schema) return null;
        if (this._sig.length < schema.slotSigLen) this._sig = new Int32Array(schema.slotSigLen);
        const dynUboCount = schema.dynamicUboCount;
        if (this._dynBlockOffsets.length < dynUboCount) this._dynBlockOffsets = new Uint32Array(dynUboCount);

        const staticSlot = schema.staticUboCount ? this._getOrCreateStaticSlot(schema.schemaKey, ownerKey, schema) : null;
        const ringChunk = dynUboCount ? this._ring.allocDynamicBlocks(schema.uboBlocks, this._dynBlockOffsets, this._serial) : null;
        this._writeUniforms(schema, staticSlot, ringChunk, dataChain);
        this._resolveResources(schema, dataChain, computeViews);

        const dynTotalCount = schema.dynamicEntries.length;
        const dynStart = dynTotalCount ? this._dynamicOffsetCursor : 0;
        if (dynTotalCount) {
            const end = dynStart + dynTotalCount;
            if (end > this._dynamicOffsets.length) {
                const offsets = new Uint32Array(Math.max(end, this._dynamicOffsets.length * 2));
                offsets.set(this._dynamicOffsets);
                this._dynamicOffsets = offsets;
            }
            for (let i = 0; i < dynTotalCount; ++i) {
                const { uboSlotIndex, blockIndex } = schema.dynamicEntries[i];
                this._dynamicOffsets[dynStart + i] = (uboSlotIndex >= 0 && !this._resolvedUboBufs[blockIndex])
                    ? this._dynBlockOffsets[uboSlotIndex]
                    : 0;
            }
            this._dynamicOffsetCursor = end;
        }

        const hasExternalTexture = this._writeSignature(schema, staticSlot, ringChunk);
        this._bindResult.bindGroup = this._getOrCreateBindGroup(schema, staticSlot, ringChunk, hasExternalTexture);
        this._bindResult.dynamicOffsets = dynTotalCount ? this._dynamicOffsets : EMPTY_DYNAMIC_OFFSETS;
        this._bindResult.dynamicOffsetStart = dynStart;
        this._bindResult.dynamicOffsetCount = dynTotalCount;

        this._resolvedUboBufs.fill(null);
        this._resolvedRawTexs.fill(null);
        this._resolvedTexViews.fill(null);
        this._resolvedBufs.fill(null);
        this._resolvedSamplers.fill(null);
        this._resolvedExtTexs.fill(null);
        return this._bindResult;
    }

    public flushUniforms (): void {
        for (const block of this._dirtyStaticBlocks) {
            const { cpuBuffer, byteSize } = block;
            this._host.device.queue.writeBuffer(block.gpuBuffer, 0, cpuBuffer, 0, byteSize);
            block.dirty = false;
            block.queued = false;
        }
        this._dirtyStaticBlocks.length = 0;
        this._ring.flush();
    }

    public onResourceEvict (uid: number): void {
        this._evictBindGroups((entry): boolean => entry.boundUIDs.includes(uid));
    }

    public endFrameSweep (maxIdleFrames = DEFAULT_PASS_BINDING_IDLE_FRAMES): void {
        this._sweepUnused(this._serial - maxIdleFrames);
        this._ring.endFrameReconcile((uid): void => this.onResourceEvict(uid));
    }

    public flushUnused (): void {
        this._sweepUnused(this._serial);
        this._ring.trimUnused(this._serial, (uid): void => this.onResourceEvict(uid));
    }

    public release (): void {
        for (const slots of this._staticSlots.values()) {
            for (const slot of slots.values()) {
                for (const block of slot.blocks) block.gpuBuffer.destroy();
            }
        }
        this._staticSlots.clear();
        this._dirtyStaticBlocks.length = 0;
        this._ring.release();
        this._cache.clear();
        this._schemas.clear();
        this._entriesScratch.length = 0;
        this._boundUIDsScratch.length = 0;
        this._explicitSamplers.clear();
        this._resolvedUboBufs.length = 0;
        this._resolvedRawTexs.length = 0;
        this._resolvedTexViews.length = 0;
        this._resolvedBufs.length = 0;
        this._resolvedSamplers.length = 0;
        this._resolvedExtTexs.length = 0;
        this._bindResult.bindGroup = null!;
        this._bindResult.dynamicOffsets = EMPTY_DYNAMIC_OFFSETS;
        this._bindResult.dynamicOffsetStart = 0;
        this._bindResult.dynamicOffsetCount = 0;
        this._dynamicOffsetCursor = 0;
    }

    private _getOrCompileSchema (layoutID: number, frequency: UpdateFrequency): PassLayoutSchema | null {
        const schemaKey = (layoutID << 2) | frequency;
        const cached = this._schemas.get(schemaKey);
        if (cached !== undefined) return cached;

        const setData = this._layoutGraph.getLayout(layoutID).getSet(frequency);
        if (!setData) {
            this._schemas.set(schemaKey, null);
            return null;
        }
        const bgl = this._host.getBindGroupLayout(layoutID, frequency);
        if (!bgl) return null;

        const { descriptorBlocks, bindingMap, uniformBlocks } = setData.descriptorSetLayoutData;
        const uboBlocks: UniformBlockSchema[] = [];
        const resources: ResourceBindingDesc[] = [];
        for (const block of descriptorBlocks) {
            const isDynamic = block.type === DescriptorTypeOrder.DYNAMIC_UNIFORM_BUFFER;
            const isUniform = isDynamic || block.type === DescriptorTypeOrder.UNIFORM_BUFFER;
            for (let i = 0; i < block.descriptors.length; ++i) {
                const { descriptorID } = block.descriptors[i];
                const binding = bindingMap.get(descriptorID) ?? (block.offset + i);
                if (isUniform) {
                    let totalFloats = 0;
                    const uniform = uniformBlocks.get(descriptorID);
                    if (uniform) {
                        for (const member of uniform.members) {
                            const layout = getUboMemberLayout(member.type, member.count);
                            if (!layout.sizeFloats) continue;
                            totalFloats = alignUp(totalFloats, layout.alignFloats) + layout.sizeFloats;
                        }
                    }
                    const byteSize = Math.max(16, alignUp(totalFloats * 4, 16));
                    uboBlocks.push({ descriptorID, binding, byteSize, floatCount: byteSize >>> 2, isDynamic, slotIndex: 0 });
                } else {
                    const kind = getResourceBindingKind(block.type);
                    resources.push({
                        descriptorID,
                        binding,
                        kind,
                        isDynamic: block.type === DescriptorTypeOrder.DYNAMIC_STORAGE_BUFFER,
                        samplerBinding: kind === WebGPUBindingKind.SAMPLER_TEXTURE ? binding + this._samplerBindingOffset : 0,
                        viewDimension: toWGPUViewDimension(block.viewDimension),
                        index: 0,
                    });
                }
            }
        }

        uboBlocks.sort((a, b): number => a.binding - b.binding);
        const uboByID = new Map<number, number>();
        const constants = new Map<number, ConstantFieldLoc>();
        const dynamicEntries: DynamicOffsetEntry[] = [];
        let staticUboCount = 0;
        let dynamicUboCount = 0;
        for (let blockIndex = 0; blockIndex < uboBlocks.length; ++blockIndex) {
            const block = uboBlocks[blockIndex];
            uboByID.set(block.descriptorID, blockIndex);
            block.slotIndex = block.isDynamic ? dynamicUboCount++ : staticUboCount++;
            if (block.isDynamic) dynamicEntries.push({ binding: block.binding, uboSlotIndex: block.slotIndex, blockIndex });
            const uniform = uniformBlocks.get(block.descriptorID);
            if (!uniform) continue;
            let floatOffset = 0;
            for (const member of uniform.members) {
                const layout = getUboMemberLayout(member.type, member.count);
                if (!layout.sizeFloats) continue;
                floatOffset = alignUp(floatOffset, layout.alignFloats);
                const constID = this._layoutGraph.constantIndex.get(member.name);
                if (constID !== undefined) {
                    constants.set(constID, {
                        blockIndex,
                        floatOffset,
                        floatCount: layout.elemFloats,
                        chunkFloats: layout.chunkFloats,
                        chunkStride: layout.chunkStride,
                        scalarKind: constantScalarKind(member.type),
                    });
                }
                floatOffset += layout.sizeFloats;
            }
        }

        resources.sort((a, b): number => a.binding - b.binding);
        const resourceByID = new Map<number, ResourceBindingDesc>();
        let slotSigLen = 2 + uboBlocks.length;
        for (let i = 0; i < resources.length; ++i) {
            const resource = resources[i];
            resource.index = i;
            resourceByID.set(resource.descriptorID, resource);
            if (resource.isDynamic) dynamicEntries.push({ binding: resource.binding, uboSlotIndex: -1, blockIndex: -1 });
            slotSigLen += resource.kind === WebGPUBindingKind.SAMPLER_TEXTURE ? 2 : 1;
        }
        dynamicEntries.sort((a, b): number => a.binding - b.binding);
        const schema: PassLayoutSchema = {
            schemaKey,
            bgl,
            bglUID: this._host.uidOf(bgl),
            uboBlocks,
            uboByID,
            staticUboCount,
            dynamicUboCount,
            dynamicEntries,
            resources,
            constants,
            resourceByID,
            slotSigLen,
        };
        this._schemas.set(schemaKey, schema);
        return schema;
    }

    private _getOrCreateStaticSlot (schemaKey: number, ownerKey: string, schema: PassLayoutSchema): StaticUniformSlot {
        let slots = this._staticSlots.get(schemaKey);
        if (!slots) {
            slots = new Map<string, StaticUniformSlot>();
            this._staticSlots.set(schemaKey, slots);
        }
        let slot = slots.get(ownerKey);
        if (!slot) {
            const blocks: StaticUniformBlock[] = [];
            for (const block of schema.uboBlocks) {
                if (block.isDynamic) continue;
                const gpuBuffer = this._host.device.createBuffer({ size: block.byteSize, usage: UNIFORM_BUFFER_USAGE });
                const cpuBuffer = new ArrayBuffer(block.byteSize);
                blocks.push({
                    binding: block.binding,
                    byteSize: block.byteSize,
                    gpuBuffer,
                    bufferUID: this._host.uidOf(gpuBuffer),
                    cpuBuffer,
                    cpuU32: new Uint32Array(cpuBuffer),
                    dirty: false,
                    queued: false,
                });
            }
            slot = { blocks, lastSerial: this._serial };
            slots.set(ownerKey, slot);
        }
        slot.lastSerial = this._serial;
        return slot;
    }

    private _writeUniforms (
        schema: PassLayoutSchema,
        staticSlot: StaticUniformSlot | null,
        ringChunk: UniformRingChunk | null,
        dataChain: readonly (RenderData | null | undefined)[],
    ): void {
        for (const data of dataChain) {
            if (!data) continue;
            for (const [constID, values] of data.constants) {
                const loc = schema.constants.get(constID);
                if (!loc) continue;
                const block = schema.uboBlocks[loc.blockIndex];
                const len = Math.min(values.length, loc.floatCount);
                const { scalarKind, chunkFloats, chunkStride } = loc;
                if (block.isDynamic) {
                    const baseDst = (this._dynBlockOffsets[block.slotIndex] >>> 2) + loc.floatOffset;
                    const staging = scalarKind === ConstantScalarKind.FLOAT
                        ? ringChunk!.stagingF32
                        : (scalarKind === ConstantScalarKind.INT ? ringChunk!.stagingI32 : ringChunk!.stagingU32);
                    if (chunkFloats === chunkStride) {
                        for (let i = 0; i < len; ++i) staging[baseDst + i] = values[i];
                    } else {
                        for (let src = 0, dst = baseDst; src < len; dst += chunkStride) {
                            const end = Math.min(src + chunkFloats, len);
                            for (let k = 0; src < end; ++src, ++k) staging[dst + k] = values[src];
                        }
                    }
                } else {
                    const staticBlock = staticSlot!.blocks[block.slotIndex];
                    const dstU32 = staticBlock.cpuU32;
                    const scalarView = scalarKind === ConstantScalarKind.FLOAT
                        ? this._scalarF32
                        : (scalarKind === ConstantScalarKind.INT ? this._scalarI32 : this._scalarU32);
                    const scalarU32 = this._scalarU32;
                    if (chunkFloats === chunkStride) {
                        const baseDst = loc.floatOffset;
                        for (let i = 0; i < len; ++i) {
                            scalarView[0] = values[i];
                            const bits = scalarU32[0];
                            const dst = baseDst + i;
                            if (dstU32[dst] === bits) continue;
                            dstU32[dst] = bits;
                            staticBlock.dirty = true;
                        }
                    } else {
                        for (let src = 0, dstBase = loc.floatOffset; src < len; dstBase += chunkStride) {
                            const end = Math.min(src + chunkFloats, len);
                            for (let k = 0; src < end; ++src, ++k) {
                                scalarView[0] = values[src];
                                const bits = scalarU32[0];
                                const dst = dstBase + k;
                                if (dstU32[dst] === bits) continue;
                                dstU32[dst] = bits;
                                staticBlock.dirty = true;
                            }
                        }
                    }
                }
            }
        }

        if (!staticSlot) return;
        for (const block of staticSlot.blocks) {
            if (block.dirty && !block.queued) {
                block.queued = true;
                this._dirtyStaticBlocks.push(block);
            }
        }
    }

    private _resolveResources (
        schema: PassLayoutSchema,
        dataChain: readonly (RenderData | null | undefined)[],
        computeViews: ReadonlyMap<string, readonly ComputeView[]> | null | undefined,
    ): void {
        const uboCount = schema.uboBlocks.length;
        this._resolvedUboBufs.length = uboCount;
        this._resolvedUboBufs.fill(null);

        const resCount = schema.resources.length;
        this._resolvedRawTexs.length = resCount;
        this._resolvedTexViews.length = resCount;
        this._resolvedBufs.length = resCount;
        this._resolvedSamplers.length = resCount;
        this._resolvedExtTexs.length = resCount;
        this._resolvedRawTexs.fill(null);
        this._resolvedTexViews.fill(null);
        this._resolvedBufs.fill(null);
        this._resolvedSamplers.fill(null);
        this._resolvedExtTexs.fill(null);
        this._explicitSamplers.clear();

        for (const data of dataChain) {
            if (!data) continue;
            for (const [id, texture] of data.textures) {
                const resource = schema.resourceByID.get(id);
                if (resource && (resource.kind === WebGPUBindingKind.TEXTURE || resource.kind === WebGPUBindingKind.SAMPLER_TEXTURE
                    || resource.kind === WebGPUBindingKind.STORAGE_TEXTURE)) {
                    this._resolvedRawTexs[resource.index] = texture;
                    const view = this._host.unwrapTextureView(texture);
                    if (view) this._resolvedTexViews[resource.index] = view;
                }
            }
            for (const [id, buffer] of data.buffers) {
                const uboBlockIndex = schema.uboByID.get(id);
                if (uboBlockIndex !== undefined) {
                    const gpuBuffer = this._host.unwrapBuffer(buffer);
                    if (gpuBuffer) this._resolvedUboBufs[uboBlockIndex] = gpuBuffer;
                    continue;
                }
                const resource = schema.resourceByID.get(id);
                if (resource?.kind === WebGPUBindingKind.STORAGE_BUFFER) {
                    const gpuBuffer = this._host.unwrapBuffer(buffer);
                    if (gpuBuffer) this._resolvedBufs[resource.index] = gpuBuffer;
                }
            }
            for (const [id, sampler] of data.samplers) {
                const resource = schema.resourceByID.get(id);
                if (resource?.kind !== WebGPUBindingKind.SAMPLER && resource?.kind !== WebGPUBindingKind.SAMPLER_TEXTURE) continue;
                const gpuSampler = this._host.unwrapSampler(sampler, data.textures.get(id) ?? this._resolvedRawTexs[resource.index]);
                if (gpuSampler) {
                    this._resolvedSamplers[resource.index] = gpuSampler;
                    this._explicitSamplers.add(id);
                }
            }
        }

        if (!computeViews) return;
        for (const [name, views] of computeViews) {
            for (const view of views) {
                const id = this._layoutGraph.attributeIndex.get(view.name);
                if (id === undefined) continue;
                const resource = schema.resourceByID.get(id);
                if (resource) this._resolveNamedResource(name, resource);
            }
        }
    }

    private _resolveNamedResource (name: string, resource: ResourceBindingDesc): void {
        const { index, kind, descriptorID } = resource;
        if (kind === WebGPUBindingKind.STORAGE_BUFFER) {
            const gpuBuffer = this._host.getBufferByName(name);
            if (gpuBuffer) this._resolvedBufs[index] = gpuBuffer;
            return;
        }
        if (kind !== WebGPUBindingKind.SAMPLER) {
            const external = kind === WebGPUBindingKind.STORAGE_TEXTURE ? null : this._host.getExternalTextureByName?.(name);
            if (external) {
                this._resolvedExtTexs[index] = external;
                this._resolvedTexViews[index] = null;
            } else {
                const view = this._host.getTextureViewByName(name);
                if (view) {
                    this._resolvedTexViews[index] = view;
                    this._resolvedExtTexs[index] = null;
                }
            }
        }
        if ((kind === WebGPUBindingKind.SAMPLER || kind === WebGPUBindingKind.SAMPLER_TEXTURE) && !this._explicitSamplers.has(descriptorID)) {
            const sampler = this._host.getSamplerByName(name);
            if (sampler) this._resolvedSamplers[index] = sampler;
        }
    }

    private _writeSignature (schema: PassLayoutSchema, staticSlot: StaticUniformSlot | null, ringChunk: UniformRingChunk | null): boolean {
        const sig = this._sig;
        const boundUIDs = this._boundUIDsScratch;
        boundUIDs.length = 0;
        sig[0] = schema.schemaKey;
        sig[1] = schema.bglUID;

        let n = 2;
        const uboBlocks = schema.uboBlocks;
        for (let blockIndex = 0; blockIndex < uboBlocks.length; ++blockIndex) {
            const extBuf = this._resolvedUboBufs[blockIndex];
            const block = uboBlocks[blockIndex];
            const uid = extBuf
                ? this._host.uidOf(extBuf)
                : (block.isDynamic ? ringChunk!.bufferUID : staticSlot!.blocks[block.slotIndex].bufferUID);
            sig[n++] = uid;
            boundUIDs.push(uid);
        }

        let hasExternalTexture = false;
        for (const { kind, index, viewDimension } of schema.resources) {
            if (kind === WebGPUBindingKind.STORAGE_BUFFER) {
                const buffer = this._resolvedBufs[index] ?? this._host.getDefaultStorageBuffer();
                this._resolvedBufs[index] = buffer;
                const uid = this._host.uidOf(buffer);
                sig[n++] = uid;
                boundUIDs.push(uid);
                continue;
            }
            if (kind !== WebGPUBindingKind.SAMPLER) {
                const external = this._resolvedExtTexs[index];
                if (external) {
                    hasExternalTexture = true;
                } else {
                    const view = this._resolvedTexViews[index] ?? this._host.getDefaultTextureView(viewDimension);
                    this._resolvedTexViews[index] = view;
                    const uid = this._host.uidOf(view);
                    sig[n++] = uid;
                    boundUIDs.push(uid);
                }
            }
            if (kind === WebGPUBindingKind.SAMPLER || kind === WebGPUBindingKind.SAMPLER_TEXTURE) {
                const sampler = this._resolvedSamplers[index] ?? this._host.getDefaultSampler();
                this._resolvedSamplers[index] = sampler;
                const uid = this._host.uidOf(sampler);
                sig[n++] = uid;
                boundUIDs.push(uid);
            }
        }
        return hasExternalTexture;
    }

    private _getOrCreateBindGroup (
        schema: PassLayoutSchema,
        staticSlot: StaticUniformSlot | null,
        ringChunk: UniformRingChunk | null,
        hasExternalTexture: boolean,
    ): GPUBindGroup {
        if (hasExternalTexture) {
            return this._createBindGroup(schema, staticSlot, ringChunk);
        }

        const sigLen = schema.slotSigLen;
        const hash = hashSignature(this._sig, sigLen);
        const bucket = this._cache.get(hash);
        if (bucket) {
            for (const entry of bucket) {
                if (signatureEquals(entry.sig, this._sig, sigLen)) {
                    entry.lastSerial = this._serial;
                    return entry.bindGroup;
                }
            }
        }

        const bindGroup = this._createBindGroup(schema, staticSlot, ringChunk);
        const entry: BindGroupEntry = {
            bindGroup,
            sig: this._sig.slice(0, sigLen),
            boundUIDs: this._boundUIDsScratch.slice(),
            lastSerial: this._serial,
        };
        if (bucket) bucket.push(entry);
        else this._cache.set(hash, [entry]);
        return bindGroup;
    }

    private _createBindGroup (schema: PassLayoutSchema, staticSlot: StaticUniformSlot | null, ringChunk: UniformRingChunk | null): GPUBindGroup {
        const entries = this._entriesScratch;
        entries.length = 0;
        const uboBlocks = schema.uboBlocks;
        for (let blockIndex = 0; blockIndex < uboBlocks.length; ++blockIndex) {
            const { binding, isDynamic, slotIndex, byteSize } = uboBlocks[blockIndex];
            const buffer = this._resolvedUboBufs[blockIndex]
                ?? (isDynamic ? ringChunk!.gpuBuffer : staticSlot!.blocks[slotIndex].gpuBuffer);
            entries.push({ binding, resource: { buffer, offset: 0, size: byteSize } });
        }
        for (const { binding, kind, index, samplerBinding } of schema.resources) {
            if (kind === WebGPUBindingKind.STORAGE_BUFFER) {
                entries.push({ binding, resource: { buffer: this._resolvedBufs[index]! } });
            } else if (kind === WebGPUBindingKind.SAMPLER) {
                entries.push({ binding, resource: this._resolvedSamplers[index]! });
            } else {
                entries.push({ binding, resource: this._resolvedExtTexs[index] ?? this._resolvedTexViews[index]! });
                if (kind === WebGPUBindingKind.SAMPLER_TEXTURE) {
                    entries.push({ binding: samplerBinding, resource: this._resolvedSamplers[index]! });
                }
            }
        }
        const bindGroup = this._host.device.createBindGroup({ layout: schema.bgl, entries });
        entries.length = 0;
        return bindGroup;
    }

    private _sweepUnused (minActiveSerial: number): void {
        this._evictBindGroups((entry): boolean => entry.lastSerial < minActiveSerial);
        for (const [schemaKey, slots] of this._staticSlots) {
            for (const [ownerKey, slot] of slots) {
                if (slot.lastSerial >= minActiveSerial) continue;
                for (const block of slot.blocks) {
                    if (block.queued) {
                        const idx = this._dirtyStaticBlocks.indexOf(block);
                        if (idx >= 0) this._dirtyStaticBlocks.splice(idx, 1);
                        block.queued = false;
                        block.dirty = false;
                    }
                    block.gpuBuffer.destroy();
                }
                slots.delete(ownerKey);
            }
            if (!slots.size) this._staticSlots.delete(schemaKey);
        }
    }

    private _evictBindGroups (shouldEvict: (entry: BindGroupEntry) => boolean): void {
        for (const [hash, bucket] of this._cache) {
            for (let i = bucket.length - 1; i >= 0; --i) {
                if (shouldEvict(bucket[i])) bucket.splice(i, 1);
            }
            if (!bucket.length) this._cache.delete(hash);
        }
    }
}
