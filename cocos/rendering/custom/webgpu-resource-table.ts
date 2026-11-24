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

import {
    BufferInfo, BufferUsageBit, Color, ColorAttachment, DepthStencilAttachment, FramebufferInfo,
    LoadOp, MemoryUsageBit, RenderPassInfo, StoreOp, TextureInfo, TextureUsageBit,
} from '../../gfx';
import type { Buffer, Device, Framebuffer, RenderPass, Texture } from '../../gfx';
import type { RasterPass, RasterView, ResourceDesc, ResourceGraph } from './render-graph';
import { ResourceGraphValue } from './render-graph';
import { AttachmentType, ResourceDimension, ResourceFlags, ResourceResidency } from './types';
import type { WebGPUGraphCompilerPlan } from './webgpu-executor';

const INVALID_ID = 0xFFFFFFFF;

//=================================================================
// Step 2B: WebGPU 物理资源表与 RenderPass 缓存
//
// 外部资源直接绑定，持久资源按名称复用，瞬态资源按规格与使用区间复用。
// RenderPass 缓存签名包含附件尺寸与 MSAA resolve 目标，资源淘汰时清理关联缓存。
//=================================================================

/** 默认空闲保活帧数（60fps ≈ 0.33s；30fps 可传 15 稳控 0.5s 内，防 iOS Jetsam / 微信小游戏 OOM）。 */
const DEFAULT_MAX_IDLE_FRAMES = 20;
/** 单池空闲条目数上限（不算字节，纯条数封顶，防动态缩放或频繁切换管线时池子短时膨胀）。 */
const DEFAULT_MAX_IDLE_ENTRIES = 16;
/** Buffer 分档最小字节数（对齐 WebGPU 256B Uniform/Storage 偏移粒度）。 */
const MIN_BUFFER_BUCKET_BYTES = 256;
const MAX_COLOR_ATTACHMENTS = 8;
/** RenderPass 签名头部：[colorCount, width, height]（编入宽高以捕获 Swapchain 原地 resize）。 */
const SIG_HEADER = 3;
/** RenderPass 签名单附件槽位：[uid, format, samples, loadOp, storeOp, resolveUID]。 */
const SIG_SLOT = 6;
/** 签名定长缓冲容量：SIG_HEADER + MAX_COLOR_ATTACHMENTS 个颜色槽 + 1 个深度槽。 */
const SIG_CAPACITY = SIG_HEADER + (MAX_COLOR_ATTACHMENTS + 1) * SIG_SLOT;

interface Destroyable {
    destroy (): void;
}

/** WebGPU 后端通过多采样纹理 gpuTexture.resolveTex 传递 MSAA resolveTarget。 */
interface ResolvableTexture {
    readonly gpuTexture?: { resolveTex: unknown };
}

/**
 * 瞬态池里的单张物理资源包装对象（仅 4 个字段）：
 * - `frameLastUse === -1`：本帧尚未被使用（空闲）；
 * - `frameLastUse >= 0`：本帧已被借出，占用至第 `frameLastUse` 步。
 */
interface PoolEntry<T> {
    /** 规格字符串（尺寸/格式/采样数/最小 flags），相同的物理对象才能互相顶替。 */
    readonly key: string;
    /** 真正的底层 GFX Texture 或 Buffer 对象。 */
    readonly res: T;
    /** 最近一次被借出是在第几帧（用于优先挑刚用过的对象 & TTL 超时淘汰）。 */
    lastSerial: number;
    /** 本帧当前占用到的最晚 step 步号（帧末统一刷回 -1）。 */
    frameLastUse: number;
}

/** Path B 持久托管条目（按逻辑资源名索引，规格 key 变化时先触发淘汰回调再重建）。 */
interface PersistentEntry<T> {
    readonly key: string;
    readonly res: T;
}

/** 单帧物理资源与缓存观测指标（衔接后续 Step 5 WebGPUFrameStats）。 */
export interface WebGPUDeviceResourceStats {
    texturesCreated: number;
    texturesAliased: number;
    texturesSameResourceHits: number;
    buffersCreated: number;
    buffersAliased: number;
    passesCreated: number;
    ttlEvictions: number;
    idleTextures: number;
    idleBuffers: number;
}

//-----------------------------------------------------------------
// 2B-1: Pure helpers (usage mapping, bucket keys, signature hash)
//-----------------------------------------------------------------

function bindResolveTarget (tex: Texture, target: Texture | null): void {
    const gpuTex = (tex as unknown as ResolvableTexture).gpuTexture;
    if (gpuTex) {
        gpuTex.resolveTex = target ? (target as unknown as ResolvableTexture).gpuTexture ?? null : null;
    }
}

function toTextureUsage (flags: ResourceFlags): TextureUsageBit {
    let usage = TextureUsageBit.NONE;
    if (flags & ResourceFlags.COLOR_ATTACHMENT) usage |= TextureUsageBit.COLOR_ATTACHMENT;
    if (flags & ResourceFlags.DEPTH_STENCIL_ATTACHMENT) usage |= TextureUsageBit.DEPTH_STENCIL_ATTACHMENT;
    if (flags & ResourceFlags.INPUT_ATTACHMENT) usage |= TextureUsageBit.INPUT_ATTACHMENT;
    if (flags & ResourceFlags.SAMPLED) usage |= TextureUsageBit.SAMPLED;
    if (flags & ResourceFlags.STORAGE) usage |= TextureUsageBit.STORAGE;
    if (flags & ResourceFlags.SHADING_RATE) usage |= TextureUsageBit.SHADING_RATE;
    if (flags & ResourceFlags.TRANSFER_SRC) usage |= TextureUsageBit.TRANSFER_SRC;
    if (flags & ResourceFlags.TRANSFER_DST) usage |= TextureUsageBit.TRANSFER_DST;
    return usage;
}

function toBufferUsage (flags: ResourceFlags): BufferUsageBit {
    let usage = BufferUsageBit.NONE;
    if (flags & ResourceFlags.UNIFORM) usage |= BufferUsageBit.UNIFORM;
    if (flags & ResourceFlags.STORAGE) usage |= BufferUsageBit.STORAGE;
    if (flags & ResourceFlags.INDIRECT) usage |= BufferUsageBit.INDIRECT;
    if (flags & ResourceFlags.TRANSFER_SRC) usage |= BufferUsageBit.TRANSFER_SRC;
    if (flags & ResourceFlags.TRANSFER_DST) usage |= BufferUsageBit.TRANSFER_DST;
    return usage;
}

/** 将逻辑 Buffer 字节数向上取整为 2 的幂分档容量（最小 MIN_BUFFER_BUCKET_BYTES = 256B），收敛动态尺寸桶数量。 */
function roundUpBufferSize (bytes: number): number {
    if (bytes <= MIN_BUFFER_BUCKET_BYTES) return MIN_BUFFER_BUCKET_BYTES;
    const n = Math.ceil(bytes);
    if (n > 0x80000000) return Math.ceil(n / MIN_BUFFER_BUCKET_BYTES) * MIN_BUFFER_BUCKET_BYTES;
    return (1 << (32 - Math.clz32(n - 1))) >>> 0;
}

/** 纹理规格桶键：包含最小有效 flags，天然将 STORAGE 读写纹理与纯光栅 AFBC/DCC RT 分桶隔离。 */
function textureKey (desc: ResourceDesc, effectiveFlags: ResourceFlags): string {
    return `${desc.viewType}|${desc.format}|${desc.width}x${desc.height}x${desc.depthOrArraySize}`
        + `|${desc.mipLevels}|${desc.sampleCount}|${effectiveFlags}|${desc.textureFlags}`;
}

/** Buffer 规格桶键：按有效 flags（Usage Scope 隔离）与 2 的幂分档字节数 roundUpBufferSize(desc.width) 分桶。 */
function bufferKey (desc: ResourceDesc, effectiveFlags: ResourceFlags): string {
    return `${effectiveFlags}|${roundUpBufferSize(desc.width)}`;
}

function isTransient (resGraph: ResourceGraph, resID: number): boolean {
    const residency = resGraph.getTraits(resID).residency;
    return residency === ResourceResidency.MANAGED || residency === ResourceResidency.MEMORYLESS;
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

//-----------------------------------------------------------------
// 2B-2: Minimal Transient Pool (_buckets + _lastAssigned) & Persistent Helpers
//-----------------------------------------------------------------

/**
 * 极简瞬态资源池（管理所有 Path C 的 Texture 或 Buffer）：
 * - `_buckets`：按规格 key 分桶的总仓库，靠 `entry.frameLastUse` 区分本帧空闲(-1)与已借出(>=0)；
 * - `_lastAssigned`：上次分配记录表（资源名 -> PoolEntry），优先把同一张物理对象发给同一资源名，稳态 0 BindGroup 重建；
 * - `_leased`：本帧发出去过的对象名单，帧末 O(K) 把 `frameLastUse` 刷回 -1。
 */
class WebGPUTransientPool<T extends Destroyable> {
    public aliased = 0;
    public created = 0;
    public sameResourceHits = 0;
    public evicted = 0;
    public idleCount = 0;

    /** 总仓库：规格 key -> 该规格下的全部物理对象列表。 */
    private readonly _buckets = new Map<string, PoolEntry<T>[]>();
    /** 上次分配记录表：资源名(如 'ForwardColor') -> 上一帧分给它的 PoolEntry（仅作优先提示）。 */
    private readonly _lastAssigned = new Map<string, PoolEntry<T>>();
    /** 本帧已借出名单：帧末遍历把每项的 frameLastUse 复位为 -1。 */
    private readonly _leased: PoolEntry<T>[] = [];

    constructor (private readonly _create: (desc: ResourceDesc, flags: ResourceFlags) => T) {}

    public acquire (
        key: string,
        name: string,
        desc: ResourceDesc,
        effectiveFlags: ResourceFlags,
        first: number,
        last: number,
        serial: number,
    ): T {
        let chosen: PoolEntry<T> | null = null;
        const prev = this._lastAssigned.get(name);
        if (prev && prev.key === key && prev.frameLastUse < first) {
            ++this.sameResourceHits;
            chosen = prev;
        } else {
            let bucket = this._buckets.get(key);
            if (bucket) {
                for (const entry of bucket) {
                    if (entry.frameLastUse >= first) continue;
                    if (entry.frameLastUse >= 0) {
                        chosen = entry;
                        break;
                    }
                    if (!chosen || entry.lastSerial > chosen.lastSerial) chosen = entry;
                }
            }
            if (!chosen) {
                chosen = { key, res: this._create(desc, effectiveFlags), lastSerial: serial, frameLastUse: -1 };
                if (!bucket) {
                    bucket = [];
                    this._buckets.set(key, bucket);
                }
                bucket.push(chosen);
                ++this.created;
            }
        }

        if (chosen.frameLastUse >= 0) {
            ++this.aliased;
        } else {
            this._leased.push(chosen);
        }
        chosen.frameLastUse = last;
        chosen.lastSerial = serial;
        this._lastAssigned.set(name, chosen);
        return chosen.res;
    }

    public endFrame (serial: number, maxIdleFrames: number, maxIdleEntries: number, onEvict?: (res: T) => void): void {
        this.evicted = 0;
        for (const entry of this._leased) entry.frameLastUse = -1;
        this._leased.length = 0;

        let idleCount = 0;
        for (const [key, bucket] of this._buckets) {
            for (let i = bucket.length - 1; i >= 0; --i) {
                const entry = bucket[i];
                if (serial - entry.lastSerial > maxIdleFrames) {
                    bucket.splice(i, 1);
                    this._evictEntry(entry, onEvict);
                } else if (entry.lastSerial < serial) {
                    ++idleCount;
                }
            }
            if (!bucket.length) this._buckets.delete(key);
        }

        while (idleCount > maxIdleEntries) {
            let oldestBucket: PoolEntry<T>[] | null = null;
            let oldestIndex = -1;
            let oldestSerial = serial;
            for (const bucket of this._buckets.values()) {
                for (let i = 0; i < bucket.length; ++i) {
                    if (bucket[i].lastSerial < oldestSerial) {
                        oldestBucket = bucket;
                        oldestIndex = i;
                        oldestSerial = bucket[i].lastSerial;
                    }
                }
            }
            if (!oldestBucket) break;
            const entry = oldestBucket[oldestIndex];
            oldestBucket.splice(oldestIndex, 1);
            if (!oldestBucket.length) this._buckets.delete(entry.key);
            this._evictEntry(entry, onEvict);
            --idleCount;
        }
        this.idleCount = idleCount;
    }

    /** 立即销毁所有当前未占用的闲置对象（供切场景、关闭后处理或退后台 visibilitychange 主动释放显存防 Jetsam OOM）。 */
    public flushUnused (onEvict?: (res: T) => void): void {
        for (const [key, bucket] of this._buckets) {
            for (let i = bucket.length - 1; i >= 0; --i) {
                const entry = bucket[i];
                if (entry.frameLastUse === -1) {
                    bucket.splice(i, 1);
                    this._evictEntry(entry, onEvict);
                }
            }
            if (!bucket.length) this._buckets.delete(key);
        }
        this.idleCount = 0;
    }

    public release (onEvict?: (res: T) => void): void {
        for (const bucket of this._buckets.values()) {
            for (const entry of bucket) {
                if (onEvict) onEvict(entry.res);
                entry.res.destroy();
            }
        }
        this._buckets.clear();
        this._lastAssigned.clear();
        this._leased.length = 0;
        this.aliased = 0;
        this.created = 0;
        this.sameResourceHits = 0;
        this.evicted = 0;
        this.idleCount = 0;
    }

    private _evictEntry (entry: PoolEntry<T>, onEvict?: (res: T) => void): void {
        for (const [name, assigned] of this._lastAssigned) {
            if (assigned === entry) this._lastAssigned.delete(name);
        }
        if (onEvict) onEvict(entry.res);
        entry.res.destroy();
        ++this.evicted;
    }
}

function acquirePersistent<T extends Destroyable> (
    map: Map<string, PersistentEntry<T>>,
    name: string,
    key: string,
    desc: ResourceDesc,
    create: (desc: ResourceDesc, flags: ResourceFlags) => T,
    onEvict?: (res: T) => void,
): T {
    const entry = map.get(name);
    if (entry) {
        if (entry.key === key) return entry.res;
        if (onEvict) onEvict(entry.res);
        entry.res.destroy();
    }
    const res = create(desc, desc.flags);
    map.set(name, { key, res });
    return res;
}

function sweepPersistent<T extends Destroyable> (
    map: Map<string, PersistentEntry<T>>,
    resGraph: ResourceGraph,
    onEvict?: (res: T) => void,
): void {
    for (const [name, entry] of map) {
        if (resGraph.find(name) === INVALID_ID) {
            if (onEvict) onEvict(entry.res);
            entry.res.destroy();
            map.delete(name);
        }
    }
}

//-----------------------------------------------------------------
// 2B-3 ~ 2B-5: RenderPass cache entry & physical resource table
//-----------------------------------------------------------------

/**
 * 缓存的物理 RenderPass + Framebuffer 条目：
 * - `clearColors / clearDepth / clearStencil` 每次 acquire 原地刷新，不参与缓存签名；
 * - `textures` 记录引用的全部物理纹理（color + depth + resolve），供淘汰纹理时精确清理对应条目。
 */
export class WebGPURenderPassEntry {
    public clearDepth = 1;
    public clearStencil = 0;
    public lastSerial = 0;

    constructor (
        public readonly renderPass: RenderPass,
        public readonly framebuffer: Framebuffer,
        public readonly clearColors: Color[],
        public readonly sig: Int32Array,
        /** 引用的全部物理纹理（color + 可选 depth + MSAA resolve），textures[0] 提供渲染区宽高。 */
        public readonly textures: Texture[],
    ) {}

    public get width (): number {
        return this.textures[0].width;
    }

    public get height (): number {
        return this.textures[0].height;
    }

    public destroy (): void {
        this.framebuffer.destroy();
        this.renderPass.destroy();
    }
}

/**
 * WebGPU 物理资源总表与 RP/FBO 缓存（衔接上游 CompilerPlan 与下游 UniformTable）：
 *
 * 三条所有权路径（严格对应 webgpu-pipeline.ts 真实顶点类型，死资源 `resFirstUse[r] < 0` 零分配）：
 * - Path A（外部直绑，不拥有）：Swapchain (color / depthStencil)、window Framebuffer、PersistentTexture (addExternalTexture)；
 * - Path B（持久托管，按 name 拥有）：非瞬态 Managed 纹理、PersistentBuffer（及非瞬态 ManagedBuffer）；
 * - Path C（瞬态池化，按规格桶共享 + 优先沿用上次对象）：MANAGED / MEMORYLESS 纹理与 ManagedBuffer。
 */
export class WebGPUDeviceResourceTable {
    public readonly stats: WebGPUDeviceResourceStats = {
        texturesCreated: 0,
        texturesAliased: 0,
        texturesSameResourceHits: 0,
        buffersCreated: 0,
        buffersAliased: 0,
        passesCreated: 0,
        ttlEvictions: 0,
        idleTextures: 0,
        idleBuffers: 0,
    };

    /** 物理资源淘汰通知钩子：供下游 Step 3 (WebGPUUniformTable) 按 uid 在冷路径精准摘除关联 BindGroup（严禁全量 ResetCache）。 */
    public onTextureEvict: ((tex: Texture, uid: number) => void) | null = null;
    public onBufferEvict: ((buf: Buffer, uid: number) => void) | null = null;

    /** 帧递增序号与当帧图引用。 */
    private _serial = 0;
    private _resGraph: ResourceGraph | null = null;
    private _transientOrder = new Int32Array(32);

    /** 当帧逻辑 resID -> 物理 Texture / Buffer O(1) 映射表（帧末清零引用）。 */
    private readonly _frameTex: (Texture | null)[] = [];
    private readonly _frameBuf: (Buffer | null)[] = [];

    /** Path B：持久资源表（name -> PersistentEntry）。 */
    private readonly _persistentTexs = new Map<string, PersistentEntry<Texture>>();
    private readonly _persistentBufs = new Map<string, PersistentEntry<Buffer>>();

    /** Path C：极简瞬态规格桶池及内部工厂/淘汰回调。 */
    private readonly _texPool: WebGPUTransientPool<Texture>;
    private readonly _bufPool: WebGPUTransientPool<Buffer>;
    private readonly _createTex: (desc: ResourceDesc, flags: ResourceFlags) => Texture;
    private readonly _createBuf: (desc: ResourceDesc, flags: ResourceFlags) => Buffer;
    private readonly _handleTexEvict: (tex: Texture) => void;
    private readonly _handleBufEvict: (buf: Buffer) => void;

    /** 全局物理对象 UID 表与零 GC RP/FBO 签名缓存（移动端仅 5~20 条，冷路径线性淘汰即可）。 */
    private _nextUID = 1;
    private readonly _uids = new WeakMap<object, number>();
    private readonly _passes = new Map<number, WebGPURenderPassEntry[]>();
    private readonly _sig = new Int32Array(SIG_CAPACITY);
    private readonly _colorTexs: Texture[] = [];
    private readonly _colorViews: RasterView[] = [];
    private readonly _colorStoreOps: StoreOp[] = [];
    private readonly _resolveTexs: Texture[] = [];
    private readonly _boundResolves = new Map<Texture, Texture>();

    constructor (private readonly _device: Device) {
        this._createTex = (desc, flags): Texture => this._createTexture(desc, flags);
        this._createBuf = (desc, flags): Buffer => this._createBuffer(desc, flags);
        this._handleTexEvict = (tex): void => this._onEvictTexture(tex);
        this._handleBufEvict = (buf): void => this._onEvictBuffer(buf);
        this._texPool = new WebGPUTransientPool(this._createTex);
        this._bufPool = new WebGPUTransientPool(this._createBuf);
    }

    public get serial (): number {
        return this._serial;
    }

    /** 为任意物理 Texture / Buffer 分配稳定的全局单调整型 UID（与下游 BindGroup 缓存共享同一命名空间）。 */
    public uidOf (obj: object): number {
        let id = this._uids.get(obj);
        if (id === undefined) {
            id = this._nextUID++;
            this._uids.set(obj, id);
        }
        return id;
    }

    /**
     * 兑现当帧全部存活逻辑资源的物理句柄（每帧在 plan.compile 之后调用一次）。
     * @param resUsedFlags 可选：由编译器聚合的当帧最小有效 ResourceFlags（未传时回退 desc.flags）。
     */
    public beginFramePlan (plan: WebGPUGraphCompilerPlan, resGraph: ResourceGraph, resUsedFlags?: Uint16Array): void {
        ++this._serial;
        this._resGraph = resGraph;
        this.stats.passesCreated = 0;
        this.stats.ttlEvictions = 0;
        this._texPool.created = 0;
        this._texPool.aliased = 0;
        this._texPool.sameResourceHits = 0;
        this._bufPool.created = 0;
        this._bufPool.aliased = 0;
        this._bufPool.sameResourceHits = 0;

        const resCount = resGraph.nv();
        this._frameTex.length = resCount;
        this._frameTex.fill(null);
        this._frameBuf.length = resCount;
        this._frameBuf.fill(null);
        if (this._transientOrder.length < resCount) {
            this._transientOrder = new Int32Array(Math.max(resCount, this._transientOrder.length * 2));
        }

        const order = this._transientOrder;
        const firstUses = plan.resFirstUse;
        let transientCount = 0;
        for (let r = 0; r < resCount; ++r) {
            const first = firstUses[r];
            if (first === undefined || first < 0) continue;
            switch (resGraph.w(r)) {
            case ResourceGraphValue.Swapchain: {
                const sc = resGraph.value(ResourceGraphValue.Swapchain, r);
                if (sc.swapchain) this._frameTex[r] = sc.isDepthStencil ? sc.swapchain.depthStencilTexture : sc.swapchain.colorTexture;
                break;
            }
            case ResourceGraphValue.Framebuffer: {
                const fb = resGraph.value(ResourceGraphValue.Framebuffer, r);
                this._frameTex[r] = fb.colorTextures[0] ?? fb.depthStencilTexture ?? null;
                break;
            }
            case ResourceGraphValue.PersistentTexture:
                this._frameTex[r] = resGraph.value(ResourceGraphValue.PersistentTexture, r).texture;
                break;
            case ResourceGraphValue.ManagedTexture: {
                const mt = resGraph.value(ResourceGraphValue.ManagedTexture, r);
                if (mt.texture) this._frameTex[r] = mt.texture;
                break;
            }
            case ResourceGraphValue.Managed:
                if (isTransient(resGraph, r)) {
                    order[transientCount++] = r;
                } else {
                    const desc = resGraph.getDesc(r);
                    this._frameTex[r] = acquirePersistent(
                        this._persistentTexs,
                        resGraph.getName(r),
                        textureKey(desc, desc.flags),
                        desc,
                        this._createTex,
                        this._handleTexEvict,
                    );
                }
                break;
            case ResourceGraphValue.PersistentBuffer: {
                const pb = resGraph.value(ResourceGraphValue.PersistentBuffer, r);
                if (pb.buffer) {
                    this._frameBuf[r] = pb.buffer;
                } else {
                    const desc = resGraph.getDesc(r);
                    this._frameBuf[r] = acquirePersistent(
                        this._persistentBufs,
                        resGraph.getName(r),
                        bufferKey(desc, desc.flags),
                        desc,
                        this._createBuf,
                        this._handleBufEvict,
                    );
                }
                break;
            }
            case ResourceGraphValue.ManagedBuffer: {
                const mb = resGraph.value(ResourceGraphValue.ManagedBuffer, r);
                if (mb.buffer) {
                    this._frameBuf[r] = mb.buffer;
                } else if (isTransient(resGraph, r)) {
                    order[transientCount++] = r;
                } else {
                    const desc = resGraph.getDesc(r);
                    this._frameBuf[r] = acquirePersistent(
                        this._persistentBufs,
                        resGraph.getName(r),
                        bufferKey(desc, desc.flags),
                        desc,
                        this._createBuf,
                        this._handleBufEvict,
                    );
                }
                break;
            }
            default:
                break;
            }
        }

        for (let i = 1; i < transientCount; ++i) {
            const r = order[i];
            const first = firstUses[r];
            let j = i;
            while (j > 0 && firstUses[order[j - 1]] > first) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = r;
        }
        for (let i = 0; i < transientCount; ++i) {
            const r = order[i];
            const desc = resGraph.getDesc(r);
            const name = resGraph.getName(r);
            const used = resUsedFlags && r < resUsedFlags.length ? resUsedFlags[r] : 0;
            const effectiveFlags = used ? desc.flags & used : desc.flags;
            const first = firstUses[r];
            const last = plan.resLastUse[r];
            if (resGraph.w(r) === ResourceGraphValue.ManagedBuffer) {
                this._frameBuf[r] = this._bufPool.acquire(bufferKey(desc, effectiveFlags), name, desc, effectiveFlags, first, last, this._serial);
            } else {
                this._frameTex[r] = this._texPool.acquire(textureKey(desc, effectiveFlags), name, desc, effectiveFlags, first, last, this._serial);
            }
        }
        this.stats.texturesCreated = this._texPool.created;
        this.stats.texturesAliased = this._texPool.aliased;
        this.stats.texturesSameResourceHits = this._texPool.sameResourceHits;
        this.stats.buffersCreated = this._bufPool.created;
        this.stats.buffersAliased = this._bufPool.aliased;
    }

    public getTexture (resID: number): Texture | null {
        return this._frameTex[resID] ?? null;
    }

    public getBuffer (resID: number): Buffer | null {
        return this._frameBuf[resID] ?? null;
    }

    public getTextureByName (name: string): Texture | null {
        return this._resGraph ? this.getTexture(this._resGraph.find(name)) : null;
    }

    public getBufferByName (name: string): Buffer | null {
        return this._resGraph ? this.getBuffer(this._resGraph.find(name)) : null;
    }

    /**
     * 获取或创建物理 RenderPass + Framebuffer，同步现绑 MSAA resolveTarget。
     * 若传入 `step` 与 `plan`（且当前处于物理 Pass 末尾），自动对 `resLastUse <= step` 的瞬态附件执行死尾 `StoreOp.DISCARD` 降级。
     */
    public acquireRenderPass (pass: RasterPass, tailStep = -1, plan: WebGPUGraphCompilerPlan | undefined = undefined): WebGPURenderPassEntry | null {
        this._colorTexs.length = 0;
        this._colorViews.length = 0;
        this._colorStoreOps.length = 0;
        this._resolveTexs.length = 0;
        const resGraph = this._resGraph;
        if (!resGraph) return null;

        let n = SIG_HEADER;
        let depthTex: Texture | null = null;
        let depthView: RasterView | null = null;
        let depthStoreOp = StoreOp.STORE;
        for (const [name, view] of pass.rasterViews) {
            const resID = resGraph.find(name);
            const tex = resID !== INVALID_ID ? this._frameTex[resID] : null;
            if (!tex) continue;

            let storeOp = view.storeOp;
            if (plan && tailStep >= 0 && storeOp === StoreOp.STORE && isTransient(resGraph, resID) && plan.resLastUse[resID] <= tailStep) {
                storeOp = StoreOp.DISCARD;
            }
            if (view.attachmentType === AttachmentType.RENDER_TARGET) {
                if (this._colorTexs.length >= MAX_COLOR_ATTACHMENTS) continue;
                const resolve = this._resolveTargetOf(pass, name);
                bindResolveTarget(tex, resolve);
                if (resolve) this._boundResolves.set(tex, resolve);
                else this._boundResolves.delete(tex);
                this._colorTexs.push(tex);
                this._colorViews.push(view);
                this._colorStoreOps.push(storeOp);
                if (resolve) this._resolveTexs.push(resolve);
                n = this._writeSlot(n, tex, view, storeOp, resolve);
            } else if (view.attachmentType === AttachmentType.DEPTH_STENCIL) {
                depthTex = tex;
                depthView = view;
                depthStoreOp = storeOp;
            }
        }

        const colorCount = this._colorTexs.length;
        const primary = colorCount > 0 ? this._colorTexs[0] : depthTex;
        if (!primary) return null;
        this._sig[0] = colorCount;
        this._sig[1] = primary.width;
        this._sig[2] = primary.height;
        n = this._writeSlot(n, depthTex, depthView, depthStoreOp, null);

        const hash = hashSignature(this._sig, n);
        const bucket = this._passes.get(hash);
        let entry: WebGPURenderPassEntry | undefined;
        if (bucket) {
            for (const candidate of bucket) {
                if (signatureEquals(candidate.sig, this._sig, n)) {
                    entry = candidate;
                    break;
                }
            }
        }
        if (!entry) {
            entry = this._createRenderPass(n, depthTex, depthView, depthStoreOp);
            if (bucket) bucket.push(entry);
            else this._passes.set(hash, [entry]);
        }

        entry.lastSerial = this._serial;
        for (let i = 0; i < colorCount; ++i) entry.clearColors[i].copy(this._colorViews[i].clearColor);
        if (depthView) {
            entry.clearDepth = depthView.clearColor.x;
            entry.clearStencil = depthView.clearColor.y;
        }
        return entry;
    }

    /** 帧末回收：执行瞬态池 TTL + 条数上限淘汰、过期 RP 缓存清理、已移除持久顶点清扫，并重置帧引用数组。 */
    public endFrameSweep (
        maxIdleFrames = DEFAULT_MAX_IDLE_FRAMES,
        maxIdleEntries = DEFAULT_MAX_IDLE_ENTRIES,
    ): void {
        this._texPool.endFrame(this._serial, maxIdleFrames, maxIdleEntries, this._handleTexEvict);
        this._bufPool.endFrame(this._serial, maxIdleFrames, maxIdleEntries, this._handleBufEvict);
        this.stats.idleTextures = this._texPool.idleCount;
        this.stats.idleBuffers = this._bufPool.idleCount;
        this.stats.ttlEvictions = this._texPool.evicted + this._bufPool.evicted;

        for (const [hash, bucket] of this._passes) {
            for (let i = bucket.length - 1; i >= 0; --i) {
                const entry = bucket[i];
                if (this._serial - entry.lastSerial > maxIdleFrames) {
                    entry.destroy();
                    bucket.splice(i, 1);
                }
            }
            if (!bucket.length) this._passes.delete(hash);
        }
        if (this._resGraph) {
            sweepPersistent(this._persistentTexs, this._resGraph, this._handleTexEvict);
            sweepPersistent(this._persistentBufs, this._resGraph, this._handleBufEvict);
        }
        this._frameTex.fill(null);
        this._frameBuf.fill(null);
    }

    /** 主动排空全部闲置池对象与冷门 FBO（切场景、关闭全屏后效或退后台 visibilitychange 时调用，防止移动端 Jetsam OOM）。 */
    public flushUnused (): void {
        this._texPool.flushUnused(this._handleTexEvict);
        this._bufPool.flushUnused(this._handleBufEvict);
        for (const [hash, bucket] of this._passes) {
            for (let i = bucket.length - 1; i >= 0; --i) {
                const entry = bucket[i];
                if (entry.lastSerial < this._serial) {
                    entry.destroy();
                    bucket.splice(i, 1);
                }
            }
            if (!bucket.length) this._passes.delete(hash);
        }
        this.stats.idleTextures = 0;
        this.stats.idleBuffers = 0;
    }

    /** 外部纹理更换或移除时，解除帧绑定并销毁引用它的 FBO 与通知下游清理 BindGroup。 */
    public invalidateExternalTexture (name: string, texture: Texture): void {
        if (this._resGraph) {
            const id = this._resGraph.find(name);
            if (id !== INVALID_ID && this._frameTex[id] === texture) this._frameTex[id] = null;
        }
        this._onEvictTexture(texture);
    }

    /** 销毁执行器拥有的全部池化/持久资源与缓存（绝不触碰 Path A 外部资源）。 */
    public release (): void {
        this._texPool.release(this._handleTexEvict);
        this._bufPool.release(this._handleBufEvict);
        for (const entry of this._persistentTexs.values()) {
            this._onEvictTexture(entry.res);
            entry.res.destroy();
        }
        this._persistentTexs.clear();
        for (const entry of this._persistentBufs.values()) {
            this._onEvictBuffer(entry.res);
            entry.res.destroy();
        }
        this._persistentBufs.clear();
        for (const bucket of this._passes.values()) {
            for (const entry of bucket) entry.destroy();
        }
        this._passes.clear();

        for (const src of this._boundResolves.keys()) bindResolveTarget(src, null);
        this._boundResolves.clear();
        this._frameTex.length = 0;
        this._frameBuf.length = 0;
        this._colorTexs.length = 0;
        this._colorViews.length = 0;
        this._colorStoreOps.length = 0;
        this._resolveTexs.length = 0;
        this._resGraph = null;
        this.stats.texturesCreated = 0;
        this.stats.texturesAliased = 0;
        this.stats.texturesSameResourceHits = 0;
        this.stats.buffersCreated = 0;
        this.stats.buffersAliased = 0;
        this.stats.passesCreated = 0;
        this.stats.ttlEvictions = 0;
        this.stats.idleTextures = 0;
        this.stats.idleBuffers = 0;
    }

    private _onEvictTexture (texture: Texture): void {
        for (const [src, target] of this._boundResolves) {
            if (target === texture) {
                bindResolveTarget(src, null);
                this._boundResolves.delete(src);
            } else if (src === texture) {
                this._boundResolves.delete(src);
            }
        }
        for (const [hash, bucket] of this._passes) {
            for (let i = bucket.length - 1; i >= 0; --i) {
                const entry = bucket[i];
                if (entry.textures.includes(texture)) {
                    entry.destroy();
                    bucket.splice(i, 1);
                }
            }
            if (!bucket.length) this._passes.delete(hash);
        }
        const uid = this._uids.get(texture);
        if (uid !== undefined && this.onTextureEvict) this.onTextureEvict(texture, uid);
    }

    private _onEvictBuffer (buffer: Buffer): void {
        const uid = this._uids.get(buffer);
        if (uid !== undefined && this.onBufferEvict) this.onBufferEvict(buffer, uid);
    }

    private _writeSlot (n: number, tex: Texture | null, view: RasterView | null, storeOp: StoreOp, resolve: Texture | null): number {
        if (!tex || !view) {
            this._sig.fill(0, n, n + SIG_SLOT);
        } else {
            this._sig[n] = this.uidOf(tex);
            this._sig[n + 1] = tex.format;
            this._sig[n + 2] = tex.samples;
            this._sig[n + 3] = view.loadOp;
            this._sig[n + 4] = storeOp;
            this._sig[n + 5] = resolve ? this.uidOf(resolve) : 0;
        }
        return n + SIG_SLOT;
    }

    private _resolveTargetOf (pass: RasterPass, source: string): Texture | null {
        for (const subpass of pass.subpassGraph._subpasses) {
            for (const pair of subpass.resolvePairs) {
                if (pair.source === source) return this.getTextureByName(pair.target);
            }
        }
        return null;
    }

    private _createRenderPass (
        sigLen: number,
        depthTex: Texture | null,
        depthView: RasterView | null,
        depthStoreOp: StoreOp,
    ): WebGPURenderPassEntry {
        const colorAttachments: ColorAttachment[] = [];
        const clearColors: Color[] = [];
        const textures = this._colorTexs.slice();
        for (let i = 0; i < this._colorTexs.length; ++i) {
            const tex = this._colorTexs[i];
            colorAttachments.push(new ColorAttachment(tex.format, tex.samples, this._colorViews[i].loadOp, this._colorStoreOps[i]));
            clearColors.push(new Color());
        }
        const depthAttachment = depthTex && depthView
            ? new DepthStencilAttachment(depthTex.format, depthTex.samples, depthView.loadOp, depthStoreOp, depthView.loadOp, depthStoreOp)
            : new DepthStencilAttachment();
        if (depthTex) textures.push(depthTex);
        textures.push(...this._resolveTexs);

        const renderPass = this._device.createRenderPass(new RenderPassInfo(colorAttachments, depthAttachment));
        const framebuffer = this._device.createFramebuffer(new FramebufferInfo(renderPass, this._colorTexs.slice(), depthTex));
        ++this.stats.passesCreated;
        return new WebGPURenderPassEntry(renderPass, framebuffer, clearColors, this._sig.slice(0, sigLen), textures);
    }

    private _createTexture (desc: ResourceDesc, effectiveFlags: ResourceFlags): Texture {
        const is3D = desc.dimension === ResourceDimension.TEXTURE3D;
        const size = Math.max(1, desc.depthOrArraySize);
        return this._device.createTexture(new TextureInfo(
            desc.viewType,
            toTextureUsage(effectiveFlags),
            desc.format,
            desc.width,
            desc.height,
            desc.textureFlags,
            is3D ? 1 : size,
            Math.max(1, desc.mipLevels),
            desc.sampleCount,
            is3D ? size : 1,
        ));
    }

    private _createBuffer (desc: ResourceDesc, effectiveFlags: ResourceFlags): Buffer {
        return this._device.createBuffer(new BufferInfo(
            toBufferUsage(effectiveFlags),
            MemoryUsageBit.DEVICE,
            roundUpBufferSize(desc.width),
        ));
    }
}
