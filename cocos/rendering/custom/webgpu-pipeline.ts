/*
 Copyright (c) 2026 Xiamen Yaji Software Co., Ltd.

 https://www.cocos.com/

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

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

import { DEBUG, EDITOR } from 'internal:constants';
import { assert, cclegacy, RecyclePool, Vec4 } from '../../core';
import type { Mat4, Quat, Vec2 } from '../../core/math';
import type {
    Buffer, BufferInfo, CommandBuffer, DescriptorSet, DescriptorSetLayout,
    Device, Swapchain, TextureInfo,
} from '../../gfx';
import {
    Address, API, ClearFlagBit, Color, Filter, Format, LoadOp, ResolveMode, Sampler, SamplerInfo, SampleCount,
    ShaderStageFlagBit, StoreOp, Texture, TextureType, Viewport,
} from '../../gfx';
import type { MacroRecord, RenderScene } from '../../render-scene';
import type { RenderWindow } from '../../render-scene/core/render-window';
import type {
    Camera, DirectionalLight, Model, PointLight, RangedDirectionalLight, SphereLight, SpotLight,
} from '../../render-scene/scene';
import { Light, LightType, ProbeType } from '../../render-scene/scene';
import type { ReflectionProbeManager } from '../../3d';
import type { Scene } from '../../scene-graph';
import type { Director } from '../../game';
import type { Root } from '../../root';
import { Material } from '../../asset/assets';
import { decideProfilerCamera } from '../pipeline-funcs';
import { DebugViewCompositeType } from '../debug-view';
import type { GeometryRenderer } from '../geometry-renderer';
import type { GlobalDSManager } from '../global-descriptor-set-manager';
import { getDefaultShadowTexture } from '../define';
import type { PipelineSceneData } from '../pipeline-scene-data';
import type { LayoutGraphData } from './layout-graph';
import type { ComputePassBuilder, ComputeQueueBuilder, ComputeSubpassBuilder, MultisampleRenderPassBuilder, MultisampleRenderSubpassBuilder, Pipeline, RenderPassBuilder, RenderQueueBuilder, RenderSubpassBuilder, SceneBuilder } from './pipeline';
import { PipelineCapabilities, PipelineType } from './pipeline';
import {
    BlitType, ComputePass, CullingFlags, ManagedResource, MovePass, PersistentTexture, RasterPass, RasterSubpass,
    RenderData, RenderGraph, RenderGraphObjectPool, RenderGraphValue, RenderQueue, RenderSwapchain, ResolvePass, ResourceDesc, ResourceGraph,
    ResourceGraphValue, ResourceStates, ResourceTraits, SceneData,
} from './render-graph';
import { AccessType, AttachmentType, QueueHint, RenderCommonObjectPool, ResolveFlags, ResourceDimension, ResourceFlags, ResourceResidency, SceneFlags } from './types';
import type { CopyPair, LightInfo, MovePair, UpdateFrequency, UploadPair } from './types';
import { ResolvePair } from './types';
import { Executor } from './executor';
import { buildReflectionProbePass } from './define';
import {
    WebSetter, setCameraUBOValues, setShadowUBOLightView, setShadowUBOView, setTextureUBOView,
} from './web-pipeline-types';

/**
 * Fallback sampler chosen from the resource's usage, matching WebPipeline and native conventions:
 * depth/stencil defaults to point filtering; everything else uses SamplerInfo defaults (linear/wrap).
 * Builders can always override per binding via addTexture(name, slot, sampler) or setSampler().
 */
function defaultSamplerInfoFor (flags: ResourceFlags): SamplerInfo {
    if (flags & ResourceFlags.DEPTH_STENCIL_ATTACHMENT) {
        return new SamplerInfo(Filter.POINT, Filter.POINT, Filter.NONE);
    }
    return new SamplerInfo();
}

const _uboVec = new Vec4();
const emptyMaterial = new Material();
const emptyRenderData = new RenderData();

/** Point/clamp sampler info for the pipeline default sampler, same as WebPipeline's. */
const _defaultPointSamplerInfo = new SamplerInfo(
    Filter.POINT,
    Filter.POINT,
    Filter.NONE,
    Address.CLAMP,
    Address.CLAMP,
    Address.CLAMP,
);

/**
 * Cursor pool for one builder kind: acquire() hands out a builder rebound to the
 * requested node, beginFrame() rewinds the cursor so next frame reuses the instances.
 * Builders are rebound through reset(vertID); no per-use constructor threading and no
 * module-level singleton like WebPipeline's pipelinePool.
 */
export class WebGPUBuilderPool<T extends WebGPUSetter> {
    private readonly _items: T[] = [];
    private _used = 0;

    constructor (private readonly _factory: (vertID: number) => T) {}

    /** Returns a builder bound to vertID, reusing a pooled instance when available. */
    public acquire (vertID: number): T {
        let item: T;
        if (this._used < this._items.length) {
            item = this._items[this._used];
            item.reset(vertID);
        } else {
            item = this._factory(vertID);
            this._items.push(item);
        }
        ++this._used;
        return item;
    }

    /** Rewinds the cursor; pooled instances keep their memory for the next frame. */
    public reset (): void {
        this._used = 0;
    }
}

function textureDimensionOf (type: TextureType): ResourceDimension {
    switch (type) {
    case TextureType.TEX1D:
    case TextureType.TEX1D_ARRAY:
        return ResourceDimension.TEXTURE1D;
    case TextureType.TEX2D:
    case TextureType.TEX2D_ARRAY:
    case TextureType.CUBE:
        return ResourceDimension.TEXTURE2D;
    case TextureType.TEX3D:
        return ResourceDimension.TEXTURE3D;
    default:
        throw new Error(`Unsupported texture type: ${String(type)}.`);
    }
}

/**
 * Instance-owned per-frame pool. Replaces WebPipeline's module-level
 * `pipelinePool` singleton: each pipeline owns its graph-object, builder and
 * small-object reuse state. Builders are pooled per kind and rebound through
 * WebGPUSetter.reset(); node objects come from graphObjects' factories.
 */
export class WebGPUFramePool {
    public readonly graphObjects = new RenderGraphObjectPool(new RenderCommonObjectPool());
    public readonly renderPassBuilders: WebGPUBuilderPool<WebGPURenderPassBuilder>;
    public readonly renderSubpassBuilders: WebGPUBuilderPool<WebGPURenderSubpassBuilder>;
    public readonly renderQueueBuilders: WebGPUBuilderPool<WebGPURenderQueueBuilder>;
    public readonly sceneBuilders: WebGPUBuilderPool<WebGPUSceneBuilder>;
    public readonly computePassBuilders: WebGPUBuilderPool<WebGPUComputePassBuilder>;
    public readonly computeQueueBuilders: WebGPUBuilderPool<WebGPUComputeQueueBuilder>;
    public readonly viewports = new RecyclePool(() => new Viewport(), 16);
    public readonly resolvePairs = new RecyclePool(() => new ResolvePair(), 16);
    public readonly colors = new RecyclePool(() => new Color(), 16);
    public readonly samplerInfos = new RecyclePool(() => new SamplerInfo(), 16);

    /** Pooled clear color for pass/view setup, same role as WebPipeline's pipelinePool.createColor. */
    public createColor (x = 0, y = 0, z = 0, w = 0): Color {
        const color = this.colors.add();
        color.set(x, y, z, w);
        return color;
    }

    /** Pooled sampler info, same role as WebPipeline's pipelinePool.createSamplerInfo. */
    public createSamplerInfo (
        minFilter: Filter = Filter.LINEAR,
        magFilter: Filter = Filter.LINEAR,
        mipFilter: Filter = Filter.NONE,
        addressU: Address = Address.WRAP,
        addressV: Address = Address.WRAP,
        addressW: Address = Address.WRAP,
    ): SamplerInfo {
        const info = this.samplerInfos.add();
        info.minFilter = minFilter;
        info.magFilter = magFilter;
        info.mipFilter = mipFilter;
        info.addressU = addressU;
        info.addressV = addressV;
        info.addressW = addressW;
        return info;
    }

    constructor (pipeline: WebGPUPipeline) {
        this.renderPassBuilders = new WebGPUBuilderPool((vertID) => new WebGPURenderPassBuilder(pipeline, vertID));
        this.renderSubpassBuilders = new WebGPUBuilderPool((vertID) => new WebGPURenderSubpassBuilder(pipeline, vertID));
        this.renderQueueBuilders = new WebGPUBuilderPool((vertID) => new WebGPURenderQueueBuilder(pipeline, vertID));
        this.sceneBuilders = new WebGPUBuilderPool((vertID) => new WebGPUSceneBuilder(pipeline, vertID));
        this.computePassBuilders = new WebGPUBuilderPool((vertID) => new WebGPUComputePassBuilder(pipeline, vertID));
        this.computeQueueBuilders = new WebGPUBuilderPool((vertID) => new WebGPUComputeQueueBuilder(pipeline, vertID));
    }

    /** Resets pooled graph objects and rewinds every cursor at the start of each frame. */
    public beginFrame (): void {
        this.graphObjects.reset();
        this.renderPassBuilders.reset();
        this.renderSubpassBuilders.reset();
        this.renderQueueBuilders.reset();
        this.sceneBuilders.reset();
        this.computePassBuilders.reset();
        this.computeQueueBuilders.reset();
        this.viewports.reset();
        this.resolvePairs.reset();
        this.colors.reset();
        this.samplerInfos.reset();
    }
}

/**
 * Base class for all WebGPU builders. Extends the engine's WebSetter, so every
 * constant/texture/sampler write and all builtin camera/light/shadow UBO logic
 * (setCameraUBOValues / setShadowUBOLightView / setShadowUBOView / setTextureUBOView) is inherited unchanged.
 * The only adjustment: layout lookup uses this pipeline's own render graph,
 * not director.root.pipeline's.
 *
 * Builders are created fresh each frame (a handful per frame, not worth pooling)
 * and used synchronously during setup only — same convention as WebPipeline.
 *
 * Caveat: the inherited builtin helpers read director.root.pipeline internally
 * (device/pipelineSceneData/defaultSampler/getCombineSignY), so setBuiltin*Constants
 * require this pipeline to be installed as Root's pipeline.
 */
export class WebGPUSetter extends WebSetter {
    constructor (pipeline: WebGPUPipeline, vertID: number) {
        super(pipeline.renderGraph.getData(vertID), pipeline.layoutGraph);
        this._pipeline = pipeline;
        this._vertID = vertID;
    }

    /**
     * @engineInternal Frame-pool rebind: repoints this builder at another graph node.
     * Subclasses caching node-derived state (e.g. layoutID) must override and recompute.
     */
    public reset (vertID: number): void {
        this._vertID = vertID;
        this._data = this._pipeline.renderGraph.getData(vertID);
    }

    public override getParentLayout (): string {
        const graph = this._pipeline.renderGraph;
        return graph.getLayout(graph.getParent(this._vertID));
    }

    public override getCurrentLayout (): string {
        return this._pipeline.renderGraph.getLayout(this._vertID);
    }

    protected readonly _pipeline: WebGPUPipeline;
}

const enum PipelineState {
    CREATED,
    READY,
    FRAME,
    BUILDING,
    EXECUTING,
    DESTROYED,
}

/**
 * WebGPU pipeline extension framework. Factory hookup pending: createCustomPipeline()
 * currently hardcodes WebPipeline; once wired, the backend selector will create a
 * WebGPUPipeline subclass for WebGPU devices and WebPipeline for WebGL devices.
 *
 * Owns logical graphs and frame transitions, not native GPU resources. A concrete implementation
 * supplies its render-graph build (buildRenderGraph) and backend resource factories; execution is
 * unified on the shared WebPipeline Executor.
 * Root remains responsible for acquire/present; buildRenderGraph can call Director.buildRenderPipeline.
 */
export abstract class WebGPUPipeline extends WebSetter implements Pipeline {
    public readonly type = PipelineType.STANDARD;
    public readonly capabilities = new PipelineCapabilities();
    public readonly renderGraph = new RenderGraph();
    public readonly resourceGraph = new ResourceGraph();
    public readonly macros: MacroRecord = {};
    public enableCpuLightCulling = true;
    public profiler: Model | null = null;

    public abstract readonly globalDSManager: GlobalDSManager;
    public abstract readonly descriptorSetLayout: DescriptorSetLayout;
    public abstract readonly descriptorSet: DescriptorSet;
    public abstract readonly commandBuffers: CommandBuffer[];
    public abstract readonly constantMacros: string;
    public abstract readonly geometryRenderer: GeometryRenderer | null;

    private _state = PipelineState.CREATED;
    private _executor: Executor | null = null;
    private _width = 0;
    private _height = 0;
    private readonly _resourceUses: string[] = [];
    private readonly _copyPassMat: Material = new Material();
    private _defaultSampler: Sampler | null = null;

    /**
     * Per-frame pool: graph node objects, per-kind builders and small objects.
     * Instance-owned, never shared across pipelines. Builders acquire from it
     * during graph construction; beginFrame() rewinds everything.
     */
    public readonly framePool = new WebGPUFramePool(this);

    protected constructor (
        public readonly device: Device,
        public readonly layoutGraph: LayoutGraphData,
        public readonly pipelineSceneData: PipelineSceneData,
    ) {
        // Same pattern as WebPipeline: satisfy the WebSetter constructor first,
        // then rebind to the render graph's global data for pipeline-scope setters.
        super(new RenderData(), layoutGraph);
        this._data = this.renderGraph.globalRenderData;
    }

    /**
     * @engineInternal Factory for the base raster-pass builder, pooled per frame.
     * Subclasses can override/extend once concrete pass data structures land.
     */
    protected createRenderPassBuilder (vertID: number): RenderPassBuilder {
        return this.framePool.renderPassBuilders.acquire(vertID);
    }

    public get width (): number {
        return this._width;
    }

    public get height (): number {
        return this._height;
    }

    /** Resource names touched by the executor this frame; the executor clears it after sweeping. */
    public get resourceUses (): string[] {
        return this._resourceUses;
    }

    /** Same screenSpace/clipSpace Y-sign packing as WebPipeline. */
    public getCombineSignY (): number {
        const caps = this.device.capabilities;
        return ((caps.screenSpaceSignY * 0.5 + 0.5) << 1) | (caps.clipSpaceSignY * 0.5 + 0.5);
    }

    public get defaultSampler (): Sampler {
        if (!this._defaultSampler) {
            this._defaultSampler = this.device.getSampler(_defaultPointSamplerInfo);
        }
        return this._defaultSampler;
    }

    public get defaultShadowTexture (): Texture {
        return getDefaultShadowTexture(this.device);
    }

    public get shadingScale (): number {
        return this.pipelineSceneData.shadingScale;
    }

    public set shadingScale (value: number) {
        this.pipelineSceneData.shadingScale = value;
    }

    public activate (swapchain: Swapchain): boolean {
        this.requireState(PipelineState.CREATED, 'activate');
        if (this.device.gfxAPI !== API.WEBGPU) {
            throw new Error('WebGPUPipeline requires a WebGPU device.');
        }
        this._compileMaterial();
        this._state = PipelineState.READY;
        return true;
    }

    public destroy (): boolean {
        if (this._state === PipelineState.DESTROYED) {
            return true;
        }
        if (this._state !== PipelineState.CREATED && this._state !== PipelineState.READY) {
            throw new Error('WebGPUPipeline cannot be destroyed during a frame.');
        }
        this._executor = null;
        this.renderGraph.clear();
        this.resourceGraph.clear();
        this.profiler = null;
        this._state = PipelineState.DESTROYED;
        return true;
    }

    public render (cameras: Camera[]): void {
        this.requireState(PipelineState.READY, 'render');
        if (cameras.length === 0) {
            return;
        }
        this._applySize(cameras);
        decideProfilerCamera(cameras);
        this.beginFrame();
        try {
            this.buildRenderGraph(cameras);
            this.execute();
        } finally {
            this.endFrame();
        }
    }

    public beginFrame (): void {
        this.requireState(PipelineState.READY, 'beginFrame');
        this.renderGraph.clear();
        // Resets pooled graph node objects and rewinds builder cursors for this frame.
        this.framePool.beginFrame();
        this._state = PipelineState.FRAME;
    }

    public beginSetup (): void {
        this.requireState(PipelineState.FRAME, 'beginSetup');
        this._state = PipelineState.BUILDING;
    }

    public endSetup (): void {
        this.requireState(PipelineState.BUILDING, 'endSetup');
        this._state = PipelineState.EXECUTING;
    }

    public execute (): void {
        this.requireState(PipelineState.EXECUTING, 'execute');
        // Unified execution: reuse the existing WebPipeline Executor.
        if (!this._executor) {
            this._executor = new Executor(
                this,
                this.device,
                this.resourceGraph,
                this.layoutGraph,
                this._width,
                this._height,
            );
        }
        this._executor.resize(this._width, this._height);
        this._executor.execute(this.renderGraph);
    }

    public endFrame (): void {
        if (this._state !== PipelineState.FRAME && this._state !== PipelineState.BUILDING
            && this._state !== PipelineState.EXECUTING) {
            throw new Error('WebGPUPipeline has no completed or abortable frame.');
        }
        this.renderGraph.clear();
        this._state = PipelineState.READY;
    }

    /** Calls beginSetup/endSetup around the selected builder, directly or through Director. */
    protected abstract buildRenderGraph (cameras: Camera[]): void;
    public abstract update (camera: Camera): void;
    public abstract onGlobalPipelineStateChanged (): void;

    /** Builders call this before adding graph nodes or mutating their frame data. */
    protected requireSetup (): void {
        this.requireState(PipelineState.BUILDING, 'build render graph');
    }

    private requireState (state: PipelineState, operation: string): void {
        if (this._state !== state) {
            throw new Error(`WebGPUPipeline cannot ${operation} in state ${this._state}.`);
        }
    }

    /** Resource imports implemented by subclasses use the same mutation boundary. */
    protected requireResourceMutation (): void {
        if (this._state !== PipelineState.CREATED && this._state !== PipelineState.READY
            && this._state !== PipelineState.FRAME && this._state !== PipelineState.BUILDING) {
            throw new Error('WebGPUPipeline resource descriptions are immutable after setup.');
        }
    }

    public getMacroString (name: string): string {
        const value = this.macros[name];
        return typeof value === 'string' ? value : '';
    }

    public getMacroInt (name: string): number {
        const value = this.macros[name];
        return typeof value === 'number' ? value : 0;
    }

    public getMacroBool (name: string): boolean {
        return this.macros[name] === true;
    }

    public setMacroString (name: string, value: string): void {
        this.macros[name] = value;
    }

    public setMacroInt (name: string, value: number): void {
        this.macros[name] = value;
    }

    public setMacroBool (name: string, value: boolean): void {
        this.macros[name] = value;
    }

    public getDescriptorSetLayout (shaderName: string, frequency: UpdateFrequency): DescriptorSetLayout | undefined {
        const phaseID = this.layoutGraph.shaderLayoutIndex.get(shaderName);
        if (phaseID === undefined) {
            return undefined;
        }
        return this.layoutGraph.getLayout(phaseID).getSet(frequency)?.descriptorSetLayout ?? undefined;
    }

    public containsResource (name: string): boolean {
        return this.resourceGraph.contains(name);
    }

    public addBuffer (name: string, size: number, flags: ResourceFlags, residency: ResourceResidency): number {
        return this.addResource(name, ResourceDimension.BUFFER, Format.UNKNOWN, size, 1, 1, 1, 1, SampleCount.X1, flags, residency);
    }

    public addStorageBuffer (name: string, format: Format, size: number, residency = ResourceResidency.MANAGED): number {
        return this.addResource(name, ResourceDimension.BUFFER, format, size, 1, 1, 1, 1, SampleCount.X1, ResourceFlags.STORAGE, residency);
    }

    public addRenderTarget (name: string, format: Format, width: number, height: number, residency = ResourceResidency.MANAGED): number {
        return this.addTexture(
            name,
            TextureType.TEX2D,
            format,
            width,
            height,
            1,
            1,
            1,
            SampleCount.X1,
            ResourceFlags.COLOR_ATTACHMENT | ResourceFlags.SAMPLED,
            residency,
        );
    }

    public addDepthStencil (name: string, format: Format, width: number, height: number, residency = ResourceResidency.MANAGED): number {
        return this.addTexture(
            name,
            TextureType.TEX2D,
            format,
            width,
            height,
            1,
            1,
            1,
            SampleCount.X1,
            ResourceFlags.DEPTH_STENCIL_ATTACHMENT | ResourceFlags.SAMPLED,
            residency,
        );
    }

    public addStorageTexture (name: string, format: Format, width: number, height: number, residency = ResourceResidency.MANAGED): number {
        return this.addTexture(
            name,
            TextureType.TEX2D,
            format,
            width,
            height,
            1,
            1,
            1,
            SampleCount.X1,
            ResourceFlags.STORAGE | ResourceFlags.SAMPLED,
            residency,
        );
    }

    public addTexture (
        name: string,
        type: TextureType,
        format: Format,
        width: number,
        height: number,
        depth: number,
        arraySize: number,
        mipLevels: number,
        sampleCount: SampleCount,
        flags: ResourceFlags,
        residency: ResourceResidency,
    ): number {
        // eslint-disable-next-line max-len
        return this.registerResource(name, textureDimensionOf(type), type, format, width, height, depth, arraySize, mipLevels, sampleCount, flags, residency);
    }

    public addResource (
        name: string,
        dimension: ResourceDimension,
        format: Format,
        width: number,
        height: number,
        depth: number,
        arraySize: number,
        mipLevels: number,
        sampleCount: SampleCount,
        flags: ResourceFlags,
        residency: ResourceResidency,
    ): number {
        let type = TextureType.TEX2D;
        switch (dimension) {
        case ResourceDimension.BUFFER:
            break;
        case ResourceDimension.TEXTURE1D:
            type = arraySize > 1 ? TextureType.TEX1D_ARRAY : TextureType.TEX1D;
            break;
        case ResourceDimension.TEXTURE2D:
            type = arraySize > 1 ? TextureType.TEX2D_ARRAY : TextureType.TEX2D;
            break;
        case ResourceDimension.TEXTURE3D:
            type = TextureType.TEX3D;
            break;
        default:
            throw new Error(`Unsupported resource dimension: ${String(dimension)}.`);
        }
        return this.registerResource(name, dimension, type, format, width, height, depth, arraySize, mipLevels, sampleCount, flags, residency);
    }

    private registerResource (
        name: string,
        dimension: ResourceDimension,
        viewType: TextureType,
        format: Format,
        width: number,
        height: number,
        depth: number,
        arraySize: number,
        mipLevels: number,
        sampleCount: SampleCount,
        flags: ResourceFlags,
        residency: ResourceResidency,
    ): number {
        this.requireResourceMutation();
        if (residency !== ResourceResidency.MANAGED && residency !== ResourceResidency.MEMORYLESS
            && residency !== ResourceResidency.PERSISTENT) {
            throw new Error(`Resource '${name}' must use a window or external import API for this residency.`);
        }
        const graph = this.resourceGraph;
        const id = graph.find(name);
        if (id !== graph.N) {
            const desc = this.getOwnedResource(name);
            if (graph.getTraits(id).residency !== residency || desc.dimension !== dimension || desc.viewType !== viewType) {
                throw new Error(`Resource '${name}' cannot change residency, dimension or view type after registration.`);
            }
            this.assignDescription(desc, format, width, height, depth, arraySize, mipLevels, sampleCount, flags);
            return id;
        }
        const desc = new ResourceDesc();
        desc.dimension = dimension;
        desc.viewType = viewType;
        this.assignDescription(desc, format, width, height, depth, arraySize, mipLevels, sampleCount, flags);
        return graph.addVertex(
            ResourceGraphValue.Managed,
            new ManagedResource(),
            name,
            desc,
            new ResourceTraits(residency),
            new ResourceStates(),
            defaultSamplerInfoFor(flags),
        );
    }

    private getOwnedResource (name: string): ResourceDesc {
        const graph = this.resourceGraph;
        const id = graph.find(name);
        if (id === graph.N) {
            throw new Error(`Resource '${name}' is not registered.`);
        }
        const residency = graph.getTraits(id).residency;
        if (residency === ResourceResidency.EXTERNAL || residency === ResourceResidency.BACKBUFFER) {
            throw new Error(`Resource '${name}' must be updated through its import API.`);
        }
        return graph.getDesc(id);
    }

    private assignDescription (
        desc: ResourceDesc,
        format: Format,
        width: number,
        height: number,
        depth: number,
        arraySize: number,
        mipLevels: number,
        sampleCount: SampleCount,
        flags: ResourceFlags,
    ): void {
        const depthOrArraySize = desc.dimension === ResourceDimension.TEXTURE3D ? depth : arraySize;
        if (desc.format === format && desc.width === width && desc.height === height
            && desc.depthOrArraySize === depthOrArraySize && desc.mipLevels === mipLevels
            && desc.sampleCount === sampleCount && desc.flags === flags) {
            return;
        }
        desc.format = format;
        desc.width = width;
        desc.height = height;
        desc.depthOrArraySize = depthOrArraySize;
        desc.mipLevels = mipLevels;
        desc.sampleCount = sampleCount;
        desc.flags = flags;
        // Logical description revision, not a GPU object generation or a content version.
        ++this.resourceGraph.version;
    }

    public updateResource (
        name: string,
        format: Format,
        width: number,
        height: number,
        depth: number,
        arraySize: number,
        mipLevels: number,
        sampleCount: SampleCount,
    ): void {
        this.requireResourceMutation();
        const desc = this.getOwnedResource(name);
        this.assignDescription(
            desc,
            format === Format.UNKNOWN ? desc.format : format,
            width,
            height,
            depth,
            arraySize,
            mipLevels,
            sampleCount,
            desc.flags,
        );
    }

    public updateTexture (
        name: string,
        format: Format,
        width: number,
        height: number,
        depth: number,
        arraySize: number,
        mipLevels: number,
        sampleCount: SampleCount,
    ): void {
        this.requireResourceMutation();
        if (this.getOwnedResource(name).dimension === ResourceDimension.BUFFER) {
            throw new Error(`Resource '${name}' is not a texture.`);
        }
        this.updateResource(name, format, width, height, depth, arraySize, mipLevels, sampleCount);
    }

    public updateBuffer (name: string, size: number): void {
        this.updateStorageBuffer(name, size);
    }

    public updateStorageBuffer (name: string, size: number, format = Format.UNKNOWN): void {
        this.requireResourceMutation();
        const desc = this.getOwnedResource(name);
        if (desc.dimension !== ResourceDimension.BUFFER) {
            throw new Error(`Resource '${name}' is not a buffer.`);
        }
        this.assignDescription(
            desc,
            format === Format.UNKNOWN ? desc.format : format,
            size,
            desc.height,
            1,
            desc.depthOrArraySize,
            desc.mipLevels,
            desc.sampleCount,
            desc.flags,
        );
    }

    public updateRenderTarget (name: string, width: number, height: number, format = Format.UNKNOWN): void {
        this.requireResourceMutation();
        const desc = this.getOwnedResource(name);
        if (desc.dimension !== ResourceDimension.TEXTURE2D) {
            throw new Error(`Resource '${name}' is not a 2D texture.`);
        }
        this.assignDescription(
            desc,
            format === Format.UNKNOWN ? desc.format : format,
            width,
            height,
            1,
            desc.depthOrArraySize,
            desc.mipLevels,
            desc.sampleCount,
            desc.flags,
        );
    }

    public updateDepthStencil (name: string, width: number, height: number, format = Format.UNKNOWN): void {
        this.updateRenderTarget(name, width, height, format);
    }

    public updateStorageTexture (name: string, width: number, height: number, format = Format.UNKNOWN): void {
        this.updateRenderTarget(name, width, height, format);
    }

    // ------------------------------------------------------------------
    // Resource import: logical registration only, no GPU work.
    // These hold references; ownership stays with the caller.
    // ------------------------------------------------------------------

    public addRenderWindow (
        name: string,
        format: Format,
        width: number,
        height: number,
        renderWindow: RenderWindow,
        depthStencilName?: string,
    ): number {
        this.requireResourceMutation();
        const graph = this.resourceGraph;
        const existing = graph.find(name);
        if (existing !== graph.N) {
            this.updateRenderWindow(name, renderWindow, depthStencilName);
            return existing;
        }

        if (depthStencilName) {
            this.addWindowDepthStencil(depthStencilName, width, height, renderWindow.swapchain);
        }

        const desc = new ResourceDesc();
        desc.dimension = ResourceDimension.TEXTURE2D;
        desc.width = width;
        desc.height = height;
        desc.depthOrArraySize = 1;
        desc.mipLevels = 1;
        desc.format = renderWindow.framebuffer.colorTextures[0]!.format;
        desc.flags = ResourceFlags.COLOR_ATTACHMENT;

        if (!renderWindow.swapchain) {
            desc.sampleCount = renderWindow.framebuffer.colorTextures[0]!.info.samples;
            return graph.addVertex(
                ResourceGraphValue.Framebuffer,
                renderWindow.framebuffer,
                name,
                desc,
                new ResourceTraits(ResourceResidency.EXTERNAL),
                new ResourceStates(),
                new SamplerInfo(),
            );
        }
        return graph.addVertex(
            ResourceGraphValue.Swapchain,
            new RenderSwapchain(renderWindow.swapchain),
            name,
            desc,
            new ResourceTraits(ResourceResidency.BACKBUFFER),
            new ResourceStates(),
            new SamplerInfo(),
        );
    }

    public updateRenderWindow (name: string, renderWindow: RenderWindow, depthStencilName?: string): void {
        this.requireResourceMutation();
        const graph = this.resourceGraph;
        const id = graph.find(name);
        if (id === graph.N) {
            throw new Error(`Render window '${name}' is not registered.`);
        }
        const desc = graph.getDesc(id);
        if (desc.width !== renderWindow.width || desc.height !== renderWindow.height) {
            desc.width = renderWindow.width;
            desc.height = renderWindow.height;
            ++graph.version;
        }
        const current = graph.object(id);
        if (current !== renderWindow.framebuffer) {
            graph.x[id].j = renderWindow.framebuffer;
        }
        if (depthStencilName) {
            this.addWindowDepthStencil(depthStencilName, renderWindow.width, renderWindow.height, renderWindow.swapchain);
        }
    }

    private addWindowDepthStencil (name: string, width: number, height: number, swapchain: Swapchain | null): number {
        const graph = this.resourceGraph;
        const existing = graph.find(name);
        if (existing !== graph.N) {
            const desc = graph.getDesc(existing);
            const current = graph.object(existing);
            if (swapchain && current instanceof RenderSwapchain) {
                current.swapchain = swapchain;
                desc.format = swapchain.depthStencilTexture.format;
            }
            if (desc.width !== width || desc.height !== height) {
                desc.width = width;
                desc.height = height;
                ++graph.version;
            }
            return existing;
        }

        const desc = new ResourceDesc();
        desc.dimension = ResourceDimension.TEXTURE2D;
        desc.width = width;
        desc.height = height;
        desc.depthOrArraySize = 1;
        desc.mipLevels = 1;
        desc.sampleCount = SampleCount.X1;
        desc.flags = ResourceFlags.DEPTH_STENCIL_ATTACHMENT | ResourceFlags.SAMPLED;

        if (swapchain) {
            desc.format = swapchain.depthStencilTexture.format;
            return graph.addVertex(
                ResourceGraphValue.Swapchain,
                new RenderSwapchain(swapchain, true),
                name,
                desc,
                new ResourceTraits(ResourceResidency.BACKBUFFER),
                new ResourceStates(),
                defaultSamplerInfoFor(desc.flags),
            );
        }
        desc.format = Format.DEPTH_STENCIL;
        return graph.addVertex(
            ResourceGraphValue.Managed,
            new ManagedResource(),
            name,
            desc,
            new ResourceTraits(ResourceResidency.MANAGED),
            new ResourceStates(),
            defaultSamplerInfoFor(desc.flags),
        );
    }

    public addExternalTexture (name: string, texture: Texture, flags: ResourceFlags): number {
        this.requireResourceMutation();
        const graph = this.resourceGraph;
        const existing = graph.find(name);
        if (existing !== graph.N) {
            this.updateExternalTexture(name, texture);
            return existing;
        }
        const info = texture.info;
        const desc = new ResourceDesc();
        desc.dimension = textureDimensionOf(info.type);
        desc.width = info.width;
        desc.height = info.height;
        desc.depthOrArraySize = desc.dimension === ResourceDimension.TEXTURE3D ? info.depth : info.layerCount;
        desc.mipLevels = info.levelCount;
        desc.format = texture.format;
        desc.sampleCount = SampleCount.X1;
        desc.textureFlags = info.flags;
        desc.flags = flags;
        desc.viewType = info.type;
        return graph.addVertex(
            ResourceGraphValue.PersistentTexture,
            new PersistentTexture(texture),
            name,
            desc,
            new ResourceTraits(ResourceResidency.EXTERNAL),
            new ResourceStates(),
            new SamplerInfo(),
        );
    }

    public updateExternalTexture (name: string, texture: Texture): void {
        this.requireResourceMutation();
        const graph = this.resourceGraph;
        const id = graph.find(name);
        if (id === graph.N) {
            return;
        }
        const persistent = graph.value(ResourceGraphValue.PersistentTexture, id);
        persistent.texture = texture;
        const desc = graph.getDesc(id);
        if (desc.width !== texture.info.width || desc.height !== texture.info.height) {
            desc.width = texture.info.width;
            desc.height = texture.info.height;
            ++graph.version;
        }
    }

    // ------------------------------------------------------------------
    // Shading rate: WebGPU has no VRS, but registration is inert metadata.
    // Tolerate registration so cross-platform builders can run; the
    // executor must reject any pass that actually attaches it.
    // (WebPipeline had the same registration but recursed into itself on
    // re-registration; this version updates instead.)
    // ------------------------------------------------------------------

    public addShadingRateTexture (name: string, width: number, height: number, residency = ResourceResidency.MANAGED): number {
        const graph = this.resourceGraph;
        const existing = graph.find(name);
        if (existing !== graph.N) {
            this.updateShadingRateTexture(name, width, height);
            return existing;
        }
        this.requireResourceMutation();
        return this.registerResource(
            name,
            ResourceDimension.TEXTURE2D,
            TextureType.TEX2D,
            Format.R8UI,
            width,
            height,
            1,
            1,
            1,
            SampleCount.X1,
            ResourceFlags.SHADING_RATE | ResourceFlags.STORAGE | ResourceFlags.SAMPLED,
            residency,
        );
    }

    public updateShadingRateTexture (name: string, width: number, height: number): void {
        this.updateRenderTarget(name, width, height);
    }

    // ------------------------------------------------------------------
    // Backend-specific resource factories: `type` selects the concrete
    // physical resource kind; only the backend implementation knows the map.
    // ------------------------------------------------------------------

    public abstract addCustomBuffer (name: string, info: BufferInfo, type: string): number;
    public abstract addCustomTexture (name: string, info: TextureInfo, type: string): number;

    // ------------------------------------------------------------------
    // Graph construction: ports of WebPipeline's implementations onto the
    // instance-owned framePool. Builders come from per-kind builder pools.
    // ------------------------------------------------------------------

    protected addRenderPassImpl (width: number, height: number, layoutName: string, count = 1, quality = 0): WebGPURenderPassBuilder {
        const pool = this.framePool.graphObjects;
        const pass = pool.createRasterPass();
        pass.viewport.width = width;
        pass.viewport.height = height;
        pass.count = count;
        pass.quality = quality;
        const data = pool.createRenderData();
        const vertID = this.renderGraph.addVertex<RenderGraphValue.RasterPass>(
            RenderGraphValue.RasterPass, pass, 'Raster', layoutName, data, !DEBUG,
        );
        const builder = this.framePool.renderPassBuilders.acquire(vertID);
        this._updateRasterPassConstants(builder, width, height);
        setTextureUBOView(builder, this.pipelineSceneData);
        return builder;
    }

    public addRenderPass (width: number, height: number, passName = 'default'): RenderPassBuilder {
        this.requireSetup();
        return this.addRenderPassImpl(width, height, passName);
    }

    public addMultisampleRenderPass (width: number, height: number, count: number,
        quality = 0, passName = 'default'): MultisampleRenderPassBuilder {
        this.requireSetup();
        const builder = this.addRenderPassImpl(width, height, passName, count, quality);
        builder.addRenderSubpass();
        return builder;
    }

    public addComputePass (passName: string): ComputePassBuilder {
        this.requireSetup();
        const pool = this.framePool.graphObjects;
        const pass = pool.createComputePass();
        const data = pool.createRenderData();
        const vertID = this.renderGraph.addVertex<RenderGraphValue.Compute>(
            RenderGraphValue.Compute, pass, 'Compute', passName, data, !DEBUG,
        );
        // WebPipeline's setComputeConstants is an empty stub; nothing to port there.
        return this.framePool.computePassBuilders.acquire(vertID);
    }

    public addCopyPass (copyPairs: CopyPair[]): void {
        this.requireSetup();
        for (const pair of copyPairs) {
            const targetName = pair.target;
            const tarVerId = this.resourceGraph.find(targetName);
            const resDesc = this.resourceGraph.getDesc(tarVerId);
            const builder = this.addRenderPassImpl(resDesc.width, resDesc.height, 'copy-pass');
            builder.addRenderTarget(targetName, LoadOp.CLEAR, StoreOp.STORE, this.framePool.createColor());
            builder.setFloat('flip', this.getCombineSignY());
            builder.addTexture(pair.source, 'outputResultMap');
            builder.addQueue(QueueHint.NONE).addFullscreenQuad(this._copyPassMat, 0, SceneFlags.NONE);
        }
    }

    public addUploadPass (uploadPairs: UploadPair[]): void {
        this.requireSetup();
        const pool = this.framePool.graphObjects;
        const pass = pool.createCopyPass();
        for (const pair of uploadPairs) {
            pass.uploadPairs.push(pair);
        }
        this.renderGraph.addVertex<RenderGraphValue.Copy>(
            RenderGraphValue.Copy, pass, 'UploadPass', '', pool.createRenderData(), !DEBUG,
        );
    }

    public addMovePass (movePairs: MovePair[]): void {
        this.requireSetup();
        const pool = this.framePool.graphObjects;
        const pass = pool.createMovePass();
        for (const pair of movePairs) {
            pass.movePairs.push(pair);
        }
        this.renderGraph.addVertex<RenderGraphValue.Move>(
            RenderGraphValue.Move, pass, 'MovePass', '', pool.createRenderData(), !DEBUG,
        );
    }

    public addResolvePass (resolvePairs: ResolvePair[]): void {
        this.requireSetup();
        // Rare enough to stay unpooled; the graph object pool has no ResolvePass factory.
        const pass = new ResolvePass();
        for (const pair of resolvePairs) {
            pass.resolvePairs.push(pair);
        }
        this.renderGraph.addVertex<RenderGraphValue.Resolve>(
            RenderGraphValue.Resolve, pass, 'ResolvePass', '', this.framePool.graphObjects.createRenderData(), !DEBUG,
        );
    }

    public addBuiltinReflectionProbePass (camera: Camera): void {
        this.requireSetup();
        const reflectionProbeManager = cclegacy.internal.reflectionProbeManager as ReflectionProbeManager;
        if (!reflectionProbeManager) return;
        const probes = reflectionProbeManager.getProbes();
        if (probes.length === 0) return;
        for (let i = 0; i < probes.length; i++) {
            const probe = probes[i];
            if (probe.needRender) {
                if (probes[i].probeType === ProbeType.PLANAR) {
                    buildReflectionProbePass(camera, this, probe, probe.realtimePlanarTexture!.window!, 0);
                } else if (EDITOR) {
                    for (let faceIdx = 0; faceIdx < probe.bakedCubeTextures.length; faceIdx++) {
                        probe.updateCameraDir(faceIdx);
                        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
                        buildReflectionProbePass(camera, this, probe, probe.bakedCubeTextures[faceIdx].window!, faceIdx);
                    }
                    probe.needRender = false;
                }
            }
        }
    }

    private _compileMaterial (): void {
        this._copyPassMat.initialize({
            effectName: 'pipeline/copy-pass',
        });
        for (let i = 0; i < this._copyPassMat.passes.length; ++i) {
            this._copyPassMat.passes[i].tryCompile();
        }
    }

    private _applySize (cameras: Camera[]): void {
        let newWidth = this._width;
        let newHeight = this._height;
        for (const camera of cameras) {
            const window = camera.window;
            newWidth = Math.max(window.width, newWidth);
            newHeight = Math.max(window.height, newHeight);
        }
        if (newWidth !== this._width || newHeight !== this._height) {
            this._width = newWidth;
            this._height = newHeight;
        }
    }

    /** Per-pass global constants (time/screen size/debug view), same content as WebPipeline. */
    private _updateRasterPassConstants (setter: WebSetter, width: number, height: number): void {
        const director: Director = cclegacy.director;
        const root: Root = director.root!;
        _uboVec.set(root.cumulativeTime, root.frameTime, director.getTotalFrames());
        setter.setVec4('cc_time', _uboVec);
        _uboVec.set(width, height, 1.0 / width, 1.0 / height);
        setter.setVec4('cc_screenSize', _uboVec);
        _uboVec.set(width, height, 1.0 / width, 1.0 / height);
        setter.setVec4('cc_nativeSize', _uboVec);
        const debugView = root.debugView;
        _uboVec.set(0.0, 0.0, 0.0, 0.0);
        if (debugView) {
            const debugPackVec: number[] = [debugView.singleMode as number, 0.0, 0.0, 0.0];
            for (let i = DebugViewCompositeType.DIRECT_DIFFUSE as number; i < (DebugViewCompositeType.MAX_BIT_COUNT as number); i++) {
                const idx = i >> 3;
                const bit = i % 8;
                debugPackVec[idx + 1] += (debugView.isCompositeModeEnabled(i) ? 1.0 : 0.0) * (10.0 ** bit);
            }
            debugPackVec[3] += (debugView.lightingWithAlbedo ? 1.0 : 0.0) * (10.0 ** 6.0);
            debugPackVec[3] += (debugView.csmLayerColoration ? 1.0 : 0.0) * (10.0 ** 7.0);
            _uboVec.set(debugPackVec[0], debugPackVec[1], debugPackVec[2], debugPackVec[3]);
        }
        setter.setVec4('cc_debug_view_mode', _uboVec);
    }

    // Pipeline-scope setters (setMat4/setTexture/setBuiltin*Constants/...) are
    // inherited from WebSetter and write into renderGraph.globalRenderData.
}

/**
 * Scene builder. Node object is derived from the graph per access, so pool
 * rebind (reset) only needs the vertID swap in WebGPUSetter.
 */
export class WebGPUSceneBuilder extends WebGPUSetter implements SceneBuilder {
    private get _scene (): SceneData {
        return this._pipeline.renderGraph.object(this._vertID) as SceneData;
    }

    public useLightFrustum (light: Light, csmLevel = 0, optCamera: Camera | undefined = undefined): void {
        const scene = this._scene;
        scene.light.light = light;
        scene.light.level = csmLevel;
        scene.light.culledByLight = true;
        if (optCamera) {
            scene.camera = optCamera;
        }
        if (scene.flags & SceneFlags.NON_BUILTIN) {
            return;
        }
        const graph = this._pipeline.renderGraph;
        const queueId = graph.getParent(this._vertID);
        const passId = graph.getParent(queueId);
        const layoutName = graph.getLayout(passId);
        setShadowUBOLightView(this, scene.camera, light, csmLevel, layoutName);
    }
}

/**
 * Render queue builder: port of WebPipeline's WebRenderQueueBuilder onto the
 * instance-owned framePool. Node object derived from the graph per access.
 */
export class WebGPURenderQueueBuilder extends WebGPUSetter implements RenderQueueBuilder {
    private get _queue (): RenderQueue {
        return this._pipeline.renderGraph.object(this._vertID) as RenderQueue;
    }

    public override get name (): string {
        return this._pipeline.renderGraph.getName(this._vertID);
    }

    public override set name (value: string) {
        this._pipeline.renderGraph.setName(this._vertID, value);
    }

    public addSceneOfCamera (camera: Camera, light: LightInfo, sceneFlags = SceneFlags.NONE, name = 'Camera'): void {
        const lightTarget = light.light;
        const scene = light.probe?.node?.scene?.renderScene || undefined;
        this._addScene(camera, sceneFlags, lightTarget, scene, light);
    }

    public addScene (camera: Camera, sceneFlags = SceneFlags.NONE, light: Light | undefined | null = null, scene: RenderScene | undefined = undefined): SceneBuilder {
        return this._addScene(camera, sceneFlags, light, scene);
    }

    private _addScene (
        camera: Camera,
        sceneFlags: SceneFlags,
        light: Light | undefined | null,
        scene: RenderScene | undefined,
        lightInfo: LightInfo | null = null,
    ): SceneBuilder {
        const pool = this._pipeline.framePool.graphObjects;
        const sceneData = pool.createSceneData(
            scene || camera.scene,
            camera,
            sceneFlags,
            light && !(sceneFlags & SceneFlags.SHADOW_CASTER) ? CullingFlags.CAMERA_FRUSTUM | CullingFlags.LIGHT_BOUNDS : CullingFlags.CAMERA_FRUSTUM,
            light,
        );
        if (lightInfo) {
            sceneData.light.reset(lightInfo.light, lightInfo.level, lightInfo.culledByLight, lightInfo.probe);
        }
        const renderData = pool.createRenderData();
        const graph = this._pipeline.renderGraph;
        const sceneId = graph.addVertex<RenderGraphValue.Scene>(RenderGraphValue.Scene, sceneData, 'Scene', '', renderData, !DEBUG, this._vertID);
        if (!(sceneFlags & SceneFlags.NON_BUILTIN)) {
            const layoutName = this.getParentLayout();
            setCameraUBOValues(
                this,
                camera,
                this._pipeline.pipelineSceneData,
                scene || camera.scene,
                layoutName,
            );
            if (light && light.type !== LightType.DIRECTIONAL) setShadowUBOLightView(this, camera, light, 0, layoutName);
            else if (!(sceneFlags & SceneFlags.SHADOW_CASTER)) setShadowUBOView(this, camera, layoutName);
        }
        const passOrSubpassId = graph.getParent(this._vertID);
        if (sceneFlags & SceneFlags.UI) {
            const queueId = graph.addVertex<RenderGraphValue.Queue>(
                RenderGraphValue.Queue,
                this._queue,
                'UI Queue',
                'default',
                this._data,
                !DEBUG,
                passOrSubpassId,
            );
            graph.addVertex<RenderGraphValue.Blit>(
                RenderGraphValue.Blit,
                pool.createBlit(emptyMaterial, graph.N, SceneFlags.NONE, camera, BlitType.DRAW_2D),
                'UI',
                '',
                emptyRenderData,
                !DEBUG,
                queueId,
            );
        }
        if (sceneFlags & SceneFlags.PROFILER) {
            this.addProfiler(camera);
        }
        return this._pipeline.framePool.sceneBuilders.acquire(sceneId);
    }

    public addFullscreenQuad (material: Material, passID: number, sceneFlags = SceneFlags.NONE, name = 'Quad'): void {
        const pool = this._pipeline.framePool.graphObjects;
        this._pipeline.renderGraph.addVertex<RenderGraphValue.Blit>(
            RenderGraphValue.Blit,
            pool.createBlit(material, passID, sceneFlags, null),
            name,
            '',
            pool.createRenderData(),
            !DEBUG,
            this._vertID,
        );
        const layoutName = this.getParentLayout();
        const scene: Scene | null = cclegacy.director.getScene();
        setCameraUBOValues(
            this,
            null,
            this._pipeline.pipelineSceneData,
            scene ? scene.renderScene : null,
            layoutName,
        );
        if (!(sceneFlags & SceneFlags.SHADOW_CASTER)) {
            setShadowUBOView(this, null, layoutName);
        }
    }

    public addCameraQuad (camera: Camera, material: Material, passID: number, sceneFlags = SceneFlags.NONE): void {
        const pool = this._pipeline.framePool.graphObjects;
        this._pipeline.renderGraph.addVertex<RenderGraphValue.Blit>(
            RenderGraphValue.Blit,
            pool.createBlit(material, passID, sceneFlags, camera),
            'CameraQuad',
            '',
            pool.createRenderData(),
            !DEBUG,
            this._vertID,
        );
        const layoutName = this.getParentLayout();
        const scene: Scene | null = cclegacy.director.getScene();
        setCameraUBOValues(
            this,
            camera,
            this._pipeline.pipelineSceneData,
            camera.scene || (scene ? scene.renderScene : null),
            layoutName,
        );
        if (!(sceneFlags & SceneFlags.SHADOW_CASTER)) {
            setShadowUBOView(this, camera, layoutName);
        }
    }

    public addDraw3D (camera: Camera, models: Model[], sceneFlags = SceneFlags.NON_BUILTIN): void {
        const pool = this._pipeline.framePool.graphObjects;
        const blit = pool.createBlit(emptyMaterial, this._pipeline.renderGraph.N, SceneFlags.NONE, camera, BlitType.DRAW_3D);
        for (const model of models) {
            blit.models.push(model);
        }
        this._pipeline.renderGraph.addVertex<RenderGraphValue.Blit>(
            RenderGraphValue.Blit,
            blit,
            'Draw3D',
            '',
            pool.createRenderData(),
            !DEBUG,
            this._vertID,
        );
        if (!(sceneFlags & SceneFlags.NON_BUILTIN)) {
            const layoutName = this.getParentLayout();
            setCameraUBOValues(
                this,
                camera,
                this._pipeline.pipelineSceneData,
                camera.scene,
                layoutName,
            );
            if (!(sceneFlags & SceneFlags.SHADOW_CASTER)) setShadowUBOView(this, camera, layoutName);
        }
    }

    public addDraw2D (camera: Camera): void {
        const layoutName = this.getParentLayout();
        setCameraUBOValues(
            this,
            camera,
            this._pipeline.pipelineSceneData,
            camera.scene,
            layoutName,
        );
        this._pipeline.renderGraph.addVertex<RenderGraphValue.Blit>(
            RenderGraphValue.Blit,
            this._pipeline.framePool.graphObjects.createBlit(emptyMaterial, this._pipeline.renderGraph.N, SceneFlags.NONE, camera, BlitType.DRAW_2D),
            'Draw2D',
            '',
            emptyRenderData,
            !DEBUG,
            this._vertID,
        );
    }

    public addProfiler (camera: Camera): void {
        const graph = this._pipeline.renderGraph;
        const passOrSubpassId = graph.getParent(this._vertID);
        const queueId = graph.addVertex<RenderGraphValue.Queue>(
            RenderGraphValue.Queue,
            this._queue,
            'UI Queue',
            'default',
            this._data,
            !DEBUG,
            passOrSubpassId,
        );
        const blitID = graph.addVertex<RenderGraphValue.Blit>(
            RenderGraphValue.Blit,
            this._pipeline.framePool.graphObjects.createBlit(emptyMaterial, graph.N, SceneFlags.NONE, camera, BlitType.DRAW_PROFILE),
            'Profiler',
            '',
            emptyRenderData,
            !DEBUG,
            queueId,
        );
        const data = graph.getData(blitID);
        WebSetter.setMat4(this._lg, data, 'cc_matProj', camera.matProj);
    }

    public clearRenderTarget (name: string, color: Color = new Color()): void {
        const pool = this._pipeline.framePool.graphObjects;
        const clearView = pool.createClearView(name, ClearFlagBit.COLOR);
        clearView.clearColor.copy(color);
        this._pipeline.renderGraph.addVertex<RenderGraphValue.Clear>(
            RenderGraphValue.Clear,
            [clearView],
            'ClearRenderTarget',
            '',
            pool.createRenderData(),
            !DEBUG,
            this._vertID,
        );
    }

    public setViewport (viewport: Viewport): void {
        const currViewport = this._pipeline.framePool.viewports.add();
        this._queue.viewport = currViewport.copy(viewport);
    }

    public addCustomCommand (customBehavior: string): void {
        throw new Error('Method not implemented.');
    }
}

/**
 * Raster subpass builder. Mirrors WebPipeline: most attachment methods are
 * unsupported and throw; addQueue and resolve pairs are the working surface.
 */
export class WebGPURenderSubpassBuilder extends WebGPUSetter implements RenderSubpassBuilder, MultisampleRenderSubpassBuilder {
    private _layoutID = 0xFFFFFFFF;

    constructor (pipeline: WebGPUPipeline, vertID: number) {
        super(pipeline, vertID);
        this._updateLayoutID();
    }

    public override reset (vertID: number): void {
        super.reset(vertID);
        this._updateLayoutID();
    }

    private _updateLayoutID (): void {
        const layoutName = this._pipeline.renderGraph.getLayout(this._vertID);
        this._layoutID = this._pipeline.layoutGraph.locateChild(this._pipeline.layoutGraph.N, layoutName);
    }

    private get _subpass (): RasterSubpass {
        return this._pipeline.renderGraph.object(this._vertID) as RasterSubpass;
    }

    public override get name (): string {
        return this._pipeline.renderGraph.getName(this._vertID);
    }

    public override set name (value: string) {
        this._pipeline.renderGraph.setName(this._vertID, value);
    }

    public addRenderTarget (name: string, accessType: AccessType, slotName?: string, loadOp?: LoadOp, storeOp?: StoreOp, color?: Color): void {
        throw new Error('Method not implemented.');
    }

    public addDepthStencil (name: string, accessType: AccessType, depthSlotName = '', stencilSlotName = '', loadOp = LoadOp.CLEAR, storeOp = StoreOp.STORE, depth = 1, stencil = 0, clearFlag = ClearFlagBit.DEPTH_STENCIL): void {
        throw new Error('Method not implemented.');
    }

    public addTexture (name: string, slotName: string, sampler?: Sampler, plane?: number): void {
        throw new Error('Method not implemented.');
    }

    public addStorageBuffer (name: string, accessType: AccessType, slotName: string): void {
        throw new Error('Method not implemented.');
    }

    public addStorageImage (name: string, accessType: AccessType, slotName: string): void {
        throw new Error('Method not implemented.');
    }

    public setViewport (viewport: Viewport): void {
        throw new Error('Method not implemented.');
    }

    public setCustomShaderStages (name: string, stageFlags: ShaderStageFlagBit): void {
        throw new Error('Method not implemented.');
    }

    public addQueue (hint: QueueHint = QueueHint.RENDER_OPAQUE, phaseName = 'default', passName = ''): RenderQueueBuilder {
        const pool = this._pipeline.framePool.graphObjects;
        const layoutId = this._lg.locateChild(this._layoutID, phaseName);
        const queue = pool.createRenderQueue(hint, layoutId);
        const data = pool.createRenderData();
        const queueID = this._pipeline.renderGraph.addVertex<RenderGraphValue.Queue>(RenderGraphValue.Queue, queue, '', phaseName, data, !DEBUG, this._vertID);
        return this._pipeline.framePool.renderQueueBuilders.acquire(queueID);
    }

    public resolveRenderTarget (source: string, target: string): void {
        const graph = this._pipeline.renderGraph;
        const pass = graph.object(graph.getParent(this._vertID)) as RasterPass;
        const subpassData = pass.subpassGraph.getSubpass(this._subpass.subpassID);
        const resolve = this._pipeline.framePool.resolvePairs.add();
        resolve.reset(source, target, ResolveFlags.COLOR, ResolveMode.AVERAGE, ResolveMode.NONE);
        this._subpass.resolvePairs.push(resolve);
        subpassData.resolvePairs.push(resolve);
    }

    public resolveDepthStencil (source: string, target: string, depthMode?: ResolveMode, stencilMode?: ResolveMode): void {
        const graph = this._pipeline.renderGraph;
        const pass = graph.object(graph.getParent(this._vertID)) as RasterPass;
        const subpassData = pass.subpassGraph.getSubpass(this._subpass.subpassID);
        let flags = ResolveFlags.NONE;
        if (depthMode !== ResolveMode.NONE) {
            flags |= ResolveFlags.DEPTH;
        }
        if (stencilMode !== ResolveMode.NONE) {
            flags |= ResolveFlags.STENCIL;
        }
        const resolve = this._pipeline.framePool.resolvePairs.add();
        resolve.reset(source, target, flags, depthMode!, stencilMode!);
        this._subpass.resolvePairs.push(resolve);
        subpassData.resolvePairs.push(resolve);
    }

    public get showStatistics (): boolean {
        return this._subpass.showStatistics;
    }

    public set showStatistics (enable: boolean) {
        this._subpass.showStatistics = enable;
    }

    public get subpassID (): number {
        return this._vertID;
    }

    public get subpassLayoutID (): number {
        return this._layoutID;
    }
}

/**
 * Raster pass builder: port of WebPipeline's WebRenderPassBuilder. All Setter
 * methods inherited from WebGPUSetter; node object derived from the graph.
 */
export class WebGPURenderPassBuilder extends WebGPUSetter implements RenderPassBuilder, MultisampleRenderPassBuilder {
    private _layoutID = 0xFFFFFFFF;
    private _subpassID = -1;

    constructor (pipeline: WebGPUPipeline, vertID: number) {
        super(pipeline, vertID);
        this._updateLayoutID();
    }

    public override reset (vertID: number): void {
        super.reset(vertID);
        this._subpassID = -1;
        this._updateLayoutID();
    }

    private _updateLayoutID (): void {
        const layoutName = this._pipeline.renderGraph.getLayout(this._vertID);
        this._layoutID = this._pipeline.layoutGraph.locateChild(this._pipeline.layoutGraph.N, layoutName);
    }

    private get _pass (): RasterPass {
        return this._pipeline.renderGraph.object(this._vertID) as RasterPass;
    }

    public override get name (): string {
        return this._pipeline.renderGraph.getName(this._vertID);
    }

    public override set name (value: string) {
        this._pipeline.renderGraph.setName(this._vertID, value);
    }

    public get passID (): number {
        return this._vertID;
    }

    public get passLayoutID (): number {
        return this._layoutID;
    }

    public get showStatistics (): boolean {
        return this._pass.showStatistics;
    }

    public set showStatistics (enable: boolean) {
        this._pass.showStatistics = enable;
    }

    public setVersion (name: string, version: number): void {
        this._pass.versionName = name;
        this._pass.version = version;
    }

    public addRenderTarget (name: string, loadOp = LoadOp.CLEAR, storeOp = StoreOp.STORE, clearColor: Color = new Color()): void {
        let clearFlag = ClearFlagBit.COLOR;
        if (loadOp === LoadOp.LOAD) {
            clearFlag = ClearFlagBit.NONE;
        }
        const view = this._pipeline.framePool.graphObjects.createRasterView(
            '',
            AccessType.WRITE,
            AttachmentType.RENDER_TARGET,
            loadOp,
            storeOp,
            clearFlag,
        );
        view.clearColor.copy(clearColor);
        this._pass.rasterViews.set(name, view);
    }

    public addDepthStencil (name: string, loadOp = LoadOp.CLEAR, storeOp = StoreOp.STORE, depth = 1, stencil = 0, clearFlag = ClearFlagBit.DEPTH_STENCIL): void {
        const view = this._pipeline.framePool.graphObjects.createRasterView(
            '',
            AccessType.WRITE,
            AttachmentType.DEPTH_STENCIL,
            loadOp,
            storeOp,
            clearFlag,
        );
        view.clearColor.set(depth, stencil, 0, 0);
        this._pass.rasterViews.set(name, view);
    }

    public resolveRenderTarget (source: string, target: string): void {
        assert(this._subpassID !== -1);
        const graph = this._pipeline.renderGraph;
        const rasterPass = this._pass;
        const subpass = graph.object(this._subpassID) as RasterSubpass;
        const subpassData = rasterPass.subpassGraph.getSubpass(subpass.subpassID);
        const resolve = this._pipeline.framePool.resolvePairs.add();
        resolve.reset(source, target, ResolveFlags.COLOR, ResolveMode.AVERAGE, ResolveMode.NONE);
        subpass.resolvePairs.push(resolve);
        subpassData.resolvePairs.push(resolve);
    }

    public resolveDepthStencil (source: string, target: string, depthMode?: ResolveMode, stencilMode?: ResolveMode): void {
        assert(this._subpassID !== -1);
        const graph = this._pipeline.renderGraph;
        const subpass = graph.object(this._subpassID) as RasterSubpass;
        let flags = ResolveFlags.NONE;
        if (depthMode !== ResolveMode.NONE) {
            flags |= ResolveFlags.DEPTH;
        }
        if (stencilMode !== ResolveMode.NONE) {
            flags |= ResolveFlags.STENCIL;
        }
        const subpassData = this._pass.subpassGraph.getSubpass(subpass.subpassID);
        const resolve = this._pipeline.framePool.resolvePairs.add();
        resolve.reset(source, target, flags, depthMode!, stencilMode!);
        subpass.resolvePairs.push(resolve);
        subpassData.resolvePairs.push(resolve);
    }

    private _addComputeResource (name: string, accessType: AccessType, slotName: string, plane = 0): void {
        const view = this._pipeline.framePool.graphObjects.createComputeView(slotName);
        view.accessType = accessType;
        view.plane = plane;
        const views = this._pass.computeViews.get(name);
        if (views) {
            views.push(view);
        } else {
            this._pass.computeViews.set(name, [view]);
        }
    }

    public addTexture (name: string, slotName: string, sampler: Sampler | null = null, plane?: number): void {
        this._addComputeResource(name, AccessType.READ, slotName, plane);
        if (sampler) {
            const descriptorID = this._lg.attributeIndex.get(slotName)!;
            this._data.samplers.set(descriptorID, sampler);
        }
    }

    public addStorageBuffer (name: string, accessType: AccessType, slotName: string): void {
        this._addComputeResource(name, accessType, slotName);
    }

    public addStorageImage (name: string, accessType: AccessType, slotName: string): void {
        this._addComputeResource(name, accessType, slotName);
    }

    public addMaterialTexture (resourceName: string, flags?: ShaderStageFlagBit): void {
        throw new Error('Method not implemented.');
    }

    public setCustomShaderStages (name: string, stageFlags: ShaderStageFlagBit): void {
        throw new Error('Method not implemented.');
    }

    public addRenderSubpass (subpassName = ''): RenderSubpassBuilder {
        return this._addSubpass(1, 0, subpassName);
    }

    public addMultisampleRenderSubpass (count: number, quality: number, subpassName = ''): MultisampleRenderSubpassBuilder {
        return this._addSubpass(count, quality, subpassName);
    }

    private _addSubpass (count: number, quality: number, layoutName: string): WebGPURenderSubpassBuilder {
        const pool = this._pipeline.framePool.graphObjects;
        const pass = this._pass;
        const subpassID = pass.subpassGraph.nv();
        pass.subpassGraph.addVertex('Raster', pool.createSubpass());
        const subpass = pool.createRasterSubpass(subpassID, count, quality);
        const data = pool.createRenderData();
        const vertID = this._pipeline.renderGraph.addVertex<RenderGraphValue.RasterSubpass>(RenderGraphValue.RasterSubpass, subpass, 'Raster', layoutName, data, !DEBUG);
        this._subpassID = vertID;
        return this._pipeline.framePool.renderSubpassBuilders.acquire(vertID);
    }

    public addComputeSubpass (subpassName?: string): ComputeSubpassBuilder {
        throw new Error('Method not implemented.');
    }

    public addQueue (hint: QueueHint = QueueHint.RENDER_OPAQUE, phaseName = 'default', passName = ''): RenderQueueBuilder {
        const pool = this._pipeline.framePool.graphObjects;
        const layoutId = this._lg.locateChild(this._layoutID, phaseName);
        const queue = pool.createRenderQueue(hint, layoutId);
        const data = pool.createRenderData();
        const queueID = this._pipeline.renderGraph.addVertex<RenderGraphValue.Queue>(RenderGraphValue.Queue, queue, '', phaseName, data, !DEBUG, this._vertID);
        return this._pipeline.framePool.renderQueueBuilders.acquire(queueID);
    }

    public addFullscreenQuad (material: Material, passID: number, sceneFlags = SceneFlags.NONE, name = 'FullscreenQuad'): void {
        const pool = this._pipeline.framePool.graphObjects;
        const graph = this._pipeline.renderGraph;
        const queue = pool.createRenderQueue(QueueHint.RENDER_TRANSPARENT);
        const queueId = graph.addVertex<RenderGraphValue.Queue>(
            RenderGraphValue.Queue,
            queue,
            'Queue',
            '',
            pool.createRenderData(),
            !DEBUG,
            this._vertID,
        );
        graph.addVertex<RenderGraphValue.Blit>(
            RenderGraphValue.Blit,
            pool.createBlit(material, passID, sceneFlags, null),
            name,
            '',
            pool.createRenderData(),
            !DEBUG,
            queueId,
        );
    }

    public addCameraQuad (camera: Camera, material: Material, passID: number, sceneFlags: SceneFlags, name = 'CameraQuad'): void {
        const pool = this._pipeline.framePool.graphObjects;
        const graph = this._pipeline.renderGraph;
        const queue = pool.createRenderQueue(QueueHint.RENDER_TRANSPARENT);
        const queueId = graph.addVertex<RenderGraphValue.Queue>(
            RenderGraphValue.Queue,
            queue,
            'Queue',
            '',
            pool.createRenderData(),
            !DEBUG,
            this._vertID,
        );
        graph.addVertex<RenderGraphValue.Blit>(
            RenderGraphValue.Blit,
            pool.createBlit(material, passID, sceneFlags, camera),
            name,
            '',
            pool.createRenderData(),
            !DEBUG,
            queueId,
        );
    }

    public setViewport (viewport: Viewport): void {
        this._pass.viewport.copy(viewport);
    }
}

/**
 * Compute queue builder: port of WebPipeline's WebComputeQueueBuilder.
 */
export class WebGPUComputeQueueBuilder extends WebGPUSetter implements ComputeQueueBuilder {
    public override get name (): string {
        return this._pipeline.renderGraph.getName(this._vertID);
    }

    public override set name (value: string) {
        this._pipeline.renderGraph.setName(this._vertID, value);
    }

    public addDispatch (
        threadGroupCountX: number,
        threadGroupCountY: number,
        threadGroupCountZ: number,
        material: Material | null = null,
        passID = 0,
        name = 'Dispatch',
    ): void {
        const pool = this._pipeline.framePool.graphObjects;
        this._pipeline.renderGraph.addVertex<RenderGraphValue.Dispatch>(
            RenderGraphValue.Dispatch,
            pool.createDispatch(material, passID, threadGroupCountX, threadGroupCountY, threadGroupCountZ),
            name,
            '',
            pool.createRenderData(),
            !DEBUG,
            this._vertID,
        );
    }
}

/**
 * Compute pass builder: port of WebPipeline's WebComputePassBuilder.
 */
export class WebGPUComputePassBuilder extends WebGPUSetter implements ComputePassBuilder {
    private _layoutID = 0xFFFFFFFF;

    constructor (pipeline: WebGPUPipeline, vertID: number) {
        super(pipeline, vertID);
        this._updateLayoutID();
    }

    public override reset (vertID: number): void {
        super.reset(vertID);
        this._updateLayoutID();
    }

    private _updateLayoutID (): void {
        const layoutName = this._pipeline.renderGraph.getLayout(this._vertID);
        this._layoutID = this._pipeline.layoutGraph.locateChild(this._pipeline.layoutGraph.N, layoutName);
    }

    private get _pass (): ComputePass {
        return this._pipeline.renderGraph.object(this._vertID) as ComputePass;
    }

    public override get name (): string {
        return this._pipeline.renderGraph.getName(this._vertID);
    }

    public override set name (value: string) {
        this._pipeline.renderGraph.setName(this._vertID, value);
    }

    public addTexture (name: string, slotName: string, sampler: Sampler | null = null, plane?: number): void {
        this._addComputeResource(name, AccessType.READ, slotName, plane);
        if (sampler) {
            const descriptorID = this._lg.attributeIndex.get(slotName)!;
            this._data.samplers.set(descriptorID, sampler);
        }
    }

    public addStorageBuffer (name: string, accessType: AccessType, slotName: string): void {
        this._addComputeResource(name, accessType, slotName);
    }

    public addStorageImage (name: string, accessType: AccessType, slotName: string): void {
        this._addComputeResource(name, accessType, slotName);
    }

    public addMaterialTexture (resourceName: string, flags?: ShaderStageFlagBit): void {
        throw new Error('Method not implemented.');
    }

    public setCustomShaderStages (name: string, stageFlags: ShaderStageFlagBit): void {
        throw new Error('Method not implemented.');
    }

    public addQueue (phaseName = 'default', passName = ''): ComputeQueueBuilder {
        const pool = this._pipeline.framePool.graphObjects;
        const layoutId = this._lg.locateChild(this._layoutID, phaseName);
        const queue = pool.createRenderQueue(QueueHint.RENDER_OPAQUE, layoutId);
        const data = pool.createRenderData();
        const queueID = this._pipeline.renderGraph.addVertex<RenderGraphValue.Queue>(RenderGraphValue.Queue, queue, '', phaseName, data, !DEBUG, this._vertID);
        return this._pipeline.framePool.computeQueueBuilders.acquire(queueID);
    }

    private _addComputeResource (name: string, accessType: AccessType, slotName: string, plane = 0): void {
        const view = this._pipeline.framePool.graphObjects.createComputeView(slotName);
        view.accessType = accessType;
        view.plane = plane;
        const views = this._pass.computeViews.get(name);
        if (views) {
            views.push(view);
        } else {
            this._pass.computeViews.set(name, [view]);
        }
    }
}
