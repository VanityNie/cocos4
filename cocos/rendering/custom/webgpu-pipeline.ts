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

import { systemInfo } from 'pal/system-info';
import { DEBUG, EDITOR } from 'internal:constants';
import { OS } from '../../../pal/system-info/enum-type';
import { assert, cclegacy, macro, Vec4 } from '../../core';
import type {
    BufferInfo, CommandBuffer, DescriptorSet, DescriptorSetLayout,
    Device, Swapchain, TextureInfo,
} from '../../gfx';
import {
    Address, API, deviceManager, Feature, Filter, Format, FormatFeatureBit, LoadOp,
    Sampler, SamplerInfo, SampleCount, StoreOp, Texture, TextureType,
} from '../../gfx';
import type { MacroRecord } from '../../render-scene';
import type { RenderWindow } from '../../render-scene/core/render-window';
import type { Camera, Model } from '../../render-scene/scene';
import { PCFType, ProbeType } from '../../render-scene/scene';
import type { ReflectionProbeManager } from '../../3d';
import type { Director } from '../../game';
import type { Root } from '../../root';
import { Material } from '../../asset/assets';
import { decideProfilerCamera } from '../pipeline-funcs';
import { DebugViewCompositeType } from '../debug-view';
import type { GeometryRenderer } from '../geometry-renderer';
import type { GlobalDSManager } from '../global-descriptor-set-manager';
import { getDefaultShadowTexture, supportsR32FloatTexture, supportsRGBA16HalfFloatTexture, UBOSkinning } from '../define';
import { PipelineSceneData } from '../pipeline-scene-data';
import type { DescriptorSetData, LayoutGraphData } from './layout-graph';
import type {
    ComputePassBuilder, MultisampleRenderPassBuilder, Pipeline, RenderPassBuilder,
} from './pipeline';
import { PipelineCapabilities, PipelineType } from './pipeline';
import {
    ManagedBuffer, ManagedResource, PersistentBuffer, PersistentTexture, RasterPass,
    RenderData, RenderGraph, RenderGraphValue, RenderSwapchain, ResourceDesc, ResourceGraph,
    ResourceGraphValue, ResourceStates, ResourceTraits,
} from './render-graph';
import {
    LightingMode, QueueHint, ResolvePair, ResourceDimension,
    ResourceFlags, ResourceResidency, SceneFlags, UpdateFrequency,
} from './types';
import type { CopyPair, MovePair, UploadPair } from './types';
import { Compiler } from './compiler';
import { Executor } from './executor';
import { buildReflectionProbePass, genHashValue, resetPassMGState, rpMergeInfos } from './define';
import { createGfxDescriptorSetsAndPipelines } from './layout-graph-utils';
import { WebSetter, setTextureUBOView } from './web-pipeline-types';
import { WebGPUFramePool } from './webgpu-pipeline-builders';
import type { WebGPURenderPassBuilder } from './webgpu-pipeline-builders';

export {
    WebGPUBuilderPool,
    WebGPUFramePool,
    WebGPUSetter,
    WebGPUSceneBuilder,
    WebGPURenderQueueBuilder,
    WebGPURenderSubpassBuilder,
    WebGPURenderPassBuilder,
    WebGPUComputeQueueBuilder,
    WebGPUComputePassBuilder,
} from './webgpu-pipeline-builders';

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

/** Point/clamp sampler info for the pipeline default sampler, same as WebPipeline's. */
const _defaultPointSamplerInfo = new SamplerInfo(
    Filter.POINT,
    Filter.POINT,
    Filter.NONE,
    Address.CLAMP,
    Address.CLAMP,
    Address.CLAMP,
);

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

/** Writes the description implied by an external texture; returns whether anything changed. */
function describeExternalTexture (desc: ResourceDesc, texture: Texture): boolean {
    const info = texture.info;
    const dimension = textureDimensionOf(info.type);
    const depthOrArraySize = dimension === ResourceDimension.TEXTURE3D ? info.depth : info.layerCount;
    if (desc.dimension === dimension && desc.viewType === info.type && desc.format === texture.format
        && desc.width === info.width && desc.height === info.height && desc.depthOrArraySize === depthOrArraySize
        && desc.mipLevels === info.levelCount && desc.sampleCount === info.samples && desc.textureFlags === info.flags) {
        return false;
    }
    desc.dimension = dimension;
    desc.viewType = info.type;
    desc.format = texture.format;
    desc.width = info.width;
    desc.height = info.height;
    desc.depthOrArraySize = depthOrArraySize;
    desc.mipLevels = info.levelCount;
    desc.sampleCount = info.samples;
    desc.textureFlags = info.flags;
    return true;
}

/**
 * Removes an unreferenced vertex from ResourceGraph, rebasing parallel arrays,
 * name indices, and remaining edge targets.
 */
function removeResourceVertex (graph: ResourceGraph, v: number): void {
    const vert = graph.x[v];
    assert(vert.o.length === 0 && vert.i.length === 0);
    const name = graph._names[v];
    graph.x.splice(v, 1);
    graph._names.splice(v, 1);
    graph._descs.splice(v, 1);
    graph._traits.splice(v, 1);
    graph._states.splice(v, 1);
    graph._samplerInfo.splice(v, 1);
    graph._valueIndex.delete(name);
    for (let u = v; u < graph._names.length; ++u) {
        graph._valueIndex.set(graph._names[u], u);
    }
    for (const other of graph.x) {
        for (const e of other.o) {
            if ((e.target as number) > v) e.target = (e.target as number) - 1;
        }
        for (const e of other.i) {
            if ((e.target as number) > v) e.target = (e.target as number) - 1;
        }
    }
    ++graph.version;
}

/**
 * Concrete WebGPU custom render pipeline.
 * Owns logical graphs (`renderGraph`, `resourceGraph`, `layoutGraph`) and instance-scoped `framePool`.
 */
export class WebGPUPipeline extends WebSetter implements Pipeline {
    public static readonly CSM_UNIFORM_VECTORS = 61;
    public static readonly GLOBAL_UNIFORM_VECTORS = 64;

    public readonly type = PipelineType.STANDARD;
    public readonly capabilities = new PipelineCapabilities();
    public readonly renderGraph = new RenderGraph();
    public readonly resourceGraph = new ResourceGraph();
    public readonly macros: MacroRecord = {};
    public profiler: Model | null = null;

    public globalDSManager!: GlobalDSManager;
    public descriptorSetLayout!: DescriptorSetLayout;
    public descriptorSet!: DescriptorSet;

    private _inFrame = false;
    private _compiler: Compiler | null = null;
    private _executor: Executor | null = null;
    private _width = 0;
    private _height = 0;
    private _constantMacros = '';
    private _lightingMode = LightingMode.DEFAULT;
    private _enableCpuLightCulling = true;
    private readonly _resourceUses: string[] = [];
    private readonly _pendingRemovals: string[] = [];
    private readonly _copyPassMat: Material = new Material();
    private _defaultSampler: Sampler | null = null;

    /** Baseline frame timing (ms) for profiling build vs execution phases. */
    public lastFrameBuildMs = 0;
    public lastFrameExecuteMs = 0;

    /**
     * Per-frame pool: graph node objects, per-kind builders and small objects.
     * Instance-owned, never shared across pipelines.
     */
    public readonly framePool = new WebGPUFramePool(this);

    public constructor (
        public readonly layoutGraph: LayoutGraphData,
        public readonly device: Device = deviceManager.gfxDevice,
        public readonly pipelineSceneData: PipelineSceneData = new PipelineSceneData(),
    ) {
        super(new RenderData(), layoutGraph);
        this._data = this.renderGraph.globalRenderData;
    }

    public override get name (): string {
        return 'WebGPUPipeline';
    }

    public override hasSampler (name: string): boolean {
        const id = this._lg.attributeIndex.get(name);
        if (id === undefined) {
            return false;
        }
        return this._data.samplers.has(id);
    }

    public override hasTexture (name: string): boolean {
        const id = this._lg.attributeIndex.get(name);
        if (id === undefined) {
            return false;
        }
        return this._data.textures.has(id);
    }

    public get commandBuffers (): CommandBuffer[] {
        return [this.device.commandBuffer];
    }

    public get constantMacros (): string {
        return this._constantMacros;
    }

    public get geometryRenderer (): GeometryRenderer | null {
        return null;
    }

    public get lightingMode (): LightingMode {
        return this._lightingMode;
    }

    public set lightingMode (mode: LightingMode) {
        this._lightingMode = mode;
    }

    public get enableCpuLightCulling (): boolean {
        if (!this._executor) {
            return this._enableCpuLightCulling;
        }
        return this._executor._context.culling.enableLightCulling;
    }

    public set enableCpuLightCulling (enable: boolean) {
        this._enableCpuLightCulling = enable;
        if (this._executor) {
            this._executor._context.culling.enableLightCulling = enable;
        }
    }

    public setCustomPipelineName (name: string): void {
        if (name === 'Deferred') {
            this.setMacroInt('CC_PIPELINE_TYPE', 1);
        }
    }

    public getGlobalDescriptorSetData (): DescriptorSetData | undefined {
        const stageId = this.layoutGraph.locateChild(this.layoutGraph.N, 'default');
        const layout = this.layoutGraph.getLayout(stageId);
        return layout.getSet(UpdateFrequency.PER_PASS);
    }

    public getSamplerInfo (name: string): SamplerInfo | null {
        if (this.containsResource(name)) {
            const verId = this.resourceGraph.vertex(name);
            return this.resourceGraph.getSampler(verId);
        }
        return null;
    }

    public get width (): number {
        return this._width;
    }

    public get height (): number {
        return this._height;
    }

    public get resourceUses (): string[] {
        return this._resourceUses;
    }

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

    protected _generateConstantMacros (clusterEnabled: boolean): void {
        let str = '';
        str += `#define CC_DEVICE_SUPPORT_FLOAT_TEXTURE ${this.device.getFormatFeatures(Format.RGBA32F)
            & (FormatFeatureBit.RENDER_TARGET | FormatFeatureBit.SAMPLED_TEXTURE) ? 1 : 0}\n`;
        str += `#define CC_DEVICE_MAX_VERTEX_UNIFORM_VECTORS ${this.device.capabilities.maxVertexUniformVectors}\n`;
        str += `#define CC_DEVICE_MAX_FRAGMENT_UNIFORM_VECTORS ${this.device.capabilities.maxFragmentUniformVectors}\n`;
        str += `#define CC_DEVICE_CAN_BENEFIT_FROM_INPUT_ATTACHMENT ${this.device.hasFeature(Feature.INPUT_ATTACHMENT_BENEFIT) ? 1 : 0}\n`;
        str += `#define CC_PLATFORM_ANDROID_AND_WEBGL ${systemInfo.os === OS.ANDROID && systemInfo.isBrowser ? 1 : 0}\n`;
        str += `#define CC_ENABLE_WEBGL_HIGHP_STRUCT_VALUES ${macro.ENABLE_WEBGL_HIGHP_STRUCT_VALUES ? 1 : 0}\n`;
        const jointUniformCapacity = UBOSkinning.JOINT_UNIFORM_CAPACITY;
        str += `#define CC_JOINT_UNIFORM_CAPACITY ${jointUniformCapacity}\n`;
        this._constantMacros = str;
        this.layoutGraph.constantMacros = this._constantMacros;
    }

    public activate (swapchain: Swapchain): boolean {
        createGfxDescriptorSetsAndPipelines(this.device, this.layoutGraph);
        this._compileMaterial();
        this.setMacroBool('CC_USE_HDR', this.pipelineSceneData.isHDR);
        this.setMacroBool('CC_USE_FLOAT_OUTPUT', macro.ENABLE_FLOAT_OUTPUT && supportsRGBA16HalfFloatTexture(this.device));
        this._generateConstantMacros(false);
        this.pipelineSceneData.activate(this.device);
        const isFloat = supportsR32FloatTexture(this.device) ? 0 : 1;
        this.setMacroInt('CC_SHADOWMAP_FORMAT', isFloat);
        const isLinear = this.device.gfxAPI === API.WEBGL ? 1 : 0;
        this.setMacroInt('CC_SHADOWMAP_USE_LINEAR_DEPTH', isLinear);
        this._defaultSampler = this.device.getSampler(_defaultPointSamplerInfo);
        this.pipelineSceneData.csmSupported = this.device.capabilities.maxFragmentUniformVectors
            >= (WebGPUPipeline.CSM_UNIFORM_VECTORS + WebGPUPipeline.GLOBAL_UNIFORM_VECTORS);
        this.setMacroBool('CC_SUPPORT_CASCADED_SHADOW_MAP', this.pipelineSceneData.csmSupported);
        this.setMacroInt('CC_SHADOW_TYPE', 0);
        this.setMacroInt('CC_DIR_SHADOW_PCF_TYPE', PCFType.HARD);
        this.setMacroInt('CC_DIR_LIGHT_SHADOW_TYPE', 0);
        this.setMacroBool('CC_CASCADED_LAYERS_TRANSITION', false);
        return true;
    }

    public destroy (): boolean {
        this._inFrame = false;
        this._flushPendingRemovals();
        this._executor?.release();
        this._executor = null;
        this._compiler = null;
        this._copyPassMat.destroy();
        this.renderGraph.clear();
        this.resourceGraph.clear();
        this.pipelineSceneData?.destroy();
        this.profiler = null;
        return true;
    }

    public render (cameras: Camera[]): void {
        if (cameras.length === 0) {
            return;
        }
        this._applySize(cameras);
        decideProfilerCamera(cameras);
        const t0 = performance.now();
        try {
            this.beginFrame();
            const t1 = performance.now();
            this.lastFrameBuildMs = t1 - t0;
            this.execute();
            this.lastFrameExecuteMs = performance.now() - t1;
        } finally {
            this.endFrame();
        }
    }

    public beginFrame (): void {
        this._flushPendingRemovals();
        this._inFrame = true;
        const director: Director = cclegacy.director;
        director.buildRenderPipeline();
    }

    public beginSetup (): void {
        this._inFrame = true;
        this.renderGraph.clear();
        this.framePool.beginFrame();
    }

    public endSetup (): void {
        this.compile();
    }

    public compile (): void {
        resetPassMGState();
        if (DEBUG) {
            if (!this._compiler) {
                this._compiler = new Compiler(this, this.renderGraph, this.resourceGraph, this.layoutGraph);
            }
            this._compiler.compile(this.renderGraph);
        }
        let prevRasterPass: RasterPass | null = null;
        let hasInterveningPass = false;
        this.renderGraph.x.forEach((vert, v) => {
            if (this.renderGraph.getParent(v) !== this.renderGraph.N) {
                return;
            }
            if (DEBUG && !this.renderGraph.getValid(v)) {
                return;
            }
            if (vert.t !== RenderGraphValue.RasterPass) {
                if (prevRasterPass !== null) {
                    hasInterveningPass = true;
                }
                return;
            }
            const pass = vert.j as RasterPass;
            genHashValue(pass);
            if (hasInterveningPass && prevRasterPass !== null) {
                const prevInfo = rpMergeInfos.get(prevRasterPass);
                const currInfo = rpMergeInfos.get(pass);
                if (prevInfo && currInfo && !currInfo.needBeginRP) {
                    prevInfo.needEndRP = true;
                    currInfo.needBeginRP = true;
                }
            }
            prevRasterPass = pass;
            hasInterveningPass = false;
        });
    }

    public execute (): void {
        if (!this._executor) {
            this._executor = new Executor(
                this,
                this.device,
                this.resourceGraph,
                this.layoutGraph,
                this._width,
                this._height,
            );
            this._executor._context.culling.enableLightCulling = this._enableCpuLightCulling;
        }
        this._executor.resize(this._width, this._height);
        this._executor.execute(this.renderGraph);
    }

    public endFrame (): void {
        this.renderGraph.clear();
        this._inFrame = false;
        this._flushPendingRemovals();
    }

    public update (camera: Camera): void {
        // noop
    }

    public onGlobalPipelineStateChanged (): void {
        const builder = cclegacy.rendering.getCustomPipeline(macro.CUSTOM_PIPELINE_NAME);
        if (builder) {
            if (typeof builder.onGlobalPipelineStateChanged === 'function') {
                builder.onGlobalPipelineStateChanged();
            }
            cclegacy.rendering.forceResizeAllWindows();
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

    public addBuffer (name: string, size: number, flags: ResourceFlags, residency: ResourceResidency, format = Format.UNKNOWN): number {
        const graph = this.resourceGraph;
        const id = graph.find(name);
        if (id !== graph.N) {
            this.updateStorageBuffer(name, size, format);
            return id;
        }
        const desc = new ResourceDesc();
        desc.dimension = ResourceDimension.BUFFER;
        this.assignDescription(desc, format, size, 1, 1, 1, 1, SampleCount.X1, flags);
        if (residency === ResourceResidency.PERSISTENT) {
            return graph.addVertex(
                ResourceGraphValue.PersistentBuffer,
                new PersistentBuffer(),
                name,
                desc,
                new ResourceTraits(ResourceResidency.PERSISTENT),
                new ResourceStates(),
                new SamplerInfo(),
            );
        }
        return graph.addVertex(
            ResourceGraphValue.ManagedBuffer,
            new ManagedBuffer(),
            name,
            desc,
            new ResourceTraits(residency),
            new ResourceStates(),
            new SamplerInfo(),
        );
    }

    public addStorageBuffer (name: string, format: Format, size: number, residency = ResourceResidency.MANAGED): number {
        return this.addBuffer(name, size, ResourceFlags.STORAGE, residency, format);
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
        const graph = this.resourceGraph;
        const id = graph.find(name);
        if (id !== graph.N) {
            this.updateResource(name, format, width, height, depth, arraySize, mipLevels, sampleCount);
            return id;
        }
        const desc = new ResourceDesc();
        desc.dimension = textureDimensionOf(type);
        desc.viewType = type;
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
        switch (dimension) {
        case ResourceDimension.BUFFER:
            return this.addBuffer(name, width, flags, residency, format);
        case ResourceDimension.TEXTURE1D:
            return this.addTexture(
                name,
                arraySize > 1 ? TextureType.TEX1D_ARRAY : TextureType.TEX1D,
                format,
                width,
                height,
                depth,
                arraySize,
                mipLevels,
                sampleCount,
                flags,
                residency,
            );
        case ResourceDimension.TEXTURE2D:
            return this.addTexture(
                name,
                arraySize > 1 ? TextureType.TEX2D_ARRAY : TextureType.TEX2D,
                format,
                width,
                height,
                depth,
                arraySize,
                mipLevels,
                sampleCount,
                flags,
                residency,
            );
        default:
            return this.addTexture(
                name,
                TextureType.TEX3D,
                format,
                width,
                height,
                depth,
                arraySize,
                mipLevels,
                sampleCount,
                flags,
                residency,
            );
        }
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
        const desc = this.resourceGraph.getDesc(this.resourceGraph.vertex(name));
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
        this.updateResource(name, format, width, height, depth, arraySize, mipLevels, sampleCount);
    }

    public updateBuffer (name: string, size: number): void {
        this.updateStorageBuffer(name, size);
    }

    public updateStorageBuffer (name: string, size: number, format = Format.UNKNOWN): void {
        const desc = this.resourceGraph.getDesc(this.resourceGraph.vertex(name));
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
        const desc = this.resourceGraph.getDesc(this.resourceGraph.vertex(name));
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
        const id = this.resourceGraph.find(name);
        if (id === this.resourceGraph.N) {
            return;
        }
        this.updateRenderTarget(name, width, height, format);
    }

    public updateStorageTexture (name: string, width: number, height: number, format = Format.UNKNOWN): void {
        this.updateRenderTarget(name, width, height, format);
    }

    public addRenderWindow (
        name: string,
        format: Format,
        width: number,
        height: number,
        renderWindow: RenderWindow,
        depthStencilName?: string,
    ): number {
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
        const graph = this.resourceGraph;
        const id = graph.vertex(name);
        const desc = graph.getDesc(id);
        let changed = false;
        if (desc.width !== renderWindow.width || desc.height !== renderWindow.height) {
            desc.width = renderWindow.width;
            desc.height = renderWindow.height;
            changed = true;
        }
        if (graph.h(ResourceGraphValue.Framebuffer, id)) {
            const colorTex = renderWindow.framebuffer.colorTextures[0];
            if (colorTex) {
                if (desc.format !== colorTex.format) {
                    desc.format = colorTex.format;
                    changed = true;
                }
                if (desc.sampleCount !== colorTex.info.samples) {
                    desc.sampleCount = colorTex.info.samples;
                    changed = true;
                }
            }
            if (graph.object(id) !== renderWindow.framebuffer) {
                graph.x[id].j = renderWindow.framebuffer;
                changed = true;
            }
        } else if (graph.h(ResourceGraphValue.Swapchain, id) && renderWindow.swapchain) {
            const sc = graph.value(ResourceGraphValue.Swapchain, id);
            if (sc.swapchain !== renderWindow.swapchain) {
                sc.swapchain = renderWindow.swapchain;
                changed = true;
            }
            const scFormat = renderWindow.swapchain.colorTexture.format;
            if (desc.format !== scFormat) {
                desc.format = scFormat;
                changed = true;
            }
        }
        if (changed) {
            ++graph.version;
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
            let changed = false;
            if (swapchain && current instanceof RenderSwapchain) {
                if (current.swapchain !== swapchain) {
                    current.swapchain = swapchain;
                    changed = true;
                }
                if (desc.format !== swapchain.depthStencilTexture.format) {
                    desc.format = swapchain.depthStencilTexture.format;
                    changed = true;
                }
            }
            if (desc.width !== width || desc.height !== height) {
                desc.width = width;
                desc.height = height;
                changed = true;
            }
            if (changed) {
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
        const graph = this.resourceGraph;
        const existing = graph.find(name);
        if (existing !== graph.N) {
            this.updateExternalTexture(name, texture);
            return existing;
        }
        const desc = new ResourceDesc();
        describeExternalTexture(desc, texture);
        desc.flags = flags;
        return graph.addVertex(
            ResourceGraphValue.PersistentTexture,
            new PersistentTexture(texture),
            name,
            desc,
            new ResourceTraits(ResourceResidency.EXTERNAL),
            new ResourceStates(),
            defaultSamplerInfoFor(flags),
        );
    }

    public updateExternalTexture (name: string, texture: Texture): void {
        const graph = this.resourceGraph;
        const id = graph.find(name);
        if (id === graph.N) {
            return;
        }
        const pending = this._pendingRemovals.indexOf(name);
        if (pending !== -1) {
            this._pendingRemovals.splice(pending, 1);
        }
        const persistent = graph.value(ResourceGraphValue.PersistentTexture, id);
        const previous = persistent.texture;
        persistent.texture = texture;
        const descChanged = describeExternalTexture(graph.getDesc(id), texture);
        if (descChanged) {
            ++graph.version;
        }
        if (previous && (previous !== texture || descChanged)) {
            this._executor?.invalidateExternalTexture(name, previous);
        }
    }

    /**
     * Unregisters an external texture `name`.
     * Inside an active frame, removal is deferred until `endFrame()` so recorded passes
     * still resolve the resource safely; outside a frame, removal is immediate.
     */
    public removeExternalTexture (name: string): void {
        const graph = this.resourceGraph;
        const resID = graph.find(name);
        if (resID === graph.N || !graph.h(ResourceGraphValue.PersistentTexture, resID)) {
            return;
        }
        if (this._inFrame) {
            if (!this._pendingRemovals.includes(name)) {
                this._pendingRemovals.push(name);
            }
        } else {
            this._removeExternalTextureNow(name);
        }
    }

    private _removeExternalTextureNow (name: string): void {
        const graph = this.resourceGraph;
        const resID = graph.find(name);
        if (resID === graph.N || !graph.h(ResourceGraphValue.PersistentTexture, resID)) {
            return;
        }
        const texture = graph.value(ResourceGraphValue.PersistentTexture, resID).texture;
        if (texture) {
            this._executor?.invalidateExternalTexture(name, texture);
        }
        removeResourceVertex(graph, resID);
    }

    private _flushPendingRemovals (): void {
        if (this._pendingRemovals.length === 0) {
            return;
        }
        for (const name of this._pendingRemovals) {
            this._removeExternalTextureNow(name);
        }
        this._pendingRemovals.length = 0;
    }

    public addShadingRateTexture (name: string, width: number, height: number, residency = ResourceResidency.MANAGED): number {
        return this.addTexture(
            name,
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

    public addCustomBuffer (name: string, info: BufferInfo, type: string): number {
        throw new Error('Method not implemented.');
    }

    public addCustomTexture (name: string, info: TextureInfo, type: string): number {
        throw new Error('Method not implemented.');
    }

    protected addRenderPassImpl (width: number, height: number, layoutName: string, count = 1, quality = 0): WebGPURenderPassBuilder {
        const pool = this.framePool.graphObjects;
        const pass = pool.createRasterPass();
        pass.viewport.width = width;
        pass.viewport.height = height;
        pass.count = count;
        pass.quality = quality;
        const data = pool.createRenderData();
        const vertID = this.renderGraph.addVertex<RenderGraphValue.RasterPass>(
            RenderGraphValue.RasterPass,
            pass,
            'Raster',
            layoutName,
            data,
            !DEBUG,
        );
        const builder = this.framePool.renderPassBuilders.acquire(vertID);
        this._updateRasterPassConstants(builder, width, height);
        setTextureUBOView(builder, this.pipelineSceneData);
        return builder;
    }

    public addRenderPass (width: number, height: number, passName = 'default'): RenderPassBuilder {
        return this.addRenderPassImpl(width, height, passName);
    }

    public addMultisampleRenderPass (
        width: number,
        height: number,
        count: number,
        quality = 0,
        passName = 'default',
    ): MultisampleRenderPassBuilder {
        const builder = this.addRenderPassImpl(width, height, passName, count, quality);
        builder.addRenderSubpass();
        return builder;
    }

    public addComputePass (passName: string): ComputePassBuilder {
        const pool = this.framePool.graphObjects;
        const pass = pool.createComputePass();
        const data = pool.createRenderData();
        const vertID = this.renderGraph.addVertex<RenderGraphValue.Compute>(
            RenderGraphValue.Compute,
            pass,
            'Compute',
            passName,
            data,
            !DEBUG,
        );
        return this.framePool.computePassBuilders.acquire(vertID);
    }

    public addCopyPass (copyPairs: CopyPair[]): void {
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
        const pool = this.framePool.graphObjects;
        const pass = pool.createCopyPass();
        for (const pair of uploadPairs) {
            pass.uploadPairs.push(pair);
        }
        this.renderGraph.addVertex<RenderGraphValue.Copy>(
            RenderGraphValue.Copy,
            pass,
            'UploadPass',
            '',
            pool.createRenderData(),
            !DEBUG,
        );
    }

    public addMovePass (movePairs: MovePair[]): void {
        const pool = this.framePool.graphObjects;
        const pass = pool.createMovePass();
        for (const pair of movePairs) {
            pass.movePairs.push(pair);
        }
        this.renderGraph.addVertex<RenderGraphValue.Move>(
            RenderGraphValue.Move,
            pass,
            'MovePass',
            '',
            pool.createRenderData(),
            !DEBUG,
        );
    }

    public addResolvePass (resolvePairs: ResolvePair[]): void {
        const pool = this.framePool.graphObjects;
        const pass = pool.createResolvePass();
        for (const pair of resolvePairs) {
            pass.resolvePairs.push(pair);
        }
        this.renderGraph.addVertex<RenderGraphValue.Resolve>(
            RenderGraphValue.Resolve,
            pass,
            'ResolvePass',
            '',
            pool.createRenderData(),
            !DEBUG,
        );
    }

    public addBuiltinReflectionProbePass (camera: Camera): void {
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

    /**
     * Tracks max active camera window dimensions for transitional Executor compatibility.
     * Will be removed once WebGPUExecutor replaces legacy Executor.
     */
    private _applySize (cameras: Camera[]): void {
        let newWidth = 0;
        let newHeight = 0;
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
