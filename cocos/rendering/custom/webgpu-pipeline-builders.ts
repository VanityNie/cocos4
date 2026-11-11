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

import { DEBUG } from 'internal:constants';
import { assert, cclegacy, RecyclePool } from '../../core';
import type { Sampler } from '../../gfx';
import {
    Address, ClearFlagBit, Color, Filter, LoadOp, ResolveMode,
    SamplerInfo, ShaderStageFlagBit, StoreOp, Viewport,
} from '../../gfx';
import type { RenderScene } from '../../render-scene';
import type { Camera, Model } from '../../render-scene/scene';
import { Light, LightType } from '../../render-scene/scene';
import type { Scene } from '../../scene-graph';
import { Material } from '../../asset/assets';
import type {
    ComputePassBuilder, ComputeQueueBuilder, ComputeSubpassBuilder, MultisampleRenderPassBuilder,
    MultisampleRenderSubpassBuilder, RenderPassBuilder, RenderQueueBuilder, RenderSubpassBuilder, SceneBuilder,
} from './pipeline';
import {
    BlitType, CullingFlags, RenderData, RenderGraphObjectPool, RenderGraphValue,
} from './render-graph';
import type {
    ComputePass, RasterPass, RasterSubpass, RenderQueue, SceneData,
} from './render-graph';
import {
    AccessType, AttachmentType, QueueHint, RenderCommonObjectPool, ResolveFlags, ResolvePair, SceneFlags,
} from './types';
import type { LightInfo } from './types';
import {
    WebSetter, setCameraUBOValues, setShadowUBOLightView, setShadowUBOView,
} from './web-pipeline-types';
import type { WebGPUPipeline } from './webgpu-pipeline';

const emptyMaterial = new Material();
const emptyRenderData = new RenderData();

/**
 * Cursor pool for one builder kind: acquire() hands out a builder rebound to the
 * requested node, reset() rewinds the cursor so next frame reuses the instances.
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
 * Builders are pooled per kind in `WebGPUFramePool`, rebound per node via `reset(vertID)`,
 * and used synchronously during setup only.
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

    public reset (vertID: number): void {
        this._vertID = vertID;
        this._data = this._pipeline.renderGraph.getData(vertID);
    }

    public override get name (): string {
        return this._pipeline.renderGraph.getName(this._vertID);
    }

    public override set name (value: string) {
        this._pipeline.renderGraph.setName(this._vertID, value);
    }

    public override getParentLayout (): string {
        const graph = this._pipeline.renderGraph;
        let parId = graph.getParent(this._vertID);
        if (parId !== graph.N && graph.h(RenderGraphValue.RasterSubpass, parId)) {
            const subpassLayout = graph.getLayout(parId);
            if (subpassLayout.length > 0) {
                return subpassLayout;
            }
            parId = graph.getParent(parId);
        }
        return parId === graph.N ? '' : graph.getLayout(parId);
    }

    public override getCurrentLayout (): string {
        return this._pipeline.renderGraph.getLayout(this._vertID);
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

    protected readonly _pipeline: WebGPUPipeline;
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
        let passId = graph.getParent(queueId);
        let layoutName = graph.getLayout(passId);
        if (layoutName.length === 0 && graph.h(RenderGraphValue.RasterSubpass, passId)) {
            passId = graph.getParent(passId);
            layoutName = passId === graph.N ? '' : graph.getLayout(passId);
        }
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

    public addSceneOfCamera (camera: Camera, light: LightInfo, sceneFlags = SceneFlags.NONE, name = 'Camera'): void {
        const lightTarget = light.light;
        const scene = light.probe?.node?.scene?.renderScene || undefined;
        this._addScene(camera, sceneFlags, lightTarget, scene, light);
    }

    public addScene (
        camera: Camera,
        sceneFlags = SceneFlags.NONE,
        light: Light | undefined | null = null,
        scene: RenderScene | undefined = undefined,
    ): SceneBuilder {
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
            light && !(sceneFlags & SceneFlags.SHADOW_CASTER)
                ? CullingFlags.CAMERA_FRUSTUM | CullingFlags.LIGHT_BOUNDS
                : CullingFlags.CAMERA_FRUSTUM,
            light,
        );
        if (lightInfo) {
            sceneData.light.reset(lightInfo.light, lightInfo.level, lightInfo.culledByLight, lightInfo.probe);
        }
        const renderData = pool.createRenderData();
        const graph = this._pipeline.renderGraph;
        const sceneId = graph.addVertex<RenderGraphValue.Scene>(
            RenderGraphValue.Scene,
            sceneData,
            'Scene',
            '',
            renderData,
            !DEBUG,
            this._vertID,
        );
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
            const uiQueue = pool.createRenderQueue(QueueHint.RENDER_TRANSPARENT);
            const queueId = graph.addVertex<RenderGraphValue.Queue>(
                RenderGraphValue.Queue,
                uiQueue,
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
        const pool = this._pipeline.framePool.graphObjects;
        const graph = this._pipeline.renderGraph;
        const passOrSubpassId = graph.getParent(this._vertID);
        const uiQueue = pool.createRenderQueue(QueueHint.RENDER_TRANSPARENT);
        const queueId = graph.addVertex<RenderGraphValue.Queue>(
            RenderGraphValue.Queue,
            uiQueue,
            'UI Queue',
            'default',
            this._data,
            !DEBUG,
            passOrSubpassId,
        );
        const blitData = pool.createRenderData();
        WebSetter.setMat4(this._lg, blitData, 'cc_matProj', camera.matProj);
        graph.addVertex<RenderGraphValue.Blit>(
            RenderGraphValue.Blit,
            pool.createBlit(emptyMaterial, graph.N, SceneFlags.NONE, camera, BlitType.DRAW_PROFILE),
            'Profiler',
            '',
            blitData,
            !DEBUG,
            queueId,
        );
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
        const graph = this._pipeline.renderGraph;
        const lg = this._pipeline.layoutGraph;
        const passId = graph.getParent(this._vertID);
        const passLayoutID = passId === graph.N
            ? lg.N
            : lg.locateChild(lg.N, graph.getLayout(passId));
        const layoutName = graph.getLayout(this._vertID);
        if (layoutName.length === 0) {
            this._layoutID = passLayoutID;
        } else {
            const subpassLayoutID = lg.locateChild(passLayoutID, layoutName);
            this._layoutID = subpassLayoutID !== lg.N
                ? subpassLayoutID
                : lg.locateChild(lg.N, layoutName);
        }
    }

    private get _subpass (): RasterSubpass {
        return this._pipeline.renderGraph.object(this._vertID) as RasterSubpass;
    }

    public addRenderTarget (name: string, accessType: AccessType, slotName?: string, loadOp?: LoadOp, storeOp?: StoreOp, color?: Color): void {
        throw new Error('Method not implemented.');
    }

    public addDepthStencil (
        name: string,
        accessType: AccessType,
        depthSlotName = '',
        stencilSlotName = '',
        loadOp = LoadOp.CLEAR,
        storeOp = StoreOp.STORE,
        depth = 1,
        stencil = 0,
        clearFlag = ClearFlagBit.DEPTH_STENCIL,
    ): void {
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
        const queueID = this._pipeline.renderGraph.addVertex<RenderGraphValue.Queue>(
            RenderGraphValue.Queue,
            queue,
            '',
            phaseName,
            data,
            !DEBUG,
            this._vertID,
        );
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

    public addDepthStencil (
        name: string,
        loadOp = LoadOp.CLEAR,
        storeOp = StoreOp.STORE,
        depth = 1,
        stencil = 0,
        clearFlag = ClearFlagBit.DEPTH_STENCIL,
    ): void {
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

    public addTexture (name: string, slotName: string, sampler: Sampler | null = null, plane = 0): void {
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
        const vertID = this._pipeline.renderGraph.addVertex<RenderGraphValue.RasterSubpass>(
            RenderGraphValue.RasterSubpass,
            subpass,
            'Raster',
            layoutName,
            data,
            !DEBUG,
            this._vertID,
        );
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
        const queueID = this._pipeline.renderGraph.addVertex<RenderGraphValue.Queue>(
            RenderGraphValue.Queue,
            queue,
            '',
            phaseName,
            data,
            !DEBUG,
            this._vertID,
        );
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

    public addTexture (name: string, slotName: string, sampler: Sampler | null = null, plane = 0): void {
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
        const queueID = this._pipeline.renderGraph.addVertex<RenderGraphValue.Queue>(
            RenderGraphValue.Queue,
            queue,
            '',
            phaseName,
            data,
            !DEBUG,
            this._vertID,
        );
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
