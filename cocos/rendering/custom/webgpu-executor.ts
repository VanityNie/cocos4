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

import { DEBUG } from 'internal:constants';
import { LoadOp, StoreOp } from '../../gfx';
import type { ComputePass, ComputeView, CopyPass, RasterPass, RasterView, RenderGraph, ResourceGraph } from './render-graph';
import { RenderGraphValue } from './render-graph';
import { AccessType, ResourceResidency } from './types';

const INVALID_ID = 0xFFFFFFFF;

export const enum WebGPUPassMergeFlag {
    NONE = 0,
    NEED_BEGIN = 1 << 0,
    NEED_END = 1 << 1,
    BOTH = (1 << 0) | (1 << 1),
}

function setSubtreeValid (rg: RenderGraph, vertID: number, valid: boolean): void {
    rg.setValid(vertID, valid);
    const children = rg.x[vertID].c;
    for (let i = 0; i < children.length; ++i) {
        setSubtreeValid(rg, children[i].target as number, valid);
    }
}

function canMergeRasterPasses (prev: RasterPass, curr: RasterPass): boolean {
    if (prev.count !== curr.count || prev.quality !== curr.quality
        || !prev.rasterViews.size || prev.rasterViews.size !== curr.rasterViews.size) {
        return false;
    }
    for (const sub of prev.subpassGraph._subpasses) {
        if (sub.resolvePairs.length) return false;
    }
    for (const sub of curr.subpassGraph._subpasses) {
        if (sub.resolvePairs.length) return false;
    }
    const it = prev.rasterViews.entries();
    for (const [name, cv] of curr.rasterViews) {
        const [pn, pv] = it.next().value as [string, RasterView];
        if (pn !== name || cv.attachmentType !== pv.attachmentType
            || cv.loadOp !== LoadOp.LOAD || pv.storeOp !== StoreOp.STORE || cv.storeOp !== StoreOp.STORE) {
            return false;
        }
    }
    return true;
}

/**
 * Recording-order linear compiler plan (zero steady-state GC):
 * backward value-edge dead-pass culling, DEBUG hazard check, LOAD pass merge & [firstUse, lastUse] intervals.
 */
export class WebGPUGraphCompilerPlan {
    public topPassCount = 0;
    public alivePassCount = 0;
    public culledPassCount = 0;
    public topPassVerts = new Uint32Array(32);
    public alivePasses = new Uint8Array(32);
    public mergeFlags = new Uint8Array(32);
    public resFirstUse = new Int32Array(64);
    public resLastUse = new Int32Array(64);
    private _needed = new Uint8Array(64);

    public compile (rg: RenderGraph, resGraph: ResourceGraph): void {
        const verts = rg.x;
        const resCount = resGraph.nv();
        if (verts.length > this.topPassVerts.length) {
            const cap = verts.length * 2;
            this.topPassVerts = new Uint32Array(cap);
            this.alivePasses = new Uint8Array(cap);
            this.mergeFlags = new Uint8Array(cap);
        }
        if (resCount > this.resFirstUse.length) {
            const cap = resCount * 2;
            this.resFirstUse = new Int32Array(cap);
            this.resLastUse = new Int32Array(cap);
            this._needed = new Uint8Array(cap);
        }

        let topCount = 0;
        for (let v = 0; v < verts.length; ++v) {
            if (rg.getParent(v) === INVALID_ID) this.topPassVerts[topCount++] = v;
        }
        this.topPassCount = topCount;

        this._needed.fill(0, 0, resCount);
        let aliveCount = 0;
        for (let i = topCount - 1; i >= 0; --i) {
            const alive = this._evalBackward(rg, resGraph, this.topPassVerts[i]);
            this.alivePasses[i] = alive ? 1 : 0;
            if (alive) ++aliveCount;
        }
        this.alivePassCount = aliveCount;
        this.culledPassCount = topCount - aliveCount;

        this.resFirstUse.fill(-1, 0, resCount);
        this.resLastUse.fill(-1, 0, resCount);

        let prevIdx = -1;
        let prevRaster: RasterPass | null = null;
        let step = -1;
        for (let i = 0; i < topCount; ++i) {
            const vertID = this.topPassVerts[i];
            const alive = this.alivePasses[i] === 1;
            setSubtreeValid(rg, vertID, alive);
            if (!alive) {
                this.mergeFlags[i] = WebGPUPassMergeFlag.NONE;
                continue;
            }
            const vert = verts[vertID];
            if (vert.t === RenderGraphValue.RasterPass) {
                const curr = vert.j as RasterPass;
                if (prevRaster && canMergeRasterPasses(prevRaster, curr)) {
                    this.mergeFlags[prevIdx] &= ~WebGPUPassMergeFlag.NEED_END;
                    this.mergeFlags[i] = WebGPUPassMergeFlag.NEED_END;
                } else {
                    this.mergeFlags[i] = WebGPUPassMergeFlag.BOTH;
                    ++step;
                }
                prevIdx = i;
                prevRaster = curr;
            } else {
                this.mergeFlags[i] = WebGPUPassMergeFlag.NONE;
                prevIdx = -1;
                prevRaster = null;
                ++step;
            }
            this._recordLifetimes(resGraph, vert.t, vert.j, step);
        }
    }

    private _isNeeded (resGraph: ResourceGraph, name: string): boolean {
        const id = resGraph.find(name);
        if (id === INVALID_ID || this._needed[id]) return true;
        const r = resGraph.getTraits(id).residency;
        return r !== ResourceResidency.MANAGED && r !== ResourceResidency.MEMORYLESS;
    }

    private _setNeeded (resGraph: ResourceGraph, name: string, val: number): void {
        const id = resGraph.find(name);
        if (id !== INVALID_ID) this._needed[id] = val;
    }

    private _hasNeededWrite (resGraph: ResourceGraph, viewsMap: Map<string, ComputeView[]>): boolean {
        for (const [name, views] of viewsMap) {
            for (const v of views) {
                if (v.accessType !== AccessType.READ) {
                    if (this._isNeeded(resGraph, name)) return true;
                    break;
                }
            }
        }
        return false;
    }

    private _markReads (resGraph: ResourceGraph, viewsMap: Map<string, ComputeView[]>): void {
        for (const [name, views] of viewsMap) {
            for (const v of views) {
                if (v.accessType !== AccessType.WRITE) {
                    this._setNeeded(resGraph, name, 1);
                    break;
                }
            }
        }
    }

    private _evalBackward (rg: RenderGraph, resGraph: ResourceGraph, vertID: number): boolean {
        const vert = rg.x[vertID];
        if (vert.t === RenderGraphValue.RasterPass) {
            const pass = vert.j as RasterPass;
            const subs = pass.subpassGraph._subpasses;
            let alive = this._hasNeededWrite(resGraph, pass.computeViews);
            for (const [name, v] of pass.rasterViews) {
                if (alive) break;
                if (v.accessType !== AccessType.READ && this._isNeeded(resGraph, name)) alive = true;
            }
            for (let s = 0; s < subs.length && !alive; ++s) {
                for (const rp of subs[s].resolvePairs) {
                    if (this._isNeeded(resGraph, rp.target)) { alive = true; break; }
                }
            }
            if (!alive) return false;
            if (DEBUG) {
                for (const [name, rv] of pass.rasterViews) {
                    if (rv.accessType !== AccessType.READ && pass.computeViews.has(name)) {
                        throw new Error(`WebGPU read-write hazard in "${rg.getName(vertID)}": "${name}".`);
                    }
                }
            }
            for (let s = subs.length - 1; s >= 0; --s) {
                const pairs = subs[s].resolvePairs;
                for (let r = pairs.length - 1; r >= 0; --r) {
                    this._setNeeded(resGraph, pairs[r].target, 0);
                    this._setNeeded(resGraph, pairs[r].source, 1);
                }
            }
            for (const [name, v] of pass.rasterViews) {
                this._setNeeded(resGraph, name, (v.accessType === AccessType.READ || v.loadOp === LoadOp.LOAD) ? 1 : 0);
            }
            this._markReads(resGraph, pass.computeViews);
            return true;
        }
        if (vert.t === RenderGraphValue.Compute) {
            const pass = vert.j as ComputePass;
            if (pass.computeViews.size && !this._hasNeededWrite(resGraph, pass.computeViews)) return false;
            this._markReads(resGraph, pass.computeViews);
            return true;
        }
        if (vert.t === RenderGraphValue.Copy) {
            const pass = vert.j as CopyPass;
            let alive = false;
            for (const p of pass.copyPairs) if (this._isNeeded(resGraph, p.target)) { alive = true; break; }
            for (const p of pass.uploadPairs) if (alive || this._isNeeded(resGraph, p.target)) { alive = true; break; }
            if (!alive) return false;
            for (let i = pass.copyPairs.length - 1; i >= 0; --i) {
                const p = pass.copyPairs[i];
                if (p.mipLevels === INVALID_ID && p.numSlices === INVALID_ID) this._setNeeded(resGraph, p.target, 0);
                this._setNeeded(resGraph, p.source, 1);
            }
            for (let i = pass.uploadPairs.length - 1; i >= 0; --i) {
                const p = pass.uploadPairs[i];
                if (p.mipLevels === INVALID_ID && p.numSlices === INVALID_ID) this._setNeeded(resGraph, p.target, 0);
            }
            return true;
        }
        return false;
    }

    private _recordLifetimes (resGraph: ResourceGraph, type: RenderGraphValue, obj: unknown, step: number): void {
        if (type === RenderGraphValue.RasterPass) {
            const pass = obj as RasterPass;
            for (const name of pass.rasterViews.keys()) this._touch(resGraph, name, step);
            for (const sub of pass.subpassGraph._subpasses) {
                for (const rp of sub.resolvePairs) {
                    this._touch(resGraph, rp.source, step);
                    this._touch(resGraph, rp.target, step);
                }
            }
            for (const name of pass.computeViews.keys()) this._touch(resGraph, name, step);
        } else if (type === RenderGraphValue.Compute) {
            for (const name of (obj as ComputePass).computeViews.keys()) this._touch(resGraph, name, step);
        } else if (type === RenderGraphValue.Copy) {
            const pass = obj as CopyPass;
            for (const p of pass.uploadPairs) this._touch(resGraph, p.target, step);
            for (const p of pass.copyPairs) {
                this._touch(resGraph, p.source, step);
                this._touch(resGraph, p.target, step);
            }
        }
    }

    private _touch (resGraph: ResourceGraph, name: string, step: number): void {
        const id = resGraph.find(name);
        if (id === INVALID_ID) return;
        if (this.resFirstUse[id] === -1) this.resFirstUse[id] = step;
        this.resLastUse[id] = step;
    }
}
