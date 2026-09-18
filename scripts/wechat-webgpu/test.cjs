'use strict';
const assert=require('node:assert/strict'), fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
const ts=require('typescript'), ejs=require('ejs');
const root=path.resolve(__dirname,'../..');
function manager(gpu, mode, ctor, webgl) {
    const core={cclegacy:{WebGPUDevice:ctor,WebGLDevice:webgl},settings:{querySettings:()=>mode},
        Settings:{Category:{RENDERING:'rendering'}},screen:{windowSize:{width:1,height:1}},sys:{},getError:()=>'',errorID:()=>{}};
    const module={exports:{}};
    const source=ts.transpileModule(fs.readFileSync(path.join(root,'cocos/gfx/device-manager.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;
    vm.runInNewContext(source,{exports:module.exports,navigator:gpu?{gpu}:{},require:n=>{
        if(n==='internal:constants')return{EDITOR:false,JSB:false,WECHAT:true};
        if(n==='../core')return core;
        if(n.endsWith('/define'))return {DeviceInfo:class{},SwapchainInfo:class{}};
        if(n.endsWith('/device'))return {Device:class{}};
        if(n.endsWith('enum-type'))return {BrowserType:{UC:'uc'}};
        return {};
    }});
    return module.exports.deviceManager;
}
(async()=>{
    assert.throws(()=>manager(null,4).init({},{}),/requestAdapter is unavailable/);
    assert.throws(()=>manager({requestAdapter(){}},4).init({},{}),/backend is missing/);
    let swaps=0;
    class Failed { initialize(){return Promise.resolve(false)} createSwapchain(){swaps++} }
    await assert.rejects(manager({requestAdapter(){}},4,Failed).init({},{}),/returned false/);
    assert.equal(swaps,0);
    class Good { initialize(){return true} createSwapchain(){swaps++;return {}} }
    assert.equal(manager(null,2,null,Good).init({},{}),true);
    class AsyncGood extends Good { initialize(){return Promise.resolve(true)} }
    assert.equal(await manager({requestAdapter(){}},4,AsyncGood).init({},{}),true);
    assert.equal(swaps,2);
    const template=path.join(root,'templates/wechatgame/game.ejs');
    const output=ejs.render(fs.readFileSync(template,'utf8'),{cocosTemplate:'unused',importMapFile:'src/import-map.js',applicationJs:'./application.js',alpha:'default',antialias:'default',useWebgl2:'false'},
        {filename:template,includer:(name,resolved)=>name==='unused'?{template:'// Mock SystemJS include'}:{filename:resolved}});
    new vm.Script(output);
    for(const entry of ['navigator','direct']) for(const mode of [0,4]) {
        const calls=[];const gpu={requestAdapter(){}};const host={navigator:entry==='navigator'?{gpu}:{},devicePixelRatio:1};
        if(entry==='direct')host.gpu=gpu;
        const canvas={width:1,height:1};
        const builtOutput = mode === 4 ? output.replace('const webgpuEnabled = false;', 'const webgpuEnabled = true;') : output;
        assert.ok(!builtOutput.includes("require('./src/settings.json')"));
        vm.runInNewContext(builtOutput,{GameGlobal:host,canvas,console:{log(){},warn(){}},wx:{getSystemInfoSync:()=>({platform:'devtools',screenWidth:1,screenHeight:1})},
            System:{warmup(){},import(){return new Promise(()=>{})}},require:name=>{
                calls.push(name);
                if(name.endsWith('.json'))throw Error('WeChat cannot require JSON: '+name);
                if(name==='./web-adapter'){host.navigator={};return {}}
                if(name==='./first-screen')return {start:()=>new Promise(()=>{})};
                return {default:{}};
            }});
        assert.equal(calls.includes('./first-screen'),mode===0);
        if(mode===4)assert.equal(host.navigator.gpu,gpu);
    }
    // Android must not enter app initialization until the scheduled frame fires.
    let frame;
    const bootHost={requestAnimationFrame(cb){frame=cb;}};
    vm.runInNewContext(output,{GameGlobal:bootHost,console:{log(){},warn(){}},
        wx:{getSystemInfoSync:()=>({platform:'android',screenWidth:1,screenHeight:1})},
        require:()=>({start:()=>new Promise(()=>{})}),canvas:null,
        System:{warmup(){}}});
    assert.equal(bootHost.__webgpuBootTrace.stage,'raf-scheduled');
    assert.equal(bootHost.__webgpuBootTrace.events.some(e=>e.stage==='initApp-enter'),false);
    frame();
    assert.equal(bootHost.__webgpuBootTrace.stage,'initApp-enter');
    const gpuSource=fs.readFileSync(path.join(__dirname,'platform-source/platforms/wechat/wrapper/builtin/gpu.js'),'utf8').replace('export function','function');
    const gpu={requestAdapter(){}};
    for (const host of [{gpu},{navigator:{gpu}},{}]) {
        const context={GameGlobal:host};vm.createContext(context);
        vm.runInContext(gpuSource+'; this.found=getGPU();',context);
        assert.equal(context.found,host.gpu||host.navigator?.gpu);
    }
    // Regression: Android has no global GPU; only the official wx.getGPU API.
    let wxCalls=0;
    const androidWX={getGPU(){assert.equal(this,androidWX);wxCalls++;return gpu;}};
    const android={GameGlobal:{},wx:androidWX,console:{log(){},warn(){}}};
    vm.createContext(android);
    vm.runInContext(gpuSource,android);
    android.require=name=>{
        assert.equal(name,'./web-adapter');
        android.GameGlobal.navigator={gpu:vm.runInContext('getGPU()',android)};
    };
    vm.runInContext(fs.readFileSync(path.join(__dirname,'wechat-webgpu-bootstrap.js'),'utf8'),android);
    assert.equal(wxCalls,1);
    assert.equal(android.GameGlobal.navigator.gpu,gpu);
    assert.equal(android.GameGlobal.__webgpuDiagnostic.stage,'entry-ready');
    assert.equal(android.GameGlobal.__webgpuDiagnostic.source,'wx.getGPU');
    const bootstrap=fs.readFileSync(path.join(__dirname,'wechat-webgpu-bootstrap.js'),'utf8');
    for (const [wxMock, expected] of [
        [{}, 'WX_GET_GPU_NOT_EXPOSED'],
        [{getGPU:()=>undefined}, 'WX_GET_GPU_NO_USABLE_GPU'],
        [{getGPU(){throw Error('native failure');}}, 'WX_GET_GPU_THREW'],
    ]) {
        const context={GameGlobal:{},wx:wxMock,console:{log(){},warn(){}}};
        vm.createContext(context);vm.runInContext(gpuSource,context);
        context.require=()=>{context.GameGlobal.navigator={gpu:vm.runInContext('getGPU()',context)};};
        assert.throws(()=>vm.runInContext(bootstrap,context),new RegExp(expected));
        assert.equal(context.GameGlobal.__webgpuDiagnostic.stage,expected);
    }
    for (const result of [undefined,{}, {requestAdapter:42}]) {
        const context={GameGlobal:{},wx:{getGPU:()=>result},console:{log(){},warn(){}}};
        vm.createContext(context);vm.runInContext(gpuSource+';this.found=getGPU();',context);
        assert.equal(context.found,undefined);
    }
    const fallback={GameGlobal:{navigator:{gpu}},wx:{getGPU(){throw Error('unsupported');}},console:{log(){},warn(){}}};
    vm.createContext(fallback);vm.runInContext(gpuSource+';this.found=getGPU();',fallback);
    assert.equal(fallback.found,gpu);
    const canvasSource=fs.readFileSync(path.join(__dirname,'platform-source/platforms/wechat/wrapper/builtin/Canvas.js'),'utf8')
        .replace(/^import .*$/mg,'').replace('export default function Canvas','function Canvas');
    const nativeContext={};const requests=[];
    const nativeCanvas={getContext(type){assert.equal(this,nativeCanvas);requests.push(type);return nativeContext;}};
    const context={wx:{createCanvas:()=>nativeCanvas},innerWidth:640,innerHeight:480,window:{}};
    vm.createContext(context);vm.runInContext(canvasSource+';this.canvas=Canvas();',context);
    assert.equal(requests.length,0,'Adapter must not acquire a WebGL context at startup');
    assert.equal(context.canvas.getContext('webgpu'),nativeContext);
    assert.deepEqual(requests,['webgpu']);
    context.GameGlobal={__cocosWebGPUEnabled:true,screencanvas:context.canvas};
    for (const type of ['webgl','webgl2','2d']) assert.throws(()=>context.canvas.getContext(type), /reserved for webgpu/);
    assert.deepEqual(requests,['webgpu'],'Rejected contexts must never reach the native canvas');
    context.GameGlobal.__cocosWebGPUEnabled=false;
    context.canvas.getContext('webgl');
    assert.deepEqual(requests,['webgpu','webgl'],'WebGL-only builds remain supported');
    console.log('PASS: device failure/success, WebGL regression, EJS rendering, GPU preservation and splash branches');
})().catch(e=>{console.error(e);process.exitCode=1});
