# 微信小游戏 WebGPU 真机黑屏调查与引擎维护方法

## 1. 文档范围

本文记录 Cocos Creator 3.8.8 自定义引擎在微信小游戏 WebGPU 真机环境中的完整调查过程，包括现象、证据、引擎修改、构建流程、真机调试方法和后续维护规则。

验证环境：

- 引擎：`D:/code/cocos/own_cocos`
- 项目：`D:/code/project/compute/_demo`
- 微信产物：`D:/code/project/compute/_demo/build/wechatgame`
- 场景：`morph-bench`
- 手机：vivo V2283A
- GPU：Adreno 642L
- 系统：Android 15
- 微信：8.0.77
- 调试日期：2026-09-20 至 2026-09-21

本次最终确认了四项引擎兼容问题：

1. 自定义渲染管线遗漏微信平台，微信 WebGPU 选择了普通 descriptor 布局。
2. BindGroup 的 `entries` 使用迭代器，微信桥接要求带有 `length` 的数组。
3. WebGPU Pipeline 携带无用途的 `a_vertexId`，占用 `location=16`。
4. 微信报告的 uniform buffer 地址间隔为 64 字节，真机有效执行要求至少 256 字节。

四项修改共同进入引擎源码。项目业务脚本无需承担平台兼容处理。

## 2. 初始现象

### 2.1 cube texture 布局警告

微信真机最初输出：

```text
[WebGPU-MB][WARN] Auto-fixing BindGroup 161b8 binding 4: expected 2d but got cube.
[WebGPU-MB][WARN] Auto-fixing BindGroup 16359 binding 5: expected 2d but got cube.
```

浏览器和其他平台运行正常。微信模拟器能够进入游戏，真机显示黑色画面。

### 2.2 微信进程因内存不足退出

Android `ApplicationExitInfo` 显示：

```text
process=com.tencent.mm:appbrand1 reason=3 (LOW_MEMORY) rss=663MB
process=com.tencent.mm reason=3 (LOW_MEMORY) rss=807MB
```

为建立稳定的真机调查环境，项目启动参数调整为：

- 初始关闭阴影。
- 初始角色数量设置为 1。
- 初始 morph target 数量设置为 3。
- 角色和 mesh 分阶段创建。

这些设置控制启动资源规模。GPU 黑屏仍需继续检查。

### 2.3 进程稳定后持续黑屏

降低启动资源后，微信进程可以持续运行，手机仍然显示黑色画面。Console 没有明确的 shader、Pipeline 或 BindGroup 错误。

真机运行状态显示：

- `cc.director.getScene().name` 为 `morph-bench`。
- `cc.director.getTotalFrames()` 持续增长。
- `cc.game.isPaused()` 为 `false`。
- `device.gfxAPI` 为 8，即 `API.WEBGPU`。
- DrawCall 约为 15。
- Triangle 约为 8536。
- Canvas 尺寸为 1260 × 2800。
- `cc.game.canvas === GameGlobal.screencanvas`。
- 场景 Camera、UI Camera、角色节点均已启用。

这些信息确认 JavaScript 主循环、场景加载和渲染调度正在运行，调查范围进入 GPU 命令创建与执行阶段。

## 3. 引擎修改

### 3.1 自定义管线补充微信 WebGPU 布局选择

修改文件：

- `cocos/rendering/custom/layout-graph.ts`
- `cocos/rendering/custom/layout-graph-utils.ts`

原条件为：

```ts
(COCOS_RUNTIME || HTML5) && Layout.isWebGPU
```

修改后为：

```ts
(COCOS_RUNTIME || HTML5 || WECHAT) && Layout.isWebGPU
```

修改范围包括：

- `PipelineLayoutData.getSets()`。
- `PipelineLayoutData.getSet()`。
- uniform block 布局创建。
- samplerTexture 布局创建。
- sampler 布局创建。
- texture 布局创建。
- buffer 布局创建。
- storage image 布局创建。
- subpass input 布局创建。
- descriptor block 排序。

实际资源中同时保存了两套布局：

| 布局 | binding | 资源 | shader 类型 | viewDimension |
| --- | ---: | --- | --- | --- |
| `descriptorSets` | 4 | `cc_diffuseMap` | `SAMPLER_CUBE` | `UNKNOWN` |
| `descriptorSets` | 5 | `cc_environment` | `SAMPLER_CUBE` | `UNKNOWN` |
| `descriptorGroups` | 4 | `cc_diffuseMap` | `SAMPLER_CUBE` | `TEXCUBE` |
| `descriptorGroups` | 5 | `cc_environment` | `SAMPLER_CUBE` | `TEXCUBE` |

微信构建遗漏平台条件后选择 `descriptorSets`，`UNKNOWN` 最终转换为 `2d`，实际绑定资源则为 cube TextureView。加入 `WECHAT` 后，微信 WebGPU 选择 `descriptorGroups`，布局和资源维度保持一致。

### 3.2 BindGroup entries 转换为数组

修改文件：

`cocos/gfx/webgpu/webgpu-descriptor-set.ts`

修改内容：

```ts
entries: Array.from(this._bindGroupEntries.values()),
```

微信桥接创建 BindGroup 时直接读取：

```js
entries.length
```

`Map.values()` 返回迭代器，没有 `length`。浏览器实现接受 sequence，当前微信桥接需要真实数组。转换后资源数量、binding 和原生对象 ID 可以完整序列化。

该处理放在 `WebGPUDescriptorSet._createBindGroup()`，覆盖引擎创建的全部 WebGPU BindGroup。

### 3.3 WebGPU 排除内置 a_vertexId

修改文件：

`cocos/gfx/webgpu/webgpu-shader.ts`

修改内容：

```ts
this._attributes = info.attributes.filter(
    (attribute) => attribute.name !== 'a_vertexId',
);
```

morph mesh 会提供 `a_vertexId`，其 shader location 为 16。WebGPU shader 已通过以下 builtin 获取顶点编号：

```wgsl
@builtin(vertex_index)
```

因此 WebGPU Pipeline 无需声明 `a_vertexId` 顶点输入。真机测试中：

- 保留 `location=16` 时，角色 Pipeline 无法完成有效绘制。
- 移除对应 vertex stream 后，同一 shader 和其他 Pipeline 参数可以完成绘制。

过滤逻辑位于 `WebGPUShader`，WebGL1 所需的 `a_vertexId` 行为继续保留。

### 3.4 微信 uniform buffer 地址间隔至少为 256 字节

修改文件：

`cocos/gfx/webgpu/webgpu-device.ts`

修改内容：

```ts
this._caps.uboOffsetAlignment = WECHAT
    ? Math.max(256, device.limits.minUniformBufferOffsetAlignment)
    : device.limits.minUniformBufferOffsetAlignment;
```

当前微信真机报告：

```text
minUniformBufferOffsetAlignment = 64
```

性能统计绘制使用了 `offset=64` 的 uniform buffer。真实命令重放显示，执行该 BindGroup 后，包含它的 GPU 提交无法产生有效结果。

将同一段数据复制到 `offset=0` 的独立 Buffer 后：

- 全部 37 条场景命令执行完成。
- 中心像素读取为 `[22, 76, 84, 255]`。
- 手机显示游戏场景。

独立 Buffer 复制仅用于调查。正式引擎从初始化阶段将地址间隔设置为至少 256 字节，让资源分配和动态 uniform buffer 自然使用有效地址，不增加每帧复制。

## 4. 真机问题定位过程

### 4.1 建立可访问的真机运行环境

微信开发者工具通过本地 CDP 代理连接真机 JavaScript 环境。游戏运行在独立 context 中，可以通过 `Runtime.evaluate` 读取 Cocos 对象和执行诊断代码。

诊断脚本位于：

```text
scripts/wechat-webgpu/inspect-live.cjs
scripts/wechat-webgpu/probe-clear.js
scripts/wechat-webgpu/probe-buffer.js
scripts/wechat-webgpu/probe-texture.js
scripts/wechat-webgpu/probe-pipeline.js
scripts/wechat-webgpu/probe-pipeline-variants.js
scripts/wechat-webgpu/probe-passes.js
scripts/wechat-webgpu/probe-engine-frame.js
scripts/wechat-webgpu/probe-bind-groups.js
```

这些脚本直接使用真机上的 GPUDevice、Canvas、Pipeline 和游戏资源。临时函数替换均在测试结束后恢复。

### 4.2 验证 Canvas 和呈现流程

在当前 Canvas 上创建 RenderPass，并使用紫色作为 clear color。

测试内容覆盖：

- 当前 swapchain texture。
- 游戏正在使用的 depth/stencil texture。
- RenderPassEncoder。
- CommandEncoder。
- Queue submit。
- 手机屏幕呈现。

用户两次确认手机显示紫色，其中一次使用游戏当前的 depth/stencil texture。这证明 Canvas、交换链、depth/stencil 和呈现流程可用。

### 4.3 排除 Camera 和 current texture 复用问题

执行过以下实验：

- 同一帧缓存 `context.getCurrentTexture()`。
- 临时关闭 UI Camera，只保留场景 Camera。
- 恢复所有 Camera 和原始 current texture 获取方式。

两项实验期间仍然黑屏，因此继续检查真实 Pipeline 和 BindGroup。

### 4.4 验证 Buffer、Texture 和基础 Pipeline

真机独立测试结果：

- mapped buffer 写入后复制到 MAP_READ Buffer，字节结果正确。
- Texture 上传 `[17, 34, 51, 255]` 后读取正确。
- Texture 紫色 clear 后读取正确。
- 简单 WGSL 三角形 Pipeline 有效。
- `depth24plus-stencil8` RenderPass 有效。
- 使用游戏相同 depth/stencil 设置的简单 Pipeline 有效。

这些证据确认基础 Buffer、Texture、ShaderModule、Pipeline、depth/stencil 和复制命令可用。

### 4.5 比较真实角色 Pipeline 变体

使用角色的实际 shader、PipelineLayout 和 vertex layouts 创建以下变体：

1. 原始 Pipeline。
2. 自动 PipelineLayout。
3. 简单 vertex shader。
4. 简单 fragment shader。
5. 移除包含 `location=16` 的 vertex stream。

结果：

- 简单 shader Pipeline 有效。
- 原始角色 Pipeline 无效。
- 只移除 `location=16` 后 Pipeline 有效。

角色主 vertex stream 包含 position、normal、uv、tangent、joints 和 weights。第二个 stream 仅包含 `a_vertexId`。WGSL vertex 输入没有读取 location 16，由此确定 WebGPU Pipeline 应排除该属性。

### 4.6 记录真实 RenderPass 命令

拦截游戏的 CommandEncoder 和 RenderPassEncoder，记录真实场景调用：

```text
setViewport
setScissorRect
setPipeline
setStencilReference
setBindGroup
setVertexBuffer
setIndexBuffer
drawIndexed
```

记录显示场景 pass 和 UI pass 的 attachment、viewport、scissor、depth/stencil 和 draw 参数均位于有效范围。

### 4.7 逐条重放绘制命令

创建与游戏 Canvas 相同尺寸的离屏 Texture，并使用实际 depth/stencil attachment。随后从 0 条命令开始，每次增加一条真实命令：

1. 执行前 N 条命令。
2. 结束 RenderPass。
3. 将中心像素复制到 MAP_READ Buffer。
4. 读取像素并记录结果。

第一次重放定位到 `setBindGroup`。检查微信桥接实现后确认 `createBindGroup()` 读取 `entries.length`，引擎传入的迭代器无法满足该格式。

应用数组转换后，角色和场景前部绘制恢复。继续重放定位到性能统计使用的 BindGroup，其中 uniform buffer resource 为：

```text
offset = 64
size = 352
```

将其复制到地址为 0 的独立 Buffer 后，全部 37 条命令保持有效，最终像素为游戏颜色。

### 4.8 联合真机验证

在运行中的游戏临时同时应用：

- BindGroup entries 数组转换。
- `a_vertexId` 排除。
- 64 字节 uniform buffer 数据复制到 256 字节边界。

用户确认手机出现游戏画面。该结果与离屏像素读取一致。随后将对应规则写入引擎源码。

## 5. 微信桥接层的已确认行为

### 5.1 requestDevice 参数没有完整应用

引擎会提交 `requiredLimits` 和 `requiredFeatures`。当前微信桥接的 `requestDevice()` 包装没有完整使用传入参数，因此 Adapter 报告的能力、请求能力和原生 Device 的实际执行要求可能不同。

引擎初始化后应读取返回 Device 的 limits，并对真机确认过的不一致值进行平台处理。

### 5.2 BindGroup 描述要求数组

当前微信桥接读取 `entries.length`，随后遍历 entries 并序列化资源 ID。所有进入桥接边界的 WebGPU sequence 都应使用数组。

### 5.3 错误信息接口内容有限

当前真机环境中：

```js
device.popErrorScope()
```

返回 `null`。

```js
shaderModule.getCompilationInfo()
```

返回空 `messages`。

因此 Pipeline、BindGroup 或 GPU 提交无效时，Console 可能没有任何明确错误。引擎帧数和 DrawCall 统计仍会增长。

### 5.4 current texture 包装对象

每次调用 `context.getCurrentTexture()` 都会得到新的 JavaScript 包装对象和对象 ID。同一帧缓存 current texture 的实验没有改变本次黑屏结果。

### 5.5 Texture 数据传输参数

Texture copy 和 write 路径需要明确提供 `rowsPerImage`。本次引擎纹理上传路径已经提供该参数，独立纹理测试读取结果正确。

## 6. 微信 WebGPU 构建流程

### 6.1 选择自定义引擎

Creator 项目必须使用：

```text
D:/code/cocos/own_cocos
```

功能裁剪必须包含 `gfx-webgpu`。

### 6.2 微信平台插件生成 WebGPU 资源

微信平台插件负责：

- 提供 `useWebGPU` 构建选项。
- 将 WebGPU 标志传入 `game.ejs`。
- 保留 EffectAsset 的 GLSL4。
- 包含 WebGPU 引擎模块。
- 生成场景、脚本、资源 Bundle 和分包。

### 6.3 构建微信 adapter

adapter 源码位于：

```text
scripts/wechat-webgpu/platform-source
```

构建命令：

```powershell
node scripts/wechat-webgpu/build-adapter.cjs
```

生成：

```text
bin/adapter/minigame/wechat/web-adapter.js
bin/adapter/minigame/wechat/web-adapter.min.js
```

### 6.4 项目构建扩展

项目扩展 `engine-wechat-webgpu` 调用引擎中的：

```text
scripts/wechat-webgpu/builder.cjs
scripts/wechat-webgpu/hooks.cjs
scripts/wechat-webgpu/prepare.cjs
```

构建前检查：

- `useWebGPU` 已启用。
- `md5Cache` 已关闭。
- 引擎插件分离已关闭。
- WASM 压缩已关闭。

构建后检查：

- 使用正确的自定义引擎。
- `cocos-js` 包含 WebGPUDevice 和 requestAdapter。
- 产物包含 `glslang` 和 `twgsl` WASM。
- shader JSON 包含 GLSL4。
- `game.js` 包含 WebGPU 构建标志。
- `settings.json` 的 `renderMode` 为 4。
- CommonJS 模块包含微信全局对象和 GPU 枚举别名。

任何必要条件缺失时立即终止构建，避免生成只能在模拟器运行的产物。

### 6.5 微信运行顺序

最终 `game.js` 的关键流程：

1. 读取微信设备信息。
2. 检查 `wx.getGPU()`、`GameGlobal.gpu` 和 `navigator.gpu`。
3. 加载 `web-adapter.js`。
4. 建立标准 `navigator.gpu` 入口。
5. 补充 GPUTextureUsage、GPUBufferUsage 等枚举。
6. WebGPU 分支跳过占用主 Canvas 的 WebGL 首屏。
7. 加载 polyfills 和 SystemJS。
8. 加载 Application。
9. `System.import('cc')`。
10. 加载 `engine-adapter.js`。
11. 执行 `application.init(cc)`。
12. 执行 `application.start()`。
13. 验证实际 `gfxAPI` 为 `API.WEBGPU`。
14. 加载场景并开始绘制。

Shader 运行时转换流程：

```text
GLSL4 → glslang → SPIR-V → twgsl → WGSL → GPUShaderModule
```

## 7. 可复用的真机调试方法

### 7.1 第一级：检查进程与内存

收集：

- Android `ApplicationExitInfo`。
- 微信主进程和 appbrand 进程状态。
- RSS、PSS 和 graphics 内存。
- `LOW_MEMORY`、系统终止和主动退出原因。

进程持续退出时，先控制 Canvas 尺寸、纹理、角色、阴影和启动资源数量。

### 7.2 第二级：检查引擎启动状态

确认：

- WebGPU 入口存在。
- Adapter 创建成功。
- Device 创建成功。
- Canvas context 配置成功。
- `gfxAPI` 为 WebGPU。
- 场景已经启动。
- Camera 和节点有效。
- 主循环持续运行。

### 7.3 第三级：执行单色清屏

在游戏 Canvas 上提交单色 clear，并使用游戏当前的 depth/stencil attachment。

- 手机显示颜色：Canvas、RenderPass、submit 和呈现可用。
- 离屏读取有效、手机无颜色：检查交换链和 Canvas 生命周期。
- 离屏读取也无效：检查 Device、CommandEncoder、attachment 和 Queue。

### 7.4 第四级：验证基础 GPU 资源

按照以下顺序验证：

1. Buffer 写入和回读。
2. Texture 写入和回读。
3. Texture clear。
4. 简单 WGSL Pipeline。
5. depth/stencil。
6. BindGroup。
7. draw 或 drawIndexed。

每项测试都要得到屏幕颜色、Buffer 字节或 Texture 像素证据。

### 7.5 第五级：复用真实游戏资源

基础能力通过后，使用游戏中的：

- ShaderModule。
- PipelineLayout。
- BindGroupLayout。
- BindGroup。
- VertexBuffer。
- IndexBuffer。
- depth/stencil 配置。
- RenderPass attachment。

每次只调整一个参数，并记录结果变化。

### 7.6 第六级：逐条重放命令

对于持续黑屏且 Console 无错误的情况，记录真实 RenderPass 命令，从 0 条开始逐条增加并读取像素。

判断规则：

```text
前 N 条命令有效
前 N+1 条命令无效
第 N+1 条命令作为调查入口
```

该方法适用于定位：

- Pipeline 描述问题。
- vertex attribute 问题。
- BindGroup 问题。
- dynamic offset 问题。
- uniform buffer 地址问题。
- TextureView 维度问题。
- draw 参数问题。

### 7.7 第七级：联合验证

分别确认每项修正后，在同一运行环境联合启用全部修正，并完成：

- 离屏像素读取。
- 手机画面确认。
- 多帧运行。
- UI 和场景共同绘制。
- 内存稳定性检查。
- 前后台切换检查。

## 8. 引擎后续维护规则

### 8.1 平台处理集中在引擎层

微信 WebGPU 差异应集中在：

- `WebGPUDevice`：Device 能力和资源地址间隔。
- `WebGPUShader`：后端专用 shader 输入。
- `WebGPUDescriptorSet`：桥接数据结构。
- `WebGPUPipelineState`：Pipeline 描述。
- 自定义管线 Layout：图形后端布局选择。
- 微信 adapter：宿主 API 到标准 WebGPU API 的转换。

项目组件、材质和场景保持平台无关。

### 8.2 使用 Device 的实际能力

设备初始化应执行：

1. 读取 Adapter 能力。
2. 提交 requiredLimits 和 requiredFeatures。
3. 获取 GPUDevice。
4. 读取 GPUDevice 的实际 limits 和 features。
5. 对真机确认过的不一致值进行微信平台处理。
6. 使用统一结果初始化 GFX caps。

后续 Buffer、Texture、Pipeline 和 BindGroup 均使用同一份 caps。

### 8.3 Pipeline 只声明 shader 使用的输入

WebGPU Pipeline 的 vertex layouts 应来源于 WGSL 实际输入。使用 `builtin(vertex_index)` 时，不应声明 `a_vertexId`。

未来可以由 shader 转换阶段输出可靠的 WGSL 输入反射信息。反射需要覆盖：

- `@location`。
- `@builtin`。
- `@interpolate`。
- struct 输入。
- 多个 attribute 修饰。
- instancing 属性。

反射数据完整后，`WebGPUPipelineState` 可以只创建 shader 实际消费的 vertex attributes。

### 8.4 桥接边界统一转换为数组

以下 sequence 在进入微信桥接前统一使用数组：

- BindGroup entries。
- BindGroupLayout entries。
- VertexBuffer layouts。
- Vertex attributes。
- Color attachments。
- CommandBuffer 列表。
- dynamic offsets。

### 8.5 构建阶段快速失败

以下情况应立即终止微信 WebGPU 构建：

- 未启用 `useWebGPU`。
- 功能裁剪不包含 `gfx-webgpu`。
- GLSL4 缺失。
- `glslang` 或 `twgsl` 缺失。
- 使用了其他引擎目录。
- 启用了当前不支持的引擎插件分离。
- WASM 压缩方式不受支持。
- `game.js` 未进入 WebGPU 分支。
- `renderMode` 不正确。

### 8.6 建立固定真机回归场景

回归场景至少覆盖：

- 2D Sprite。
- 3D Mesh。
- cube texture。
- morph target。
- dynamic uniform buffer。
- 多个 BindGroup。
- depth/stencil。
- UI Camera。
- Texture 上传。
- Compute shader。
- 多帧 Buffer 和 Texture 更新。
- 前后台切换。

每次修改微信 adapter、GFX WebGPU 或自定义渲染管线后，执行：

1. 单色清屏。
2. Buffer 回读。
3. Texture 回读。
4. 游戏命令逐条重放。
5. 完整场景显示确认。
6. 连续运行检查。
7. 内存检查。

## 9. 验收标准

微信 WebGPU 版本发布前应同时满足：

- 真机 `gfxAPI` 为 `API.WEBGPU`。
- 场景和 UI 均可显示。
- cube texture 不再触发维度自动修正。
- 角色 Pipeline 可以绘制。
- dynamic uniform buffer 地址符合 256 字节要求。
- BindGroup entries 均为数组。
- morph 和 compute 功能正常更新。
- 前后台切换后画面可以恢复。
- 连续运行期间内存保持稳定。
- Android 没有新的系统终止记录。
- 构建产物包含 GLSL4、glslang、twgsl 和 WebGPU 后端。

## 10. 当前验证结果

已经完成：

- 自定义管线微信布局修正。
- BindGroup entries 数组修正。
- WebGPU `a_vertexId` 修正。
- 微信 uniform buffer 256 字节地址间隔修正。
- TypeScript 单文件转译检查。
- 相关文件 ESLint 检查。
- `git diff --check`。
- 真机紫色清屏验证。
- 真机 Buffer 和 Texture 回读。
- 真机基础 Pipeline 和 depth/stencil 验证。
- 真机角色 Pipeline 变体验证。
- 真机真实 RenderPass 命令逐条重放。
- 三项运行时修正联合验证。
- 用户确认手机显示游戏画面。

运行时临时测试在结束后恢复原函数。引擎源码修改会在下一次 Creator 编译并重新构建微信小游戏时进入正式产物。
