# 引擎维护的微信 WebGPU 接入（实验）

> 2026-09-15 更新：已接入同事提供的 Creator 微信平台插件，以下早期纯 post-build 方案说明保留作历史记录。当前必须启用平台 `useWebGPU`；GLSL4 由平台插件序列化，prepare 只验证。适配器使用 `build-adapter.cjs` 从 `platform-source` 编译。完整接入说明见 `D:/code/project/webgpu-engine-modify/INTEGRATION.md`。

实现集中在当前引擎仓库。项目只安装三个小型扩展入口文件，构建完成后自动调用引擎代码。
无需修改编辑器 app.asar，也无需每次手工修包。当前支持 Creator 3.8.8 的标准微信模板与 JSON 资源。

## 一次性接入

```powershell
node D:\code\cocos\own_cocos\scripts\wechat-webgpu\install.cjs <项目目录>
```

在 Creator 的扩展管理器“项目”页刷新并启用 `engine-wechat-webgpu`，重新打开构建面板。
该扩展的 Experimental WebGPU 默认开启；未安装或未启用扩展的项目不参与处理。
项目需要选择当前自定义引擎，并在功能裁剪中包含 gfx-webgpu。
关闭 md5Cache、WASM 压缩和引擎插件分离。暂不支持远程/压缩 WASM 与二进制资源。
移动引擎目录后需更新项目扩展入口路径。

已安装入口：NewProject_7 与 compute/_demo。编辑器是否已加载并启用仍需在扩展管理器确认。

## 实现

- game.ejs 根据构建时写入的布尔常量选择 WebGPU 首屏分支，不通过 require 加载 JSON。
- bootstrap 保存并恢复宿主 GPU，补齐标准枚举值，不伪造 GPU 对象。
- prepare.cjs 在 onAfterBuild 检查后端及 WASM，按名称/hash 补回 glsl4，
  补充微信隔离模块的变量引用并设置显式 WebGPU 模式。旧标准模板也可兼容；未知模板报错。
- patch-adapter.cjs 为微信两个预打包 adapter 添加模块级 window 引用，spread-adapter 后自动重建。
- DeviceManager 在创建交换链前检查成功状态；微信显式 WebGPU 模式缺入口时直接报错。
- WebGPUDevice 检查 adapter/context 空值；WASM 初始化不再吞掉异常。

原有 PC 模板及项目模块配置没有在这次接入中修改。
当前 post-build 方案并没有通过私有 API 注入 WEBGPU 编译常量；依赖项目显式包含 gfx-webgpu，
并验证实际后端代码与 WASM 已生成。不符合条件会使构建失败。

## 验证与限制

```powershell
node scripts/wechat-webgpu/test.cjs
```

已通过：设备缺失/失败/成功模拟测试、WebGL 首屏分支、EJS 渲染、GPU 保留；
在 NewProject_7 现有微信包上执行引擎 post-build 逻辑通过，检查 33 个 shader。
早先项目原型已在微信模拟器验证 gfxAPI=8 和 ubo-bug 场景显示。
尚未完成：扩展启用后的全新 Creator 构建、改动后完整 PC 场景回归、真机验证。
因此当前是已实现且完成针对性测试的实验接入，不能宣称所有环境已兼容。

期望日志：`[WebGPU] active backend=8, expected=8`、`scene started`。
手机由用户测试；失败请保留这些日志以及第一条业务/引擎错误。
