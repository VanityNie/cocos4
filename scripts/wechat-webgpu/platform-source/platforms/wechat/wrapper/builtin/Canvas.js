// import HTMLCanvasElement from './HTMLCanvasElement'
import { innerWidth, innerHeight } from './WindowProperties';

const hasModifiedCanvasPrototype = false;
const hasInit2DContextConstructor = false;
const hasInitWebGLContextConstructor = false;

export default function Canvas () {
  const canvas = wx.createCanvas();

  canvas.type = 'canvas';

  // canvas.__proto__.__proto__.__proto__ = new HTMLCanvasElement()

  const _getContext = canvas.getContext.bind(canvas);

  // 透传 getContext：绑定到 canvas 并保留 'webgpu' 类型。
  // 主 Canvas 最终可能被 WebGPU 后端通过 getContext('webgpu') 使用，
  // 这里不拦截、不伪造，交回微信运行时原生实现处理；未提供时自然失败。
  canvas.getContext = function (type, ...args) {
    if (typeof GameGlobal !== 'undefined' && GameGlobal.__cocosWebGPUEnabled === true
      && (canvas === GameGlobal.screencanvas || canvas === GameGlobal.canvas)
      && ['webgl', 'webgl2', 'experimental-webgl', 'experimental-webgl2', '2d'].includes(type)) {
      throw new Error('[WebGPU] Main canvas is reserved for webgpu; rejected getContext(' + type + ')');
    }
    if (type === 'webgpu') {
      return _getContext('webgpu', ...args);
    }
    return _getContext(type, ...args);
  };

  canvas.getBoundingClientRect = () => {
    const ret = {
      top: 0,
      left: 0,
      width: window.innerWidth,
      height: window.innerHeight,
    };
    return ret;
  };

  canvas.style = {
    top: '0px',
    left: '0px',
    width: `${innerWidth}px`,
    height: `${innerHeight}px`,
  };

  canvas.addEventListener = function (type, listener, options = {}) {
    if (typeof getApp === 'function') {
      // for wechat miniprogram
      GameGlobal.document.addEventListener(type, listener, options);
    } else {
      // for wechat minigame
      document.addEventListener(type, listener, options);
    }
  };

  canvas.removeEventListener = function (type, listener) {
    if (typeof getApp === 'function') {
      // for wechat miniprogram
      GameGlobal.document.removeEventListener(type, listener);
    } else {
      // for wechat minigame
      document.removeEventListener(type, listener);
    }
  };

  canvas.dispatchEvent = function (event = {}) {
    console.log('canvas.dispatchEvent', event.type, event);
    // nothing to do
  };

  Object.defineProperty(canvas, 'clientWidth', {
    enumerable: true,
    get: function get () {
      return innerWidth;
    },
  });

  Object.defineProperty(canvas, 'clientHeight', {
    enumerable: true,
    get: function get () {
      return innerHeight;
    },
  });

  return canvas;
}
