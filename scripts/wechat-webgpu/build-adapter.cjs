'use strict';
// Creator 3.8.8 adapter sources plus the colleague's GPU/Canvas patch.
const fs = require('node:fs'), path = require('node:path');
const browserify = require('browserify'), babelify = require('babelify');
async function build() {
    const entry = path.join(__dirname, 'platform-source/platforms/wechat/wrapper/builtin/index.js');
    const code = await new Promise((resolve, reject) => browserify(entry).transform(babelify, {
        babelrc: false, configFile: false,
        presets: [[require('@babel/preset-env'), { targets: { chrome: '80' } }]],
        plugins: [require('@babel/plugin-proposal-class-properties'), require('@babel/plugin-proposal-export-default-from')],
    }).bundle((err, data) => err ? reject(err) : resolve(data.toString())));
    const source = 'var window=GameGlobal;\n' + code;
    const minified = require('terser').minify(source);
    const result = await minified;
    if (!result.code) throw Error('Adapter minification failed');
    const dest = path.resolve(__dirname, '../../bin/adapter/minigame/wechat');
    fs.mkdirSync(dest, { recursive: true });
    fs.writeFileSync(path.join(dest, 'web-adapter.js'), source);
    fs.writeFileSync(path.join(dest, 'web-adapter.min.js'), result.code);
    console.log('Built WeChat adapter from source:', dest);
}
exports.build = build;
if (require.main === module) build().catch(e => { console.error(e); process.exitCode = 1; });
