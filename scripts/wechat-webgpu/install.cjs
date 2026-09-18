'use strict';
const fs = require('node:fs'), path = require('node:path');
const project = path.resolve(process.argv[2] || '.');
if (!fs.existsSync(path.join(project, 'assets')) || !fs.existsSync(path.join(project, 'package.json'))) throw Error('Pass a Creator project directory');
const dest = path.join(project, 'extensions/engine-wechat-webgpu');
const files = {
    'package.json': JSON.stringify({ name: 'engine-wechat-webgpu', version: '0.1.0', package_version: 2,
        description: 'Experimental WeChat WebGPU integration maintained by the custom engine',
        contributions: { builder: './builder.cjs' } }, null, 2),
    'builder.cjs': `module.exports = require(${JSON.stringify(path.join(__dirname, 'builder.cjs'))});\n`,
    'hooks.cjs': `module.exports = require(${JSON.stringify(path.join(__dirname, 'hooks.cjs'))});\n`,
};
for (const [name, content] of Object.entries(files)) {
    const file = path.join(dest, name);
    if (fs.existsSync(file) && fs.readFileSync(file, 'utf8') !== content) throw Error('Existing extension differs: ' + file);
}
fs.mkdirSync(dest, { recursive: true });
for (const [name, content] of Object.entries(files)) fs.writeFileSync(path.join(dest, name), content);
console.log('Installed project extension:', dest);
