'use strict';
const fs = require('node:fs'), path = require('node:path');
// This repository consumes prebuilt engine-platforms adapters. Keep this wrapper
// reproducible after spread-adapter, until the platform package sources are available.
function patch() {
    const dir = path.resolve(__dirname, '../../bin/adapter/minigame/wechat');
    const prefix = 'var window=GameGlobal;\n';
    for (const name of ['web-adapter.js', 'web-adapter.min.js']) {
        const file = path.join(dir, name), text = fs.readFileSync(file, 'utf8');
        if (!text.startsWith(prefix)) fs.writeFileSync(file, prefix + text);
    }
}
module.exports = { patch };
if (require.main === module) patch();
