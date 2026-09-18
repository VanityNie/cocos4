'use strict';
// Creator's quick compiler resolves bare PAL imports through ModuleQuery.
// The separately distributed PAL keeps cc.config overrides but lacks workspace exports.
// Describe the existing platform files without changing their implementations.
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const config=JSON.parse(fs.readFileSync(path.join(root,'cc.config.json')));
const groups=config.moduleOverrides;
const group=flag=>groups.find(g=>g.isVirtualModule && g.test.includes('.'+flag))?.overrides;
const web=group('HTML5'), native=group('NATIVE'), mini=group('MINIGAME');
if(!web||!native||!mini)throw Error('PAL platform overrides are missing');
const dir=path.join(__dirname,'creator-pal-exports');
const relative=p=>path.relative(dir,path.join(root,p)).replaceAll('\\','/');
const exportsMap={};
for(const [name,target] of Object.entries(web)) {
    if(!name.startsWith('pal/'))continue;
    const entry={web:relative(target),native:relative(native[name])};
    const miniTarget=mini[name];
    if(miniTarget.includes('{{')) {
        entry.minigame={};
        for(const platform of fs.readdirSync(path.join(root,'pal/minigame'))) {
            if(!platform.endsWith('.js'))continue;
            const name=platform.slice(0,-3);
            entry.minigame[name]=relative(miniTarget.replace('{{context.platform.toLowerCase()}}',name));
        }
    } else entry.minigame=relative(miniTarget);
    exportsMap['./'+name.slice(4)]=entry;
}
fs.mkdirSync(dir,{recursive:true});
fs.writeFileSync(path.join(dir,'package.json'),JSON.stringify({name:'pal',version:'0.0.0',private:true,exports:exportsMap},null,2)+'\n');
console.log('Generated Creator PAL module exports from cc.config.json');
