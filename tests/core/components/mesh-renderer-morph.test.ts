import { Mesh } from '../../../cocos/3d/assets/mesh';
import { MeshRenderer } from '../../../cocos/3d/framework/mesh-renderer';
import { MorphModel } from '../../../cocos/3d/models/morph-model';
import { Model } from '../../../cocos/render-scene/scene/model';
import { Node, Scene } from '../../../cocos/scene-graph';
import { director } from '../../../cocos/game';
import { Attribute, AttributeName, Format, PrimitiveMode } from '../../../cocos/gfx';
import { Vec3 } from '../../../cocos/core';

function morphMesh(): Mesh {
    const data = new Uint8Array(108);
    new Float32Array(data.buffer, 0, 9).set([-1, 0, 0, 1, 0, 0, 0, 1, 0]);
    new Float32Array(data.buffer, 36, 9).fill(0.5);
    const mesh = new Mesh();
    mesh.reset({ data, struct: {
        vertexBundles: [{ attributes: [new Attribute(AttributeName.ATTR_POSITION, Format.RGB32F)], view: { offset: 0, length: 36, count: 3, stride: 12 } }],
        primitives: [{ vertexBundelIndices: [0], primitiveMode: PrimitiveMode.TRIANGLE_LIST }],
        morph: { subMeshMorphs: [{ attributes: [AttributeName.ATTR_POSITION], targets: [36, 72].map((offset) => ({ displacements: [{ offset, length: 36, count: 9, stride: 4 }] })), weights: [0, 0] }] },
        minPosition: new Vec3(-2, -2, -2), maxPosition: new Vec3(2, 2, 2),
    } });
    mesh.initialize();
    return mesh;
}

test('assigning a morph mesh to an already active renderer creates a MorphModel and preserves it across mode switches', () => {
    const scene = new Scene();
    director.runSceneImmediate(scene);
    const node = new Node(); scene.addChild(node);
    const renderer = node.addComponent(MeshRenderer);
    expect(renderer.model).toBeInstanceOf(Model);
    expect(renderer.model).not.toBeInstanceOf(MorphModel);
    const mesh = morphMesh();
    renderer.morphRenderingMode = 'cpu';
    renderer.mesh = mesh;
    expect(renderer.model).toBeInstanceOf(MorphModel);
    renderer.setWeights([0.25, 0.75], 0);
    for (const mode of ['vs', 'cpu'] as const) {
        renderer.morphRenderingMode = mode;
        expect(renderer.model).toBeInstanceOf(MorphModel);
        expect(renderer.getWeight(0, 1)).toBe(0.75);
        expect(renderer.model!.getMacroPatches(0)).toEqual(expect.arrayContaining([{ name: 'CC_USE_MORPH', value: true }]));
    }
    node.destroy(); Node._deferredDestroy(); mesh.destroy();
});

test('a deserialized inactive renderer can select its mode before onLoad initializes morph weights', () => {
    const node = new Node(); node.active = false;
    const renderer = node.addComponent(MeshRenderer);
    const mesh = morphMesh();
    // Deserialization assigns the serialized field directly, bypassing the mesh setter.
    (renderer as unknown as { _mesh: Mesh })._mesh = mesh;
    expect(() => { renderer.morphRenderingMode = 'cpu'; }).not.toThrow();
    expect(renderer.getWeight(0, 0)).toBe(0);
    node.destroy(); Node._deferredDestroy(); mesh.destroy();
});
