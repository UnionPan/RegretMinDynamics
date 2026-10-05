import { beforeEach, expect, it, vi } from "vitest";
import * as THREE from "three";
import { extractTrajectory } from "./trajectoryData";
import { mountCube } from "./trajectoryCube";

const state = vi.hoisted(() => ({ render: vi.fn(), dispose: vi.fn(), contextLoss: vi.fn(), pixelRatio: vi.fn(), size: vi.fn(), controlsDispose: vi.fn(), reset: vi.fn(), disconnect: vi.fn(), change: null as (() => void) | null, resize: null as (() => void) | null }));
vi.mock("three", async (original) => {
  const actual = await original<typeof import("three")>();
  return { ...actual, WebGLRenderer: class {
    domElement = document.createElement("canvas");
    setPixelRatio = state.pixelRatio; setClearColor() {}
    setSize = state.size; render = state.render; dispose = state.dispose; forceContextLoss = state.contextLoss;
  } };
});
vi.mock("three/addons/controls/OrbitControls.js", () => ({
  OrbitControls: class {
    target = new THREE.Vector3(); enableDamping = false; enablePan = false; minZoom = 0; maxZoom = 0;
    addEventListener(_event: string, callback: () => void) { state.change = callback; }
    removeEventListener() { state.change = null; }
    update() {} saveState() {} reset = state.reset; dispose = state.controlsDispose;
  },
}));
const data = extractTrajectory({ layout: {}, data: [
  { type: "scatter3d", meta: { role: "history", algorithm: "Hedge", run: 0 }, x: [0, 1], y: [1, 0], z: [0.5, 0.5], customdata: [[1, 0, 1, 0.5], [9, 1, 0, 0.5]], line: { color: "#3866e9", dash: "dot" }, opacity: 0.5 },
  { type: "scatter3d", meta: { role: "equilibrium" }, x: [0.5], y: [0.5], z: [0.5], hovertemplate: "Uniform reference<br><extra></extra>" },
] });
beforeEach(() => {
  vi.clearAllMocks();
  state.render.mockImplementation((scene: THREE.Scene, camera: THREE.Camera) => { scene.updateMatrixWorld(); camera.updateMatrixWorld(); });
  state.change = null;
  vi.stubGlobal("ResizeObserver", class {
    constructor(callback: () => void) { state.resize = callback; }
    observe() {} disconnect = state.disconnect;
  });
});
function host() {
  const node = document.createElement("div");
  Object.defineProperties(node, { clientWidth: { value: 360 }, clientHeight: { value: 400 } });
  document.body.append(node);
  return node;
}

it("renders on demand, preserves probability axes and disposes every owned GPU resource", () => {
  const frame = vi.spyOn(window, "requestAnimationFrame");
  const geometryDispose = vi.spyOn(THREE.BufferGeometry.prototype, "dispose");
  const materialDispose = vi.spyOn(THREE.Material.prototype, "dispose");
  const container = host();
  const mounted = mountCube(container, data, new AbortController().signal)!;
  expect(state.pixelRatio).toHaveBeenCalledWith(Math.min(window.devicePixelRatio, 1.5));
  expect(container.querySelector("canvas")).toHaveAttribute("role", "img");
  expect(container.textContent).toContain("P1 · P(Action 0)");
  expect(container.querySelectorAll(".trajectory-axis-label")).toHaveLength(3);
  const scene = state.render.mock.calls.at(-1)![0] as THREE.Scene;
  const camera = state.render.mock.calls.at(-1)![1] as THREE.PerspectiveCamera;
  expect(camera.up.toArray()).toEqual([0, 0, 1]);
  const cursor = scene.children.find((node) => node instanceof THREE.Sprite)!;
  mounted.seek(5);
  expect(cursor.position.toArray()).toEqual([0.5, 0.5, 0.5]);
  mounted.setVisible("Hedge", false);
  expect(cursor.visible).toBe(false);
  mounted.setVisible("Hedge", true);
  expect(cursor.visible).toBe(true);
  mounted.reset();
  expect(state.reset).toHaveBeenCalledTimes(1);
  state.change!(); state.resize!();
  container.dispatchEvent(new MouseEvent("pointermove", { clientX: 180, clientY: 200 }));
  expect(container.querySelector("canvas")?.title).toContain("Probabilities");
  const canvas = container.querySelector("canvas")!;
  expect(canvas).toHaveAttribute("tabindex", "0");
  const previous = camera.position.clone();
  const browserZoom = new KeyboardEvent("keydown", { key: "+", metaKey: true, cancelable: true });
  canvas.dispatchEvent(browserZoom);
  expect(camera.zoom).toBe(1);
  expect(browserZoom.defaultPrevented).toBe(false);
  canvas.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft" }));
  expect(camera.position.equals(previous)).toBe(false);
  canvas.dispatchEvent(new KeyboardEvent("keydown", { key: "+" }));
  expect(camera.zoom).toBeGreaterThan(1);
  canvas.dispatchEvent(new KeyboardEvent("keydown", { key: "-" }));
  canvas.dispatchEvent(new KeyboardEvent("keydown", { key: "Home" }));
  expect(state.reset).toHaveBeenCalledTimes(2);
  expect(frame).not.toHaveBeenCalled();
  mounted.dispose(); mounted.dispose();
  expect(state.disconnect).toHaveBeenCalledTimes(1);
  expect(state.controlsDispose).toHaveBeenCalledTimes(1);
  expect(state.dispose).toHaveBeenCalledTimes(1);
  expect(state.contextLoss).toHaveBeenCalledTimes(1);
  expect(geometryDispose.mock.calls.length).toBeGreaterThan(12);
  expect(materialDispose.mock.calls.length).toBeGreaterThan(12);
  expect(container.childElementCount).toBe(0);
  const count = state.render.mock.calls.length;
  mounted.seek(7);
  expect(state.render).toHaveBeenCalledTimes(count);
  container.remove();
});

it("honors cancellation before allocating a scene and releases a failed render", () => {
  const controller = new AbortController(); controller.abort();
  const container = host();
  expect(mountCube(container, data, controller.signal)).toBeNull();
  expect(state.render).not.toHaveBeenCalled();
  state.render.mockImplementationOnce(() => { throw new Error("GPU context lost"); });
  expect(() => mountCube(container, data, new AbortController().signal)).toThrow("GPU context lost");
  expect(state.dispose).toHaveBeenCalledTimes(1);
  expect(container.childElementCount).toBe(0);
  container.remove();
});

it("gives the cube depth with inward faces that leave trajectories visible from every viewing side", () => {
  const container = host();
  const mounted = mountCube(container, data, new AbortController().signal)!;
  const scene = state.render.mock.calls.at(-1)![0] as THREE.Scene;
  const faces = scene.children.filter((node): node is THREE.Mesh => node instanceof THREE.Mesh && node.geometry.type === "PlaneGeometry");
  expect(faces).toHaveLength(6);
  const center = new THREE.Vector3(0.5, 0.5, 0.5);
  for (const face of faces) {
    const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(face.quaternion);
    expect(normal.dot(face.position.clone().sub(center))).toBeCloseTo(-0.5);
    expect((face.material as THREE.Material).side).toBe(THREE.FrontSide);
    // An observer outside any wall sees its back, so the nearest wall is culled.
    const outside = face.position.clone().sub(normal);
    expect(normal.dot(outside.sub(face.position))).toBeLessThan(0);
  }
  mounted.seek(5);
  const cursor = scene.children.find((node) => node instanceof THREE.Sprite)!;
  expect(cursor.position.toArray()).toEqual([0.5, 0.5, 0.5]);
  mounted.dispose();
  container.remove();
});

it("uses perspective so parallel world edges converge on screen and fits a narrow viewport", () => {
  const container = host();
  const mounted = mountCube(container, data, new AbortController().signal)!;
  const camera = state.render.mock.calls.at(-1)![1] as THREE.PerspectiveCamera;
  expect(camera.isPerspectiveCamera).toBe(true);
  const project = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z).project(camera);
  const nearEdge = project(1, 0, 0).sub(project(0, 0, 0));
  const farEdge = project(1, 1, 0).sub(project(0, 1, 0));
  expect(Math.abs(nearEdge.x * farEdge.y - nearEdge.y * farEdge.x)).toBeGreaterThan(0.01);
  for (const x of [0, 1]) for (const y of [0, 1]) for (const z of [0, 1]) {
    const projected = project(x, y, z);
    expect(Math.abs(projected.x)).toBeLessThan(0.85);
    expect(Math.abs(projected.y)).toBeLessThan(0.85);
  }
  mounted.dispose(); container.remove();
});

it("draws continuous fine paths and compact screen-sized markers without changing recorded points", () => {
  const container = host();
  const mounted = mountCube(container, data, new AbortController().signal)!;
  const scene = state.render.mock.calls.at(-1)![0] as THREE.Scene;
  const line = scene.children.find((node) => node.type === "Line2" && (node as THREE.Mesh).geometry.getAttribute("instanceStart").getZ(0) === 0.5) as THREE.Mesh<THREE.BufferGeometry, THREE.Material & { dashed: boolean; linewidth: number }>;
  expect(line.material.dashed).toBe(false);
  expect(line.material.linewidth).toBeLessThanOrEqual(1.5);
  expect(Array.from(line.geometry.getAttribute("instanceStart").array).slice(0, 6)).toEqual([0, 1, 0.5, 1, 0, 0.5]);
  const markers = scene.children.filter((node): node is THREE.Sprite => node instanceof THREE.Sprite);
  expect(markers).toHaveLength(2);
  expect(markers.every((marker) => !marker.material.sizeAttenuation)).toBe(true);
  expect(markers.every((marker) => !marker.material.depthTest)).toBe(true);
  mounted.dispose(); container.remove();
});

it("reveals only a fading recent tail during replay and restores complete recorded geometry afterward", () => {
  const container = host();
  const mounted = mountCube(container, data, new AbortController().signal)!;
  expect(mounted.setReplay).toBeTypeOf("function");
  const scene = state.render.mock.calls.at(-1)![0] as THREE.Scene;
  const line = scene.children.find((node) => node.type === "Line2" && (node as THREE.Mesh).geometry.getAttribute("instanceStart").getZ(0) === 0.5) as THREE.Mesh<THREE.InstancedBufferGeometry>;
  const geometry = line.geometry;
  const storage = geometry.getAttribute("instanceStart");
  mounted.setReplay(true); mounted.seek(5);
  expect(geometry.getAttribute("instanceEnd").getX(0)).toBeCloseTo(0.5);
  expect(geometry.getAttribute("instanceStart").getX(0)).toBeGreaterThan(0);
  expect(geometry.getAttribute("instanceOpacityStart").getX(0)).toBe(0);
  expect(geometry.getAttribute("instanceOpacityEnd").getX(0)).toBe(1);
  mounted.seek(1);
  expect(geometry.instanceCount).toBe(0);
  mounted.seek(5); mounted.setReplay(true, false);
  expect(geometry.getAttribute("instanceStart").getX(0)).toBe(0);
  expect(geometry.getAttribute("instanceEnd").getX(0)).toBeCloseTo(0.5);
  expect(geometry.getAttribute("instanceOpacityStart").getX(0)).toBe(1);
  mounted.setReplay(false);
  expect(geometry.getAttribute("instanceStart")).toBe(storage);
  expect(geometry.getAttribute("instanceStart").getX(0)).toBe(0);
  expect(geometry.getAttribute("instanceEnd").getX(0)).toBe(1);
  expect(geometry.getAttribute("instanceOpacityStart").getX(0)).toBe(1);
  mounted.dispose(); container.remove();
});
