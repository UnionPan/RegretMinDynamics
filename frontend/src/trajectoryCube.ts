import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { createTrajectoryLine } from "./trajectoryLine";
import { createMarkerTexture } from "./trajectoryMarkers";
import { layoutLabels, PLAYER_COLORS, sampleAt, sampleDescription } from "./trajectoryData";
import type { CubeHandle, Position, TrajectoryData } from "./trajectoryData";

/** Flat, antialiased geometry. Rendering happens only on resize, rotation or seek. */
export function mountCube(host: HTMLElement, data: TrajectoryData, signal: AbortSignal): CubeHandle | null {
  if (signal.aborted) return null;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.setClearColor(0xffffff, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 40);
  camera.up.set(0, 0, 1);
  camera.position.set(1.9, -3.7, 1.6).normalize().multiplyScalar(3.2).addScalar(0.5);
  camera.lookAt(0.5, 0.5, 0.5);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.5, 0.5, 0.5);
  controls.enableDamping = false;
  controls.enablePan = false;
  controls.minZoom = 0.6;
  controls.maxZoom = 1.7;
  controls.minDistance = 2.2;
  controls.maxDistance = 7;
  const resources: { dispose: () => void }[] = [];
  const materials: LineMaterial[] = [];
  const groups = new Map<string, THREE.Object3D[]>();
  const overlays: { node: HTMLSpanElement; position: Position; offset: number; outside?: boolean }[] = [];
  const markers: THREE.Sprite[] = [];
  const sprites: { marker: THREE.Sprite; diameter: number }[] = [];
  const trajectories: ReturnType<typeof createTrajectoryLine>[] = [];
  let width = 1, height = 1, disposed = false, iteration = data.lastIteration, replay = false, fading = true;
  let observer: ResizeObserver | undefined;
  const group = (name: string, object: THREE.Object3D) => {
    groups.set(name, [...(groups.get(name) ?? []), object]);
    scene.add(object);
  };
  const addLine = (points: number[], color: string, opacity = 1, dash = "solid", lineWidth = 2.2) => {
    const geometry = new LineGeometry();
    geometry.setPositions(points);
    const patterns: Record<string, [number, number]> = { dash: [0.035, 0.025], dot: [0.008, 0.024], dashdot: [0.055, 0.017], longdash: [0.08, 0.028] };
    const [dashSize, gapSize] = patterns[dash] ?? [0.035, 0.025];
    const material = new LineMaterial({ color: new THREE.Color(color).getHex(), linewidth: lineWidth, transparent: opacity < 1, opacity, depthWrite: opacity === 1, dashed: dash !== "solid", dashSize, gapSize });
    const line = new Line2(geometry, material);
    line.computeLineDistances();
    resources.push(geometry, material);
    materials.push(material);
    return line;
  };
  const draw = () => {
    if (disposed) return;
    const pixelScale = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / (height * camera.zoom);
    sprites.forEach(({ marker, diameter }) => marker.scale.setScalar(diameter * pixelScale));
    renderer.render(scene, camera);
    const center = new THREE.Vector3(0.5, 0.5, 0.5).project(camera);
    const labels = overlays.map(({ node, position, offset, outside }) => {
      const projected = new THREE.Vector3(...position).project(camera);
      const labelWidth = node.offsetWidth || Math.max(12, node.textContent!.length * 6.5 + 10), labelHeight = node.offsetHeight || 18;
      let x = (projected.x + 1) * width / 2, y = (1 - projected.y) * height / 2 + offset;
      if (outside) {
        const direction = new THREE.Vector2((projected.x - center.x) * width, (center.y - projected.y) * height).normalize();
        const clearance = Math.abs(direction.x) * labelWidth / 2 + Math.abs(direction.y) * labelHeight / 2 + 18;
        x += direction.x * clearance; y += direction.y * clearance - labelHeight / 2;
      }
      return { x, y, width: labelWidth, height: labelHeight };
    });
    layoutLabels(labels, width, height).forEach(([x, y], index) => {
      overlays[index].node.style.transform = "translate(" + x + "px, " + y + "px)";
    });
  };
  const seek = (value: number) => {
    if (disposed) return;
    iteration = value;
    trajectories.forEach((trajectory) => trajectory.update(value, replay, fading));
    markers.forEach((marker, index) => marker.position.set(...sampleAt(data.paths[index], value).position));
    draw();
  };
  const resize = () => {
    width = host.clientWidth || 800;
    height = host.clientHeight || 440;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    // Preserve horizontal breathing room when the chart is taller than it is wide.
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(20)) / Math.min(camera.aspect, 1)));
    camera.updateProjectionMatrix();
    materials.forEach((material) => material.resolution.set(width, height));
    draw();
  };
  const hover = (event: PointerEvent) => {
    const rect = host.getBoundingClientRect();
    let nearest = 18, title = "Drag to rotate. Scroll to zoom.";
    const inspect = (position: Position, description: string, visible: boolean) => {
      if (!visible) return;
      const point = new THREE.Vector3(...position).project(camera);
      const distance = Math.hypot((point.x + 1) * width / 2 - (event.clientX - rect.left), (1 - point.y) * height / 2 - (event.clientY - rect.top));
      if (distance < nearest) { nearest = distance; title = description; }
    };
    data.paths.forEach((path, index) => {
      const sample = sampleAt(path, iteration);
      inspect(sample.position, sampleDescription(path, sample), markers[index].visible);
    });
    data.references.forEach((reference) => inspect(reference.position, reference.label + "\n" + data.axes.map((label, axis) => label + ": " + reference.position[axis].toFixed(3)).join("\n"), groups.get("Equilibrium references")?.[0]?.visible ?? false));
    renderer.domElement.title = title;
  };
  const keyboard = (event: KeyboardEvent) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const turn: Record<string, [number, number]> = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] };
    if (turn[event.key]) {
      event.preventDefault();
      const alignment = new THREE.Quaternion().setFromUnitVectors(camera.up, new THREE.Vector3(0, 1, 0));
      const offset = camera.position.clone().sub(controls.target).applyQuaternion(alignment);
      const sphere = new THREE.Spherical().setFromVector3(offset);
      sphere.theta += turn[event.key][0];
      sphere.phi = Math.max(0.15, Math.min(Math.PI - 0.15, sphere.phi + turn[event.key][1]));
      offset.setFromSpherical(sphere).applyQuaternion(alignment.invert());
      camera.position.copy(controls.target).add(offset);
      camera.lookAt(controls.target); controls.update(); draw();
    } else if (["+", "=", "-"].includes(event.key)) {
      event.preventDefault();
      camera.zoom = Math.max(0.6, Math.min(1.7, camera.zoom * (event.key === "-" ? 1 / 1.12 : 1.12)));
      camera.updateProjectionMatrix(); draw();
    } else if (event.key === "Home") { event.preventDefault(); controls.reset(); draw(); }
  };
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    observer?.disconnect();
    controls.removeEventListener("change", draw);
    controls.dispose();
    host.removeEventListener("pointermove", hover);
    renderer.domElement.removeEventListener("keydown", keyboard);
    resources.forEach((resource) => resource.dispose());
    renderer.dispose();
    renderer.forceContextLoss();
    renderer.domElement.remove();
    overlays.forEach((overlay) => overlay.node.remove());
  };
  try {
    // Inward normals cull the near walls at every angle, leaving the data exposed.
    // Flat materials give the volume depth without lighting or continuous rendering.
    const faceGeometry = new THREE.PlaneGeometry(1, 1);
    resources.push(faceGeometry);
    const faces: { position: Position; rotation: Position; color: string }[] = [
      { position: [0, 0.5, 0.5], rotation: [0, Math.PI / 2, 0], color: "#f3f5f7" },
      { position: [1, 0.5, 0.5], rotation: [0, -Math.PI / 2, 0], color: "#f3f5f7" },
      { position: [0.5, 0, 0.5], rotation: [-Math.PI / 2, 0, 0], color: "#fafbfc" },
      { position: [0.5, 1, 0.5], rotation: [Math.PI / 2, 0, 0], color: "#fafbfc" },
      { position: [0.5, 0.5, 0], rotation: [0, 0, 0], color: "#eaf0f4" },
      { position: [0.5, 0.5, 1], rotation: [Math.PI, 0, 0], color: "#eaf0f4" },
    ];
    faces.forEach(({ position, rotation, color }) => {
      const material = new THREE.MeshBasicMaterial({ color, side: THREE.FrontSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
      resources.push(material);
      const face = new THREE.Mesh(faceGeometry, material);
      face.position.set(...position); face.rotation.set(...rotation);
      scene.add(face);
    });
    const corners: Position[] = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]];
    for (let a = 0; a < corners.length; a++) for (let b = a + 1; b < corners.length; b++) {
      if (corners[a].filter((value, axis) => value !== corners[b][axis]).length === 1) scene.add(addLine([...corners[a], ...corners[b]], "#ccd5dd", 1, "solid", 0.9));
    }
    const axisStarts: Position[] = [[0,0,0], [1,0,0], [0,0,0]];
    const captionPositions: Position[] = [[0.5,0,0], [1,0.5,0], [0,0,0.5]];
    PLAYER_COLORS.forEach((color, axis) => {
      const end = [...axisStarts[axis]]; end[axis] = 1;
      scene.add(addLine([...axisStarts[axis],...end], color, 1, "solid", 1.3));
      const label = document.createElement("span");
      label.className = "trajectory-axis-label"; label.textContent = data.axes[axis]; label.style.color = color;
      overlays.push({ node: label, position: captionPositions[axis], offset: 0, outside: true });
    });
    PLAYER_COLORS.forEach((color, axis) => {
      for (const value of axis === 0 ? [0, 0.5, 1] : [0.5, 1]) {
        const node = document.createElement("span"), position: Position = [...axisStarts[axis]];
        position[axis] = value; node.className = "trajectory-tick-label"; node.textContent = String(value); node.style.color = color;
        overlays.push({ node, position, offset: -14 });
      }
    });
    data.paths.forEach((path) => {
      const trajectory = createTrajectoryLine(path, data.lastIteration);
      trajectories.push(trajectory); resources.push(trajectory);
      materials.push(trajectory.line.material);
      group(path.algorithm, trajectory.line);
    });
    const markerMaterials = new Map<string, THREE.SpriteMaterial>();
    const addMarker = (color: string, shape: "circle" | "diamond", diameter: number) => {
      const key = color + shape;
      let material = markerMaterials.get(key);
      if (!material) {
        const texture = createMarkerTexture(color, shape);
        // Markers annotate exact projected positions. Overlaying them prevents
        // a boundary policy from cutting its own billboard against a cube wall.
        material = new THREE.SpriteMaterial({ map: texture, sizeAttenuation: false, depthTest: false, depthWrite: false });
        markerMaterials.set(key, material); resources.push(texture, material);
      }
      const marker = new THREE.Sprite(material);
      marker.renderOrder = 3;
      sprites.push({ marker, diameter });
      return marker;
    };
    data.paths.forEach((path) => {
      const marker = addMarker(path.color, "circle", path.run === 0 ? 6 : 4.5);
      markers.push(marker); group(path.algorithm, marker);
    });
    data.references.forEach((reference) => {
      const marker = addMarker("#64748b", "diamond", 7);
      marker.position.set(...reference.position);
      group("Equilibrium references", marker);
    });
    renderer.domElement.setAttribute("aria-label", "Joint policy cube. Drag or use arrow keys to rotate. Scroll or use plus and minus to zoom. Home resets the view.");
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.tabIndex = 0;
    renderer.domElement.addEventListener("keydown", keyboard);
    host.append(renderer.domElement, ...overlays.map((overlay) => overlay.node));
    observer = new ResizeObserver(resize); observer.observe(host);
    controls.addEventListener("change", draw); host.addEventListener("pointermove", hover);
    controls.update(); controls.saveState(); resize(); seek(data.lastIteration);
    return { seek, setReplay: (active, tail = true) => { replay = active; fading = tail; seek(iteration); }, setVisible: (name, visible) => { groups.get(name)?.forEach((object) => { object.visible = visible; }); draw(); }, reset: () => { controls.reset(); draw(); }, dispose };
  } catch (failure) { dispose(); throw failure; }
}
