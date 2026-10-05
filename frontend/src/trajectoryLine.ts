import * as THREE from "three";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { pathAppearance, TRAIL_FRACTION, trailAt } from "./trajectoryData";
import type { TrajectoryPath } from "./trajectoryData";

/** Reuse GPU storage for a continuous path or a fading, time-clipped trail. */
export function createTrajectoryLine(path: TrajectoryPath, lastIteration: number) {
  const geometry = new LineGeometry();
  const capacity = path.points.length + 1;
  geometry.setPositions(new Float32Array((capacity + 1) * 3));
  const start = geometry.getAttribute("instanceStart") as THREE.InterleavedBufferAttribute;
  const end = geometry.getAttribute("instanceEnd") as THREE.InterleavedBufferAttribute;
  start.data.setUsage(THREE.DynamicDrawUsage);
  const alphaStart = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1).setUsage(THREE.DynamicDrawUsage);
  const alphaEnd = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("instanceOpacityStart", alphaStart);
  geometry.setAttribute("instanceOpacityEnd", alphaEnd);
  const appearance = pathAppearance(path.run);
  const material = new LineMaterial({ color: path.color, linewidth: appearance.width, opacity: appearance.opacity, transparent: true, depthWrite: false });
  // Line2 expands each sampled segment into a screen-width strip. Interpolate
  // age across that strip so fading follows trajectory time, including loops.
  material.vertexShader = "attribute float instanceOpacityStart; attribute float instanceOpacityEnd; varying float vTrailOpacity;\n" + material.vertexShader.replace("void main() {", "void main() { vTrailOpacity = (position.y < 0.5) ? instanceOpacityStart : instanceOpacityEnd;");
  material.fragmentShader = "varying float vTrailOpacity;\n" + material.fragmentShader.replace("float alpha = opacity;", "float alpha = opacity * vTrailOpacity;");
  const line = new Line2(geometry, material);
  line.frustumCulled = false;
  const update = (iteration: number, replay: boolean, fading = true) => {
    const duration = Math.max(1, lastIteration * TRAIL_FRACTION);
    const points = replay ? trailAt(path, iteration, fading ? duration : Infinity) : path.points;
    const head = points.at(-1)!.iteration;
    const alpha = (at: number) => replay && fading ? Math.pow(Math.max(0, 1 - (head - at) / duration), 1.6) : 1;
    geometry.instanceCount = Math.max(0, points.length - 1);
    for (let index = 0; index < geometry.instanceCount; index++) {
      start.setXYZ(index, ...points[index].position);
      end.setXYZ(index, ...points[index + 1].position);
      alphaStart.setX(index, alpha(points[index].iteration));
      alphaEnd.setX(index, alpha(points[index + 1].iteration));
    }
    start.data.needsUpdate = true; alphaStart.needsUpdate = true; alphaEnd.needsUpdate = true;
  };
  update(lastIteration, false);
  return { line, update, dispose: () => { geometry.dispose(); material.dispose(); } };
}
