import { expect, it } from "vitest";
import * as THREE from "three";
import { createMarkerTexture } from "./trajectoryMarkers";

function pixel(texture: THREE.DataTexture, x: number, y: number) {
  const data = texture.image.data as Uint8Array;
  return Array.from(data.slice((y * 64 + x) * 4, (y * 64 + x + 1) * 4));
}

it.each(["circle", "diamond"] as const)("draws an outlined %s with a white core and antialiased transparent edges", (shape) => {
  const texture = createMarkerTexture("#3866e9", shape);
  expect(texture.image.width).toBe(64);
  expect(texture.image.height).toBe(64);
  expect(pixel(texture, 32, 32)).toEqual([255, 255, 255, 255]);
  expect(pixel(texture, 57, 32)).toEqual([56, 102, 233, 255]);
  expect(pixel(texture, 0, 0)[3]).toBe(0);
  expect(pixel(texture, 63, 32)[3]).toBe(0);
  const [edgeX, edgeY] = shape === "circle" ? [53, 52] : [61, 32];
  const edgeAlpha = pixel(texture, edgeX, edgeY)[3];
  expect(edgeAlpha).toBeGreaterThan(0);
  expect(edgeAlpha).toBeLessThan(255);
  expect(pixel(texture, edgeX, edgeY)).toEqual(pixel(texture, 63 - edgeX, 63 - edgeY));
  texture.dispose();
});

it("configures a reusable sRGB texture without mipmaps and supports caller-owned disposal", () => {
  const texture = createMarkerTexture("#c78322", "circle");
  expect(texture.colorSpace).toBe(THREE.SRGBColorSpace);
  expect(texture.minFilter).toBe(THREE.LinearFilter);
  expect(texture.magFilter).toBe(THREE.LinearFilter);
  expect(texture.generateMipmaps).toBe(false);
  expect(texture.version).toBeGreaterThan(0);
  let disposed = false;
  texture.addEventListener("dispose", () => { disposed = true; });
  texture.dispose();
  expect(disposed).toBe(true);
});
