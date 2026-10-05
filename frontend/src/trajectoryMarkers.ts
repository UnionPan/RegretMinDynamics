import * as THREE from "three";

const TEXTURE_SIZE = 64;
const SAMPLE_GRID = 4;
const OUTER_RADIUS = 30;
const OUTLINE_WIDTH = (1.2 / 7) * TEXTURE_SIZE;

/** A screen-sized, outlined marker; callers share and dispose the texture. */
export function createMarkerTexture(color: string, shape: "circle" | "diamond"): THREE.DataTexture {
  const hex = new THREE.Color(color).getHex(THREE.SRGBColorSpace);
  const channels = [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255];
  const pixels = new Uint8Array(TEXTURE_SIZE * TEXTURE_SIZE * 4);
  const sampleCount = SAMPLE_GRID * SAMPLE_GRID;
  for (let y = 0; y < TEXTURE_SIZE; y++) {
    for (let x = 0; x < TEXTURE_SIZE; x++) {
      let covered = 0;
      let interior = 0;
      for (let sy = 0; sy < SAMPLE_GRID; sy++) {
        for (let sx = 0; sx < SAMPLE_GRID; sx++) {
          const dx = x + (sx + 0.5) / SAMPLE_GRID - TEXTURE_SIZE / 2;
          const dy = y + (sy + 0.5) / SAMPLE_GRID - TEXTURE_SIZE / 2;
          const distance = shape === "circle"
            ? Math.hypot(dx, dy) - OUTER_RADIUS
            : (Math.abs(dx) + Math.abs(dy) - OUTER_RADIUS) / Math.SQRT2;
          if (distance <= 0) covered++;
          if (distance <= -OUTLINE_WIDTH) interior++;
        }
      }
      const offset = (y * TEXTURE_SIZE + x) * 4;
      const whiteFraction = covered ? interior / covered : 0;
      channels.forEach((channel, index) => {
        pixels[offset + index] = Math.round(channel + (255 - channel) * whiteFraction);
      });
      pixels[offset + 3] = Math.round((covered / sampleCount) * 255);
    }
  }
  const texture = new THREE.DataTexture(pixels, TEXTURE_SIZE, TEXTURE_SIZE);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}
