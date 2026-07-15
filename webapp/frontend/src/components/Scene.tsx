import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

const reduced =
  typeof window !== "undefined" &&
  !!window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** A field of points that undulates like a speech signal in 3-D. */
function SignalField() {
  const ref = useRef<THREE.Points>(null!);
  const COLS = 96;
  const ROWS = 46;
  const N = COLS * ROWS;

  const { positions, base } = useMemo(() => {
    const positions = new Float32Array(N * 3);
    const base = new Float32Array(N * 2);
    let i = 0;
    for (let z = 0; z < ROWS; z++) {
      for (let x = 0; x < COLS; x++) {
        const px = (x / (COLS - 1) - 0.5) * 16;
        const pz = (z / (ROWS - 1) - 0.5) * 9;
        positions[i * 3] = px;
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = pz;
        base[i * 2] = px;
        base[i * 2 + 1] = pz;
        i++;
      }
    }
    return { positions, base };
  }, [N]);

  useFrame((state) => {
    if (reduced || !ref.current) return;
    const t = state.clock.elapsedTime;
    const attr = ref.current.geometry.attributes.position as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    for (let k = 0; k < N; k++) {
      const px = base[k * 2];
      const pz = base[k * 2 + 1];
      // pseudo-speech envelope (bursts + gaps) modulating a travelling wave
      const env = Math.max(0, Math.sin(px * 0.42 + t * 0.5)) * 0.6 + 0.22;
      arr[k * 3 + 1] =
        Math.sin(px * 1.25 - t * 1.5) * env * 0.55 + Math.sin(pz * 1.05 + t * 0.8) * 0.16;
    }
    attr.needsUpdate = true;
    ref.current.rotation.y = Math.sin(t * 0.05) * 0.12;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={N} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color="#4FD1C5"
        transparent
        opacity={0.8}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}

export default function Scene() {
  return (
    <Canvas
      camera={{ position: [0, 3.4, 9.6], fov: 42 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      frameloop={reduced ? "demand" : "always"}
      style={{ pointerEvents: "none" }}
    >
      <SignalField />
    </Canvas>
  );
}
