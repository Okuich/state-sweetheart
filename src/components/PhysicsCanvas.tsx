import { useEffect, useRef } from "react";

export type SimParams = {
  gravity: number;
  damping: number;
  attractor: number;
  particleCount: number;
  trail: number;
  paused: boolean;
};

type State = {
  N: number;
  D: number;
  x: Float32Array;   // [N*2]
  v: Float32Array;
  m: Float32Array;
  f: Float32Array;
  hue: Float32Array;
};

function initState(N: number, w: number, h: number): State {
  const x = new Float32Array(N * 2);
  const v = new Float32Array(N * 2);
  const m = new Float32Array(N);
  const f = new Float32Array(N * 2);
  const hue = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    x[i * 2] = Math.random() * w;
    x[i * 2 + 1] = Math.random() * h;
    const a = Math.random() * Math.PI * 2;
    const s = 30 + Math.random() * 40;
    v[i * 2] = Math.cos(a) * s;
    v[i * 2 + 1] = Math.sin(a) * s;
    m[i] = 0.6 + Math.random() * 1.8;
    hue[i] = Math.random();
  }
  return { N, D: 2, x, v, m, f, hue };
}

export function PhysicsCanvas({
  params,
  pointerRef,
}: {
  params: SimParams;
  pointerRef: React.MutableRefObject<{ x: number; y: number; active: boolean; mode: 1 | -1 }>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<State | null>(null);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let last = performance.now();

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const r = canvas.getBoundingClientRect();
      canvas.width = r.width * dpr;
      canvas.height = r.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (!stateRef.current) {
        stateRef.current = initState(paramsRef.current.particleCount, r.width, r.height);
      }
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const step = (now: number) => {
      const dt = Math.min(0.033, (now - last) / 1000);
      last = now;
      const p = paramsRef.current;
      const r = canvas.getBoundingClientRect();
      const w = r.width, h = r.height;

      let s = stateRef.current!;
      if (s.N !== p.particleCount) {
        s = initState(p.particleCount, w, h);
        stateRef.current = s;
      }

      // Trail fade
      ctx.fillStyle = `oklch(0.16 0.02 260 / ${1 - p.trail})`;
      ctx.fillRect(0, 0, w, h);

      if (!p.paused) {
        // Reset forces
        s.f.fill(0);
        // Gravity (down)
        for (let i = 0; i < s.N; i++) {
          s.f[i * 2 + 1] += p.gravity * s.m[i];
        }
        // Pointer attractor
        if (pointerRef.current.active) {
          const px = pointerRef.current.x, py = pointerRef.current.y;
          const sign = pointerRef.current.mode;
          const G = p.attractor * sign;
          for (let i = 0; i < s.N; i++) {
            const dx = px - s.x[i * 2];
            const dy = py - s.x[i * 2 + 1];
            const r2 = dx * dx + dy * dy + 400;
            const inv = 1 / Math.sqrt(r2);
            const a = (G * s.m[i]) / r2;
            s.f[i * 2] += dx * inv * a * 1000;
            s.f[i * 2 + 1] += dy * inv * a * 1000;
          }
        }
        // Integrate
        for (let i = 0; i < s.N; i++) {
          const ax = s.f[i * 2] / s.m[i];
          const ay = s.f[i * 2 + 1] / s.m[i];
          s.v[i * 2] = (s.v[i * 2] + ax * dt) * (1 - p.damping * dt);
          s.v[i * 2 + 1] = (s.v[i * 2 + 1] + ay * dt) * (1 - p.damping * dt);
          s.x[i * 2] += s.v[i * 2] * dt;
          s.x[i * 2 + 1] += s.v[i * 2 + 1] * dt;
          // Walls
          if (s.x[i * 2] < 0) { s.x[i * 2] = 0; s.v[i * 2] *= -0.7; }
          else if (s.x[i * 2] > w) { s.x[i * 2] = w; s.v[i * 2] *= -0.7; }
          if (s.x[i * 2 + 1] < 0) { s.x[i * 2 + 1] = 0; s.v[i * 2 + 1] *= -0.7; }
          else if (s.x[i * 2 + 1] > h) { s.x[i * 2 + 1] = h; s.v[i * 2 + 1] *= -0.7; }
        }
      }

      // Render
      for (let i = 0; i < s.N; i++) {
        const sp = Math.hypot(s.v[i * 2], s.v[i * 2 + 1]);
        const radius = 1.5 + s.m[i] * 1.6;
        const hueDeg = (s.hue[i] * 80 + 140) % 360; // mint→lavender
        const light = Math.min(0.92, 0.55 + sp / 600);
        ctx.beginPath();
        ctx.arc(s.x[i * 2], s.x[i * 2 + 1], radius, 0, Math.PI * 2);
        ctx.fillStyle = `oklch(${light} 0.18 ${hueDeg})`;
        ctx.fill();
      }

      // Pointer indicator
      if (pointerRef.current.active) {
        const sign = pointerRef.current.mode;
        ctx.strokeStyle =
          sign > 0 ? "oklch(0.82 0.18 165 / 0.8)" : "oklch(0.72 0.20 35 / 0.8)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(pointerRef.current.x, pointerRef.current.y, 28, 0, Math.PI * 2);
        ctx.stroke();
      }

      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [pointerRef]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full"
      onPointerMove={(e) => {
        const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
        pointerRef.current.x = e.clientX - r.left;
        pointerRef.current.y = e.clientY - r.top;
      }}
      onPointerDown={(e) => {
        const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
        pointerRef.current.x = e.clientX - r.left;
        pointerRef.current.y = e.clientY - r.top;
        pointerRef.current.active = true;
        pointerRef.current.mode = e.button === 2 ? -1 : 1;
      }}
      onPointerUp={() => { pointerRef.current.active = false; }}
      onPointerLeave={() => { pointerRef.current.active = false; }}
      onContextMenu={(e) => e.preventDefault()}
    />
  );
}
