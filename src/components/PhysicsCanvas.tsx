import { useEffect, useRef } from "react";

export type SimParams = {
  gravity: number;
  damping: number;
  attractor: number;
  particleCount: number;
  trail: number;
  paused: boolean;
  springK: number;
  restLength: number;
  edgesPerNode: number;
  showEdges: boolean;
};

type State = {
  N: number;
  D: number;
  x: Float32Array;
  v: Float32Array;
  m: Float32Array;
  f: Float32Array;
  hue: Float32Array;
  edges: Int32Array;       // [E*2]
  edgeRest: Float32Array;  // [E]
  E: number;
};

function buildEdges(N: number, perNode: number) {
  // Random sparse graph: each node connects to `perNode` neighbors
  const set = new Set<number>();
  const list: number[] = [];
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < perNode; k++) {
      const j = Math.floor(Math.random() * N);
      if (j === i) continue;
      const a = Math.min(i, j), b = Math.max(i, j);
      const key = a * 100000 + b;
      if (set.has(key)) continue;
      set.add(key);
      list.push(a, b);
    }
  }
  return new Int32Array(list);
}

function initState(N: number, w: number, h: number, perNode: number, rest: number): State {
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
  const edges = buildEdges(N, perNode);
  const E = edges.length / 2;
  const edgeRest = new Float32Array(E);
  edgeRest.fill(rest);
  return { N, D: 2, x, v, m, f, hue, edges, edgeRest, E };
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
        const p = paramsRef.current;
        stateRef.current = initState(p.particleCount, r.width, r.height, p.edgesPerNode, p.restLength);
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
        s = initState(p.particleCount, w, h, p.edgesPerNode, p.restLength);
        stateRef.current = s;
      }

      // Trail fade
      ctx.fillStyle = `oklch(0.16 0.02 260 / ${1 - p.trail})`;
      ctx.fillRect(0, 0, w, h);

      if (!p.paused) {
        s.f.fill(0);

        // Gravity
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
            s.f[i * 2]     += dx * inv * a * 1000;
            s.f[i * 2 + 1] += dy * inv * a * 1000;
          }
        }

        // Spring forces (Hooke's law) — forces.py
        const k = p.springK;
        for (let e = 0; e < s.E; e++) {
          const i = s.edges[e * 2];
          const j = s.edges[e * 2 + 1];
          const dx = s.x[i * 2]     - s.x[j * 2];
          const dy = s.x[i * 2 + 1] - s.x[j * 2 + 1];
          const dist = Math.sqrt(dx * dx + dy * dy) + 1e-8;
          const dirx = dx / dist;
          const diry = dy / dist;
          const mag = -k * (dist - s.edgeRest[e]);
          const fx = dirx * mag;
          const fy = diry * mag;
          s.f[i * 2]     += fx;
          s.f[i * 2 + 1] += fy;
          s.f[j * 2]     -= fx;
          s.f[j * 2 + 1] -= fy;
        }

        // Integrate
        for (let i = 0; i < s.N; i++) {
          const ax = s.f[i * 2] / s.m[i];
          const ay = s.f[i * 2 + 1] / s.m[i];
          s.v[i * 2]     = (s.v[i * 2]     + ax * dt) * (1 - p.damping * dt);
          s.v[i * 2 + 1] = (s.v[i * 2 + 1] + ay * dt) * (1 - p.damping * dt);
          s.x[i * 2]     += s.v[i * 2]     * dt;
          s.x[i * 2 + 1] += s.v[i * 2 + 1] * dt;
          if (s.x[i * 2] < 0)     { s.x[i * 2] = 0; s.v[i * 2] *= -0.7; }
          else if (s.x[i * 2] > w){ s.x[i * 2] = w; s.v[i * 2] *= -0.7; }
          if (s.x[i * 2 + 1] < 0) { s.x[i * 2 + 1] = 0; s.v[i * 2 + 1] *= -0.7; }
          else if (s.x[i * 2 + 1] > h){ s.x[i * 2 + 1] = h; s.v[i * 2 + 1] *= -0.7; }
        }
      }

      // Render edges
      if (p.showEdges && s.E > 0) {
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        for (let e = 0; e < s.E; e++) {
          const i = s.edges[e * 2];
          const j = s.edges[e * 2 + 1];
          ctx.moveTo(s.x[i * 2], s.x[i * 2 + 1]);
          ctx.lineTo(s.x[j * 2], s.x[j * 2 + 1]);
        }
        ctx.strokeStyle = "oklch(0.78 0.16 280 / 0.25)";
        ctx.stroke();
      }

      // Render nodes
      for (let i = 0; i < s.N; i++) {
        const sp = Math.hypot(s.v[i * 2], s.v[i * 2 + 1]);
        const radius = 1.5 + s.m[i] * 1.6;
        const hueDeg = (s.hue[i] * 80 + 140) % 360;
        const light = Math.min(0.92, 0.55 + sp / 600);
        ctx.beginPath();
        ctx.arc(s.x[i * 2], s.x[i * 2 + 1], radius, 0, Math.PI * 2);
        ctx.fillStyle = `oklch(${light} 0.18 ${hueDeg})`;
        ctx.fill();
      }

      if (pointerRef.current.active) {
        const sign = pointerRef.current.mode;
        ctx.strokeStyle = sign > 0 ? "oklch(0.82 0.18 165 / 0.8)" : "oklch(0.72 0.20 35 / 0.8)";
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
