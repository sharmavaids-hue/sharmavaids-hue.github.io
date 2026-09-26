/* ==========================================================================
   Aryan Sharma — portfolio interactions
   Three small numerical simulations, written in plain JavaScript:
     1. Heat equation       (explicit finite differences)
     2. Gradient descent    (heavy-ball momentum on a non-convex surface)
     3. Monte Carlo         (geometric Brownian motion, 5-year horizon)
   ========================================================================== */

(() => {
  "use strict";

  document.documentElement.classList.add("js");

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const COLORS = {
    paper: [243, 237, 226],
    ink: [29, 28, 25],
    inkCss: "#1d1c19",
    muted: "#6c665b",
    accent: "#a13a28",
    navy: "#23344d",
  };

  /* ---------- helpers ---------- */

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Size a canvas to its CSS box at device pixel ratio. */
  function fit(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h, dpr };
  }

  /** Run `frame` on rAF only while the element is on screen. */
  function loopWhileVisible(el, frame) {
    let visible = false;
    let raf = 0;
    const tick = (t) => {
      frame(t);
      if (visible) raf = requestAnimationFrame(tick);
    };
    const io = new IntersectionObserver((entries) => {
      const now = entries[0].isIntersecting;
      if (now && !visible) {
        visible = true;
        raf = requestAnimationFrame(tick);
      } else if (!now) {
        visible = false;
        cancelAnimationFrame(raf);
      }
    }, { rootMargin: "80px" });
    io.observe(el);
  }

  function localPoint(canvas, e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  /* ======================================================================
     1. Heat equation   ∂u/∂t = α ∇²u
     ====================================================================== */

  function heatSim(canvas) {
    const off = document.createElement("canvas");
    const offCtx = off.getContext("2d");
    let cols = 0, rows = 0, u, next, img;
    let size = fit(canvas);
    let lastTouch = -1e9;
    let lastAuto = 0;
    let pressed = false;
    let prev = null;

    // warm colour ramp: paper → sand → terracotta → vermillion → ink
    const stops = [
      [0.0, COLORS.paper],
      [0.18, [236, 214, 178]],
      [0.42, [214, 142, 96]],
      [0.68, [161, 58, 40]],
      [1.0, [44, 33, 28]],
    ];
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      let k = 0;
      while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
      const [t0, c0] = stops[k];
      const [t1, c1] = stops[k + 1];
      const f = clamp((t - t0) / (t1 - t0), 0, 1);
      const s = f * f * (3 - 2 * f);
      lut[i * 3] = lerp(c0[0], c1[0], s);
      lut[i * 3 + 1] = lerp(c0[1], c1[1], s);
      lut[i * 3 + 2] = lerp(c0[2], c1[2], s);
    }

    function allocate() {
      size = fit(canvas);
      const newCols = clamp(Math.round(size.w / 4.5), 60, 220);
      const newRows = clamp(Math.round(newCols * (size.h / size.w)), 16, 120);
      const old = u, oc = cols, or = rows;
      cols = newCols; rows = newRows;
      u = new Float32Array(cols * rows);
      next = new Float32Array(cols * rows);
      if (old) { // resample so resizes don't wipe the field
        for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
          const sx = Math.floor((x / cols) * oc), sy = Math.floor((y / rows) * or);
          u[y * cols + x] = old[sy * oc + sx] || 0;
        }
      }
      off.width = cols; off.height = rows;
      img = offCtx.createImageData(cols, rows);
    }

    function deposit(px, py, radius, amount) {
      const gx = (px / size.w) * cols;
      const gy = (py / size.h) * rows;
      const r = radius;
      const r2 = 2 * (r / 2.2) * (r / 2.2);
      for (let y = Math.max(1, Math.floor(gy - r)); y < Math.min(rows - 1, Math.ceil(gy + r)); y++) {
        for (let x = Math.max(1, Math.floor(gx - r)); x < Math.min(cols - 1, Math.ceil(gx + r)); x++) {
          const d2 = (x - gx) ** 2 + (y - gy) ** 2;
          if (d2 > r * r) continue;
          const i = y * cols + x;
          u[i] = Math.min(1, u[i] + amount * Math.exp(-d2 / r2));
        }
      }
    }

    function step() {
      const a = 0.15; // α·Δt/Δx² — must stay ≤ 0.25 for stability
      for (let y = 1; y < rows - 1; y++) {
        const row = y * cols;
        for (let x = 1; x < cols - 1; x++) {
          const i = row + x;
          const lap = u[i - 1] + u[i + 1] + u[i - cols] + u[i + cols] - 4 * u[i];
          next[i] = (u[i] + a * lap) * 0.9978;
        }
      }
      // Neumann boundary (insulated edges): ∂u/∂n = 0
      for (let x = 0; x < cols; x++) { next[x] = next[cols + x]; next[(rows - 1) * cols + x] = next[(rows - 2) * cols + x]; }
      for (let y = 0; y < rows; y++) { next[y * cols] = next[y * cols + 1]; next[y * cols + cols - 1] = next[y * cols + cols - 2]; }
      const t = u; u = next; next = t;
    }

    function draw() {
      const d = img.data;
      for (let i = 0, n = cols * rows; i < n; i++) {
        const k = Math.min(255, Math.round(Math.pow(u[i], 0.7) * 255)) * 3;
        const j = i * 4;
        d[j] = lut[k]; d[j + 1] = lut[k + 1]; d[j + 2] = lut[k + 2]; d[j + 3] = 255;
      }
      offCtx.putImageData(img, 0, 0);
      const { ctx, w, h } = size;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(off, 0, 0, w, h);

      // measurement grid, for a lab-notebook feel
      ctx.fillStyle = "rgba(29,28,25,0.16)";
      const gap = 22;
      for (let y = gap / 2; y < h; y += gap) {
        for (let x = gap / 2; x < w; x += gap) ctx.fillRect(x, y, 1, 1);
      }
    }

    function autoDrop(t) {
      if (t - lastTouch < 3500 || t - lastAuto < 1150) return;
      lastAuto = t;
      deposit(
        size.w * (0.08 + Math.random() * 0.84),
        size.h * (0.2 + Math.random() * 0.6),
        4 + Math.random() * 5,
        1.0 + Math.random() * 0.5
      );
    }

    // pointer interaction
    canvas.addEventListener("pointerdown", (e) => {
      pressed = true;
      prev = localPoint(canvas, e);
      lastTouch = performance.now();
      deposit(prev.x, prev.y, 5, 1.2);
    });
    const release = () => { pressed = false; prev = null; };
    window.addEventListener("pointerup", release);
    canvas.addEventListener("pointerleave", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("pointermove", (e) => {
      const p = localPoint(canvas, e);
      lastTouch = performance.now();
      if (pressed && prev) {
        // interpolate so fast strokes stay continuous
        const dist = Math.hypot(p.x - prev.x, p.y - prev.y);
        const n = Math.max(1, Math.ceil(dist / 5));
        for (let k = 1; k <= n; k++) {
          deposit(lerp(prev.x, p.x, k / n), lerp(prev.y, p.y, k / n), 3.6, 0.55);
        }
        prev = p;
      } else if (e.pointerType === "mouse") {
        deposit(p.x, p.y, 2.6, 0.07);
      }
    });

    allocate();
    // seed a few sources so the strip is alive on first paint
    [[0.12, 0.45], [0.3, 0.62], [0.52, 0.38], [0.7, 0.6], [0.88, 0.4]].forEach(([fx, fy], i) =>
      deposit(size.w * fx, size.h * fy, 6 + (i % 3) * 2, 1.4));
    for (let i = 0; i < 30; i++) step();

    new ResizeObserver(() => { allocate(); draw(); }).observe(canvas);

    if (reduceMotion) {
      draw();
      canvas.addEventListener("pointerup", () => { for (let i = 0; i < 20; i++) step(); draw(); });
      return;
    }

    loopWhileVisible(canvas, (t) => {
      autoDrop(t);
      for (let i = 0; i < 2; i++) step();
      draw();
    });
  }

  /* ======================================================================
     2. Gradient descent with momentum on a non-convex loss surface
     ====================================================================== */

  function gradientSim(canvas, readout) {
    const wells = [
      { x: -0.5, y: 0.42, a: 0.95, s: 0.21 },
      { x: 0.52, y: 0.38, a: 0.72, s: 0.17 },
      { x: 0.28, y: -0.52, a: 1.1, s: 0.24 },
      { x: -0.5, y: -0.42, a: 0.6, s: 0.15 },
      { x: 0.02, y: 0.02, a: 0.35, s: 0.12 },
    ];
    const raw = (x, y) => {
      let f = 0.62 * (x * x + y * y);
      for (const w of wells) {
        const d2 = (x - w.x) ** 2 + (y - w.y) ** 2;
        f -= w.a * Math.exp(-d2 / (2 * w.s * w.s));
      }
      f += 0.045 * Math.sin(5 * x + 1) * Math.cos(4 * y);
      return f;
    };

    // normalise to [0,1] over the domain
    let fMin = Infinity, fMax = -Infinity;
    for (let i = 0; i <= 200; i++) for (let j = 0; j <= 200; j++) {
      const v = raw(-1 + (2 * i) / 200, -1 + (2 * j) / 200);
      if (v < fMin) fMin = v;
      if (v > fMax) fMax = v;
    }
    const f = (x, y) => (raw(x, y) - fMin) / (fMax - fMin);
    const grad = (x, y) => {
      const h = 1e-3;
      return [(f(x + h, y) - f(x - h, y)) / (2 * h), (f(x, y + h) - f(x, y - h)) / (2 * h)];
    };

    let size = fit(canvas);
    const base = document.createElement("canvas");
    let pad = 0;
    const toPx = (x, y) => [pad + ((x + 1) / 2) * (size.w - 2 * pad), pad + ((1 - y) / 2) * (size.h - 2 * pad)];
    const toDomain = (px, py) => [((px - pad) / (size.w - 2 * pad)) * 2 - 1, 1 - ((py - pad) / (size.h - 2 * pad)) * 2];

    function renderBase() {
      size = fit(canvas);
      pad = Math.round(Math.min(size.w, size.h) * 0.06);
      const dpr = size.dpr;
      base.width = Math.round(size.w * dpr);
      base.height = Math.round(size.h * dpr);
      const b = base.getContext("2d");
      b.setTransform(dpr, 0, 0, dpr, 0, 0);

      // shaded relief
      const N = 150;
      const vals = new Float32Array((N + 1) * (N + 1));
      for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) {
        vals[j * (N + 1) + i] = f(-1 + (2 * i) / N, 1 - (2 * j) / N);
      }
      const shade = document.createElement("canvas");
      shade.width = N + 1; shade.height = N + 1;
      const sctx = shade.getContext("2d");
      const im = sctx.createImageData(N + 1, N + 1);
      for (let k = 0; k < vals.length; k++) {
        const t = 1 - vals[k]; // low loss → deeper tone
        const r = lerp(243, 214, t * t), g = lerp(237, 200, t * t), bl = lerp(226, 178, t * t);
        im.data[k * 4] = r; im.data[k * 4 + 1] = g; im.data[k * 4 + 2] = bl; im.data[k * 4 + 3] = 255;
      }
      sctx.putImageData(im, 0, 0);
      b.fillStyle = "rgb(243,237,226)";
      b.fillRect(0, 0, size.w, size.h);
      b.imageSmoothingEnabled = true;
      b.drawImage(shade, pad, pad, size.w - 2 * pad, size.h - 2 * pad);

      // contour lines — marching squares
      const cw = (size.w - 2 * pad) / N, ch = (size.h - 2 * pad) / N;
      const levels = 22;
      for (let L = 1; L < levels; L++) {
        const iso = L / levels;
        b.beginPath();
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
          const a = vals[j * (N + 1) + i], bb = vals[j * (N + 1) + i + 1];
          const c = vals[(j + 1) * (N + 1) + i + 1], d = vals[(j + 1) * (N + 1) + i];
          const idx = (a > iso ? 8 : 0) | (bb > iso ? 4 : 0) | (c > iso ? 2 : 0) | (d > iso ? 1 : 0);
          if (idx === 0 || idx === 15) continue;
          const x0 = pad + i * cw, y0 = pad + j * ch;
          const top = [x0 + cw * ((iso - a) / (bb - a)), y0];
          const right = [x0 + cw, y0 + ch * ((iso - bb) / (c - bb))];
          const bottom = [x0 + cw * ((iso - d) / (c - d)), y0 + ch];
          const left = [x0, y0 + ch * ((iso - a) / (d - a))];
          const seg = (p, q) => { b.moveTo(p[0], p[1]); b.lineTo(q[0], q[1]); };
          switch (idx) {
            case 1: case 14: seg(left, bottom); break;
            case 2: case 13: seg(bottom, right); break;
            case 3: case 12: seg(left, right); break;
            case 4: case 11: seg(top, right); break;
            case 5: seg(left, top); seg(bottom, right); break;
            case 6: case 9: seg(top, bottom); break;
            case 7: case 8: seg(left, top); break;
            case 10: seg(left, bottom); seg(top, right); break;
          }
        }
        b.strokeStyle = L % 4 === 0 ? "rgba(29,28,25,0.42)" : "rgba(29,28,25,0.17)";
        b.lineWidth = L % 4 === 0 ? 1 : 0.8;
        b.stroke();
      }

      // frame ticks
      b.strokeStyle = "rgba(29,28,25,0.3)";
      b.lineWidth = 1;
      for (let k = 0; k <= 10; k++) {
        const x = pad + (k / 10) * (size.w - 2 * pad);
        const y = pad + (k / 10) * (size.h - 2 * pad);
        b.beginPath(); b.moveTo(x, size.h - pad); b.lineTo(x, size.h - pad + (k % 5 ? 4 : 8)); b.stroke();
        b.beginPath(); b.moveTo(pad, y); b.lineTo(pad - (k % 5 ? 4 : 8), y); b.stroke();
      }
      b.strokeRect(pad, pad, size.w - 2 * pad, size.h - 2 * pad);
    }

    const runs = [];         // { path: [[x,y]...], v:[vx,vy], done, steps }
    let lastUser = -1e9, doneAt = 0;

    function launch(x, y) {
      runs.push({ path: [[x, y]], v: [0, 0], done: false, steps: 0 });
      if (runs.length > 5) runs.shift();
    }

    function advance() {
      const r = runs[runs.length - 1];
      if (!r || r.done) return;
      const beta = 0.9, lr = 0.0065;
      for (let k = 0; k < 2; k++) {
        const [x, y] = r.path[r.path.length - 1];
        const [gx, gy] = grad(x, y);
        r.v[0] = beta * r.v[0] - lr * gx;
        r.v[1] = beta * r.v[1] - lr * gy;
        const nx = clamp(x + r.v[0], -1, 1), ny = clamp(y + r.v[1], -1, 1);
        r.path.push([nx, ny]);
        r.steps++;
        if ((Math.hypot(r.v[0], r.v[1]) < 2e-4 && Math.hypot(gx, gy) < 0.02) || r.steps > 900) {
          r.done = true;
          doneAt = performance.now();
          break;
        }
      }
    }

    function draw() {
      const { ctx, w, h } = size;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(base, 0, 0, w, h);

      runs.forEach((r, idx) => {
        const current = idx === runs.length - 1;
        const alpha = current ? 1 : 0.22 + 0.12 * idx;
        ctx.beginPath();
        r.path.forEach(([x, y], i) => {
          const [px, py] = toPx(x, y);
          i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        });
        ctx.strokeStyle = current ? COLORS.accent : `rgba(161,58,40,${alpha * 0.6})`;
        ctx.lineWidth = current ? 2 : 1.2;
        ctx.lineJoin = "round";
        ctx.stroke();

        // start point
        const [sx, sy] = toPx(r.path[0][0], r.path[0][1]);
        ctx.fillStyle = current ? COLORS.navy : `rgba(35,52,77,${alpha})`;
        ctx.beginPath(); ctx.arc(sx, sy, current ? 5 : 3, 0, Math.PI * 2); ctx.fill();

        // end marker
        const [ex, ey] = toPx(...r.path[r.path.length - 1]);
        if (current) {
          ctx.fillStyle = "rgba(29,28,25,0.18)";
          ctx.beginPath(); ctx.arc(ex, ey, 12, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = COLORS.inkCss;
          ctx.beginPath(); ctx.arc(ex, ey, 7, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = "rgba(243,237,226,0.7)";
          ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(ex, ey, 4, 0, Math.PI * 2); ctx.stroke();
        } else {
          ctx.strokeStyle = `rgba(29,28,25,${alpha})`;
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(ex, ey, 3.5, 0, Math.PI * 2); ctx.stroke();
        }
      });

      const r = runs[runs.length - 1];
      if (r && readout) {
        const [x, y] = r.path[r.path.length - 1];
        readout.textContent = `step ${String(r.steps).padStart(3, "0")} · loss ${Math.max(0, f(x, y)).toFixed(3)}${r.done ? " · converged" : ""}`;
      }
    }

    function randomStart() {
      const edge = Math.random() * 4;
      const t = Math.random() * 1.7 - 0.85;
      if (edge < 1) return [t, 0.88];
      if (edge < 2) return [0.88, t];
      if (edge < 3) return [t, -0.88];
      return [-0.88, t];
    }

    canvas.addEventListener("pointerdown", (e) => {
      const p = localPoint(canvas, e);
      const [x, y] = toDomain(p.x, p.y);
      if (Math.abs(x) > 1 || Math.abs(y) > 1) return;
      lastUser = performance.now();
      launch(x, y);
      if (reduceMotion) { const r = runs[runs.length - 1]; while (!r.done) advance(); draw(); }
    });

    renderBase();
    launch(...randomStart());
    new ResizeObserver(() => { renderBase(); draw(); }).observe(canvas);

    if (reduceMotion) {
      const r = runs[0]; while (!r.done) advance(); draw();
      return;
    }

    loopWhileVisible(canvas, (t) => {
      advance();
      const r = runs[runs.length - 1];
      if (r && r.done && t - doneAt > 2600 && t - lastUser > 7000) launch(...randomStart());
      draw();
    });
  }

  /* ======================================================================
     3. Monte Carlo — geometric Brownian motion
        S(t+Δt) = S(t) · exp((μ − σ²/2)Δt + σ√Δt · Z)
     ====================================================================== */

  function monteCarloSim(canvas, readout) {
    const N_PATHS = 56, STEPS = 120, YEARS = 5, MU = 0.07, SIGMA = 0.22;
    const dt = YEARS / STEPS;
    let paths = [], bands = [], hero = 0, started = 0, lastUser = -1e9;
    let yMin = 0.4, yMax = 2.6;
    let size = fit(canvas);

    const quantile = (sorted, q) => {
      const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
      return lerp(sorted[lo], sorted[hi], pos - lo);
    };

    function simulate() {
      paths = [];
      for (let p = 0; p < N_PATHS; p++) {
        const s = new Float32Array(STEPS + 1);
        s[0] = 1;
        for (let k = 1; k <= STEPS; k++) {
          s[k] = s[k - 1] * Math.exp((MU - 0.5 * SIGMA * SIGMA) * dt + SIGMA * Math.sqrt(dt) * randn());
        }
        paths.push(s);
      }
      bands = [];
      for (let k = 0; k <= STEPS; k++) {
        const col = paths.map((s) => s[k]).sort((a, b) => a - b);
        bands.push([quantile(col, 0.05), quantile(col, 0.5), quantile(col, 0.95)]);
      }
      const finals = paths.map((s) => s[STEPS]).sort((a, b) => a - b);
      hero = Math.floor(Math.random() * N_PATHS);
      yMin = Math.min(0.5, Math.floor(quantile(finals, 0.01) * 4) / 4);
      yMax = Math.max(2.5, Math.ceil(quantile(finals, 0.985) * 2) / 2);
      started = performance.now();
      if (readout) {
        readout.textContent =
          `median ${quantile(finals, 0.5).toFixed(2)}× · P5 ${quantile(finals, 0.05).toFixed(2)}× · P95 ${quantile(finals, 0.95).toFixed(2)}×`;
      }
    }

    function draw(t) {
      const { ctx, w, h } = size;
      const m = { l: 46, r: 58, t: 58, b: 64 };
      const pw = w - m.l - m.r, ph = h - m.t - m.b;
      const X = (k) => m.l + (k / STEPS) * pw;
      const Y = (v) => m.t + (1 - (clamp(v, yMin, yMax) - yMin) / (yMax - yMin)) * ph;
      const progress = reduceMotion ? 1 : clamp((t - started) / 1800, 0, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const upto = Math.max(1, Math.round(eased * STEPS));

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "rgb(243,237,226)";
      ctx.fillRect(0, 0, w, h);

      // grid + axes
      ctx.font = `500 10px "JetBrains Mono", ui-monospace, monospace`;
      ctx.textBaseline = "middle";
      const stepY = yMax - yMin > 3 ? 1 : 0.5;
      for (let v = Math.ceil(yMin / stepY) * stepY; v <= yMax + 1e-9; v += stepY) {
        const y = Y(v);
        ctx.strokeStyle = Math.abs(v - 1) < 1e-9 ? "rgba(29,28,25,0.45)" : "rgba(29,28,25,0.1)";
        ctx.setLineDash(Math.abs(v - 1) < 1e-9 ? [4, 4] : []);
        ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + pw, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS.muted;
        ctx.textAlign = "right";
        ctx.fillText(`${v.toFixed(1)}×`, m.l - 8, y);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let yr = 0; yr <= YEARS; yr++) {
        const x = X((yr / YEARS) * STEPS);
        ctx.strokeStyle = "rgba(29,28,25,0.3)";
        ctx.beginPath(); ctx.moveTo(x, m.t + ph); ctx.lineTo(x, m.t + ph + 5); ctx.stroke();
        ctx.fillStyle = COLORS.muted;
        ctx.fillText(`Y${yr}`, x, m.t + ph + 10);
      }

      // P5–P95 band
      ctx.beginPath();
      for (let k = 0; k <= upto; k++) ctx.lineTo(X(k), Y(bands[k][2]));
      for (let k = upto; k >= 0; k--) ctx.lineTo(X(k), Y(bands[k][0]));
      ctx.closePath();
      ctx.fillStyle = "rgba(161,58,40,0.1)";
      ctx.fill();

      // all paths
      ctx.lineWidth = 0.9;
      ctx.strokeStyle = "rgba(29,28,25,0.13)";
      paths.forEach((s, p) => {
        if (p === hero) return;
        ctx.beginPath();
        for (let k = 0; k <= upto; k++) k ? ctx.lineTo(X(k), Y(s[k])) : ctx.moveTo(X(k), Y(s[k]));
        ctx.stroke();
      });

      // median
      ctx.beginPath();
      for (let k = 0; k <= upto; k++) k ? ctx.lineTo(X(k), Y(bands[k][1])) : ctx.moveTo(X(k), Y(bands[k][1]));
      ctx.strokeStyle = COLORS.inkCss;
      ctx.lineWidth = 1.6;
      ctx.stroke();

      // one highlighted path
      const s = paths[hero];
      ctx.beginPath();
      for (let k = 0; k <= upto; k++) k ? ctx.lineTo(X(k), Y(s[k])) : ctx.moveTo(X(k), Y(s[k]));
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath(); ctx.arc(X(upto), Y(s[upto]), 3.5, 0, Math.PI * 2); ctx.fill();

      // terminal distribution (histogram on the right margin)
      if (progress >= 1) {
        const bins = 22, counts = new Array(bins).fill(0);
        paths.forEach((p) => {
          const b = Math.floor(((clamp(p[STEPS], yMin, yMax - 1e-6) - yMin) / (yMax - yMin)) * bins);
          counts[b]++;
        });
        const maxC = Math.max(...counts);
        const bh = ph / bins;
        counts.forEach((c, b) => {
          const len = (c / maxC) * (m.r - 18);
          ctx.fillStyle = "rgba(161,58,40,0.45)";
          ctx.fillRect(m.l + pw + 8, m.t + ph - (b + 1) * bh + 1, len, bh - 2);
        });
      }

      // legend (top-right)
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      if (w < 470) return;
      const ly = 24, lx = w - 176;
      ctx.fillStyle = COLORS.inkCss; ctx.fillRect(lx, ly - 1, 14, 2);
      ctx.fillStyle = COLORS.muted; ctx.fillText("median", lx + 20, ly);
      ctx.fillStyle = "rgba(161,58,40,0.25)"; ctx.fillRect(lx + 82, ly - 5, 14, 10);
      ctx.fillStyle = COLORS.muted; ctx.fillText("P5–P95", lx + 102, ly);
    }

    canvas.addEventListener("pointerdown", () => {
      lastUser = performance.now();
      simulate();
      if (reduceMotion) draw(performance.now());
    });

    simulate();
    new ResizeObserver(() => { size = fit(canvas); draw(performance.now()); }).observe(canvas);

    if (reduceMotion) { draw(performance.now()); return; }

    loopWhileVisible(canvas, (t) => {
      if (t - started > 7500 && t - lastUser > 9000) simulate();
      draw(t);
    });
  }

  /* ---------- boot ---------- */

  const heat = document.getElementById("heat-canvas");
  const gd = document.getElementById("gd-canvas");
  const mc = document.getElementById("mc-canvas");
  if (heat) heatSim(heat);
  if (gd) gradientSim(gd, document.getElementById("gd-readout"));
  if (mc) monteCarloSim(mc, document.getElementById("mc-readout"));

  // reveal-on-scroll
  const revealIO = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add("is-in");
        revealIO.unobserve(e.target);
      }
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
  document.querySelectorAll(".reveal").forEach((el) => revealIO.observe(el));

  const year = document.getElementById("year");
  if (year) year.textContent = new Date().getFullYear();
})();
