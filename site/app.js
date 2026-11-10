/* Distributed Training Demo — animated strategy diagrams + charts of real runs. */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------------------------------------------------------------- diagram
  const N_BLOCKS = 4;
  const LAYERS = ["Embed", "Block 1", "Block 2", "Block 3", "Block 4", "Head"];
  const BATCH = 8; // squares drawn for the global batch (the real runs use 32 rows)

  const PHASES = {
    single: [
      ["Batch", (w) => "One process loads the whole batch of sequences."],
      ["Forward", () => "Activations flow through every layer, Embed to Head, and the loss is computed."],
      ["Backward", () => "Gradients flow back through every layer, Head to Embed."],
      ["Step", () => "AdamW updates every weight. This is the reference run every strategy must reproduce."],
    ],
    ddp: [
      ["Split batch", (w) => `Each of the ${w} ranks takes ${BATCH / w} of the ${BATCH} rows. Every rank already holds a full, identical copy of the model.`],
      ["Forward", () => "Each rank runs the whole model on its own rows — in parallel, no communication."],
      ["Backward", () => "Each rank computes gradients from its rows. They differ, because the rows differ."],
      ["All-reduce", (w) => `Gradients are averaged across all ${w} ranks (a ring all-reduce). DDP overlaps this with backward, bucket by bucket.`],
      ["Step", () => "Same averaged gradients + same starting weights ⇒ every rank takes the identical step. The copies never drift."],
    ],
    fsdp: [
      ["Shard", (w) => `Each rank permanently stores just 1/${w} of every layer — and only that slice's gradients and Adam state.`],
      ["Split batch", (w) => `As in DDP, each rank takes ${BATCH / w} of the ${BATCH} rows.`],
      ["Forward", () => "Just before a layer runs, its slices are all-gathered into a full copy; right after, the copy is freed."],
      ["Backward", () => "Backward gathers each layer again (it was freed), computes the gradient, frees it."],
      ["Reduce-scatter", (w) => `Gradients are summed across ranks and scattered: each rank receives only the averaged gradient for its own 1/${w} slice.`],
      ["Step", () => "Each rank updates only its slice. Memory per rank shrinks ~W×; the price is more communication."],
    ],
    pipeline: [
      ["Cut into stages", (w) => `The ${LAYERS.length} layers are cut into ${w} consecutive stages, one per rank. The batch is cut into microbatches.`],
      ["Forward (fill)", () => "Microbatch activations hop stage to stage with send/recv. Later stages wait while the pipe fills."],
      ["Backward (drain)", () => "Gradients travel back the other way. Early stages wait while it drains. Grey cells are the bubble."],
      ["Step", () => "Gradients summed over microbatches equal the full-batch gradient, so each stage steps its own layers exactly as the single run would."],
    ],
  };

  const st = { s: "ddp", w: 4, m: 4, phase: 0, p: 0, playing: false, auto: false };
  const svg = $("#diagram");

  function stageBounds(nBlocks, world) {
    const sizes = Array.from({ length: world }, (_, i) => Math.floor(nBlocks / world) + (i < nBlocks % world ? 1 : 0));
    const out = [];
    let start = 1;
    sizes.forEach((s, i) => {
      out.push([i === 0 ? 0 : start, start + s + (i === world - 1 ? 1 : 0)]);
      start += s;
    });
    return out;
  }

  const el = (tag, attrs, text) => {
    const a = Object.entries(attrs).map(([k, v]) => `${k}="${v}"`).join(" ");
    return text == null ? `<${tag} ${a}/>` : `<${tag} ${a}>${text}</${tag}>`;
  };
  const txt = (x, y, s, style = "", anchor = "middle") =>
    el("text", { x, y, "text-anchor": anchor, style: `font: 500 12px Inter, sans-serif; fill: var(--text-2); ${style}` }, s);

  function drawColumns() {
    const { s, w, phase, p } = st;
    const world = s === "single" ? 1 : w;
    const name = PHASES[s][phase][0];
    const W = 960, gap = 24, gutter = 70;   // layer names live in the left gutter
    const colW = Math.min(210, (W - gutter - gap * world) / world);
    const x0 = gutter + (W - gutter - (colW * world + gap * (world - 1))) / 2;
    const top = 40, lh = 26, lg = 7;
    const fsdp = s === "fsdp";
    const order = name === "Backward" ? [5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5];
    const active = (name === "Forward" || name === "Backward") ? order[Math.min(5, Math.floor(p * 6))] : -1;
    const doneUpTo = (name === "Forward" || name === "Backward") ? Math.floor(p * 6) : -1;
    const ph = PHASES[s].map((q) => q[0]);
    const after = (n) => ph.indexOf(n) >= 0 && phase > ph.indexOf(n);
    const gradsExist = after("Backward") || (name === "Backward");
    let out = "";

    // Communication: a ring through every rank, drawn under the columns.
    const ringY = top + 6 * (lh + lg) + 22 + 66;
    const comm = { "All-reduce": "ring all-reduce of every gradient", "Reduce-scatter": "reduce-scatter: each rank receives its slice's averaged gradient" }[name]
      || (fsdp && active >= 0 ? `all-gather: rebuild ${LAYERS[active]} from its ${w} slices` : "");
    if (world > 1 && comm) {
      const a = x0 + colW / 2, b = x0 + (world - 1) * (colW + gap) + colW / 2;
      out += el("path", {
        d: `M ${a} ${ringY} L ${b} ${ringY} Q ${b} ${ringY + 22} ${b - 22} ${ringY + 22} L ${a + 22} ${ringY + 22} Q ${a} ${ringY + 22} ${a} ${ringY} Z`,
        style: `fill:none; stroke: var(--accent); stroke-width: 2; stroke-dasharray: 6 6; stroke-dashoffset: ${-p * 120}`,
      });
      for (let r = 0; r < world; r++) {
        out += el("circle", { cx: x0 + r * (colW + gap) + colW / 2, cy: ringY, r: 5, style: "fill: var(--accent)" });
      }
      out += txt(480, ringY + 40, comm, "fill: var(--accent); font-weight: 600;");
    }

    for (let r = 0; r < world; r++) {
      const cx = x0 + r * (colW + gap);
      out += txt(cx + colW / 2, 32, world === 1 ? "One process" : `Rank ${r}`, "fill: var(--text); font-weight: 600;");
      LAYERS.forEach((L, i) => {
        const y = top + i * (lh + lg);
        const isActive = i === active;
        const color = name === "Backward" ? "var(--bwd)" : "var(--fwd)";
        if (fsdp) {
          const segW = colW / w;
          const gathered = isActive;
          for (let k = 0; k < w; k++) {
            const own = k === r;
            const fill = own ? "var(--s3)" : gathered ? "var(--s3)" : "transparent";
            const op = own ? 0.85 : gathered ? 0.35 : 1;
            out += el("rect", {
              x: cx + k * segW + 1, y, width: segW - 2, height: lh, rx: 4,
              style: `fill:${fill}; fill-opacity:${op}; stroke: ${isActive ? color : "var(--border)"}; stroke-width:${isActive ? 2 : 1}; ${own || gathered ? "" : "stroke-dasharray: 3 3;"}`,
            });
          }
        } else {
          out += el("rect", {
            x: cx, y, width: colW, height: lh, rx: 5,
            style: `fill:${isActive ? color : "var(--surface-2)"}; fill-opacity:${isActive ? 0.25 : 1}; stroke:${isActive ? color : "var(--border)"}; stroke-width:${isActive ? 2 : 1}`,
          });
        }
        if (r === 0) out += txt(x0 - 12, y + 17, L, "fill: var(--text);", "end");
        // gradient marker
        const pos = order.indexOf(i);
        const hasGrad = gradsExist && !(name === "Backward" && pos > doneUpTo);
        if (hasGrad && name !== "Step") {
          const avg = after("All-reduce") || name === "All-reduce" || after("Reduce-scatter") || name === "Reduce-scatter";
          const label = avg ? "∇ avg" : "∇";
          if (fsdp) {   // a rank only keeps the gradient for its own slice
            const segW = colW / w;
            out += txt(cx + r * segW + segW / 2, y + 17, label, "fill: #fff; font-weight: 600; font-size: 11px;");
          } else {
            out += txt(cx + colW - 8, y + 17, label, "fill: var(--bwd); font-weight: 600;", "end");
          }
        }
        if (name === "Step") {
          out += el("rect", { x: cx, y, width: colW, height: lh, rx: 5, style: `fill: var(--s3); fill-opacity: ${0.35 * Math.sin(Math.PI * p)}` });
        }
      });

      // batch rows
      const by = top + 6 * (lh + lg) + 22;
      const sq = Math.min(18, (colW - (BATCH - 1) * 4) / BATCH);
      const per = BATCH / world;
      const split = ph.indexOf("Split batch");
      for (let b = 0; b < BATCH; b++) {
        const mine = Math.floor(b / per) === r;
        const shown = world === 1 || phase > split || (phase === split && p > 0) || split < 0;
        const fade = phase === split ? p : 1;
        const op = mine ? 1 : shown ? 1 - fade * 0.85 : 1;
        out += el("rect", {
          x: cx + b * (sq + 4), y: by, width: sq, height: sq, rx: 3,
          style: `fill: ${mine || !shown ? "var(--s1)" : "var(--surface-2)"}; fill-opacity:${mine || !shown ? 1 : 1}; stroke: var(--border); opacity:${world === 1 ? 1 : op}`,
        });
      }
      const rows = world === 1 ? `${BATCH} rows` : `${per} of ${BATCH} rows`;
      out += txt(cx, by + sq + 16, rows, "font-size: 11px;", "start");
      const holds = s === "fsdp" ? `stores 1/${w} of each layer` : "stores the full model";
      out += txt(cx, by + sq + 32, holds, "font-size: 11px; fill: var(--text-3);", "start");
    }
    svg.setAttribute("viewBox", `0 0 960 ${ringY + (world > 1 ? 52 : 0)}`);
    svg.innerHTML = out;
  }

  function drawPipeline() {
    const { w, m, phase, p } = st;
    const bounds = stageBounds(N_BLOCKS, w);
    const W = 960, gap = 28;
    const colW = (W - gap * (w - 1) - 40) / w;
    let out = "";
    // stage boxes with their layers
    bounds.forEach(([lo, hi], r) => {
      const x = 20 + r * (colW + gap);
      out += txt(x + colW / 2, 16, `Rank ${r} · stage ${r}`, "fill: var(--text); font-weight: 600;");
      out += el("rect", { x, y: 24, width: colW, height: 38, rx: 6, style: "fill: var(--surface-2); stroke: var(--border);" });
      out += txt(x + colW / 2, 47, LAYERS.slice(lo, hi).join(" · "), "fill: var(--text); font-size: 11px;");
      if (r < w - 1) {
        out += el("path", { d: `M ${x + colW + 4} 43 L ${x + colW + gap - 4} 43`, style: "stroke: var(--text-3); stroke-width: 1.5; marker-end: url(#arr);" });
      }
    });

    // GPipe schedule: forward of microbatch i on stage s at slot s+i; backward mirrored.
    const tf = m + w - 1, total = 2 * tf;
    const gx = 70, gy = 96, cw = Math.min(56, (W - gx - 10) / total), ch = 28;
    let now = 0;
    if (phase === 0) now = 0;
    else if (phase === 1) now = p * tf;
    else if (phase === 2) now = tf + p * tf;
    else now = total;
    let busy = 0;
    for (let s = 0; s < w; s++) {
      out += txt(gx - 10, gy + s * (ch + 6) + 18, `Stage ${s}`, "font-size: 11px;", "end");
      for (let t = 0; t < total; t++) {
        const x = gx + t * cw, y = gy + s * (ch + 6);
        let label = "", kind = "idle";
        if (t < tf) { const i = t - s; if (i >= 0 && i < m) { label = `F${i}`; kind = "f"; } }
        else { const i = t - tf - (w - 1 - s); if (i >= 0 && i < m) { label = `B${i}`; kind = "b"; } }
        const reached = t < now;
        if (kind !== "idle") busy++;
        const fill = !reached ? "transparent" : kind === "f" ? "var(--fwd)" : kind === "b" ? "var(--bwd)" : "var(--surface-2)";
        out += el("rect", {
          x: x + 1, y, width: cw - 2, height: ch, rx: 4,
          style: `fill:${fill}; fill-opacity:${kind === "idle" ? 1 : 0.85}; stroke: var(--border); ${reached ? "" : "stroke-dasharray: 3 3;"}`,
        });
        if (reached && label) out += txt(x + cw / 2, y + 18, label, "fill: #fff; font-size: 11px; font-weight: 600;");
        if (reached && kind === "idle") out += txt(x + cw / 2, y + 18, "idle", "fill: var(--text-3); font-size: 9px;");
      }
    }
    const gh = w * (ch + 6);
    if (now > 0 && now < total) {
      const px = gx + now * cw;
      out += el("line", { x1: px, y1: gy - 6, x2: px, y2: gy + gh, style: "stroke: var(--text); stroke-width: 1.5;" });
    }
    out += txt(gx, gy + gh + 18, "time →", "font-size: 11px; fill: var(--text-3);", "start");
    const bubble = (w - 1) / (m + w - 1);
    out += txt(W - 10, gy + gh + 18, `bubble = (W−1)/(M+W−1) = ${(bubble * 100).toFixed(0)}% of each stage’s time is idle`, "font-size: 12px; fill: var(--text);", "end");
    const defs = `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-3)"/></marker></defs>`;
    svg.setAttribute("viewBox", `0 0 960 ${gy + gh + 32}`);
    svg.innerHTML = defs + out;
  }

  function draw() {
    const phases = PHASES[st.s];
    $("#phases").innerHTML = phases.map((q, i) =>
      `<span data-i="${i}" class="${i === st.phase ? "on" : i < st.phase ? "done" : ""}">${i + 1}. ${q[0]}</span>`).join("");
    $("#phaseText").textContent = phases[st.phase][1](st.s === "single" ? 1 : st.w);
    $("#mbCtl").style.display = st.s === "pipeline" ? "" : "none";
    $("#worldSel").disabled = st.s === "single";
    if (st.s === "pipeline") drawPipeline(); else drawColumns();
    $("#playBtn").textContent = st.playing ? "❚❚ Pause" : "▶ Play";
  }

  let last = 0;
  function tick(t) {
    const dt = last ? (t - last) / 1000 : 0;
    last = t;
    if (st.playing) {
      st.p += dt / (st.s === "pipeline" && (st.phase === 1 || st.phase === 2) ? 3.2 : 2.2);
      if (st.p >= 1) {
        if (st.auto && st.phase < PHASES[st.s].length - 1) { st.phase++; st.p = 0; }
        else { st.p = 1; st.playing = false; }
      }
      draw();
    }
    requestAnimationFrame(tick);
  }

  function reset() { st.phase = 0; st.p = 0; st.playing = false; draw(); }

  $("#stratSeg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    document.querySelectorAll("#stratSeg button").forEach((x) => x.setAttribute("aria-selected", x === b));
    st.s = b.dataset.s; reset();
  });
  $("#worldSel").addEventListener("change", (e) => { st.w = +e.target.value; reset(); });
  $("#mbRange").addEventListener("input", (e) => { st.m = +e.target.value; $("#mbOut").textContent = st.m; draw(); });
  $("#playBtn").addEventListener("click", () => {
    if (st.playing) { st.playing = false; draw(); return; }
    if (st.phase === PHASES[st.s].length - 1 && st.p >= 1) { st.phase = 0; st.p = 0; }
    st.auto = true; st.playing = true; if (st.p >= 1) { st.phase = Math.min(st.phase + 1, PHASES[st.s].length - 1); st.p = 0; }
  });
  $("#stepBtn").addEventListener("click", () => {
    const n = PHASES[st.s].length;
    if (st.p > 0) st.phase = (st.phase + 1) % n;   // otherwise play the phase we're on
    st.p = 0; st.auto = false; st.playing = true;
  });
  $("#phases").addEventListener("click", (e) => {
    const sp = e.target.closest("span"); if (!sp) return;
    st.phase = +sp.dataset.i; st.p = 0; st.auto = false; st.playing = true;
  });

  st.p = 1; st.phase = 0; draw();
  requestAnimationFrame(tick);

  // ---------------------------------------------------------------- results
  const data = window.RESULTS;
  if (!data || !window.Chart) {
    $("#runNote").textContent = "No results yet — run scripts/benchmark.py to generate site/data/results.js.";
    return;
  }
  const runs = data.runs;
  const NAMES = { single: "Single", ddp: "DDP", fsdp: "FSDP", pipeline: "Pipeline" };
  const SLOT = { single: "--s1", ddp: "--s2", fsdp: "--s3", pipeline: "--s4" };
  const label = (r) => r.world === 1 ? "Single" : `${NAMES[r.config.strategy]} ×${r.world}`;
  const MB = (b) => b / 1e6;
  const busiest = (r) => r.per_rank.reduce((a, x) => {
    const t = x.memory.params + x.memory.grads + x.memory.optimizer;
    return t > a.t ? { t, m: x.memory } : a;
  }, { t: -1 }).m;
  const single = runs.find((r) => r.world === 1);
  const maxDiff = Math.max(...runs.filter((r) => r.world > 1).map((r) => r.max_diff_vs_single));
  const fsdp4 = runs.find((r) => r.config.strategy === "fsdp" && r.world === 4) || runs[runs.length - 1];
  const memTot = (m) => m.params + m.grads + m.optimizer;
  const cfg = single.config;

  $("#runNote").innerHTML = `${cfg.steps} steps of a ${(single.n_params / 1e6).toFixed(2)}M-parameter transformer, global batch ${cfg.global_batch} × ${cfg.seq_len} tokens, on ${single.host === "i386" ? "an Intel Mac CPU" : single.host} with PyTorch ${single.torch}. Measured by <code>scripts/benchmark.py</code>.`;

  $("#tiles").innerHTML = [
    [maxDiff.toExponential(0).replace("e-", "×10⁻").replace(/(\d)$/, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]), "largest loss gap between any distributed run and the single process, over every step"],
    [`${Math.round(100 * memTot(busiest(fsdp4)) / memTot(busiest(single)))}%`, `memory per rank with FSDP across ${fsdp4.world} processes, vs one process`],
    [`${runs.length}`, "real runs: 4 strategies, 1–4 processes, one training loop"],
    [`${single.final_loss.toFixed(2)}`, `final loss (best possible: ${single.entropy_floor.toFixed(2)}, random guessing: ${single.uniform_loss.toFixed(2)})`],
  ].map(([n, l]) => `<div class="tile"><div class="num">${n}</div><div class="lbl">${l}</div></div>`).join("");

  let charts = [];
  function buildCharts() {
    charts.forEach((c) => c.destroy());
    const text2 = css("--text-2"), grid = css("--grid"), text3 = css("--text-3");
    Chart.defaults.font.family = "Inter, system-ui, sans-serif";
    Chart.defaults.color = text2;
    const scales = (yTitle, extra = {}) => ({
      x: { grid: { display: false }, border: { color: grid }, ticks: { color: text2 }, ...extra.x },
      y: { grid: { color: grid }, border: { display: false }, ticks: { color: text2 }, title: { display: !!yTitle, text: yTitle, color: text3 }, ...extra.y },
    });
    const tooltip = { backgroundColor: css("--surface"), titleColor: css("--text"), bodyColor: text2, borderColor: css("--border"), borderWidth: 1, padding: 10, boxPadding: 4 };

    // loss curves
    const steps = single.losses.map((_, i) => i);
    const lossSets = runs.map((r) => ({
      label: label(r),
      data: r.losses,
      borderColor: css(SLOT[r.config.strategy]),
      backgroundColor: css(SLOT[r.config.strategy]),
      borderWidth: r.world === 1 ? 4 : 2,
      borderDash: r.world === 4 ? [5, 4] : [],
      pointRadius: 0, pointHoverRadius: 4, tension: 0,
    }));
    lossSets.push({ label: "Best possible", data: steps.map(() => single.entropy_floor), borderColor: text3, borderWidth: 1.5, borderDash: [2, 4], pointRadius: 0, pointHoverRadius: 0 });
    charts.push(new Chart($("#lossChart"), {
      type: "line",
      data: { labels: steps, datasets: lossSets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false }, tooltip: { ...tooltip, callbacks: { title: (i) => `Step ${i[0].label}`, label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}` } } },
        scales: scales("loss (nats)", { x: { ticks: { color: text2, maxTicksLimit: 8 }, title: { display: true, text: "step", color: text3 } } }),
      },
    }));
    $("#lossLegend").innerHTML = runs.map((r) =>
      `<span><i style="background:${css(SLOT[r.config.strategy])};${r.world === 4 ? "background:repeating-linear-gradient(90deg," + css(SLOT[r.config.strategy]) + " 0 5px,transparent 5px 8px)" : ""}"></i>${label(r)}</span>`).join("") +
      `<span><i style="background:repeating-linear-gradient(90deg,${text3} 0 2px,transparent 2px 5px)"></i>Best possible</span>`;

    // checkpoint demo: first half under one strategy, second half under another
    const rs = data.resume;
    if (rs) {
      const cut = rs.second.start_step;
      const pad = (part) => steps.map((i) => (i >= part.start_step && i < part.start_step + part.losses.length ? part.losses[i - part.start_step] : null));
      const name = (part) => `${NAMES[part.strategy]} ×${part.world}`;
      const cutLine = {
        id: "cut",
        afterDatasetsDraw(c) {
          const x = c.scales.x.getPixelForValue(cut), { top, bottom } = c.chartArea, g = c.ctx;
          g.save(); g.strokeStyle = text3; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke();
          g.setLineDash([]); g.fillStyle = css("--text"); g.font = "500 12px Inter, sans-serif";
          g.fillText(`checkpoint at step ${cut}`, x + 6, top + 14); g.restore();
        },
      };
      charts.push(new Chart($("#resumeChart"), {
        type: "line",
        data: {
          labels: steps,
          datasets: [
            // The reference is a wide translucent band underneath; the stitched run is drawn on top of it.
            { label: "Single, uninterrupted", data: single.losses, borderColor: css("--s1") + "55", borderWidth: 8, borderJoinStyle: "round", pointRadius: 0, pointHoverRadius: 4, order: 2 },
            { label: `${name(rs.first)}, saved`, data: pad(rs.first), borderColor: css(SLOT[rs.first.strategy]), borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, order: 1 },
            { label: `${name(rs.second)}, resumed`, data: pad(rs.second), borderColor: css(SLOT[rs.second.strategy]), borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, order: 1 },
          ],
        },
        plugins: [cutLine],
        options: {
          responsive: true, maintainAspectRatio: false, animation: false, spanGaps: false,
          interaction: { mode: "index", intersect: false },
          plugins: { legend: { display: false }, tooltip: { ...tooltip, filter: (i) => i.parsed.y != null, callbacks: { title: (i) => `Step ${i[0].label}`, label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}` } } },
          scales: scales("loss (nats)", { x: { ticks: { color: text2, maxTicksLimit: 8 }, title: { display: true, text: "step", color: text3 } } }),
        },
      }));
      $("#resumeLegend").innerHTML = [["--s1", "Single, uninterrupted"], [SLOT[rs.first.strategy], `${name(rs.first)}, steps 0–${cut - 1}, then saved`], [SLOT[rs.second.strategy], `${name(rs.second)}, resumed from the file`]]
        .map(([c, n], i) => `<span><i style="background:${css(c)}${i ? "" : "55;height:7px"}"></i>${n}</span>`).join("");
      $("#resumeNote").textContent = `One checkpoint file holds the whole model and Adam state under plain layer names, so any strategy can pick it up. The stitched run never leaves the single-process curve (largest gap ${rs.max_diff_vs_single.toExponential(1)}).`;
      $("#resumeCard").hidden = false;
    }

    // mixed precision: bf16 runs against the fp32 single run
    const pr = data.precision;
    if (pr) {
      const ref = css("--text-3");
      const pSteps = steps.slice(0, pr.steps);
      const sets = [{ label: "fp32 single (reference)", data: single.losses.slice(0, pr.steps), borderColor: ref + "66", borderWidth: 8, borderJoinStyle: "round", pointRadius: 0, pointHoverRadius: 4, order: 2 }]
        .concat(pr.runs.map((r) => ({
          label: `bf16 ${r.world === 1 ? "Single" : `${NAMES[r.strategy]} ×${r.world}`}`,
          data: r.losses, borderColor: css(SLOT[r.strategy]), borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, order: 1,
        })));
      charts.push(new Chart($("#bf16Chart"), {
        type: "line",
        data: { labels: pSteps, datasets: sets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          interaction: { mode: "index", intersect: false },
          plugins: { legend: { display: false }, tooltip: { ...tooltip, callbacks: { title: (i) => `Step ${i[0].label}`, label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}` } } },
          scales: scales("loss (nats)", { x: { ticks: { color: text2, maxTicksLimit: 8 }, title: { display: true, text: "step", color: text3 } } }),
        },
      }));
      $("#bf16Legend").innerHTML = sets.map((d, i) =>
        `<span><i style="background:${d.borderColor}${i ? "" : ";height:7px"}"></i>${d.label}</span>`).join("");
      const worst = Math.max(...pr.runs.slice(1).map((r) => r.max_diff_vs_single));
      $("#bf16Note").textContent = `Matmuls and attention run in bf16; weights, gradients and Adam state stay fp32. bf16 stays within ${pr.max_diff_vs_fp32.toFixed(3)} of fp32, and the strategies stay within ${worst.toFixed(3)} of each other — close, not bit-exact, because bf16 rounding depends on how the work is split. First ${pr.steps} steps; this CPU has no native bf16, so bf16 is slower here — the speed and memory gains show up on GPUs.`;
      $("#bf16Card").hidden = false;
    }

    const labels = runs.map(label);
    const barBase = { borderRadius: 4, borderSkipped: "bottom", maxBarThickness: 44 };

    // memory (stacked; sequential blues for the three parts)
    const parts = [["params", "Parameters", "--m1"], ["grads", "Gradients", "--m2"], ["optimizer", "Adam state", "--m3"]];
    charts.push(new Chart($("#memChart"), {
      type: "bar",
      data: { labels, datasets: parts.map(([k, n, c]) => ({ label: n, data: runs.map((r) => MB(busiest(r)[k])), backgroundColor: css(c), borderColor: css("--surface"), borderWidth: { top: 2 }, ...barBase, borderRadius: 0 })) },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false }, tooltip: { ...tooltip, mode: "index", callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(1)} MB`, footer: (items) => `Total: ${items.reduce((a, i) => a + i.parsed.y, 0).toFixed(1)} MB` } } },
        scales: scales("MB", { x: { stacked: true, ticks: { color: text2, maxRotation: 0, autoSkip: false, font: { size: 10 } } }, y: { stacked: true, grid: { color: grid }, ticks: { color: text2 }, title: { display: true, text: "MB", color: text3 } } }),
      },
    }));
    $("#memLegend").innerHTML = parts.map(([, n, c]) => `<span><i class="sq" style="background:${css(c)}"></i>${n}</span>`).join("");

    const colorFor = runs.map((r) => css(SLOT[r.config.strategy]));
    const bar = (id, values, unit, digits) => charts.push(new Chart($(id), {
      type: "bar",
      data: { labels, datasets: [{ data: values, backgroundColor: colorFor, ...barBase }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false }, tooltip: { ...tooltip, callbacks: { label: (c) => ` ${c.parsed.y.toLocaleString(undefined, { maximumFractionDigits: digits })} ${unit}` } } },
        scales: scales(unit, { x: { ticks: { color: text2, maxRotation: 0, autoSkip: false, font: { size: 10 } } } }),
      },
    }));
    bar("#commChart", runs.map((r) => MB(Math.max(...r.per_rank.map((x) => x.comm_bytes_per_step)))), "MB", 2);
    bar("#tputChart", runs.map((r) => r.tokens_per_s), "tokens/s", 0);
  }

  const best = runs.reduce((a, r) => (r.tokens_per_s > a.tokens_per_s ? r : a));
  $("#tputNote").textContent = `Tokens per second after warm-up. Here every “process” shares one laptop CPU, so this shows overhead, not real scaling — ${label(best)} was fastest at ${(best.tokens_per_s / single.tokens_per_s).toFixed(2)}× the single run. On separate GPUs, data parallelism scales close to linearly.`;

  $("#runTable").innerHTML =
    "<thead><tr><th>Run</th><th>Final loss</th><th>Max |Δ| vs single</th><th>Tokens/s</th><th>Memory / rank (MB)</th><th>Sent / step (MB)</th></tr></thead><tbody>" +
    runs.map((r) => `<tr><td>${label(r)}</td><td>${r.final_loss.toFixed(4)}</td><td>${r.world === 1 ? "—" : r.max_diff_vs_single.toExponential(1)}</td><td>${Math.round(r.tokens_per_s).toLocaleString()}</td><td>${MB(memTot(busiest(r))).toFixed(1)}</td><td>${MB(Math.max(...r.per_rank.map((x) => x.comm_bytes_per_step))).toFixed(2)}</td></tr>`).join("") +
    "</tbody>";

  buildCharts();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", buildCharts);
})();
