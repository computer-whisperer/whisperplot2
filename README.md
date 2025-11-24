Whisperplot — GPU‑accelerated time‑series explorer

Overview

Whisperplot is a high‑quality, GPU‑accelerated plotter for exploring large time‑domain datasets on both native (desktop) and web. Think of it as a faster, smoother, and more flexible alternative to the InfluxDB 2.x Explorer UI, designed to stay responsive while you zoom and pan through millions to billions of points — including datasets of 8 GB or more.

Status: early prototype. Currently loads a CSV and renders a basic line plot with zoom. The repository contains the building blocks (wgpu, winit, glyphon, egui) but architecture and modules are not yet structured for a full product.

Vision and goals

- First‑class, smooth exploration of large time‑series datasets
  - Deep, fluid zoom and pan
  - Multiple series overlaid, easy series selection and toggling
  - Accurate visual aggregation when zoomed out (no misleading plots)
- Multi‑backend data access
  - Local files: CSV, Parquet (Arrow)
  - Network sources: InfluxDB 3 server (SQL / Arrow Flight)
  - Extensible “data adapter” interface for additional sources
- Cross‑platform
  - Native desktop (Linux first; macOS/Windows later) using wgpu
  - Web via WASM with WebGPU (no WebGL fallback for MVP)
- Usability & UX
  - Brushing, range select, tooltips, value inspection, reset zoom
  - Time‑axis aware navigation and labeling
  - Minimal friction to get from raw data to useful visuals
- Open, modular core
  - Clean separation between rendering, data access, and UI
  - Leverage compute shaders for on‑GPU aggregation/decimation where possible

Scale and hardware assumptions

- Target users are likely to have enthusiast‑grade CPU/GPU/RAM. Optimize for that experience first.
- Provide reasonable fallbacks that work on minimal Vulkan features where possible.

Platform targets (current)

- Native: Linux prioritized first.
- Web: WebGPU only for now. Add a WebGL fallback later only if a compelling need appears.

Time semantics

- Assume UTC timestamps for now. Timezone conversions may be added later as needed.

Memory usage visibility and control

- VRAM/RAM usage should be first‑class: visible, logged, and user‑controllable (budgets, eviction policies).
- This may be omitted from the absolute MVP but should follow shortly thereafter.

Non‑goals (initially)

- Complex dashboarding or alerting (focus is the plotting/exploration core)
- Built‑in data transformation language (prefer adapters and query backends)
- 3D plotting or non‑time‑series charts in early milestones

Current snapshot (as of 2025‑11‑23)

- Native app runs; basic CSV load and line rendering with zoom/selection
- Mouse wheel zoom on X, drag/selection zoom on X, quick double‑click reset
- Dependencies include: wgpu, winit, glyphon for text, egui (not yet wired)
- Web scaffolding present (`index.html` and `pkg/` loader)
- Known technical debt:
  - Cargo.toml config uses `wgpu = 27` for native but `wgpu = 0.19` under wasm target with `webgl` feature; these versions are incompatible. We need a unified version strategy and a WebGPU path. Until then, web builds may not function reliably.
  - Single monolithic crate; no clear module boundaries for data adapters, render core, and frontends yet.

Fractal ticks and grid (new)

- Stateless, deterministic tick planning each frame using a 1–2–5 × 10^k ladder.
- Multiple adjacent tick levels (coarse → fine) are evaluated; their visibility fades in/out smoothly as zoom changes using `smoothstep` on pixel spacing thresholds.
- Grid is drawn via instancing using a tiny per‑level uniform buffer; no per‑frame large VB rebuilds.
- Coarser levels draw first, finer levels on top with alpha blending to avoid “popping.”
- This lays the groundwork for axis labels at the selected “label level” (coarsest level with ≥ ~80 px spacing).

Uniform alignment note (native validation fix)

- The per-level grid uniform (`GridLevelUniform`) must be 64 bytes due to WGSL/std140‑like alignment rules on uniform buffers (16‑byte alignment; `vec3<f32>` rounds up to 16 bytes).
- A mismatch between Rust struct size and WGSL struct size will trigger wgpu validation errors like: “Buffer is bound with size 48 where the shader expects 64 in group[0] binding[1]”.
- Fix implemented:
  - Rust: `#[repr(C)]` `GridLevelUniform` explicitly padded to 64 bytes; compile‑time size assert added.
  - WGSL: trailing padding upgraded to `vec4<f32>`.
  - Bind group layout enforces `min_binding_size = 64` for that binding to catch regressions early.

Architecture (planned)

Target decomposition (suggested workspace crates):

- whisperplot-core
  - Core types: time/value series, metadata, view transforms (world <-> screen)
  - Traits for data adapters and tile/LOD providers
  - Caching and query orchestration
- whisperplot-render
  - wgpu pipelines (lines/points, axes/grid, text via glyphon)
  - On‑GPU aggregation/decimation (compute) and dynamic vertex buffers
  - Multi‑resolution tiling for fast zoom/pan
- whisperplot-adapters
  - csv: out‑of‑core chunked reader, schema inference
  - parquet: Arrow/Parquet reader, zero‑copy where possible
  - influxdb3: client queries that return Arrow Tables/RecordBatches
  - (extensible via a `DataSource` trait)
- whisperplot-ui
  - egui or custom UI for series selection, query, appearance, tooltips
  - Interaction: brush zoom, pan, reset, cursor/value readout
- whisperplot-app-native
  - Desktop runner (winit event loop, file dialogs)
- whisperplot-app-web
  - WASM runner (WebGPU ideally; fallback TBD)

Key design principles

- Don’t draw what you can’t see:
  - Use multi‑resolution precomputation and/or streaming GPU aggregation to bound drawn primitives to O(pixels)
- Correctness over convenience:
  - Aggregation/decimation must preserve extrema to avoid misleading plots; consider M4, min/max bins, or LTTB variants
- Asynchronous data flow:
  - Data adapters produce chunks/tiles; the view consumes the best resolution available and refines progressively
- Stateless rendering passes:
  - Frame builds from immutable buffers and uniform state; dynamic updates use ring buffers and staging queues

Architecture‑First LLM Guidance

- Architecture is the top priority. If a requested feature does not fit cleanly or would regress performance/maintainability, stop and request a design decision.
- Triggers to escalate re‑architecture:
  - Memory usage cannot be bounded or observed within configured budgets
  - Tiling/LOD strategy doesn’t map to new data shapes or scales
  - Excessive CPU↔GPU copies or synchronization required for a change
  - Adapter interfaces leak backend details into render/UI layers
  - Web and native diverge in ways that block unification
- Expect to re‑engineer large‑scale structures when necessary to enable performant and maintainable expansion.

Performance strategy

- Level‑of‑detail (LOD): build multi‑resolution representations (offline for files; on‑the‑fly for streams)
- GPU aggregation: compute shaders to bin samples into pixel columns with min/max/mean counters
- Tiling: fixed‑size tiles over time axis to localize memory and cache behavior
- Batching: pack many series into shared buffers; minimize pipeline switches
- Text/axis rendering: cache glyphs via glyphon; reuse vertex data while panning
- No fixed SLOs at this stage. Start with a solid architecture and measure. If interactions feel slow, profile and optimize guided by data.

Data adapters (planned)

- CSV
  - Streamed, chunked reading; optional type/schema hints
  - Optional background LOD build stored as sidecar (.wpidx)
- Parquet (Arrow)
  - Columnar reads via `parquet`/`arrow` crates; predicate pushdown where possible
  - Zero‑copy to GPU staging buffers when aligned
- InfluxDB 3
  - Query via SQL and/or Arrow Flight; pagination to chunks/tiles
  - Server‑side downsampling when appropriate; client builds LOD caches
  - Note: interface design deferred until after MVP; treat as a future expansion.

User interactions (baseline)

- Wheel zoom (cursor‑centered)
- Brush to zoom a selected X range; double‑click to reset view
- Pan (mouse drag or modified key)
- Cursor crosshair + value readout; hover tooltips on series
- Series picker and legend with visibility toggles

Developer quickstart

Prerequisites

- Rust toolchain (stable; edition 2024)
- A working Vulkan/Metal/DirectX/ANGLE stack (for wgpu on your OS)
- For web builds: wasm32 target and a static file server

Native (desktop)

```
rustup update
cargo run
```

Web (WASM)

Note: current Cargo.toml specifies `wgpu = 27` for native and `wgpu = 0.19` for wasm with `webgl`. This mismatch likely prevents successful web builds. The medium‑term plan is to unify on a single recent `wgpu` and target WebGPU in browsers that support it. For the web, WebGPU is the target; no WebGL fallback is planned for MVP. Until then, if you experiment:

```
rustup target add wasm32-unknown-unknown
# Option A: wasm-pack
wasm-pack build --target web
# serve index.html from project root (any static server)
# Option B: trunk (if a Trunk.toml is added later)
```

Open `index.html` via a static server (not file://) so the `pkg/` JS can load.

LLM‑aided development guide

- When opening a task, state:
  - Background/motivation
  - Definition of done
  - Acceptance checks (commands to run; expected behavior)
  - Constraints (APIs, performance, compatibility)
- Keep patches minimal and focused; preserve style and module patterns
- Require benchmarks or a measurement plan for any performance‑affecting change
- PR checklist (copy into description):
  - [ ] Problem statement and approach documented
  - [ ] Builds on native; web build status noted
  - [ ] clippy: `cargo clippy -- -D warnings` clean
  - [ ] fmt: `cargo fmt --all`
  - [ ] (If perf‑related) numbers before/after or a repeatable bench harness
  - [ ] Tests added/updated when applicable
- Prompting tips for LLMs
  - Provide file paths and short, literal symbol names when asking for edits
  - Ask for targeted diffs; avoid sweeping refactors unless planned
  - Prefer introducing interfaces/traits first, then swap implementations

MVP (focus only on the basics before expanding)

- Rendering core
  - Lines + axes + text (Glyphon)
  - Basic frame timing HUD/logging
- Interactions
  - Wheel zoom (cursor‑centered), brush zoom, pan
  - Cursor readout, tooltips, legend (using Glyphon)
- Data
  - CSV and Parquet loaders (chunked), UTC timestamps
  - Adapter trait skeleton to keep the system extensible
- LOD scaffolding
  - Tile index format and background builder for files
  - On‑GPU binning path (compute) feeding dynamic vertex buffers
- Web build hygiene
  - Plan to unify on a single `wgpu` that supports WebGPU for native + web
  - No WebGL fallback in MVP
- Near‑term (post‑MVP)
  - VRAM/RAM usage visibility and user‑controlled budgets
  - InfluxDB interface design and adapter

Known limitations and risks

- WebGL/WebGPU portability: feature parity with native may be constrained
- Very large datasets require careful memory management and LOD caching
- Cross‑platform font/text differences (glyphon) need validation

License

- Dual‑licensed under MIT OR Apache‑2.0.
- See `LICENSE-MIT` and `LICENSE-APACHE` for details.

Acknowledgments

- Built on wgpu, winit, glyphon, egui, Arrow/Parquet ecosystems

How to contribute

- Open an issue describing the problem and the definition of done
- Keep PRs small and focused; include performance notes if relevant
- Follow the PR checklist in the LLM guide above
