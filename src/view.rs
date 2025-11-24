// Stateless tick planning for axes/grid; designed for per-frame recomputation.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Axis { X, Y }

#[derive(Copy, Clone, Debug)]
pub struct TickLevel {
    pub axis: Axis,
    pub first: f32,
    pub step: f32,
    pub count: u32,
    pub alpha: f32,
    pub label_stride: u32,
}

#[derive(Default, Debug)]
pub struct GridPlan {
    pub x: Vec<TickLevel>,
    pub y: Vec<TickLevel>,
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if x <= edge0 { return 0.0; }
    if x >= edge1 { return 1.0; }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn nice_step(target: f32) -> f32 {
    // Snap target to 1-2-5 * 10^k ladder
    if !target.is_finite() || target <= 0.0 { return 1.0; }
    let k = target.abs().log10().floor();
    let pow10 = 10.0_f32.powf(k);
    let base = target / pow10;
    let m = if base < 1.5 { 1.0 } else if base < 3.5 { 2.0 } else if base < 7.5 { 5.0 } else { 10.0 };
    m * pow10
}

fn levels_for_axis(axis: Axis, view_min: f32, view_max: f32, pixels: u32) -> Vec<TickLevel> {
    // Robust to reversed ranges: compute on sorted [a,b]
    let a = view_min.min(view_max);
    let b = view_min.max(view_max);
    let span = (b - a).max(f32::EPSILON);
    let px_per_unit = pixels.max(1) as f32 / span;

    // Aim for ~100px label spacing for a base level
    let target_px = 100.0_f32;
    let target_units = (span / (pixels.max(1) as f32 / target_px)).max(f32::EPSILON);
    let base = nice_step(target_units);

    // Build a small ladder of surrounding steps (coarser to finer)
    let steps = [base * 5.0, base * 2.0, base, base / 2.0, base / 5.0];

    let mut out_full: Vec<TickLevel> = Vec::with_capacity(steps.len());
    let mut chosen_label_set = false;
    let max_instances: u32 = 4096;


    for &s in steps.iter() {
        if s <= 0.0 || !s.is_finite() { continue; }
        let spacing_px = s * px_per_unit;

        // Alpha bands: coarse fades in around 64px, fine around 10-16px
        // Compose a smooth visibility curve by combining several bands and clamping
        let mut alpha = 0.0_f32;
        // Coarser visibility (tick/label level)
        alpha = alpha.max(smoothstep(48.0, 64.0, spacing_px));
        // Minor lines visibility
        alpha = alpha.max(smoothstep(8.0, 16.0, spacing_px));
        // Very fine lines visibility
        alpha = alpha.max(smoothstep(4.0, 10.0, spacing_px));

        if alpha < 0.02 { continue; }

        // Compute first/count deterministically over [a,b]
        let start_index = (a / s).ceil();
        let first = start_index * s;
        let count_f = ((b - first) / s).floor() + 1.0;
        let mut count = if count_f.is_finite() && count_f > 0.0 { count_f as u32 } else { 0 };
        if count > max_instances { count = max_instances; }
        if count == 0 { continue; }

        // Choose label level (coarsest with sufficient spacing)
        let label_stride = if !chosen_label_set && spacing_px >= 80.0 {
            chosen_label_set = true;
            1
        } else { u32::MAX }; // no labels at this level for now

        out_full.push(TickLevel { axis, first, step: s, count, alpha, label_stride });
    }

    // Limit density: keep at most two levels per axis — a label level and one finer minor level
    if out_full.is_empty() { return out_full; }

    // Prefer the coarsest level with spacing >= 80 px for labels
    let mut label_idx: Option<usize> = None;
    for (i, lvl) in out_full.iter().enumerate() {
        let spacing_px = lvl.step * px_per_unit;
        if spacing_px >= 80.0 {
            label_idx = Some(i);
            break;
        }
    }

    let mut filtered: Vec<TickLevel> = Vec::with_capacity(2);
    match label_idx {
        Some(i) => filtered.push(out_full[i]),
        None => filtered.push(out_full[0]), // fallback to coarsest available
    }

    // Choose one finer level with reasonable spacing (8..32 px) for minor ticks
    let start_j = label_idx.map(|i| i + 1).unwrap_or(1);
    for j in start_j..out_full.len() {
        let lvl = out_full[j];
        let spacing_px = lvl.step * px_per_unit;
        if spacing_px >= 8.0 && spacing_px <= 32.0 {
            filtered.push(lvl);
            break;
        }
    }

    filtered
}

pub fn plan_grid(view_min_x: f32, view_max_x: f32,
                 view_min_y: f32, view_max_y: f32,
                 width_px: u32, height_px: u32) -> GridPlan {
    let x = levels_for_axis(Axis::X, view_min_x, view_max_x, width_px);
    let y = levels_for_axis(Axis::Y, view_min_y, view_max_y, height_px);

    GridPlan { x, y }
}
