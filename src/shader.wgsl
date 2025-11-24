struct UniformInput {
    full_min_x: f32,
    full_max_x: f32,
    full_min_y: f32,
    full_max_y: f32,
    view_min_x: f32,
    view_max_x: f32,
    view_min_y: f32,
    view_max_y: f32
}
@group(0) @binding(0)
var<uniform> uniform_input: UniformInput;

struct GridLevelUniform {
    x_first: f32,
    x_step: f32,
    x_count: u32,
    _pad0: u32,
    y_first: f32,
    y_step: f32,
    y_count: u32,
    orient: u32, // 0 = vertical lines, 1 = horizontal lines
    alpha: f32,
    _pad1: vec4<f32>,
}
@group(0) @binding(1)
var<uniform> u_grid: GridLevelUniform;

struct VertexInput {
    @location(0) position: vec2<f32>
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>
};

fn x_to_ndc(x: f32) -> f32 {
    return ((x - uniform_input.view_min_x) / (uniform_input.view_max_x - uniform_input.view_min_x)) * 2.0 - 1.0;
}

fn y_to_ndc(y: f32) -> f32 {
    return ((y - uniform_input.view_min_y) / (uniform_input.view_max_y - uniform_input.view_min_y)) * 2.0 - 1.0;
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let x = x_to_ndc(model.position.x);
    let y = y_to_ndc(model.position.y);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@vertex
fn vs_grid(
    model: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let idx = f32(instance_index);
    if (u_grid.orient == 0u) {
        // Vertical lines: compute x from instance; y spans full NDC via model.position.x (-1..+1)
        let x_tick = u_grid.x_first + idx * u_grid.x_step;
        let x_ndc = x_to_ndc(x_tick);
        let y_ndc = model.position.x; // -1 to +1 line segment
        out.clip_position = vec4<f32>(x_ndc, y_ndc, 0.0, 1.0);
    } else {
        // Horizontal lines: compute y from instance; x spans full NDC via model.position.x (-1..+1)
        let y_tick = u_grid.y_first + idx * u_grid.y_step;
        let y_ndc = y_to_ndc(y_tick);
        let x_ndc = model.position.x; // -1 to +1 line segment
        out.clip_position = vec4<f32>(x_ndc, y_ndc, 0.0, 1.0);
    }
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
}

@fragment
fn fs_grid(in: VertexOutput) -> @location(0) vec4<f32> {
    // Color-code by orientation for debug visibility: vertical=red, horizontal=green
    if (u_grid.orient == 0u) {
        return vec4<f32>(0.9, 0.2, 0.2, u_grid.alpha);
    } else {
        return vec4<f32>(0.0, 0.7, 0.0, u_grid.alpha);
    }
}