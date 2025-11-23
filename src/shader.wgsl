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

struct VertexInput {
    @location(0) position: vec2<f32>
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    var x = ((model.position.x - uniform_input.view_min_x)/(uniform_input.view_max_x - uniform_input.view_min_x))*2 - 1;
    var y = ((model.position.y - uniform_input.view_min_y)/(uniform_input.view_max_y - uniform_input.view_min_y))*2 - 1;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@vertex
fn vs_grid(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    var plot_width = uniform_input.full_max_x - uniform_input.full_min_x;
    var plot_height = uniform_input.full_max_y - uniform_input.full_min_y;
    var x = model.position.x*plot_width + uniform_input.full_min_x;
    var y = model.position.y*plot_height + uniform_input.full_min_y;

    var x2 = ((x - uniform_input.view_min_x)/(uniform_input.view_max_x - uniform_input.view_min_x))*2 - 1;
    var y2 = ((y - uniform_input.view_min_y)/(uniform_input.view_max_y - uniform_input.view_min_y))*2 - 1;
    out.clip_position = vec4<f32>(x2, y2, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
}

@fragment
fn fs_grid(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.1, 0.0, 1.0);
}