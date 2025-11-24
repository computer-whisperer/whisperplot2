use std::fs::File;
use std::path::Path;
// Use a wasm-friendly Instant on the web, std Instant elsewhere
#[cfg(target_arch = "wasm32")]
use instant::Instant as AppInstant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant as AppInstant;
use std::sync::Arc;
#[cfg(target_arch="wasm32")]
use std::io::Cursor;
use bytemuck::{Pod, Zeroable};
use glyphon::{FontSystem, SwashCache, TextAtlas, TextRenderer, Metrics, Attrs, Family, Shaping, Resolution, TextArea, TextBounds, Cache};
use wgpu::{Backends, MultisampleState, RenderPass};
use wgpu::util::DeviceExt;
#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ControlFlow, EventLoop, ActiveEventLoop},
    window::{WindowId},
};
use winit::dpi::PhysicalPosition;

mod dataset;
use crate::dataset::{Dataset, Series};
mod view;
use crate::view::plan_grid;

// On wasm we must not block the main thread or use thread-based primitives.
// We'll initialize GPU state asynchronously and stash it in a thread-local
// until the winit event loop can pick it up.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static GLOBAL_STATE: std::cell::RefCell<Option<State>> = const { std::cell::RefCell::new(None) };
}
use winit::event::MouseScrollDelta::LineDelta;
use winit::window::Window;

// Ensure we always have at least one usable font (especially on wasm where
// system fonts are unavailable). We bundle Noto Sans Regular and load system
// fonts on native as well for broader coverage.
fn init_font_system() -> FontSystem {
    let mut fs = FontSystem::new();

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Try to load OS fonts on native; ignore if unavailable.
        fs.db_mut().load_system_fonts();
    }

    // Load bundled fallback (always), so wasm has a font and native has a
    // deterministic fallback if system fonts are missing.
    let data = include_bytes!("../assets/fonts/static/NotoSans-Regular.ttf").to_vec();
    let _ = fs.db_mut().load_font_data(data);

    fs
}

#[derive(Default, Copy, Clone)]
struct PlotMetadata {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    num_points: u32
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
struct PlotViewFrame {
    full_min_x: f32,
    full_max_x: f32,
    full_min_y: f32,
    full_max_y: f32,
    view_min_x: f32,
    view_max_x: f32,
    view_min_y: f32,
    view_max_y: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GridLevelUniform {
    x_first: f32,
    x_step: f32,
    x_count: u32,
    _pad0: u32,
    y_first: f32,
    y_step: f32,
    y_count: u32,
    orient: u32,
    alpha: f32,
    // Pad to 64 bytes total to match WGSL/std140 expectations.
    _pad1: [f32; 7],
}

// Compile-time size check to prevent uniform size regressions.
const _: [(); 64] = [(); std::mem::size_of::<GridLevelUniform>()];

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    plot_render_pipeline: wgpu::RenderPipeline,
    grid_render_pipeline: wgpu::RenderPipeline,
    plot_vertex_buffer: wgpu::Buffer,
    grid_vertex_buffer: wgpu::Buffer, // unit line vertices for instanced grid
    grid_vertex_count: usize,         // should be 2
    uniform_buffer: wgpu::Buffer,
    uniform_buffer_bind_group: wgpu::BindGroup,
    grid_level_uniform_buffer: wgpu::Buffer,
    latest_plot_metadata: PlotMetadata,
    plot_view_frame_value: PlotViewFrame,
    text_renderer: TextRenderer,
    text_buffer: glyphon::Buffer,
    font_system: FontSystem,
    text_atlas: TextAtlas,
    text_cache: SwashCache,
    glyphon_viewport: glyphon::Viewport,
    // Input state moved from run()
    last_cursor_position: PhysicalPosition<f64>,
    last_left_click_screen_position: PhysicalPosition<f64>,
    current_left_mouse_state: ElementState,
    left_mouse_last_transition_time: AppInstant,
    left_mouse_last_quick_click_time: AppInstant,
    current_right_mouse_state: ElementState,
    right_mouse_last_transition_time: AppInstant,
    window: Arc<Window>,
    dataset: Dataset,
    last_debug_log: AppInstant,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Arc<Window>) -> State {
        // On web the canvas can briefly report a 0×0 size during initial layout.
        // Guard against configuring the surface with zero dimensions.
        let initial_size = window.inner_size();
        let size = if initial_size.width == 0 || initial_size.height == 0 {
            winit::dpi::PhysicalSize::new(1, 1)
        } else {
            initial_size
        };

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                // On WebGPU (wasm32), avoid constraining to WebGL2 limits; use downlevel defaults for portability.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
                .. Default::default()
            },
        ).await.unwrap();

        const PLOT_VERTICES: &[Vertex] = &[
            Vertex { position: [0.0, 0.5]},
            Vertex { position: [-0.5, -0.5]},
            Vertex { position: [0.5, -0.5]},
        ];

        let plot_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Plot Vertex Buffer"),
                contents: bytemuck::cast_slice(PLOT_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        // Grid unit vertices: two vertices spanning -1..+1 along X, Y=0.
        // Shader uses model.position.x as the varying coordinate for both orientations:
        //   - vertical: y_ndc = model.position.x, x_ndc comes from instance x_tick
        //   - horizontal: x_ndc = model.position.x, y_ndc comes from instance y_tick
        const GRID_UNIT_VERTICES: &[Vertex] = &[
            Vertex { position: [-1.0,  0.0] },
            Vertex { position: [ 1.0,  0.0] },
        ];

        let grid_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Grid Unit Vertex Buffer"),
                contents: bytemuck::cast_slice(GRID_UNIT_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[PlotViewFrame {
                    full_min_x: 0.0,
                    full_max_x: 1.0,
                    full_min_y: 0.0,
                    full_max_y: 1.0,
                    view_min_x: 0.0,
                    view_max_x: 1.0,
                    view_min_y: 0.0,
                    view_max_y: 1.0,
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let grid_level_uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Grid Level Uniform Buffer"),
                contents: bytemuck::cast_slice(&[GridLevelUniform{
                    x_first: 0.0, x_step: 1.0, x_count: 0, _pad0: 0,
                    y_first: 0.0, y_step: 1.0, y_count: 0, orient: 0,
                    alpha: 1.0, _pad1: [0.0;7],
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let uniform_buffer_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<GridLevelUniform>() as u64),
                    },
                    count: None,
                },
            ],
            label: Some("uniform_buffer_bind_group_layout"),
        });

        let uniform_buffer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_buffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_level_uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("uniform_buffer_bind_group"),
        });

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2
        };
        surface.configure(&device, &config);

        let mut font_system = init_font_system();
        let mut text_cache = SwashCache::new();
        let mut cache = Cache::new(&device);
        let mut text_atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        let mut text_renderer = TextRenderer::new(&mut text_atlas, &device, MultisampleState::default(), None);
        let mut text_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(30.0, 42.0));

        // Ensure text layout never sees a zero-sized viewport.
        text_buffer.set_size(&mut font_system, Some(size.width.max(1) as f32), Some(size.height.max(1) as f32));
        text_buffer.set_text(
            &mut font_system,
            "Hello world! 👋\nThis is rendered with 🦅 glyphon 🦁\nThe text below should be partially clipped.\na b c d e f g h i j k l m n o p q r s t u v w x y z",
            &Attrs::new().family(Family::Name("Noto Sans")),
            Shaping::Advanced,
        );

        text_buffer.shape_until_scroll(&mut font_system, false);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into())
        });

        let render_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &uniform_buffer_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let plot_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            cache: None,
            label: Some("Plot Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                compilation_options: Default::default(),
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            }
                        ],
                    },
                ]
            },
            fragment: Some(wgpu::FragmentState {
                compilation_options: Default::default(),
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })]
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false
            },
            multiview: None,
        });

        let grid_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            cache: None,
            label: Some("Grid Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                compilation_options: Default::default(),
                module: &shader,
                entry_point: Some("vs_grid"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            }
                        ],
                    },
                ]
            },
            fragment: Some(wgpu::FragmentState {
                compilation_options: Default::default(),
                module: &shader,
                entry_point: Some("fs_grid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })]
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false
            },
            multiview: None,
        });

        State {
            glyphon_viewport: glyphon::Viewport::new(&device, &cache),
            window,
            surface,
            device,
            queue,
            config,
            size,
            plot_render_pipeline,
            grid_render_pipeline,
            plot_vertex_buffer,
            grid_vertex_buffer,
            grid_vertex_count: 2,
            uniform_buffer,
            uniform_buffer_bind_group,
            grid_level_uniform_buffer,
            latest_plot_metadata: PlotMetadata {.. Default::default() },
            plot_view_frame_value: PlotViewFrame {.. Default::default() },
            text_renderer,
            text_buffer,
            font_system,
            text_atlas,
            text_cache,
            last_cursor_position: Default::default(),
            last_left_click_screen_position: Default::default(),
            current_left_mouse_state: ElementState::Released,
            left_mouse_last_transition_time: AppInstant::now(),
            left_mouse_last_quick_click_time: AppInstant::now(),
            current_right_mouse_state: ElementState::Released,
            right_mouse_last_transition_time: AppInstant::now(),
            dataset: Dataset::new(),
            last_debug_log: AppInstant::now(),
        }
    }

    fn load_data(&mut self) {
        // CSV reader source differs between native and wasm32 (no filesystem on the web)
        #[cfg(not(target_arch = "wasm32"))]
        let mut rdr = {
            let path = Path::new("data.csv");
            let file = File::open(path).unwrap();
            csv::ReaderBuilder::new().has_headers(true).delimiter(b',').from_reader(file)
        };

        #[cfg(target_arch = "wasm32")]
        let mut rdr = {
            // Embed the CSV at compile time for the web build to avoid fetch/setup complexity initially.
            let bytes: &[u8] = include_bytes!("../data.csv");
            let cursor = Cursor::new(bytes);
            csv::ReaderBuilder::new().has_headers(true).delimiter(b',').from_reader(cursor)
        };
        // Read first column as Y, implicit X = index
        let mut y_vals: Vec<f32> = Vec::new();
        for result in rdr.records() {
            let v = result.unwrap();
            let y = v.get(0).unwrap().trim().parse::<f32>().unwrap();
            y_vals.push(y);
        }

        // Build immutable series and insert into dataset
        let series = Series::from_implicit_y("series0", y_vals);
        let len = series.y_f32.len();
        self.dataset.insert(series.name.clone(), series);

        // Prepare GPU vertex buffer with positions in data space
        if let Some(s) = self.dataset.first_series() {
            let mut raw_points: Vec<Vertex> = Vec::with_capacity(len);
            for (i, y) in s.y_f32.iter().enumerate() {
                raw_points.push(Vertex { position: [i as f32, *y] });
            }
            self.plot_vertex_buffer = self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(raw_points.as_slice()),
                    usage: wgpu::BufferUsages::VERTEX,
                }
            );

            // Update metadata and view
            let plot_metadata = PlotMetadata{
                min_x: s.min_x as f32,
                max_x: s.max_x as f32,
                min_y: s.min_y,
                max_y: s.max_y,
                num_points: len as u32,
            };
            self.latest_plot_metadata = plot_metadata;
            self.plot_view_frame_value = PlotViewFrame {
                full_min_x: self.dataset.full_min_x as f32,
                full_max_x: self.dataset.full_max_x as f32,
                full_min_y: self.dataset.full_min_y,
                full_max_y: self.dataset.full_max_y,
                view_min_x: self.dataset.full_min_x as f32,
                view_max_x: self.dataset.full_max_x as f32,
                view_min_y: self.dataset.full_min_y,
                view_max_y: self.dataset.full_max_y,
            };
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            // Keep text layout in sync with window size
            self.text_buffer.set_size(&mut self.font_system, Some(self.config.width as f32), Some(self.config.height as f32));
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseWheel{delta, ..} => {
                if let LineDelta(_x, y) = delta {
                    let middle = (self.plot_view_frame_value.view_min_x + self.plot_view_frame_value.view_max_x) / 2.0;
                    let span = (self.plot_view_frame_value.view_max_x - self.plot_view_frame_value.view_min_x) / 2.0;
                    let new_span = span * (1.0 - (*y/20f32));
                    self.plot_view_frame_value.view_min_x = middle - new_span;
                    self.plot_view_frame_value.view_max_x = middle + new_span;
                    return true;
                }
            }
            WindowEvent::MouseInput{state: mouse_state, button, ..} => {
                match button {
                    MouseButton::Left => {
                        if *mouse_state == ElementState::Released {
                            if self.left_mouse_last_transition_time.elapsed().as_millis() < 200 {
                                if self.left_mouse_last_quick_click_time.elapsed().as_millis() < 500 {
                                    self.plot_view_frame_value.view_min_x = self.latest_plot_metadata.min_x;
                                    self.plot_view_frame_value.view_max_x = self.latest_plot_metadata.max_x;
                                    self.plot_view_frame_value.view_min_y = self.latest_plot_metadata.min_y;
                                    self.plot_view_frame_value.view_max_y = self.latest_plot_metadata.max_y;
                                }
                                self.left_mouse_last_quick_click_time = AppInstant::now();
                            }
                            if self.left_mouse_last_transition_time.elapsed().as_millis() > 200 {
                                // Zoom X to selected region
                                let start_x_fraction = self.last_left_click_screen_position.x / self.size.width as f64;
                                let end_x_fraction = self.last_cursor_position.x / self.size.width as f64;
                                let x_span = self.plot_view_frame_value.view_max_x - self.plot_view_frame_value.view_min_x;
                                let start_x = start_x_fraction * x_span as f64 + self.plot_view_frame_value.view_min_x as f64;
                                let end_x = end_x_fraction * x_span as f64 + self.plot_view_frame_value.view_min_x as f64;
                                let (lo, hi) = if start_x <= end_x { (start_x as f32, end_x as f32) } else { (end_x as f32, start_x as f32) };
                                self.plot_view_frame_value.view_min_x = lo;
                                self.plot_view_frame_value.view_max_x = hi;
                            }
                        }
                        if *mouse_state == ElementState::Pressed {
                            self.last_left_click_screen_position = self.last_cursor_position;
                        }
                        self.current_left_mouse_state = *mouse_state;
                        self.left_mouse_last_transition_time = AppInstant::now();
                        return true;
                    }
                    MouseButton::Right => {
                        self.current_right_mouse_state = *mouse_state;
                        self.right_mouse_last_transition_time = AppInstant::now();
                        return true;
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved {position,..} => {
                self.last_cursor_position = *position;
                return true;
            }
            _ => {}
        }
        false
    }

    fn update(&mut self) {
        //todo!()
        self.window.request_redraw();
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // If the surface has zero size (can happen transiently on web), skip rendering this frame.
        if self.config.width == 0 || self.config.height == 0 {
            return Ok(());
        }
        // Update hot uniforms BEFORE encoding draw calls
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.plot_view_frame_value]));

        self.glyphon_viewport.update(
            &self.queue,
            Resolution{
                width: self.config.width.max(1),
                height: self.config.height.max(1),
            }
        );
        self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.text_atlas,
            &self.glyphon_viewport,
            [TextArea {
                custom_glyphs: &[],
                buffer: &self.text_buffer,
                left: 10.0,
                top: 10.0,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: 600,
                    bottom: 160
                },
                default_color: glyphon::Color::rgb(255, 255, 255),
            }],
            &mut self.text_cache
        ).unwrap();

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    depth_slice: None,
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01,
                            g: 0.01,
                            b: 0.01,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Plan fractal ticks for current view and size (stateless)
            let plan = plan_grid(
                self.plot_view_frame_value.view_min_x,
                self.plot_view_frame_value.view_max_x,
                self.plot_view_frame_value.view_min_y,
                self.plot_view_frame_value.view_max_y,
                self.config.width,
                self.config.height,
            );

            // Periodic debug dump of grid plan to diagnose missing verticals / dense horizontals
            if self.last_debug_log.elapsed().as_millis() > 500 {
                let vx = plan.x.len();
                let vy = plan.y.len();
                eprintln!(
                    "[grid debug] view_x:[{:.3},{:.3}] view_y:[{:.3},{:.3}] size:{}x{} | levels X:{} Y:{}",
                    self.plot_view_frame_value.view_min_x,
                    self.plot_view_frame_value.view_max_x,
                    self.plot_view_frame_value.view_min_y,
                    self.plot_view_frame_value.view_max_y,
                    self.config.width,
                    self.config.height,
                    vx,
                    vy
                );
                if let Some(l) = plan.x.get(0) {
                    eprintln!("[grid debug] X0 first:{:.3} step:{:.3} count:{} alpha:{:.2}", l.first, l.step, l.count, l.alpha);
                }
                if let Some(l) = plan.x.get(1) {
                    eprintln!("[grid debug] X1 first:{:.3} step:{:.3} count:{} alpha:{:.2}", l.first, l.step, l.count, l.alpha);
                }
                if let Some(l) = plan.y.get(0) {
                    eprintln!("[grid debug] Y0 first:{:.3} step:{:.3} count:{} alpha:{:.2}", l.first, l.step, l.count, l.alpha);
                }
                if let Some(l) = plan.y.get(1) {
                    eprintln!("[grid debug] Y1 first:{:.3} step:{:.3} count:{} alpha:{:.2}", l.first, l.step, l.count, l.alpha);
                }
                self.last_debug_log = AppInstant::now();
            }

            render_pass.set_pipeline(&self.grid_render_pipeline);
            render_pass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.uniform_buffer_bind_group, &[]);

            // Draw X-axis vertical lines (coarse to fine as given)
            for lvl in plan.x.iter() {
                let u = GridLevelUniform{
                    x_first: lvl.first, x_step: lvl.step, x_count: lvl.count, _pad0: 0,
                    y_first: 0.0, y_step: 1.0, y_count: 0, orient: 0,
                    alpha: lvl.alpha, _pad1: [0.0;7],
                };
                self.queue.write_buffer(&self.grid_level_uniform_buffer, 0, bytemuck::bytes_of(&u));
                render_pass.draw(0..self.grid_vertex_count as u32, 0..lvl.count);
            }

            // Draw Y-axis horizontal lines
            for lvl in plan.y.iter() {
                let u = GridLevelUniform{
                    x_first: 0.0, x_step: 1.0, x_count: 0, _pad0: 0,
                    y_first: lvl.first, y_step: lvl.step, y_count: lvl.count, orient: 1,
                    alpha: lvl.alpha, _pad1: [0.0;7],
                };
                self.queue.write_buffer(&self.grid_level_uniform_buffer, 0, bytemuck::bytes_of(&u));
                render_pass.draw(0..self.grid_vertex_count as u32, 0..lvl.count);
            }

            // Optional: debug sanity vertical line at center x
            const DEBUG_FORCE_CENTER_VERTICAL: bool = false;
            if DEBUG_FORCE_CENTER_VERTICAL {
                let center_x = (self.plot_view_frame_value.view_min_x + self.plot_view_frame_value.view_max_x) * 0.5;
                let u = GridLevelUniform{
                    x_first: center_x, x_step: 1.0, x_count: 1, _pad0: 0,
                    y_first: 0.0, y_step: 1.0, y_count: 0, orient: 0,
                    alpha: 1.0, _pad1: [0.0;7],
                };
                self.queue.write_buffer(&self.grid_level_uniform_buffer, 0, bytemuck::bytes_of(&u));
                render_pass.draw(0..self.grid_vertex_count as u32, 0..1);
            }

            render_pass.set_pipeline(&self.plot_render_pipeline);
            render_pass.set_vertex_buffer(0, self.plot_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.uniform_buffer_bind_group, &[]);
            render_pass.draw(0..self.latest_plot_metadata.num_points, 0..1);

            // Render text last to ensure it draws on top
            self.text_renderer.render(&self.text_atlas, &self.glyphon_viewport, &mut render_pass).unwrap();


        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/*8
pub fn insert_canvas_and_create_log_list(window: &Window) -> web_sys::Element {
    use winit::platform::web::WindowExtWebSys;

    let canvas = window.canvas().unwrap();
    let mut surface = Surface::from_canvas(canvas.clone()).unwrap();
    surface
        .resize(
            NonZeroU32::new(canvas.width()).unwrap(),
            NonZeroU32::new(canvas.height()).unwrap(),
        )
        .unwrap();
    let mut buffer = surface.buffer_mut().unwrap();
    buffer.fill(0xFFF0000);
    buffer.present().unwrap();

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();

    let style = &canvas.style();
    style.set_property("margin", "50px").unwrap();
    // Use to test interactions with border and padding.
    //style.set_property("border", "50px solid black").unwrap();
    //style.set_property("padding", "50px").unwrap();

    let log_header = document.create_element("h2").unwrap();
    log_header.set_text_content(Some("Event Log"));
    body.append_child(&log_header).unwrap();

    let log_list = document.create_element("ul").unwrap();
    body.append_child(&log_list).unwrap();
    log_list
}*/

struct App {
    state: Option<State>,
    // Prevent repeated window creation on wasm while async init is pending
    initialized: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.initialized {
            let window_attributes = Window::default_attributes();
            
            #[cfg(target_arch = "wasm32")]
            let window_attributes = {
                use winit::platform::web::WindowAttributesExtWebSys;
                window_attributes.with_append(true)
            };

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

            #[cfg(target_arch = "wasm32")]
            {
                // Let CSS control the canvas size (index.html sets canvas to fill the viewport).
                // Avoid forcing a fixed inner size on the web; winit will emit Resized events
                // as the canvas/layout changes, and our resize path will reconfigure the surface.
            }

            #[cfg(target_arch = "wasm32")]
            {
                // Async initialize State without blocking the single-threaded wasm main thread.
                let window_clone = window.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let mut state = State::new(window_clone.clone()).await;
                    state.load_data();
                    GLOBAL_STATE.with(|cell| {
                        *cell.borrow_mut() = Some(state);
                    });
                    // Kick a redraw once ready
                    window_clone.request_redraw();
                });
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut state = pollster::block_on(State::new(window));
                state.load_data();
                self.state = Some(state);
            }
            self.initialized = true;
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        // On wasm, async init may complete after the window starts emitting events.
        // Try to claim the prepared State if we don't have it yet.
        if self.state.is_none() {
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(state) = GLOBAL_STATE.with(|cell| cell.borrow_mut().take()) {
                    self.state = Some(state);
                }
            }
        }

        if let Some(state) = &mut self.state {
            if window_id == state.window.id() && !state.input(&event) {
                 match event {
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                            Err(e) => eprintln!("{:?}", e),
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = App { state: None, initialized: false };
    event_loop.run_app(&mut app).unwrap();
}
