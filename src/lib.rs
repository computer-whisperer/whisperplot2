use std::fs::File;
use std::path::Path;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use glyphon::{FontSystem, SwashCache, TextAtlas, TextRenderer, Metrics, Attrs, Family, Shaping, Resolution, TextArea, TextBounds};
use wgpu::{Backends, MultisampleState, RenderPass};
use wgpu::util::DeviceExt;
#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit::dpi::PhysicalPosition;
use winit::event::MouseScrollDelta::LineDelta;

use winit::window::Window;

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

struct State<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    plot_render_pipeline: wgpu::RenderPipeline,
    grid_render_pipeline: wgpu::RenderPipeline,
    plot_vertex_buffer: wgpu::Buffer,
    grid_vertex_buffer: wgpu::Buffer,
    grid_vertex_count: usize,
    uniform_buffer: wgpu::Buffer,
    uniform_buffer_bind_group: wgpu::BindGroup,
    latest_plot_metadata: PlotMetadata,
    plot_view_frame_value: PlotViewFrame,
    text_renderer: TextRenderer,
    text_buffer: glyphon::Buffer,
    font_system: FontSystem,
    text_atlas: TextAtlas,
    text_cache: SwashCache,
}

impl State<'_> {
    // Creating some of the wgpu types requires async code
    async fn new<'window>(window: &'window Window) -> State<'window> {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

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
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web, we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None, // Trace path
        ).await.unwrap();

        const VERTICES: &[Vertex] = &[
            Vertex { position: [0.0, 0.5]},
            Vertex { position: [-0.5, -0.5]},
            Vertex { position: [0.5, -0.5]},
        ];

        let plot_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Plot Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let mut grid_vertexes: Vec::<Vertex> = vec![];
        let num_per_side = 30;
        for i in 0..num_per_side {
            grid_vertexes.push(Vertex { position: [(i as f32)/num_per_side as f32, 0.0] });
            grid_vertexes.push(Vertex { position: [(i as f32)/num_per_side as f32, 1.0] });
            grid_vertexes.push(Vertex { position: [0.0, (i as f32)/num_per_side as f32] });
            grid_vertexes.push(Vertex { position: [1.0, (i as f32)/num_per_side as f32] });
        }

        let grid_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Grid Vertex Buffer"),
                contents: bytemuck::cast_slice(grid_vertexes.as_slice()),
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
                }
            ],
            label: Some("uniform_buffer_bind_group_layout"),
        });

        let uniform_buffer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_buffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }
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

        let mut font_system = FontSystem::new();
        let mut text_cache = SwashCache::new();
        let mut text_atlas = TextAtlas::new(&device, &queue, surface_format);
        let mut text_renderer = TextRenderer::new(&mut text_atlas, &device, MultisampleState::default(), None);
        let mut text_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(30.0, 42.0));

        text_buffer.set_size(&mut font_system, size.width as f32, size.height as f32);
        text_buffer.set_text(&mut font_system, "Hello world! 👋\nThis is rendered with 🦅 glyphon 🦁\nThe text below should be partially clipped.\na b c d e f g h i j k l m n o p q r s t u v w x y z", Attrs::new().family(Family::SansSerif), Shaping::Advanced);
        text_buffer.shape_until_scroll(&mut font_system);

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
            label: Some("Plot Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
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
                module: &shader,
                entry_point: "fs_main",
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
            label: Some("Grid Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_grid",
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
                module: &shader,
                entry_point: "fs_grid",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
            surface,
            device,
            queue,
            config,
            size,
            plot_render_pipeline,
            grid_render_pipeline,
            plot_vertex_buffer,
            grid_vertex_buffer,
            grid_vertex_count: grid_vertexes.len(),
            uniform_buffer,
            uniform_buffer_bind_group,
            latest_plot_metadata: PlotMetadata {.. Default::default() },
            plot_view_frame_value: PlotViewFrame {.. Default::default() },
            text_renderer,
            text_buffer,
            font_system,
            text_atlas,
            text_cache
        }
    }

    fn load_data(&mut self) {
        let path = Path::new("data.csv");
        let file = File::open(path).unwrap();
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).delimiter(b',').from_reader(file);
        let mut plot_metadata = PlotMetadata{
            min_x: 0f32,
            max_x: 0f32,
            min_y: 0f32,
            max_y: 0f32,
            num_points: 0};

        let mut raw_points: Vec<Vertex> = vec![];

        let mut i = 0;
        for result in rdr.records() {
            let v = result.unwrap();
            //let x = v.get(0).unwrap().trim().parse::<f32>().unwrap();
            let x = i as f32;
            let y = v.get(0).unwrap().trim().parse::<f32>().unwrap();

            //let y = f32::sin((i as f32)/10000f32);

            raw_points.push(Vertex{position: [x, y]});

            if i == 0 {
                plot_metadata.min_x = x;
                plot_metadata.max_x = x;
                plot_metadata.min_y = y;
                plot_metadata.max_y = y;
            }

            if x < plot_metadata.min_x {
                plot_metadata.min_x = x;
            }
            if x > plot_metadata.max_x {
                plot_metadata.max_x = x;
            }
            if y < plot_metadata.min_y {
                plot_metadata.min_y = y;
            }
            if y > plot_metadata.max_y {
                plot_metadata.max_y = y;
            }
            i += 1;
        };

        plot_metadata.num_points = i;
        self.plot_vertex_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(raw_points.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        self.latest_plot_metadata = plot_metadata;
        self.plot_view_frame_value = PlotViewFrame {
            full_min_x: plot_metadata.min_x,
            full_max_x: plot_metadata.max_x,
            full_min_y: plot_metadata.min_y,
            full_max_y: plot_metadata.max_y,
            view_min_x: plot_metadata.min_x,
            view_max_x: plot_metadata.max_x,
            view_min_y: plot_metadata.min_y,
            view_max_y: plot_metadata.max_y
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        //todo!()
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.text_atlas,
            Resolution {
                width: self.config.width,
                height: self.config.height,
            },
            [TextArea {
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

            self.text_renderer.render(&self.text_atlas, &mut render_pass).unwrap();


            render_pass.set_pipeline(&self.grid_render_pipeline);
            render_pass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.uniform_buffer_bind_group, &[]);
            render_pass.draw(0..self.grid_vertex_count as u32, 0..1);

            render_pass.set_pipeline(&self.plot_render_pipeline);
            render_pass.set_vertex_buffer(0, self.plot_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.uniform_buffer_bind_group, &[]);
            render_pass.draw(0..self.latest_plot_metadata.num_points, 0..1);


        }

        // submit will accept anything that implements IntoIter
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.plot_view_frame_value]));
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
    let builder = WindowBuilder::new();
    #[cfg(target_arch = "wasm32")]
        let builder = {
        use winit::platform::web::WindowBuilderExtWebSys;
        builder.with_append(true)
    };

    let window = builder.build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        let _ = window.request_inner_size(PhysicalSize::new(1024, 1024));
    }


    let mut state = State::new(&window).await;

    let mut last_cursor_position : PhysicalPosition<f64> = Default::default();
    let mut last_left_click_screen_position : PhysicalPosition<f64> = Default::default();

    let mut current_left_mouse_state : ElementState = ElementState::Released;
    let mut left_mouse_last_transition_time = Instant::now();
    let mut left_mouse_last_quick_click_time = Instant::now();

    let mut current_right_mouse_state : ElementState = ElementState::Released;
    let mut right_mouse_last_transition_time = Instant::now();

    state.load_data();

    event_loop.run(|event, event_loop_window_target| {
        match event {
            Event::AboutToWait => {
                // RedrawRequested will only trigger once unless we manually
                // request it.
                window.request_redraw();
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested => {event_loop_window_target.exit();}
                    WindowEvent::MouseWheel{delta, ..} => {
                        if let LineDelta(x, y) = delta {
                            let middle = (state.plot_view_frame_value.view_min_x + state.plot_view_frame_value.view_max_x) / 2.0;
                            let span = (state.plot_view_frame_value.view_max_x - state.plot_view_frame_value.view_min_x) / 2.0;
                            let new_span = span * (1.0 - (*y/20f32));
                            state.plot_view_frame_value.view_min_x = middle - new_span;
                            state.plot_view_frame_value.view_max_x = middle + new_span;
                        }
                    }
                    WindowEvent::MouseInput{state: mouse_state, button, ..} => {
                        match button {
                            MouseButton::Left => {
                                if *mouse_state == ElementState::Released {
                                    if left_mouse_last_transition_time.elapsed().as_millis() < 200 {
                                        if left_mouse_last_quick_click_time.elapsed().as_millis() < 500 {
                                            state.plot_view_frame_value.view_min_x = state.latest_plot_metadata.min_x;
                                            state.plot_view_frame_value.view_max_x = state.latest_plot_metadata.max_x;
                                            state.plot_view_frame_value.view_min_y = state.latest_plot_metadata.min_y;
                                            state.plot_view_frame_value.view_max_y = state.latest_plot_metadata.max_y;
                                        }
                                        left_mouse_last_quick_click_time = Instant::now();
                                    }
                                    if left_mouse_last_transition_time.elapsed().as_millis() > 200 {
                                        // Zoom X to selected region
                                        let start_x_fraction = (last_left_click_screen_position.x/state.size.width as f64);
                                        let end_x_fraction = (last_cursor_position.x/state.size.width as f64);
                                        let x_span = state.plot_view_frame_value.view_max_x - state.plot_view_frame_value.view_min_x;
                                        let start_x = start_x_fraction*x_span as f64 + state.plot_view_frame_value.view_min_x as f64;
                                        let end_x = end_x_fraction*x_span as f64 + state.plot_view_frame_value.view_min_x as f64;
                                        state.plot_view_frame_value.view_min_x = start_x as f32;
                                        state.plot_view_frame_value.view_max_x = end_x as f32;
                                    }
                                }
                                if *mouse_state == ElementState::Pressed {
                                    last_left_click_screen_position = last_cursor_position;
                                    dbg!(last_cursor_position);
                                }
                                current_left_mouse_state = *mouse_state;
                                left_mouse_last_transition_time = Instant::now();
                            }
                            MouseButton::Right => {
                                current_right_mouse_state = *mouse_state;
                                right_mouse_last_transition_time = Instant::now();
                            }
                            MouseButton::Middle => {}
                            MouseButton::Back => {}
                            MouseButton::Forward => {}
                            MouseButton::Other(_) => {}
                        }
                    }
                    WindowEvent::CursorMoved {position,..} => {
                        last_cursor_position = *position;
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            // Reconfigure the surface if lost
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
                            // All other errors (Outdated, Timeout) should be resolved by the next frame
                            Err(e) => eprintln!("{:?}", e),
                        }
                    }
                    WindowEvent::ScaleFactorChanged { inner_size_writer, .. } => {
                        //state.resize(inner_size_writer);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }).unwrap();
}

