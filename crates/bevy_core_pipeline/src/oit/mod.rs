//! Order Independent Transparency (OIT) for 3d rendering. See [`OrderIndependentTransparencyPlugin`] for more details.

use bevy_app::prelude::*;
use bevy_camera::Camera3d;
use bevy_core_pipeline_types::schedule::PrepareOitBuffers;
use bevy_ecs::prelude::*;
use bevy_log::trace;
use bevy_math::UVec2;
use bevy_platform::time::Instant;
use bevy_render::{
    camera::ExtractedCamera,
    extract_component::ExtractComponentPlugin,
    render_resource::{
        BufferUsages, DynamicUniformBuffer, TextureUsages, UniformBuffer, UninitBufferVec,
    },
    renderer::{RenderDevice, RenderQueue},
    view::Msaa,
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::load_shader_library;
use resolve::OitResolvePlugin;

use crate::{
    core_3d::main_transparent_pass_3d,
    oit::resolve::node::oit_resolve,
    schedule::{Core3d, Core3dSystems},
};

pub use bevy_core_pipeline_types::oit::{
    OitBuffers, OitFragmentNode, OrderIndependentTransparencySettings,
    OrderIndependentTransparencySettingsOffset,
};

/// Module that defines the necessary systems to resolve the OIT buffer and render it to the screen.
pub mod resolve;

/// A plugin that adds support for Order Independent Transparency (OIT).
/// This can correctly render some scenes that would otherwise have artifacts due to alpha blending, but uses more memory.
///
/// To enable OIT for a camera you need to add the [`OrderIndependentTransparencySettings`] component to it.
///
/// If you want to use OIT for your custom material you need to call `oit_draw(position, color)` in your fragment shader.
/// You also need to make sure that your fragment shader doesn't output any colors.
///
/// # Implementation details
/// This implementation uses 2 passes.
///
/// The first pass constructs a linked list which stores depth and color of all fragments in a big buffer.
/// The linked list capacity can be set with [`OrderIndependentTransparencySettings::fragments_per_pixel_average`].
/// This pass is essentially a forward pass.
///
/// The second pass is a single fullscreen triangle pass that sorts all the fragments then blends them together
/// and outputs the result to the screen.
pub struct OrderIndependentTransparencyPlugin;
impl Plugin for OrderIndependentTransparencyPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "oit_draw.wgsl");

        app.add_plugins((
            ExtractComponentPlugin::<OrderIndependentTransparencySettings>::default(),
            OitResolvePlugin,
        ))
        .add_systems(Update, check_msaa);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(RenderStartup, init_oit_buffers)
            .add_systems(
                Render,
                (
                    configure_camera_depth_usages
                        .in_set(RenderSystems::PrepareViews)
                        .ambiguous_with(RenderSystems::PrepareViews),
                    prepare_oit_buffers
                        .in_set(PrepareOitBuffers)
                        .in_set(RenderSystems::PrepareResources),
                ),
            );

        render_app.add_systems(
            Core3d,
            oit_resolve
                .after(main_transparent_pass_3d)
                .in_set(Core3dSystems::MainPass),
        );
    }
}

fn configure_camera_depth_usages(
    mut cameras: Query<
        &mut Camera3d,
        (
            Changed<Camera3d>,
            With<OrderIndependentTransparencySettings>,
        ),
    >,
) {
    for mut camera in &mut cameras {
        camera.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
    }
}

fn check_msaa(cameras: Query<&Msaa, With<OrderIndependentTransparencySettings>>) {
    for msaa in &cameras {
        if msaa.samples() > 1 {
            panic!("MSAA is not supported when using OrderIndependentTransparency");
        }
    }
}

fn create_nodes_buffer(
    size: usize,
    render_device: &RenderDevice,
) -> UninitBufferVec<OitFragmentNode> {
    let mut nodes = UninitBufferVec::new(BufferUsages::COPY_DST | BufferUsages::STORAGE);
    nodes.set_label(Some("oit_nodes"));
    nodes.reserve(size, render_device);
    nodes
}

fn create_heads_buffer(size: usize, render_device: &RenderDevice) -> UninitBufferVec<u32> {
    let mut nodes = UninitBufferVec::new(BufferUsages::COPY_DST | BufferUsages::STORAGE);
    nodes.set_label(Some("oit_heads"));
    nodes.reserve(size, render_device);
    nodes
}

pub fn init_oit_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    // initialize buffers with something so there's a valid binding

    let mut nodes_capacity = UniformBuffer::default();
    nodes_capacity.set_label(Some("oit_nodes_capacity"));
    nodes_capacity.set(1);
    nodes_capacity.write_buffer(&render_device, &render_queue);

    let nodes = create_nodes_buffer(1, &render_device);

    let heads = create_heads_buffer(1, &render_device);

    let mut atomic_counter = UninitBufferVec::new(BufferUsages::COPY_DST | BufferUsages::STORAGE);
    atomic_counter.set_label(Some("oit_atomic_counter"));
    atomic_counter.reserve(1, &render_device);

    let mut settings = DynamicUniformBuffer::default();
    settings.set_label(Some("oit_settings"));

    commands.insert_resource(OitBuffers {
        nodes_capacity,
        nodes,
        heads,
        atomic_counter,
        settings,
    });
}

/// This creates or resizes the oit buffers for each camera.
/// It will always create one big buffer that's as big as the biggest buffer needed.
/// Cameras with smaller viewports or less layers will simply use the big buffer and ignore the rest.
pub fn prepare_oit_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    cameras: Query<
        (&ExtractedCamera, &OrderIndependentTransparencySettings),
        (
            Changed<ExtractedCamera>,
            Changed<OrderIndependentTransparencySettings>,
        ),
    >,
    camera_oit_uniforms: Query<
        (Entity, &OrderIndependentTransparencySettings),
        With<ExtractedCamera>,
    >,
    mut buffers: ResMut<OitBuffers>,
) {
    let camera_count = camera_oit_uniforms.count();
    if camera_count == 0 {
        if buffers.nodes_capacity.get() > &1 {
            // Release oit buffers if no camera enables OIT.
            buffers.nodes = create_nodes_buffer(1, &render_device);
            buffers.heads = create_heads_buffer(1, &render_device);
            buffers.nodes_capacity.set(1);
            buffers
                .nodes_capacity
                .write_buffer(&render_device, &render_queue);
        }
        return;
    }

    // Get the max buffer size for any OIT enabled camera
    let mut max_size = UVec2::new(0, 0);
    let mut fragments_per_pixel_average = 0f32;
    for (camera, settings) in &cameras {
        let Some(size) = camera.physical_target_size else {
            continue;
        };
        max_size = max_size.max(size);
        fragments_per_pixel_average =
            fragments_per_pixel_average.max(settings.fragments_per_pixel_average);
    }

    // Create or update the heads buffer based on the max size
    let heads_size = (max_size.x * max_size.y) as usize;
    if buffers.heads.capacity() != heads_size {
        let start = Instant::now();
        buffers.heads = create_heads_buffer(heads_size, &render_device);
        trace!(
            "OIT heads buffer updated in {:.01}ms with total size {} MiB",
            start.elapsed().as_millis(),
            (buffers.heads.capacity() * size_of::<u32>()) as f32 / 1024.0 / 1024.0,
        );
    }

    // Create or update the nodes buffer based on the max size
    let nodes_size = ((max_size.x * max_size.y) as f32 * fragments_per_pixel_average) as usize;
    if buffers.nodes.capacity() != nodes_size {
        let start = Instant::now();
        buffers.nodes = create_nodes_buffer(nodes_size, &render_device);
        trace!(
            "OIT nodes buffer updated in {:.01}ms with total size {} MiB",
            start.elapsed().as_millis(),
            (buffers.nodes.capacity() * size_of::<OitFragmentNode>()) as f32 / 1024.0 / 1024.0,
        );
    }

    buffers.nodes_capacity.set(nodes_size as u32);
    buffers
        .nodes_capacity
        .write_buffer(&render_device, &render_queue);

    if let Some(mut writer) =
        buffers
            .settings
            .get_writer(camera_count, &render_device, &render_queue)
    {
        for (entity, settings) in &camera_oit_uniforms {
            let offset = writer.write(settings);
            commands
                .entity(entity)
                .insert(OrderIndependentTransparencySettingsOffset { offset });
        }
    }
}
