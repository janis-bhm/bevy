//! The core rendering pipelines schedules. These schedules define the "default" render graph
//! for 2D and 3D rendering in Bevy.
//!
//! Rendering in Bevy is "camera driven", meaning that for each camera in the world, its
//! associated rendering schedule is executed. This allows different cameras to have different
//! rendering pipelines, for example a 3D camera with post-processing effects and a 2D camera
//! with a simple clear and sprite rendering.
//!
//! The [`camera_driver`] system is responsible for iterating over all cameras in the world
//! and executing their associated schedules. In this way, the schedule for each camera is a
//! sub-schedule or sub-graph of the root render graph schedule.
use bevy_camera::{ClearColor, NormalizedRenderTarget};
use bevy_ecs::prelude::*;
use bevy_log::info_span;
use bevy_platform::collections::HashSet;
use bevy_render::{
    camera::{ExtractedCamera, SortedCameras},
    render_resource::{
        CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
        RenderPassDescriptor, StoreOp,
    },
    renderer::{CurrentView, PendingCommandBuffers, RenderDevice, RenderQueue},
    view::ExtractedWindows,
};

pub use bevy_core_pipeline_types::schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems};

/// Holds the entity of windows that are a render target for a camera
#[derive(Resource)]
struct CameraWindows(HashSet<Entity>);

/// The default entry point for camera driven rendering added to the root [`bevy_render::renderer::RenderGraph`]
/// schedule. This system iterates over all cameras in the world, executing their associated
/// rendering schedules defined by the [`bevy_render::camera::CameraRenderGraph`] component.
///
/// After executing all camera schedules, it submits any pending command buffers to the GPU
/// and clears any swap chains that were not covered by a camera. Users can order any additional
/// operations (e.g. one-off compute passes) before or after this system in the root render
/// graph schedule.
pub fn camera_driver(world: &mut World) {
    let sorted_cameras: Vec<_> = {
        let sorted = world.resource::<SortedCameras>();
        sorted.0.iter().map(|c| (c.entity, c.order)).collect()
    };

    let mut camera_windows = HashSet::default();

    for camera in sorted_cameras {
        #[cfg(feature = "trace")]
        let (camera_entity, order) = camera;
        #[cfg(not(feature = "trace"))]
        let (camera_entity, _) = camera;
        let Some(camera) = world.get::<ExtractedCamera>(camera_entity) else {
            continue;
        };

        let schedule = camera.schedule;
        let target = camera.target.clone();

        let mut run_schedule = true;
        if let Some(NormalizedRenderTarget::Window(window_ref)) = &target {
            let window_entity = window_ref.entity();
            let windows = world.resource::<ExtractedWindows>();
            if windows
                .windows
                .get(&window_entity)
                .is_some_and(|w| w.physical_width > 0 && w.physical_height > 0)
            {
                camera_windows.insert(window_entity);
            } else {
                run_schedule = false;
            }
        }

        if run_schedule {
            world.insert_resource(CurrentView(camera_entity));

            #[cfg(feature = "trace")]
            let _span = bevy_log::info_span!(
                "camera_schedule",
                camera = format!("Camera {} ({:?})", order, camera_entity)
            )
            .entered();

            world.run_schedule(schedule);
        }
    }
    world.remove_resource::<CurrentView>();

    world.insert_resource(CameraWindows(camera_windows));
}

pub(crate) fn submit_pending_command_buffers(world: &mut World) {
    let mut pending = world.resource_mut::<PendingCommandBuffers>();
    let buffer_count = pending.len();
    let buffers = pending.take();

    if !buffers.is_empty() {
        let _span = info_span!("queue_submit", count = buffer_count).entered();
        let queue = world.resource::<RenderQueue>();
        queue.submit(buffers);
    }
}

pub(crate) fn handle_uncovered_swap_chains(world: &mut World) {
    let windows_to_clear: Vec<_> = {
        let clear_color = world.resource::<ClearColor>().0.to_linear();
        let Some(camera_windows) = world.remove_resource::<CameraWindows>() else {
            return;
        };
        let windows = world.resource::<ExtractedWindows>();
        windows
            .iter()
            .filter_map(|(window_entity, window)| {
                if camera_windows.0.contains(window_entity) {
                    return None;
                }
                let swap_chain_texture = window.swap_chain_texture_view.as_ref()?;
                Some((swap_chain_texture.clone(), clear_color))
            })
            .collect()
    };

    if windows_to_clear.is_empty() {
        return;
    }

    let render_device = world.resource::<RenderDevice>();
    let render_queue = world.resource::<RenderQueue>();

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor::default());

    for (swap_chain_texture, clear_color) in &windows_to_clear {
        #[cfg(feature = "trace")]
        let _span = bevy_log::info_span!("no_camera_clear_pass").entered();

        let pass_descriptor = RenderPassDescriptor {
            label: Some("no_camera_clear_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: swap_chain_texture,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear((*clear_color).into()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        encoder.begin_render_pass(&pass_descriptor);
    }

    render_queue.submit([encoder.finish()]);
}
