mod main_opaque_pass_2d_node;
mod main_transparent_pass_2d_node;

use bevy_camera::{Camera, Camera2d};
use bevy_image::ToExtents;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    batching::gpu_preprocessing::GpuPreprocessingMode,
    camera::CameraRenderGraph,
    view::{ExtractedView, RetainedViewEntity},
};
pub use main_opaque_pass_2d_node::*;
pub use main_transparent_pass_2d_node::*;

use crate::schedule::Core2d;
use crate::tonemapping::{tonemapping, DebandDither, Tonemapping};
use crate::upscaling::upscaling;
use crate::Core2dSystems;
use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    extract_component::ExtractComponentPlugin,
    render_phase::{
        sort_phase_system, DrawFunctions, ViewBinnedRenderPhases, ViewSortedRenderPhases,
    },
    render_resource::{TextureDescriptor, TextureDimension, TextureUsages},
    renderer::RenderDevice,
    texture::TextureCache,
    view::{Msaa, ViewDepthTexture},
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
};

pub use bevy_core_pipeline_types::core_2d::{
    AlphaMask2d, Opaque2d, Transparent2d, CORE_2D_DEPTH_FORMAT,
};

pub struct Core2dPlugin;

impl Plugin for Core2dPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Camera2d, DebandDither>()
            .register_required_components_with::<Camera2d, CameraRenderGraph>(|| {
                CameraRenderGraph::new(Core2d)
            })
            .register_required_components_with::<Camera2d, Tonemapping>(|| Tonemapping::None)
            .add_plugins(ExtractComponentPlugin::<Camera2d>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<DrawFunctions<Opaque2d>>()
            .init_resource::<DrawFunctions<AlphaMask2d>>()
            .init_resource::<DrawFunctions<Transparent2d>>()
            .init_resource::<ViewSortedRenderPhases<Transparent2d>>()
            .init_resource::<ViewBinnedRenderPhases<Opaque2d>>()
            .init_resource::<ViewBinnedRenderPhases<AlphaMask2d>>()
            .allow_ambiguous_resource::<ViewSortedRenderPhases<Transparent2d>>()
            .allow_ambiguous_resource::<ViewBinnedRenderPhases<Opaque2d>>()
            .allow_ambiguous_resource::<ViewBinnedRenderPhases<AlphaMask2d>>()
            .add_systems(ExtractSchedule, extract_core_2d_camera_phases)
            .add_systems(
                Render,
                (
                    sort_phase_system::<Transparent2d>.in_set(RenderSystems::PhaseSort),
                    prepare_core_2d_depth_textures.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_schedule(Core2d::base_schedule())
            .add_systems(
                Core2d,
                (
                    (main_opaque_pass_2d, main_transparent_pass_2d)
                        .chain()
                        .in_set(Core2dSystems::MainPass),
                    tonemapping
                        .in_set(Core2dSystems::Tonemapping)
                        .in_set(Core2dSystems::PostProcess),
                    upscaling
                        .in_set(Core2dSystems::Upscaling)
                        .after(Core2dSystems::PostProcess),
                ),
            );
    }
}

pub fn extract_core_2d_camera_phases(
    mut transparent_2d_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    mut opaque_2d_phases: ResMut<ViewBinnedRenderPhases<Opaque2d>>,
    mut alpha_mask_2d_phases: ResMut<ViewBinnedRenderPhases<AlphaMask2d>>,
    cameras_2d: Extract<Query<(Entity, &Camera), With<Camera2d>>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
) {
    live_entities.clear();

    for (main_entity, camera) in &cameras_2d {
        if !camera.is_active {
            continue;
        }

        // This is the main 2D camera, so we use the first subview index (0).
        let retained_view_entity = RetainedViewEntity::new(main_entity.into(), None, 0);

        transparent_2d_phases.prepare_for_new_frame(retained_view_entity);
        opaque_2d_phases.prepare_for_new_frame(retained_view_entity, GpuPreprocessingMode::None);
        alpha_mask_2d_phases
            .prepare_for_new_frame(retained_view_entity, GpuPreprocessingMode::None);

        live_entities.insert(retained_view_entity);
    }

    // Clear out all dead views.
    transparent_2d_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
    opaque_2d_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
    alpha_mask_2d_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
}

pub fn prepare_core_2d_depth_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    transparent_2d_phases: Res<ViewSortedRenderPhases<Transparent2d>>,
    opaque_2d_phases: Res<ViewBinnedRenderPhases<Opaque2d>>,
    views_2d: Query<(Entity, &ExtractedCamera, &ExtractedView, &Msaa), (With<Camera2d>,)>,
) {
    let mut textures = <HashMap<_, _>>::default();
    for (view, camera, extracted_view, msaa) in &views_2d {
        if !opaque_2d_phases.contains_key(&extracted_view.retained_view_entity)
            || !transparent_2d_phases.contains_key(&extracted_view.retained_view_entity)
        {
            continue;
        };

        let Some(physical_target_size) = camera.physical_target_size else {
            continue;
        };

        let cached_texture = textures
            .entry(camera.target.clone())
            .or_insert_with(|| {
                let descriptor = TextureDescriptor {
                    label: Some("view_depth_texture"),
                    // The size of the depth texture
                    size: physical_target_size.to_extents(),
                    mip_level_count: 1,
                    sample_count: msaa.samples(),
                    dimension: TextureDimension::D2,
                    format: CORE_2D_DEPTH_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                };

                texture_cache.get(&render_device, descriptor)
            })
            .clone();

        commands
            .entity(view)
            .insert(ViewDepthTexture::new(cached_texture, Some(0.0)));
    }
}
