mod main_opaque_pass_3d_node;
mod main_transparent_pass_3d_node;

use bevy_camera::{Camera, Camera3d, Camera3dDepthLoadOp};
use bevy_core_pipeline_types::schedule::PrepareCore3dDepthTextures;
use bevy_diagnostic::FrameCount;
use bevy_render::{
    batching::gpu_preprocessing::{GpuPreprocessingMode, GpuPreprocessingSupport},
    camera::CameraRenderGraph,
    occlusion_culling::OcclusionCulling,
    texture::CachedTexture,
    view::{prepare_view_targets, NoIndirectDrawing, RetainedViewEntity},
};
pub use main_opaque_pass_3d_node::*;
pub use main_transparent_pass_3d_node::*;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_color::LinearRgba;
use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_log::warn;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    camera::ExtractedCamera,
    extract_component::ExtractComponentPlugin,
    prelude::Msaa,
    render_phase::{
        sort_phase_system, DrawFunctions, ViewBinnedRenderPhases, ViewSortedRenderPhases,
    },
    render_resource::{TextureDescriptor, TextureDimension, TextureUsages},
    renderer::RenderDevice,
    sync_world::RenderEntity,
    texture::{ColorAttachment, TextureCache},
    view::{ExtractedView, ViewDepthTexture},
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
};

use crate::deferred::copy_lighting_id::copy_deferred_lighting_id;
use crate::deferred::node::{early_deferred_prepass, late_deferred_prepass};
use crate::prepass::node::{early_prepass, late_prepass};
use crate::tonemapping::tonemapping;
use crate::upscaling::upscaling;
use crate::{
    deferred::{
        AlphaMask3dDeferred, Opaque3dDeferred, DEFERRED_LIGHTING_PASS_ID_FORMAT,
        DEFERRED_PREPASS_FORMAT,
    },
    prepass::{
        AlphaMask3dPrepass, DeferredPrepass, DeferredPrepassDoubleBuffer, DepthPrepass,
        DepthPrepassDoubleBuffer, MotionVectorPrepass, NormalPrepass, Opaque3dPrepass,
        ViewPrepassTextures, MOTION_VECTOR_PREPASS_FORMAT, NORMAL_PREPASS_FORMAT,
    },
    schedule::Core3d,
    skybox::SkyboxPlugin,
    tonemapping::{DebandDither, Tonemapping},
    Core3dSystems,
};

pub use bevy_core_pipeline_types::core_3d::{
    AlphaMask3d, Opaque3d, Transparent3d, CORE_3D_DEPTH_FORMAT, DEPTH_TEXTURE_SAMPLING_SUPPORTED,
};

pub struct Core3dPlugin;

impl Plugin for Core3dPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components_with::<Camera3d, DebandDither>(|| DebandDither::Enabled)
            .register_required_components_with::<Camera3d, CameraRenderGraph>(|| {
                CameraRenderGraph::new(Core3d)
            })
            .register_required_components::<Camera3d, Tonemapping>()
            .add_plugins((SkyboxPlugin, ExtractComponentPlugin::<Camera3d>::default()))
            .add_systems(PostUpdate, check_msaa);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<DrawFunctions<Opaque3d>>()
            .init_resource::<DrawFunctions<AlphaMask3d>>()
            .init_resource::<DrawFunctions<Transparent3d>>()
            .init_resource::<DrawFunctions<Opaque3dPrepass>>()
            .init_resource::<DrawFunctions<AlphaMask3dPrepass>>()
            .init_resource::<DrawFunctions<Opaque3dDeferred>>()
            .init_resource::<DrawFunctions<AlphaMask3dDeferred>>()
            .init_resource::<ViewBinnedRenderPhases<Opaque3d>>()
            .init_resource::<ViewBinnedRenderPhases<AlphaMask3d>>()
            .init_resource::<ViewBinnedRenderPhases<Opaque3dPrepass>>()
            .init_resource::<ViewBinnedRenderPhases<AlphaMask3dPrepass>>()
            .init_resource::<ViewBinnedRenderPhases<Opaque3dDeferred>>()
            .init_resource::<ViewBinnedRenderPhases<AlphaMask3dDeferred>>()
            .init_resource::<ViewSortedRenderPhases<Transparent3d>>()
            .add_systems(ExtractSchedule, extract_core_3d_camera_phases)
            .add_systems(ExtractSchedule, extract_camera_prepass_phase)
            .add_systems(
                Render,
                (
                    sort_phase_system::<Transparent3d>.in_set(RenderSystems::PhaseSort),
                    configure_occlusion_culling_view_targets
                        .after(prepare_view_targets)
                        .in_set(RenderSystems::PrepareViews)
                        .ambiguous_with(RenderSystems::PrepareViews),
                    (prepare_core_3d_depth_textures.in_set(PrepareCore3dDepthTextures),)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_prepass_textures.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_schedule(Core3d::base_schedule())
            .add_systems(
                Core3d,
                (
                    (
                        early_prepass.in_set(Core3dSystems::EarlyPrepass),
                        early_deferred_prepass,
                        late_prepass.in_set(Core3dSystems::LatePrepass),
                        late_deferred_prepass.in_set(Core3dSystems::LateDeferredPrepass),
                        copy_deferred_lighting_id,
                    )
                        .chain()
                        .in_set(Core3dSystems::Prepass),
                    (
                        main_opaque_pass_3d.in_set(Core3dSystems::MainOpaquePass),
                        main_transparent_pass_3d.in_set(Core3dSystems::MainTransparentPass),
                    )
                        .chain()
                        .in_set(Core3dSystems::MainPass),
                    tonemapping
                        .in_set(Core3dSystems::Tonemapping)
                        .in_set(Core3dSystems::PostProcess),
                    upscaling
                        .in_set(Core3dSystems::Upscaling)
                        .after(Core3dSystems::PostProcess),
                ),
            );
    }
}

pub fn extract_core_3d_camera_phases(
    mut opaque_3d_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    mut alpha_mask_3d_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3d>>,
    mut transparent_3d_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    cameras_3d: Extract<Query<(Entity, &Camera, Has<NoIndirectDrawing>), With<Camera3d>>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    live_entities.clear();

    for (main_entity, camera, no_indirect_drawing) in &cameras_3d {
        if !camera.is_active {
            continue;
        }

        // If GPU culling is in use, use it (and indirect mode); otherwise, just
        // preprocess the meshes.
        let gpu_preprocessing_mode = gpu_preprocessing_support.min(if !no_indirect_drawing {
            GpuPreprocessingMode::Culling
        } else {
            GpuPreprocessingMode::PreprocessingOnly
        });

        // This is the main 3D camera, so use the first subview index (0).
        let retained_view_entity = RetainedViewEntity::new(main_entity.into(), None, 0);

        opaque_3d_phases.prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
        alpha_mask_3d_phases.prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
        transparent_3d_phases.prepare_for_new_frame(retained_view_entity);

        live_entities.insert(retained_view_entity);
    }

    opaque_3d_phases.retain(|view_entity, _| live_entities.contains(view_entity));
    alpha_mask_3d_phases.retain(|view_entity, _| live_entities.contains(view_entity));
    transparent_3d_phases.retain(|view_entity, _| live_entities.contains(view_entity));
}

// Extract the render phases for the prepass

pub fn extract_camera_prepass_phase(
    mut commands: Commands,
    mut opaque_3d_prepass_phases: ResMut<ViewBinnedRenderPhases<Opaque3dPrepass>>,
    mut alpha_mask_3d_prepass_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3dPrepass>>,
    mut opaque_3d_deferred_phases: ResMut<ViewBinnedRenderPhases<Opaque3dDeferred>>,
    mut alpha_mask_3d_deferred_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3dDeferred>>,
    cameras_3d: Extract<
        Query<
            (
                Entity,
                RenderEntity,
                &Camera,
                Has<NoIndirectDrawing>,
                Has<DepthPrepass>,
                Has<NormalPrepass>,
                Has<MotionVectorPrepass>,
                Has<DeferredPrepass>,
                Has<DepthPrepassDoubleBuffer>,
                Has<DeferredPrepassDoubleBuffer>,
            ),
            With<Camera3d>,
        >,
    >,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    live_entities.clear();

    for (
        main_entity,
        entity,
        camera,
        no_indirect_drawing,
        depth_prepass,
        normal_prepass,
        motion_vector_prepass,
        deferred_prepass,
        depth_prepass_double_buffer,
        deferred_prepass_double_buffer,
    ) in cameras_3d.iter()
    {
        if !camera.is_active {
            continue;
        }

        // If GPU culling is in use, use it (and indirect mode); otherwise, just
        // preprocess the meshes.
        let gpu_preprocessing_mode = gpu_preprocessing_support.min(if !no_indirect_drawing {
            GpuPreprocessingMode::Culling
        } else {
            GpuPreprocessingMode::PreprocessingOnly
        });

        // This is the main 3D camera, so we use the first subview index (0).
        let retained_view_entity = RetainedViewEntity::new(main_entity.into(), None, 0);

        if depth_prepass || normal_prepass || motion_vector_prepass {
            opaque_3d_prepass_phases
                .prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
            alpha_mask_3d_prepass_phases
                .prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
        } else {
            opaque_3d_prepass_phases.remove(&retained_view_entity);
            alpha_mask_3d_prepass_phases.remove(&retained_view_entity);
        }

        if deferred_prepass {
            opaque_3d_deferred_phases
                .prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
            alpha_mask_3d_deferred_phases
                .prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
        } else {
            opaque_3d_deferred_phases.remove(&retained_view_entity);
            alpha_mask_3d_deferred_phases.remove(&retained_view_entity);
        }
        live_entities.insert(retained_view_entity);

        // Add or remove prepasses as appropriate.

        let mut camera_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");

        if depth_prepass {
            camera_commands.insert(DepthPrepass);
        } else {
            camera_commands.remove::<DepthPrepass>();
        }

        if normal_prepass {
            camera_commands.insert(NormalPrepass);
        } else {
            camera_commands.remove::<NormalPrepass>();
        }

        if motion_vector_prepass {
            camera_commands.insert(MotionVectorPrepass);
        } else {
            camera_commands.remove::<MotionVectorPrepass>();
        }

        if deferred_prepass {
            camera_commands.insert(DeferredPrepass);
        } else {
            camera_commands.remove::<DeferredPrepass>();
        }

        if depth_prepass_double_buffer {
            camera_commands.insert(DepthPrepassDoubleBuffer);
        } else {
            camera_commands.remove::<DepthPrepassDoubleBuffer>();
        }

        if deferred_prepass_double_buffer {
            camera_commands.insert(DeferredPrepassDoubleBuffer);
        } else {
            camera_commands.remove::<DeferredPrepassDoubleBuffer>();
        }
    }

    opaque_3d_prepass_phases.retain(|view_entity, _| live_entities.contains(view_entity));
    alpha_mask_3d_prepass_phases.retain(|view_entity, _| live_entities.contains(view_entity));
    opaque_3d_deferred_phases.retain(|view_entity, _| live_entities.contains(view_entity));
    alpha_mask_3d_deferred_phases.retain(|view_entity, _| live_entities.contains(view_entity));
}

pub fn prepare_core_3d_depth_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views_3d: Query<(
        Entity,
        &ExtractedCamera,
        Option<&DepthPrepass>,
        &Camera3d,
        &Msaa,
    )>,
) {
    let mut render_target_usage = <HashMap<_, _>>::default();
    for (_, camera, depth_prepass, camera_3d, _msaa) in &views_3d {
        // Default usage required to write to the depth texture
        let mut usage: TextureUsages = camera_3d.depth_texture_usages.into();
        if depth_prepass.is_some() {
            // Required to read the output of the prepass
            usage |= TextureUsages::COPY_SRC;
        }
        render_target_usage
            .entry(camera.target.clone())
            .and_modify(|u| *u |= usage)
            .or_insert_with(|| usage);
    }

    let mut textures = <HashMap<_, _>>::default();
    for (entity, camera, _, camera_3d, msaa) in &views_3d {
        let Some(physical_target_size) = camera.physical_target_size else {
            continue;
        };

        let cached_texture = textures
            .entry((camera.target.clone(), msaa))
            .or_insert_with(|| {
                let usage = *render_target_usage
                    .get(&camera.target.clone())
                    .expect("The depth texture usage should already exist for this target");

                let descriptor = TextureDescriptor {
                    label: Some("view_depth_texture"),
                    // The size of the depth texture
                    size: physical_target_size.to_extents(),
                    mip_level_count: 1,
                    sample_count: msaa.samples(),
                    dimension: TextureDimension::D2,
                    format: CORE_3D_DEPTH_FORMAT,
                    usage,
                    view_formats: &[],
                };

                texture_cache.get(&render_device, descriptor)
            })
            .clone();

        commands.entity(entity).insert(ViewDepthTexture::new(
            cached_texture,
            match camera_3d.depth_load_op {
                Camera3dDepthLoadOp::Clear(v) => Some(v),
                Camera3dDepthLoadOp::Load => None,
            },
        ));
    }
}

/// Sets the `TEXTURE_BINDING` flag on the depth texture if necessary for
/// occlusion culling.
///
/// We need that flag to be set in order to read from the texture.
fn configure_occlusion_culling_view_targets(
    mut view_targets: Query<
        &mut Camera3d,
        (
            With<OcclusionCulling>,
            Without<NoIndirectDrawing>,
            With<DepthPrepass>,
        ),
    >,
) {
    for mut camera_3d in &mut view_targets {
        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();
    }
}

// Disable MSAA and warn if using deferred rendering
pub fn check_msaa(mut deferred_views: Query<&mut Msaa, (With<Camera>, With<DeferredPrepass>)>) {
    for mut msaa in deferred_views.iter_mut() {
        match *msaa {
            Msaa::Off => (),
            _ => {
                warn!("MSAA is incompatible with deferred rendering and has been disabled.");
                *msaa = Msaa::Off;
            }
        };
    }
}

// Prepares the textures used by the prepass
pub fn prepare_prepass_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    frame_count: Res<FrameCount>,
    opaque_3d_prepass_phases: Res<ViewBinnedRenderPhases<Opaque3dPrepass>>,
    alpha_mask_3d_prepass_phases: Res<ViewBinnedRenderPhases<AlphaMask3dPrepass>>,
    opaque_3d_deferred_phases: Res<ViewBinnedRenderPhases<Opaque3dDeferred>>,
    alpha_mask_3d_deferred_phases: Res<ViewBinnedRenderPhases<AlphaMask3dDeferred>>,
    views_3d: Query<(
        Entity,
        &ExtractedCamera,
        &ExtractedView,
        &Msaa,
        Has<DepthPrepass>,
        Has<NormalPrepass>,
        Has<MotionVectorPrepass>,
        Has<DeferredPrepass>,
        Has<DepthPrepassDoubleBuffer>,
        Has<DeferredPrepassDoubleBuffer>,
    )>,
) {
    let mut depth_textures1 = <HashMap<_, _>>::default();
    let mut depth_textures2 = <HashMap<_, _>>::default();
    let mut normal_textures = <HashMap<_, _>>::default();
    let mut deferred_textures1: HashMap<_, _> = <HashMap<_, _>>::default();
    let mut deferred_textures2: HashMap<_, _> = <HashMap<_, _>>::default();
    let mut deferred_lighting_id_textures = <HashMap<_, _>>::default();
    let mut motion_vectors_textures = <HashMap<_, _>>::default();
    for (
        entity,
        camera,
        view,
        msaa,
        depth_prepass,
        normal_prepass,
        motion_vector_prepass,
        deferred_prepass,
        depth_prepass_double_buffer,
        deferred_prepass_double_buffer,
    ) in &views_3d
    {
        if !opaque_3d_prepass_phases.contains_key(&view.retained_view_entity)
            && !alpha_mask_3d_prepass_phases.contains_key(&view.retained_view_entity)
            && !opaque_3d_deferred_phases.contains_key(&view.retained_view_entity)
            && !alpha_mask_3d_deferred_phases.contains_key(&view.retained_view_entity)
        {
            commands.entity(entity).remove::<ViewPrepassTextures>();
            continue;
        };

        let Some(physical_target_size) = camera.physical_target_size else {
            continue;
        };

        let size = physical_target_size.to_extents();

        let cached_depth_texture1 = depth_prepass.then(|| {
            depth_textures1
                .entry(camera.target.clone())
                .or_insert_with(|| {
                    let descriptor = TextureDescriptor {
                        label: Some("prepass_depth_texture_1"),
                        size,
                        mip_level_count: 1,
                        sample_count: msaa.samples(),
                        dimension: TextureDimension::D2,
                        format: CORE_3D_DEPTH_FORMAT,
                        usage: TextureUsages::COPY_DST
                            | TextureUsages::RENDER_ATTACHMENT
                            | TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    };
                    texture_cache.get(&render_device, descriptor)
                })
                .clone()
        });

        let cached_depth_texture2 = depth_prepass_double_buffer.then(|| {
            depth_textures2
                .entry(camera.target.clone())
                .or_insert_with(|| {
                    let descriptor = TextureDescriptor {
                        label: Some("prepass_depth_texture_2"),
                        size,
                        mip_level_count: 1,
                        sample_count: msaa.samples(),
                        dimension: TextureDimension::D2,
                        format: CORE_3D_DEPTH_FORMAT,
                        usage: TextureUsages::COPY_DST
                            | TextureUsages::RENDER_ATTACHMENT
                            | TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    };
                    texture_cache.get(&render_device, descriptor)
                })
                .clone()
        });

        let cached_normals_texture = normal_prepass.then(|| {
            normal_textures
                .entry(camera.target.clone())
                .or_insert_with(|| {
                    texture_cache.get(
                        &render_device,
                        TextureDescriptor {
                            label: Some("prepass_normal_texture"),
                            size,
                            mip_level_count: 1,
                            sample_count: msaa.samples(),
                            dimension: TextureDimension::D2,
                            format: NORMAL_PREPASS_FORMAT,
                            usage: TextureUsages::RENDER_ATTACHMENT
                                | TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        },
                    )
                })
                .clone()
        });

        let cached_motion_vectors_texture = motion_vector_prepass.then(|| {
            motion_vectors_textures
                .entry(camera.target.clone())
                .or_insert_with(|| {
                    texture_cache.get(
                        &render_device,
                        TextureDescriptor {
                            label: Some("prepass_motion_vectors_textures"),
                            size,
                            mip_level_count: 1,
                            sample_count: msaa.samples(),
                            dimension: TextureDimension::D2,
                            format: MOTION_VECTOR_PREPASS_FORMAT,
                            usage: TextureUsages::RENDER_ATTACHMENT
                                | TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        },
                    )
                })
                .clone()
        });

        let cached_deferred_texture1 = deferred_prepass.then(|| {
            deferred_textures1
                .entry(camera.target.clone())
                .or_insert_with(|| {
                    texture_cache.get(
                        &render_device,
                        TextureDescriptor {
                            label: Some("prepass_deferred_texture_1"),
                            size,
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: TextureDimension::D2,
                            format: DEFERRED_PREPASS_FORMAT,
                            usage: TextureUsages::RENDER_ATTACHMENT
                                | TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        },
                    )
                })
                .clone()
        });

        let cached_deferred_texture2 = deferred_prepass_double_buffer.then(|| {
            deferred_textures2
                .entry(camera.target.clone())
                .or_insert_with(|| {
                    texture_cache.get(
                        &render_device,
                        TextureDescriptor {
                            label: Some("prepass_deferred_texture_2"),
                            size,
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: TextureDimension::D2,
                            format: DEFERRED_PREPASS_FORMAT,
                            usage: TextureUsages::RENDER_ATTACHMENT
                                | TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        },
                    )
                })
                .clone()
        });

        let cached_deferred_lighting_pass_id_texture = deferred_prepass.then(|| {
            deferred_lighting_id_textures
                .entry(camera.target.clone())
                .or_insert_with(|| {
                    texture_cache.get(
                        &render_device,
                        TextureDescriptor {
                            label: Some("deferred_lighting_pass_id_texture"),
                            size,
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: TextureDimension::D2,
                            format: DEFERRED_LIGHTING_PASS_ID_FORMAT,
                            usage: TextureUsages::RENDER_ATTACHMENT
                                | TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        },
                    )
                })
                .clone()
        });

        commands.entity(entity).insert(ViewPrepassTextures {
            depth: package_double_buffered_texture(
                cached_depth_texture1,
                cached_depth_texture2,
                frame_count.0,
            ),
            normal: cached_normals_texture
                .map(|t| ColorAttachment::new(t, None, None, Some(LinearRgba::BLACK.into()))),
            // Red and Green channels are X and Y components of the motion vectors
            // Blue channel doesn't matter, but set to 0.0 for possible faster clear
            // https://gpuopen.com/performance/#clears
            motion_vectors: cached_motion_vectors_texture
                .map(|t| ColorAttachment::new(t, None, None, Some(LinearRgba::BLACK.into()))),
            deferred: package_double_buffered_texture(
                cached_deferred_texture1,
                cached_deferred_texture2,
                frame_count.0,
            ),
            deferred_lighting_pass_id: cached_deferred_lighting_pass_id_texture
                .map(|t| ColorAttachment::new(t, None, None, Some(LinearRgba::BLACK.into()))),
            size,
        });
    }
}

fn package_double_buffered_texture(
    texture1: Option<CachedTexture>,
    texture2: Option<CachedTexture>,
    frame_count: u32,
) -> Option<ColorAttachment> {
    match (texture1, texture2) {
        (Some(t1), None) => Some(ColorAttachment::new(
            t1,
            None,
            None,
            Some(LinearRgba::BLACK.into()),
        )),
        (Some(t1), Some(t2)) if frame_count.is_multiple_of(2) => Some(ColorAttachment::new(
            t1,
            None,
            Some(t2),
            Some(LinearRgba::BLACK.into()),
        )),
        (Some(t1), Some(t2)) => Some(ColorAttachment::new(
            t2,
            None,
            Some(t1),
            Some(LinearRgba::BLACK.into()),
        )),
        _ => None,
    }
}
