use bevy_app::prelude::*;
use bevy_asset::{
    embedded_asset, load_embedded_asset, AssetServer, Assets, Handle, RenderAssetUsages,
};
use bevy_ecs::prelude::*;
use bevy_image::{CompressedImageFormats, Image, ImageSampler, ImageType};
#[cfg(not(feature = "tonemapping_luts"))]
use bevy_log::error;
use bevy_render::{
    extract_component::ExtractComponentPlugin,
    extract_resource::ExtractResourcePlugin,
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    view::{ExtractedView, ViewTarget, ViewUniform},
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::{load_shader_library, Shader, ShaderDefVal};
use bitflags::bitflags;

mod node;

use bevy_utils::default;
pub use node::tonemapping;

pub use bevy_core_pipeline_types::tonemapping::{
    get_lut_bind_group_layout_entries, get_lut_bindings, lut_placeholder, DebandDither,
    Tonemapping, TonemappingLuts,
};

use crate::FullscreenShader;

pub struct TonemappingPlugin;

impl Plugin for TonemappingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "tonemapping_shared.wgsl");
        load_shader_library!(app, "lut_bindings.wgsl");

        embedded_asset!(app, "tonemapping.wgsl");

        if !app.world().is_resource_added::<TonemappingLuts>() {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();

            #[cfg(feature = "tonemapping_luts")]
            let tonemapping_luts = {
                TonemappingLuts {
                    blender_filmic: images.add(setup_tonemapping_lut_image(
                        include_bytes!("luts/Blender_-11_12.ktx2"),
                        ImageType::Extension("ktx2"),
                    )),
                    agx: images.add(setup_tonemapping_lut_image(
                        include_bytes!("luts/AgX-default_contrast.ktx2"),
                        ImageType::Extension("ktx2"),
                    )),
                    tony_mc_mapface: images.add(setup_tonemapping_lut_image(
                        include_bytes!("luts/tony_mc_mapface.ktx2"),
                        ImageType::Extension("ktx2"),
                    )),
                }
            };

            #[cfg(not(feature = "tonemapping_luts"))]
            let tonemapping_luts = {
                let placeholder = images.add(lut_placeholder());
                TonemappingLuts {
                    blender_filmic: placeholder.clone(),
                    agx: placeholder.clone(),
                    tony_mc_mapface: placeholder,
                }
            };

            app.insert_resource(tonemapping_luts);
        }

        app.add_plugins(ExtractResourcePlugin::<TonemappingLuts>::default());

        app.add_plugins((
            ExtractComponentPlugin::<Tonemapping>::default(),
            ExtractComponentPlugin::<DebandDither>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<TonemappingPipeline>>()
            .add_systems(RenderStartup, init_tonemapping_pipeline)
            .add_systems(
                Render,
                prepare_view_tonemapping_pipelines.in_set(RenderSystems::Prepare),
            );
    }
}

#[derive(Resource)]
pub struct TonemappingPipeline {
    texture_bind_group: BindGroupLayoutDescriptor,
    sampler: Sampler,
    fullscreen_shader: FullscreenShader,
    fragment_shader: Handle<Shader>,
}

bitflags! {
    /// Various flags describing what tonemapping needs to do.
    ///
    /// This allows the shader to skip unneeded steps.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct TonemappingPipelineKeyFlags: u8 {
        /// The hue needs to be changed.
        const HUE_ROTATE                = 0x01;
        /// The white balance needs to be adjusted.
        const WHITE_BALANCE             = 0x02;
        /// Saturation/contrast/gamma/gain/lift for one or more sections
        /// (shadows, midtones, highlights) need to be adjusted.
        const SECTIONAL_COLOR_GRADING   = 0x04;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TonemappingPipelineKey {
    deband_dither: DebandDither,
    tonemapping: Tonemapping,
    flags: TonemappingPipelineKeyFlags,
}

impl SpecializedRenderPipeline for TonemappingPipeline {
    type Key = TonemappingPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        shader_defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_TEXTURE_BINDING_INDEX".into(),
            3,
        ));
        shader_defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_SAMPLER_BINDING_INDEX".into(),
            4,
        ));

        if let DebandDither::Enabled = key.deband_dither {
            shader_defs.push("DEBAND_DITHER".into());
        }

        // Define shader flags depending on the color grading options in use.
        if key.flags.contains(TonemappingPipelineKeyFlags::HUE_ROTATE) {
            shader_defs.push("HUE_ROTATE".into());
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::WHITE_BALANCE)
        {
            shader_defs.push("WHITE_BALANCE".into());
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::SECTIONAL_COLOR_GRADING)
        {
            shader_defs.push("SECTIONAL_COLOR_GRADING".into());
        }

        match key.tonemapping {
            Tonemapping::None => shader_defs.push("TONEMAP_METHOD_NONE".into()),
            Tonemapping::Reinhard => shader_defs.push("TONEMAP_METHOD_REINHARD".into()),
            Tonemapping::ReinhardLuminance => {
                shader_defs.push("TONEMAP_METHOD_REINHARD_LUMINANCE".into());
            }
            Tonemapping::AcesFitted => shader_defs.push("TONEMAP_METHOD_ACES_FITTED".into()),
            Tonemapping::AgX => {
                #[cfg(not(feature = "tonemapping_luts"))]
                error!(
                    "AgX tonemapping requires the `tonemapping_luts` feature.
                    Either enable the `tonemapping_luts` feature for bevy in `Cargo.toml` (recommended),
                    or use a different `Tonemapping` method for your `Camera2d`/`Camera3d`."
                );
                shader_defs.push("TONEMAP_METHOD_AGX".into());
            }
            Tonemapping::SomewhatBoringDisplayTransform => {
                shader_defs.push("TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into());
            }
            Tonemapping::TonyMcMapface => {
                #[cfg(not(feature = "tonemapping_luts"))]
                error!(
                    "TonyMcMapFace tonemapping requires the `tonemapping_luts` feature.
                    Either enable the `tonemapping_luts` feature for bevy in `Cargo.toml` (recommended),
                    or use a different `Tonemapping` method for your `Camera2d`/`Camera3d`."
                );
                shader_defs.push("TONEMAP_METHOD_TONY_MC_MAPFACE".into());
            }
            Tonemapping::BlenderFilmic => {
                #[cfg(not(feature = "tonemapping_luts"))]
                error!(
                    "BlenderFilmic tonemapping requires the `tonemapping_luts` feature.
                    Either enable the `tonemapping_luts` feature for bevy in `Cargo.toml` (recommended),
                    or use a different `Tonemapping` method for your `Camera2d`/`Camera3d`."
                );
                shader_defs.push("TONEMAP_METHOD_BLENDER_FILMIC".into());
            }
        }
        RenderPipelineDescriptor {
            label: Some("tonemapping pipeline".into()),
            layout: vec![self.texture_bind_group.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

pub fn init_tonemapping_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let mut entries = DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::FRAGMENT,
        (
            (0, uniform_buffer::<ViewUniform>(true)),
            (
                1,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
            (2, sampler(SamplerBindingType::NonFiltering)),
        ),
    );
    let lut_layout_entries = get_lut_bind_group_layout_entries();
    entries = entries.extend_with_indices(((3, lut_layout_entries[0]), (4, lut_layout_entries[1])));

    let tonemap_texture_bind_group =
        BindGroupLayoutDescriptor::new("tonemapping_hdr_texture_bind_group_layout", &entries);

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    commands.insert_resource(TonemappingPipeline {
        texture_bind_group: tonemap_texture_bind_group,
        sampler,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: load_embedded_asset!(asset_server.as_ref(), "tonemapping.wgsl"),
    });
}

#[derive(Component)]
pub struct ViewTonemappingPipeline(CachedRenderPipelineId);

pub fn prepare_view_tonemapping_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TonemappingPipeline>>,
    upscaling_pipeline: Res<TonemappingPipeline>,
    view_targets: Query<
        (
            Entity,
            &ExtractedView,
            Option<&Tonemapping>,
            Option<&DebandDither>,
        ),
        With<ViewTarget>,
    >,
) {
    for (entity, view, tonemapping, dither) in view_targets.iter() {
        // As an optimization, we omit parts of the shader that are unneeded.
        let mut flags = TonemappingPipelineKeyFlags::empty();
        flags.set(
            TonemappingPipelineKeyFlags::HUE_ROTATE,
            view.color_grading.global.hue != 0.0,
        );
        flags.set(
            TonemappingPipelineKeyFlags::WHITE_BALANCE,
            view.color_grading.global.temperature != 0.0 || view.color_grading.global.tint != 0.0,
        );
        flags.set(
            TonemappingPipelineKeyFlags::SECTIONAL_COLOR_GRADING,
            view.color_grading
                .all_sections()
                .any(|section| *section != default()),
        );

        let key = TonemappingPipelineKey {
            deband_dither: *dither.unwrap_or(&DebandDither::Disabled),
            tonemapping: *tonemapping.unwrap_or(&Tonemapping::None),
            flags,
        };
        let pipeline = pipelines.specialize(&pipeline_cache, &upscaling_pipeline, key);

        commands
            .entity(entity)
            .insert(ViewTonemappingPipeline(pipeline));
    }
}

#[expect(clippy::allow_attributes, reason = "`dead_code` is not always linted.")]
#[allow(
    dead_code,
    reason = "There is unused code when the `tonemapping_luts` feature is disabled."
)]
fn setup_tonemapping_lut_image(bytes: &[u8], image_type: ImageType) -> Image {
    let image_sampler = ImageSampler::Descriptor(bevy_image::ImageSamplerDescriptor {
        label: Some("Tonemapping LUT sampler".to_string()),
        address_mode_u: bevy_image::ImageAddressMode::ClampToEdge,
        address_mode_v: bevy_image::ImageAddressMode::ClampToEdge,
        address_mode_w: bevy_image::ImageAddressMode::ClampToEdge,
        mag_filter: bevy_image::ImageFilterMode::Linear,
        min_filter: bevy_image::ImageFilterMode::Linear,
        mipmap_filter: bevy_image::ImageFilterMode::Linear,
        ..default()
    });
    Image::from_buffer(
        bytes,
        image_type,
        CompressedImageFormats::NONE,
        false,
        image_sampler,
        // LUT must be kept in main world for render recovery reasons
        RenderAssetUsages::default(),
    )
    .unwrap()
}
