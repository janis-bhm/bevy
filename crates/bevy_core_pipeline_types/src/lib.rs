//! This crate contains types related to the core rendering pipelines.
//! The motivation for this crate is to allow dependents on types within to
//! start compiling before `bevy_core_pipeline` is ready.

mod fullscreen_vertex_shader {
    use bevy_asset::{load_embedded_asset, Handle};
    use bevy_ecs::{resource::Resource, world::FromWorld};
    use bevy_render::render_resource::VertexState;
    use bevy_shader::Shader;

    /// A shader that renders to the whole screen. Useful for post-processing.
    #[derive(Resource, Clone)]
    pub struct FullscreenShader(Handle<Shader>);

    impl FromWorld for FullscreenShader {
        fn from_world(world: &mut bevy_ecs::world::World) -> Self {
            Self(load_embedded_asset!(world, "fullscreen.wgsl"))
        }
    }

    impl FullscreenShader {
        /// Gets the raw shader handle.
        pub fn shader(&self) -> Handle<Shader> {
            self.0.clone()
        }

        /// Creates a [`VertexState`] that uses the [`FullscreenShader`] to output a
        /// ```wgsl
        /// struct FullscreenVertexOutput {
        ///     @builtin(position)
        ///     position: vec4<f32>;
        ///     @location(0)
        ///     uv: vec2<f32>;
        /// };
        /// ```
        /// from the vertex shader.
        /// The draw call should render one triangle: `render_pass.draw(0..3, 0..1);`
        pub fn to_vertex_state(&self) -> VertexState {
            VertexState {
                shader: self.0.clone(),
                shader_defs: Vec::new(),
                entry_point: Some("fullscreen_vertex_shader".into()),
                buffers: Vec::new(),
            }
        }
    }
}

pub use fullscreen_vertex_shader::FullscreenShader;

pub mod blit {
    use bevy_asset::Handle;
    use bevy_camera::CompositingSpace;
    use bevy_ecs::resource::Resource;
    use bevy_render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BlendState, ColorTargetState,
            ColorWrites, FragmentState, MultisampleState, PipelineCache, RenderPipelineDescriptor,
            Sampler, SpecializedRenderPipeline, TextureFormat, TextureView,
        },
        renderer::RenderDevice,
    };
    use bevy_shader::Shader;
    use bevy_utils::default;

    use crate::FullscreenShader;

    #[derive(Resource)]
    pub struct BlitPipeline {
        pub layout: BindGroupLayoutDescriptor,
        pub sampler: Sampler,
        pub fullscreen_shader: FullscreenShader,
        pub fragment_shader: Handle<Shader>,
    }

    impl BlitPipeline {
        pub fn create_bind_group(
            &self,
            render_device: &RenderDevice,
            src_texture: &TextureView,
            pipeline_cache: &PipelineCache,
        ) -> BindGroup {
            render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&self.layout),
                &BindGroupEntries::sequential((src_texture, &self.sampler)),
            )
        }
    }

    #[derive(PartialEq, Eq, Hash, Clone, Copy)]
    pub struct BlitPipelineKey {
        pub texture_format: TextureFormat,
        pub blend_state: Option<BlendState>,
        pub samples: u32,
        /// Color space of the source texture. When `Some(Srgb)` or `Some(Oklab)`, the blit converts
        /// to linear RGB before writing to the output target.
        pub source_space: Option<CompositingSpace>,
    }

    impl SpecializedRenderPipeline for BlitPipeline {
        type Key = BlitPipelineKey;

        fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
            let mut shader_defs = Vec::new();
            match key.source_space {
                Some(CompositingSpace::Srgb) => shader_defs.push("SRGB_TO_LINEAR".into()),
                Some(CompositingSpace::Oklab) => shader_defs.push("OKLAB_TO_LINEAR".into()),
                Some(CompositingSpace::Linear) | None => {}
            }

            RenderPipelineDescriptor {
                label: Some("blit pipeline".into()),
                layout: vec![self.layout.clone()],
                vertex: self.fullscreen_shader.to_vertex_state(),
                fragment: Some(FragmentState {
                    shader: self.fragment_shader.clone(),
                    shader_defs,
                    targets: vec![Some(ColorTargetState {
                        format: key.texture_format,
                        blend: key.blend_state,
                        write_mask: ColorWrites::ALL,
                    })],
                    ..default()
                }),
                multisample: MultisampleState {
                    count: key.samples,
                    ..default()
                },
                ..default()
            }
        }
    }
}

pub mod schedule {
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
    use bevy_ecs::schedule::{IntoScheduleConfigs, Schedule, ScheduleLabel, SystemSet};

    /// Schedule label for the Core 3D rendering pipeline.
    #[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash, Default)]
    pub struct Core3d;

    /// System sets for the Core 3D rendering pipeline, defining the main stages of rendering.
    /// These stages include and run in the following order:
    /// - `Prepass`: Initial rendering operations, such as depth pre-pass.
    /// - `MainPass`: The primary rendering operations, including drawing opaque and transparent objects.
    /// - `EarlyPostProcess`: Initial post processing effects.
    /// - `PostProcess`: Final rendering operations, such as post-processing effects.
    ///
    /// Additional systems can be added to these sets to customize the rendering pipeline, or additional
    /// sets can be created relative to these core sets.
    #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
    pub enum Core3dSystems {
        Prepass,
        MainPass,
        EarlyPostProcess,
        PostProcess,
    }

    impl Core3d {
        pub fn base_schedule() -> Schedule {
            use bevy_ecs::schedule::ScheduleBuildSettings;
            use Core3dSystems::*;

            let mut schedule = Schedule::new(Self);

            schedule.set_build_settings(ScheduleBuildSettings {
                auto_insert_apply_deferred: false,
                ..Default::default()
            });

            schedule.configure_sets((Prepass, MainPass, EarlyPostProcess, PostProcess).chain());

            schedule
        }
    }

    /// Schedule label for the Core 2D rendering pipeline.
    #[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash, Default)]
    pub struct Core2d;

    /// System sets for the Core 2D rendering pipeline, defining the main stages of rendering.
    /// These stages include and run in the following order:
    /// - `Prepass`: Initial rendering operations, such as depth pre-pass.
    /// - `MainPass`: The primary rendering operations, including drawing 2D sprites and meshes.
    /// - `EarlyPostProcess`: Initial post processing effects.
    /// - `PostProcess`: Final rendering operations, such as post-processing effects.
    ///
    /// Additional systems can be added to these sets to customize the rendering pipeline, or additional
    /// sets can be created relative to these core sets.
    #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
    pub enum Core2dSystems {
        Prepass,
        MainPass,
        EarlyPostProcess,
        PostProcess,
    }

    impl Core2d {
        pub fn base_schedule() -> Schedule {
            use bevy_ecs::schedule::ScheduleBuildSettings;
            use Core2dSystems::*;

            let mut schedule = Schedule::new(Self);

            schedule.set_build_settings(ScheduleBuildSettings {
                auto_insert_apply_deferred: false,
                ..Default::default()
            });

            schedule.configure_sets((Prepass, MainPass, EarlyPostProcess, PostProcess).chain());

            schedule
        }
    }
}

pub mod core_2d {
    use core::ops::Range;

    use bevy_asset::UntypedAssetId;
    use bevy_ecs::entity::EntityHash;
    use bevy_ecs::prelude::*;
    use bevy_math::FloatOrd;
    use bevy_render::render_resource::TextureFormat;
    use bevy_render::{render_phase::PhaseItemBatchSetKey, view::ExtractedView};
    use bevy_render::{
        render_phase::{
            BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem,
            PhaseItemExtraIndex, SortedPhaseItem,
        },
        render_resource::{BindGroupId, CachedRenderPipelineId},
        sync_world::MainEntity,
    };
    use indexmap::IndexMap;

    pub const CORE_2D_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

    /// Opaque 2D [`BinnedPhaseItem`]s.
    pub struct Opaque2d {
        /// Determines which objects can be placed into a *batch set*.
        ///
        /// Objects in a single batch set can potentially be multi-drawn together,
        /// if it's enabled and the current platform supports it.
        pub batch_set_key: BatchSetKey2d,
        /// The key, which determines which can be batched.
        pub bin_key: Opaque2dBinKey,
        /// An entity from which data will be fetched, including the mesh if
        /// applicable.
        pub representative_entity: (Entity, MainEntity),
        /// The ranges of instances.
        pub batch_range: Range<u32>,
        /// An extra index, which is either a dynamic offset or an index in the
        /// indirect parameters list.
        pub extra_index: PhaseItemExtraIndex,
    }

    /// Data that must be identical in order to batch phase items together.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Opaque2dBinKey {
        /// The identifier of the render pipeline.
        pub pipeline: CachedRenderPipelineId,
        /// The function used to draw.
        pub draw_function: DrawFunctionId,
        /// The asset that this phase item is associated with.
        ///
        /// Normally, this is the ID of the mesh, but for non-mesh items it might be
        /// the ID of another type of asset.
        pub asset_id: UntypedAssetId,
        /// The ID of a bind group specific to the material.
        pub material_bind_group_id: Option<BindGroupId>,
    }

    impl PhaseItem for Opaque2d {
        #[inline]
        fn entity(&self) -> Entity {
            self.representative_entity.0
        }

        fn main_entity(&self) -> MainEntity {
            self.representative_entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.bin_key.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }
    }

    impl BinnedPhaseItem for Opaque2d {
        // Since 2D meshes presently can't be multidrawn, the batch set key is
        // irrelevant.
        type BatchSetKey = BatchSetKey2d;

        type BinKey = Opaque2dBinKey;

        fn new(
            batch_set_key: Self::BatchSetKey,
            bin_key: Self::BinKey,
            representative_entity: (Entity, MainEntity),
            batch_range: Range<u32>,
            extra_index: PhaseItemExtraIndex,
        ) -> Self {
            Opaque2d {
                batch_set_key,
                bin_key,
                representative_entity,
                batch_range,
                extra_index,
            }
        }
    }

    /// 2D meshes aren't currently multi-drawn together, so this batch set key only
    /// stores whether the mesh is indexed.
    #[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
    pub struct BatchSetKey2d {
        /// True if the mesh is indexed.
        pub indexed: bool,
    }

    impl PhaseItemBatchSetKey for BatchSetKey2d {
        fn indexed(&self) -> bool {
            self.indexed
        }
    }

    impl CachedRenderPipelinePhaseItem for Opaque2d {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.bin_key.pipeline
        }
    }

    /// Alpha mask 2D [`BinnedPhaseItem`]s.
    pub struct AlphaMask2d {
        /// Determines which objects can be placed into a *batch set*.
        ///
        /// Objects in a single batch set can potentially be multi-drawn together,
        /// if it's enabled and the current platform supports it.
        pub batch_set_key: BatchSetKey2d,
        /// The key, which determines which can be batched.
        pub bin_key: AlphaMask2dBinKey,
        /// An entity from which data will be fetched, including the mesh if
        /// applicable.
        pub representative_entity: (Entity, MainEntity),
        /// The ranges of instances.
        pub batch_range: Range<u32>,
        /// An extra index, which is either a dynamic offset or an index in the
        /// indirect parameters list.
        pub extra_index: PhaseItemExtraIndex,
    }

    /// Data that must be identical in order to batch phase items together.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct AlphaMask2dBinKey {
        /// The identifier of the render pipeline.
        pub pipeline: CachedRenderPipelineId,
        /// The function used to draw.
        pub draw_function: DrawFunctionId,
        /// The asset that this phase item is associated with.
        ///
        /// Normally, this is the ID of the mesh, but for non-mesh items it might be
        /// the ID of another type of asset.
        pub asset_id: UntypedAssetId,
        /// The ID of a bind group specific to the material.
        pub material_bind_group_id: Option<BindGroupId>,
    }

    impl PhaseItem for AlphaMask2d {
        #[inline]
        fn entity(&self) -> Entity {
            self.representative_entity.0
        }

        #[inline]
        fn main_entity(&self) -> MainEntity {
            self.representative_entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.bin_key.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }
    }

    impl BinnedPhaseItem for AlphaMask2d {
        // Since 2D meshes presently can't be multidrawn, the batch set key is
        // irrelevant.
        type BatchSetKey = BatchSetKey2d;

        type BinKey = AlphaMask2dBinKey;

        fn new(
            batch_set_key: Self::BatchSetKey,
            bin_key: Self::BinKey,
            representative_entity: (Entity, MainEntity),
            batch_range: Range<u32>,
            extra_index: PhaseItemExtraIndex,
        ) -> Self {
            AlphaMask2d {
                batch_set_key,
                bin_key,
                representative_entity,
                batch_range,
                extra_index,
            }
        }
    }

    impl CachedRenderPipelinePhaseItem for AlphaMask2d {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.bin_key.pipeline
        }
    }

    /// Transparent 2D [`SortedPhaseItem`]s.
    pub struct Transparent2d {
        pub sort_key: FloatOrd,
        pub entity: (Entity, MainEntity),
        pub pipeline: CachedRenderPipelineId,
        pub draw_function: DrawFunctionId,
        pub batch_range: Range<u32>,
        pub extracted_index: usize,
        pub extra_index: PhaseItemExtraIndex,
        /// Whether the mesh in question is indexed (uses an index buffer in
        /// addition to its vertex buffer).
        pub indexed: bool,
    }

    impl PhaseItem for Transparent2d {
        #[inline]
        fn entity(&self) -> Entity {
            self.entity.0
        }

        #[inline]
        fn main_entity(&self) -> MainEntity {
            self.entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        #[inline]
        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        #[inline]
        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }
    }

    impl SortedPhaseItem for Transparent2d {
        type SortKey = FloatOrd;

        #[inline]
        fn sort_key(&self) -> Self::SortKey {
            self.sort_key
        }

        #[inline]
        fn sort(items: &mut IndexMap<(Entity, MainEntity), Transparent2d, EntityHash>) {
            items.sort_by_key(|_, item| item.sort_key());
        }

        fn recalculate_sort_keys(
            _: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
            _: &ExtractedView,
        ) {
            // Sort keys are precalculated for 2D phase items.
        }

        fn indexed(&self) -> bool {
            self.indexed
        }
    }

    impl CachedRenderPipelinePhaseItem for Transparent2d {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.pipeline
        }
    }
}

pub mod core_3d {
    use bevy_math::{FloatOrd, Vec3};
    use core::ops::Range;
    use indexmap::IndexMap;

    use bevy_asset::UntypedAssetId;
    use bevy_ecs::entity::{Entity, EntityHash};
    use bevy_render::{
        mesh::allocator::MeshSlabs,
        render_phase::{
            BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem,
            PhaseItemBatchSetKey, PhaseItemExtraIndex, SortedPhaseItem, ViewRangefinder3d,
        },
        render_resource::{CachedRenderPipelineId, TextureFormat},
        sync_world::MainEntity,
        view::ExtractedView,
    };
    use nonmax::NonMaxU32;

    use crate::prepass::{OpaqueNoLightmap3dBatchSetKey, OpaqueNoLightmap3dBinKey};

    // PERF: vulkan docs recommend using 24 bit depth for better performance
    pub const CORE_3D_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

    /// True if multisampled depth textures are supported on this platform.
    ///
    /// In theory, Naga supports depth textures on WebGL 2. In practice, it doesn't,
    /// because of a silly bug whereby Naga assumes that all depth textures are
    /// `sampler2DShadow` and will cheerfully generate invalid GLSL that tries to
    /// perform non-percentage-closer-filtering with such a sampler. Therefore we
    /// disable depth of field and screen space reflections entirely on WebGL 2.
    #[cfg(not(any(feature = "webgpu", not(target_arch = "wasm32"))))]
    pub const DEPTH_TEXTURE_SAMPLING_SUPPORTED: bool = false;

    /// True if multisampled depth textures are supported on this platform.
    ///
    /// In theory, Naga supports depth textures on WebGL 2. In practice, it doesn't,
    /// because of a silly bug whereby Naga assumes that all depth textures are
    /// `sampler2DShadow` and will cheerfully generate invalid GLSL that tries to
    /// perform non-percentage-closer-filtering with such a sampler. Therefore we
    /// disable depth of field and screen space reflections entirely on WebGL 2.
    #[cfg(any(feature = "webgpu", not(target_arch = "wasm32")))]
    pub const DEPTH_TEXTURE_SAMPLING_SUPPORTED: bool = true;

    /// Opaque 3D [`BinnedPhaseItem`]s.
    pub struct Opaque3d {
        /// Determines which objects can be placed into a *batch set*.
        ///
        /// Objects in a single batch set can potentially be multi-drawn together,
        /// if it's enabled and the current platform supports it.
        pub batch_set_key: Opaque3dBatchSetKey,
        /// The key, which determines which can be batched.
        pub bin_key: Opaque3dBinKey,
        /// An entity from which data will be fetched, including the mesh if
        /// applicable.
        pub representative_entity: (Entity, MainEntity),
        /// The ranges of instances.
        pub batch_range: Range<u32>,
        /// An extra index, which is either a dynamic offset or an index in the
        /// indirect parameters list.
        pub extra_index: PhaseItemExtraIndex,
    }

    /// Information that must be identical in order to place opaque meshes in the
    /// same *batch set*.
    ///
    /// A batch set is a set of batches that can be multi-drawn together, if
    /// multi-draw is in use.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Opaque3dBatchSetKey {
        /// The identifier of the render pipeline.
        pub pipeline: CachedRenderPipelineId,

        /// The function used to draw.
        pub draw_function: DrawFunctionId,

        /// The ID of a bind group specific to the material instance.
        ///
        /// In the case of PBR, this is the `MaterialBindGroupIndex`.
        pub material_bind_group_index: Option<u32>,

        /// The IDs of the slabs of GPU memory in the mesh allocator that contain
        /// the mesh data.
        ///
        /// For non-mesh items, you can fill the [`MeshSlabs::vertex_slab_id`] with
        /// 0 if your items can be multi-drawn, or with a unique value if they
        /// can't.
        pub slabs: MeshSlabs,

        /// Index of the slab that the lightmap resides in, if a lightmap is
        /// present.
        pub lightmap_slab: Option<NonMaxU32>,
    }

    impl PhaseItemBatchSetKey for Opaque3dBatchSetKey {
        fn indexed(&self) -> bool {
            self.slabs.index_slab_id.is_some()
        }
    }

    /// Data that must be identical in order to *batch* phase items together.
    ///
    /// Note that a *batch set* (if multi-draw is in use) contains multiple batches.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Opaque3dBinKey {
        /// The asset that this phase item is associated with.
        ///
        /// Normally, this is the ID of the mesh, but for non-mesh items it might be
        /// the ID of another type of asset.
        pub asset_id: UntypedAssetId,
    }

    impl PhaseItem for Opaque3d {
        #[inline]
        fn entity(&self) -> Entity {
            self.representative_entity.0
        }

        #[inline]
        fn main_entity(&self) -> MainEntity {
            self.representative_entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.batch_set_key.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }

        const AUTOMATIC_BATCHING: bool = true;
    }

    impl BinnedPhaseItem for Opaque3d {
        type BatchSetKey = Opaque3dBatchSetKey;
        type BinKey = Opaque3dBinKey;

        #[inline]
        fn new(
            batch_set_key: Self::BatchSetKey,
            bin_key: Self::BinKey,
            representative_entity: (Entity, MainEntity),
            batch_range: Range<u32>,
            extra_index: PhaseItemExtraIndex,
        ) -> Self {
            Opaque3d {
                batch_set_key,
                bin_key,
                representative_entity,
                batch_range,
                extra_index,
            }
        }
    }

    impl CachedRenderPipelinePhaseItem for Opaque3d {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.batch_set_key.pipeline
        }
    }

    pub struct AlphaMask3d {
        /// Determines which objects can be placed into a *batch set*.
        ///
        /// Objects in a single batch set can potentially be multi-drawn together,
        /// if it's enabled and the current platform supports it.
        pub batch_set_key: OpaqueNoLightmap3dBatchSetKey,
        /// The key, which determines which can be batched.
        pub bin_key: OpaqueNoLightmap3dBinKey,
        pub representative_entity: (Entity, MainEntity),
        pub batch_range: Range<u32>,
        pub extra_index: PhaseItemExtraIndex,
    }

    impl PhaseItem for AlphaMask3d {
        #[inline]
        fn entity(&self) -> Entity {
            self.representative_entity.0
        }

        fn main_entity(&self) -> MainEntity {
            self.representative_entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.batch_set_key.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        #[inline]
        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        #[inline]
        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }
    }

    impl BinnedPhaseItem for AlphaMask3d {
        type BinKey = OpaqueNoLightmap3dBinKey;
        type BatchSetKey = OpaqueNoLightmap3dBatchSetKey;

        #[inline]
        fn new(
            batch_set_key: Self::BatchSetKey,
            bin_key: Self::BinKey,
            representative_entity: (Entity, MainEntity),
            batch_range: Range<u32>,
            extra_index: PhaseItemExtraIndex,
        ) -> Self {
            Self {
                batch_set_key,
                bin_key,
                representative_entity,
                batch_range,
                extra_index,
            }
        }
    }

    impl CachedRenderPipelinePhaseItem for AlphaMask3d {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.batch_set_key.pipeline
        }
    }

    pub struct Transparent3d {
        pub sorting_info: TransparentSortingInfo3d,
        pub distance: f32,
        pub pipeline: CachedRenderPipelineId,
        pub entity: (Entity, MainEntity),
        pub draw_function: DrawFunctionId,
        pub batch_range: Range<u32>,
        pub extra_index: PhaseItemExtraIndex,
        /// Whether the mesh in question is indexed (uses an index buffer in
        /// addition to its vertex buffer).
        pub indexed: bool,
    }

    impl PhaseItem for Transparent3d {
        #[inline]
        fn entity(&self) -> Entity {
            self.entity.0
        }

        fn main_entity(&self) -> MainEntity {
            self.entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        #[inline]
        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        #[inline]
        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }
    }

    impl SortedPhaseItem for Transparent3d {
        // NOTE: Values increase towards the camera. Back-to-front ordering for transparent means we need an ascending sort.
        type SortKey = FloatOrd;

        #[inline]
        fn sort_key(&self) -> Self::SortKey {
            FloatOrd(self.distance)
        }

        #[inline]
        fn sort(items: &mut IndexMap<(Entity, MainEntity), Transparent3d, EntityHash>) {
            items.sort_by_key(|_, item| item.sort_key());
        }

        fn recalculate_sort_keys(
            items: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
            view: &ExtractedView,
        ) {
            // Determine the distance to the view for each phase item.
            let rangefinder = view.rangefinder3d();
            for item in items.values_mut() {
                item.distance = item.sorting_info.sort_distance(&rangefinder);
            }
        }

        #[inline]
        fn indexed(&self) -> bool {
            self.indexed
        }
    }

    impl CachedRenderPipelinePhaseItem for Transparent3d {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.pipeline
        }
    }

    /// Information needed to perform a depth sort.
    #[derive(Clone, Copy)]
    pub enum TransparentSortingInfo3d {
        /// No information is needed because this object should always appear on top
        /// of other objects.
        AlwaysOnTop,
        /// Information needed to sort the object based on distance to the view.
        Sorted {
            /// The center of the mesh.
            ///
            /// This is the point that is used to sort.
            mesh_center: Vec3,
            /// An additional value that's artificially added to the distance before
            /// sorting.
            depth_bias: f32,
        },
    }

    impl TransparentSortingInfo3d {
        /// Calculates the value used for distance sorting for an item.
        /// For [`Self::AlwaysOnTop`], this is [`f32::NEG_INFINITY`].
        pub fn sort_distance(&self, rangefinder: &ViewRangefinder3d) -> f32 {
            match *self {
                TransparentSortingInfo3d::AlwaysOnTop => f32::NEG_INFINITY,
                TransparentSortingInfo3d::Sorted {
                    mesh_center,
                    depth_bias,
                } => rangefinder.distance(&mesh_center) + depth_bias,
            }
        }
    }
}

pub mod prepass {
    use core::ops::Range;

    use crate::deferred::{DEFERRED_LIGHTING_PASS_ID_FORMAT, DEFERRED_PREPASS_FORMAT};
    use bevy_asset::UntypedAssetId;
    use bevy_ecs::prelude::*;
    use bevy_math::Mat4;
    use bevy_reflect::{std_traits::ReflectDefault, Reflect};
    use bevy_render::mesh::allocator::MeshSlabs;
    use bevy_render::render_phase::PhaseItemBatchSetKey;
    use bevy_render::sync_world::MainEntity;
    use bevy_render::{
        render_phase::{
            BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem,
            PhaseItemExtraIndex,
        },
        render_resource::{
            CachedRenderPipelineId, ColorTargetState, ColorWrites, DynamicUniformBuffer, Extent3d,
            ShaderType, TextureFormat, TextureView,
        },
        texture::ColorAttachment,
    };

    pub const NORMAL_PREPASS_FORMAT: TextureFormat = TextureFormat::Rgb10a2Unorm;
    pub const MOTION_VECTOR_PREPASS_FORMAT: TextureFormat = TextureFormat::Rg16Float;

    /// If added to a [`bevy_camera::Camera3d`] then depth values will be copied to a separate texture available to the main pass.
    #[derive(Component, Default, Reflect, Clone)]
    #[reflect(Component, Default, Clone)]
    pub struct DepthPrepass;

    /// If added to a [`bevy_camera::Camera3d`] then vertex world normals will be copied to a separate texture available to the main pass.
    /// Normals will have normal map textures already applied.
    #[derive(Component, Default, Reflect, Clone)]
    #[reflect(Component, Default, Clone)]
    pub struct NormalPrepass;

    /// If added to a [`bevy_camera::Camera3d`] then screen space motion vectors will be copied to a separate texture available to the main pass.
    ///
    /// Motion vectors are stored in the range -1,1, with +x right and +y down.
    /// A value of (1.0,1.0) indicates a pixel moved from the top left corner to the bottom right corner of the screen.
    #[derive(Component, Default, Reflect, Clone)]
    #[reflect(Component, Default, Clone)]
    pub struct MotionVectorPrepass;

    /// If added to a [`bevy_camera::Camera3d`] then deferred materials will be rendered to the deferred gbuffer texture and will be available to subsequent passes.
    /// Note the default deferred lighting plugin also requires `DepthPrepass` to work correctly.
    #[derive(Component, Default, Reflect)]
    #[reflect(Component, Default)]
    pub struct DeferredPrepass;

    /// Allows querying the previous frame's [`DepthPrepass`].
    #[derive(Component, Default, Reflect, Clone)]
    #[reflect(Component, Default, Clone)]
    #[require(DepthPrepass)]
    pub struct DepthPrepassDoubleBuffer;

    /// Allows querying the previous frame's [`DeferredPrepass`].
    #[derive(Component, Default, Reflect, Clone)]
    #[reflect(Component, Default, Clone)]
    #[require(DeferredPrepass)]
    pub struct DeferredPrepassDoubleBuffer;

    /// View matrices from the previous frame.
    ///
    /// Useful for temporal rendering techniques that need access to last frame's camera data.
    #[derive(Component, ShaderType, Clone)]
    pub struct PreviousViewData {
        pub view_from_world: Mat4,
        pub clip_from_world: Mat4,
        pub clip_from_view: Mat4,
        pub world_from_clip: Mat4,
        pub view_from_clip: Mat4,
    }

    #[derive(Resource, Default)]
    pub struct PreviousViewUniforms {
        pub uniforms: DynamicUniformBuffer<PreviousViewData>,
    }

    #[derive(Component)]
    pub struct PreviousViewUniformOffset {
        pub offset: u32,
    }

    /// Textures that are written to by the prepass.
    ///
    /// This component will only be present if any of the relevant prepass components are also present.
    #[derive(Component)]
    pub struct ViewPrepassTextures {
        /// The depth texture generated by the prepass.
        /// Exists only if [`DepthPrepass`] is added to the [`ViewTarget`](bevy_render::view::ViewTarget)
        pub depth: Option<ColorAttachment>,
        /// The normals texture generated by the prepass.
        /// Exists only if [`NormalPrepass`] is added to the [`ViewTarget`](bevy_render::view::ViewTarget)
        pub normal: Option<ColorAttachment>,
        /// The motion vectors texture generated by the prepass.
        /// Exists only if [`MotionVectorPrepass`] is added to the `ViewTarget`
        pub motion_vectors: Option<ColorAttachment>,
        /// The deferred gbuffer generated by the deferred pass.
        /// Exists only if [`DeferredPrepass`] is added to the `ViewTarget`
        pub deferred: Option<ColorAttachment>,
        /// A texture that specifies the deferred lighting pass id for a material.
        /// Exists only if [`DeferredPrepass`] is added to the `ViewTarget`
        pub deferred_lighting_pass_id: Option<ColorAttachment>,
        /// The size of the textures.
        pub size: Extent3d,
    }

    impl ViewPrepassTextures {
        pub fn depth_view(&self) -> Option<&TextureView> {
            self.depth.as_ref().map(|t| &t.texture.default_view)
        }

        pub fn previous_depth_view(&self) -> Option<&TextureView> {
            self.depth
                .as_ref()
                .and_then(|t| t.previous_frame_texture.as_ref().map(|t| &t.default_view))
        }

        pub fn normal_view(&self) -> Option<&TextureView> {
            self.normal.as_ref().map(|t| &t.texture.default_view)
        }

        pub fn motion_vectors_view(&self) -> Option<&TextureView> {
            self.motion_vectors
                .as_ref()
                .map(|t| &t.texture.default_view)
        }

        pub fn deferred_view(&self) -> Option<&TextureView> {
            self.deferred.as_ref().map(|t| &t.texture.default_view)
        }

        pub fn previous_deferred_view(&self) -> Option<&TextureView> {
            self.deferred
                .as_ref()
                .and_then(|t| t.previous_frame_texture.as_ref().map(|t| &t.default_view))
        }
    }

    /// Opaque phase of the 3D prepass.
    ///
    /// Sorted by pipeline, then by mesh to improve batching.
    ///
    /// Used to render all 3D meshes with materials that have no transparency.
    pub struct Opaque3dPrepass {
        /// Determines which objects can be placed into a *batch set*.
        ///
        /// Objects in a single batch set can potentially be multi-drawn together,
        /// if it's enabled and the current platform supports it.
        pub batch_set_key: OpaqueNoLightmap3dBatchSetKey,
        /// Information that separates items into bins.
        pub bin_key: OpaqueNoLightmap3dBinKey,

        /// An entity from which Bevy fetches data common to all instances in this
        /// batch, such as the mesh.
        pub representative_entity: (Entity, MainEntity),
        pub batch_range: Range<u32>,
        pub extra_index: PhaseItemExtraIndex,
    }

    /// Information that must be identical in order to place opaque meshes in the
    /// same *batch set* in the prepass and deferred pass.
    ///
    /// A batch set is a set of batches that can be multi-drawn together, if
    /// multi-draw is in use.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct OpaqueNoLightmap3dBatchSetKey {
        /// The ID of the GPU pipeline.
        pub pipeline: CachedRenderPipelineId,

        /// The function used to draw the mesh.
        pub draw_function: DrawFunctionId,

        /// The ID of a bind group specific to the material.
        ///
        /// In the case of PBR, this is the `MaterialBindGroupIndex`.
        pub material_bind_group_index: Option<u32>,

        /// The IDs of the slabs of GPU memory in the mesh allocator that contain
        /// the mesh data.
        ///
        /// For non-mesh items, you can fill the [`MeshSlabs::vertex_slab_id`] with
        /// 0 if your items can be multi-drawn, or with a unique value if they
        /// can't.
        pub slabs: MeshSlabs,
    }

    impl PhaseItemBatchSetKey for OpaqueNoLightmap3dBatchSetKey {
        fn indexed(&self) -> bool {
            self.slabs.index_slab_id.is_some()
        }
    }

    // TODO: Try interning these.
    /// The data used to bin each opaque 3D object in the prepass and deferred pass.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct OpaqueNoLightmap3dBinKey {
        /// The ID of the asset.
        pub asset_id: UntypedAssetId,
    }

    impl PhaseItem for Opaque3dPrepass {
        #[inline]
        fn entity(&self) -> Entity {
            self.representative_entity.0
        }

        fn main_entity(&self) -> MainEntity {
            self.representative_entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.batch_set_key.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        #[inline]
        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        #[inline]
        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }
    }

    impl BinnedPhaseItem for Opaque3dPrepass {
        type BatchSetKey = OpaqueNoLightmap3dBatchSetKey;
        type BinKey = OpaqueNoLightmap3dBinKey;

        #[inline]
        fn new(
            batch_set_key: Self::BatchSetKey,
            bin_key: Self::BinKey,
            representative_entity: (Entity, MainEntity),
            batch_range: Range<u32>,
            extra_index: PhaseItemExtraIndex,
        ) -> Self {
            Opaque3dPrepass {
                batch_set_key,
                bin_key,
                representative_entity,
                batch_range,
                extra_index,
            }
        }
    }

    impl CachedRenderPipelinePhaseItem for Opaque3dPrepass {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.batch_set_key.pipeline
        }
    }

    /// Alpha mask phase of the 3D prepass.
    ///
    /// Sorted by pipeline, then by mesh to improve batching.
    ///
    /// Used to render all meshes with a material with an alpha mask.
    pub struct AlphaMask3dPrepass {
        /// Determines which objects can be placed into a *batch set*.
        ///
        /// Objects in a single batch set can potentially be multi-drawn together,
        /// if it's enabled and the current platform supports it.
        pub batch_set_key: OpaqueNoLightmap3dBatchSetKey,
        /// Information that separates items into bins.
        pub bin_key: OpaqueNoLightmap3dBinKey,
        pub representative_entity: (Entity, MainEntity),
        pub batch_range: Range<u32>,
        pub extra_index: PhaseItemExtraIndex,
    }

    impl PhaseItem for AlphaMask3dPrepass {
        #[inline]
        fn entity(&self) -> Entity {
            self.representative_entity.0
        }

        fn main_entity(&self) -> MainEntity {
            self.representative_entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.batch_set_key.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        #[inline]
        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        #[inline]
        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }
    }

    impl BinnedPhaseItem for AlphaMask3dPrepass {
        type BatchSetKey = OpaqueNoLightmap3dBatchSetKey;
        type BinKey = OpaqueNoLightmap3dBinKey;

        #[inline]
        fn new(
            batch_set_key: Self::BatchSetKey,
            bin_key: Self::BinKey,
            representative_entity: (Entity, MainEntity),
            batch_range: Range<u32>,
            extra_index: PhaseItemExtraIndex,
        ) -> Self {
            Self {
                batch_set_key,
                bin_key,
                representative_entity,
                batch_range,
                extra_index,
            }
        }
    }

    impl CachedRenderPipelinePhaseItem for AlphaMask3dPrepass {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.batch_set_key.pipeline
        }
    }

    pub fn prepass_target_descriptors(
        normal_prepass: bool,
        motion_vector_prepass: bool,
        deferred_prepass: bool,
    ) -> Vec<Option<ColorTargetState>> {
        vec![
            normal_prepass.then_some(ColorTargetState {
                format: NORMAL_PREPASS_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            }),
            motion_vector_prepass.then_some(ColorTargetState {
                format: MOTION_VECTOR_PREPASS_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            }),
            deferred_prepass.then_some(ColorTargetState {
                format: DEFERRED_PREPASS_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            }),
            deferred_prepass.then_some(ColorTargetState {
                format: DEFERRED_LIGHTING_PASS_ID_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            }),
        ]
    }
}

pub mod deferred {
    pub mod copy_lighting_id {
        use bevy_ecs::component::Component;
        use bevy_render::texture::CachedTexture;

        #[derive(Component)]
        pub struct DeferredLightingIdDepthTexture {
            pub texture: CachedTexture,
        }
    }

    use core::ops::Range;

    use crate::prepass::{OpaqueNoLightmap3dBatchSetKey, OpaqueNoLightmap3dBinKey};
    use bevy_ecs::prelude::*;
    use bevy_render::sync_world::MainEntity;
    use bevy_render::{
        render_phase::{
            BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem,
            PhaseItemExtraIndex,
        },
        render_resource::{CachedRenderPipelineId, TextureFormat},
    };

    pub const DEFERRED_PREPASS_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;
    pub const DEFERRED_LIGHTING_PASS_ID_FORMAT: TextureFormat = TextureFormat::R8Uint;
    pub const DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth16Unorm;

    /// Opaque phase of the 3D Deferred pass.
    ///
    /// Sorted by pipeline, then by mesh to improve batching.
    ///
    /// Used to render all 3D meshes with materials that have no transparency.
    #[derive(PartialEq, Eq, Hash)]
    pub struct Opaque3dDeferred {
        /// Determines which objects can be placed into a *batch set*.
        ///
        /// Objects in a single batch set can potentially be multi-drawn together,
        /// if it's enabled and the current platform supports it.
        pub batch_set_key: OpaqueNoLightmap3dBatchSetKey,
        /// Information that separates items into bins.
        pub bin_key: OpaqueNoLightmap3dBinKey,
        pub representative_entity: (Entity, MainEntity),
        pub batch_range: Range<u32>,
        pub extra_index: PhaseItemExtraIndex,
    }

    impl PhaseItem for Opaque3dDeferred {
        #[inline]
        fn entity(&self) -> Entity {
            self.representative_entity.0
        }

        fn main_entity(&self) -> MainEntity {
            self.representative_entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.batch_set_key.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        #[inline]
        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        #[inline]
        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }
    }

    impl BinnedPhaseItem for Opaque3dDeferred {
        type BatchSetKey = OpaqueNoLightmap3dBatchSetKey;
        type BinKey = OpaqueNoLightmap3dBinKey;

        #[inline]
        fn new(
            batch_set_key: Self::BatchSetKey,
            bin_key: Self::BinKey,
            representative_entity: (Entity, MainEntity),
            batch_range: Range<u32>,
            extra_index: PhaseItemExtraIndex,
        ) -> Self {
            Self {
                batch_set_key,
                bin_key,
                representative_entity,
                batch_range,
                extra_index,
            }
        }
    }

    impl CachedRenderPipelinePhaseItem for Opaque3dDeferred {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.batch_set_key.pipeline
        }
    }

    /// Alpha mask phase of the 3D Deferred pass.
    ///
    /// Sorted by pipeline, then by mesh to improve batching.
    ///
    /// Used to render all meshes with a material with an alpha mask.
    pub struct AlphaMask3dDeferred {
        /// Determines which objects can be placed into a *batch set*.
        ///
        /// Objects in a single batch set can potentially be multi-drawn together,
        /// if it's enabled and the current platform supports it.
        pub batch_set_key: OpaqueNoLightmap3dBatchSetKey,
        /// Information that separates items into bins.
        pub bin_key: OpaqueNoLightmap3dBinKey,
        pub representative_entity: (Entity, MainEntity),
        pub batch_range: Range<u32>,
        pub extra_index: PhaseItemExtraIndex,
    }

    impl PhaseItem for AlphaMask3dDeferred {
        #[inline]
        fn entity(&self) -> Entity {
            self.representative_entity.0
        }

        #[inline]
        fn main_entity(&self) -> MainEntity {
            self.representative_entity.1
        }

        #[inline]
        fn draw_function(&self) -> DrawFunctionId {
            self.batch_set_key.draw_function
        }

        #[inline]
        fn batch_range(&self) -> &Range<u32> {
            &self.batch_range
        }

        #[inline]
        fn batch_range_mut(&mut self) -> &mut Range<u32> {
            &mut self.batch_range
        }

        #[inline]
        fn extra_index(&self) -> PhaseItemExtraIndex {
            self.extra_index.clone()
        }

        #[inline]
        fn batch_range_and_extra_index_mut(
            &mut self,
        ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
            (&mut self.batch_range, &mut self.extra_index)
        }
    }

    impl BinnedPhaseItem for AlphaMask3dDeferred {
        type BatchSetKey = OpaqueNoLightmap3dBatchSetKey;
        type BinKey = OpaqueNoLightmap3dBinKey;

        fn new(
            batch_set_key: Self::BatchSetKey,
            bin_key: Self::BinKey,
            representative_entity: (Entity, MainEntity),
            batch_range: Range<u32>,
            extra_index: PhaseItemExtraIndex,
        ) -> Self {
            Self {
                batch_set_key,
                bin_key,
                representative_entity,
                batch_range,
                extra_index,
            }
        }
    }

    impl CachedRenderPipelinePhaseItem for AlphaMask3dDeferred {
        #[inline]
        fn cached_pipeline(&self) -> CachedRenderPipelineId {
            self.batch_set_key.pipeline
        }
    }
}

pub mod mip_generation {
    use bevy_asset::Handle;
    use bevy_ecs::resource::Resource;
    use bevy_math::Vec2;
    use bevy_platform::collections::HashMap;
    use bevy_render::{
        render_resource::{ShaderType, TextureFormat, TextureFormatFeatureFlags},
        renderer::{RenderAdapter, RenderDevice},
    };
    use bevy_shader::Shader;

    pub mod experimental {
        pub mod depth {
            use core::array;

            use bevy_ecs::component::Component;
            use bevy_math::{uvec2, UVec2};
            use bevy_render::{
                render_resource::{
                    BindGroup, BindGroupEntries, BindGroupLayout, ComputePassDescriptor,
                    ComputePipeline, Extent3d, IntoBinding, Sampler, TextureAspect,
                    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
                    TextureViewDescriptor, TextureViewDimension,
                },
                renderer::{RenderContext, RenderDevice},
                texture::TextureCache,
            };

            /// The maximum number of mip levels that we can produce.
            ///
            /// 2^12 is 4096, so that's the maximum size of the depth buffer that we
            /// support.
            pub const DEPTH_PYRAMID_MIP_COUNT: usize = 12;

            /// Creates a placeholder texture that can be bound to a depth pyramid binding
            /// if no depth pyramid is needed.
            pub fn create_depth_pyramid_dummy_texture(
                render_device: &RenderDevice,
                texture_label: &'static str,
                texture_view_label: &'static str,
            ) -> TextureView {
                render_device
                    .create_texture(&TextureDescriptor {
                        label: Some(texture_label),
                        size: Extent3d::default(),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: TextureFormat::R32Float,
                        usage: TextureUsages::STORAGE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&TextureViewDescriptor {
                        label: Some(texture_view_label),
                        format: Some(TextureFormat::R32Float),
                        dimension: Some(TextureViewDimension::D2),
                        usage: None,
                        aspect: TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    })
            }

            /// Stores a hierarchical Z-buffer for a view, which is a series of mipmaps
            /// useful for efficient occlusion culling.
            ///
            /// This will only be present on a view when occlusion culling is enabled.
            #[derive(Component)]
            pub struct ViewDepthPyramid {
                /// A texture view containing the entire depth texture.
                pub all_mips: TextureView,
                /// A series of texture views containing one mip level each.
                pub mips: [TextureView; DEPTH_PYRAMID_MIP_COUNT],
                /// The total number of mipmap levels.
                ///
                /// This is the base-2 logarithm of the greatest dimension of the depth
                /// buffer, rounded up.
                pub mip_count: u32,
            }

            /// Returns the previous power of two of x, or, if x is exactly a power of two,
            /// returns x unchanged.
            fn previous_power_of_two(x: u32) -> u32 {
                1 << (31 - x.leading_zeros())
            }

            impl ViewDepthPyramid {
                /// Allocates a new depth pyramid for a depth buffer with the given size.
                pub fn new(
                    render_device: &RenderDevice,
                    texture_cache: &mut TextureCache,
                    depth_pyramid_dummy_texture: &TextureView,
                    size: UVec2,
                    texture_label: &'static str,
                    texture_view_label: &'static str,
                ) -> ViewDepthPyramid {
                    // Calculate the size of the depth pyramid. This is the size of the
                    // depth buffer rounded down to the previous power of two.
                    let depth_pyramid_size = Extent3d {
                        width: previous_power_of_two(size.x),
                        height: previous_power_of_two(size.y),
                        depth_or_array_layers: 1,
                    };

                    // Calculate the number of mip levels we need.
                    let depth_pyramid_mip_count = depth_pyramid_size.max_mips(TextureDimension::D2);

                    // Create the depth pyramid.
                    let depth_pyramid = texture_cache.get(
                        render_device,
                        TextureDescriptor {
                            label: Some(texture_label),
                            size: depth_pyramid_size,
                            mip_level_count: depth_pyramid_mip_count,
                            sample_count: 1,
                            dimension: TextureDimension::D2,
                            format: TextureFormat::R32Float,
                            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        },
                    );

                    // Create individual views for each level of the depth pyramid.
                    let depth_pyramid_mips = array::from_fn(|i| {
                        if (i as u32) < depth_pyramid_mip_count {
                            depth_pyramid.texture.create_view(&TextureViewDescriptor {
                                label: Some(texture_view_label),
                                format: Some(TextureFormat::R32Float),
                                dimension: Some(TextureViewDimension::D2),
                                usage: None,
                                aspect: TextureAspect::All,
                                base_mip_level: i as u32,
                                mip_level_count: Some(1),
                                base_array_layer: 0,
                                array_layer_count: Some(1),
                            })
                        } else {
                            (*depth_pyramid_dummy_texture).clone()
                        }
                    });

                    // Create the view for the depth pyramid as a whole.
                    let depth_pyramid_all_mips = depth_pyramid.default_view.clone();

                    Self {
                        all_mips: depth_pyramid_all_mips,
                        mips: depth_pyramid_mips,
                        mip_count: depth_pyramid_mip_count,
                    }
                }

                /// Creates a bind group that allows the depth buffer to be attached to the
                /// `downsample_depth.wgsl` shader.
                pub fn create_bind_group<'a, R>(
                    &'a self,
                    render_device: &RenderDevice,
                    label: &'static str,
                    bind_group_layout: &BindGroupLayout,
                    source_image: R,
                    sampler: &'a Sampler,
                ) -> BindGroup
                where
                    R: IntoBinding<'a>,
                {
                    render_device.create_bind_group(
                        label,
                        bind_group_layout,
                        &BindGroupEntries::sequential((
                            source_image,
                            &self.mips[0],
                            &self.mips[1],
                            &self.mips[2],
                            &self.mips[3],
                            &self.mips[4],
                            &self.mips[5],
                            &self.mips[6],
                            &self.mips[7],
                            &self.mips[8],
                            &self.mips[9],
                            &self.mips[10],
                            &self.mips[11],
                            sampler,
                        )),
                    )
                }

                pub fn downsample_depth_with_ctx(
                    &self,
                    label: &str,
                    ctx: &mut RenderContext,
                    view_size: UVec2,
                    downsample_depth_bind_group: &BindGroup,
                    downsample_depth_first_pipeline: &ComputePipeline,
                    downsample_depth_second_pipeline: &ComputePipeline,
                ) {
                    // We need to make sure that every mip level the single-pass
                    // downsampling (SPD) shader sees has lengths that are powers of two for
                    // correct conservative depth buffer downsampling. To do this, we
                    // maintain the fiction that we're downsampling a depth buffer scaled up
                    // so that it has side lengths rounded up to the next power of two. (If
                    // the depth buffer already has a side length that's a power of two,
                    // then we double it anyway; this ensures that we don't lose any
                    // precision in the top level of the depth pyramid.) The
                    // `downsample_depth` shader's `load_mip_0` function returns the value
                    // that sampling such a depth buffer would yield, without actually
                    // having to construct such a scaled depth buffer.
                    let virtual_view_size = uvec2(
                        (view_size.x + 1).next_power_of_two(),
                        (view_size.y + 1).next_power_of_two(),
                    );

                    let command_encoder = ctx.command_encoder();
                    let mut downsample_pass =
                        command_encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: Some(label),
                            timestamp_writes: None,
                        });
                    downsample_pass.set_pipeline(downsample_depth_first_pipeline);
                    // Pass the mip count as an immediate, for simplicity.
                    downsample_pass.set_immediates(0, &self.mip_count.to_le_bytes());
                    downsample_pass.set_bind_group(0, downsample_depth_bind_group, &[]);
                    downsample_pass.dispatch_workgroups(
                        virtual_view_size.x.div_ceil(64),
                        virtual_view_size.y.div_ceil(64),
                        1,
                    );

                    if self.mip_count >= 7 {
                        downsample_pass.set_pipeline(downsample_depth_second_pipeline);
                        downsample_pass.dispatch_workgroups(1, 1, 1);
                    }
                }
            }
        }
    }

    /// A resource that stores the shaders that perform downsampling.
    #[derive(Clone, Resource)]
    pub struct DownsampleShaders {
        /// The experimental shader that downsamples depth
        /// (`downsample_depth.wgsl`).
        pub depth: Handle<Shader>,
        /// The shaders that perform downsampling of color textures.
        ///
        /// This table maps a [`TextureFormat`] to the shader that performs
        /// downsampling for textures in that format.
        pub general: HashMap<TextureFormat, Handle<Shader>>,
    }

    /// Constants for the single-pass downsampling shader generated on the CPU and
    /// read on the GPU.
    ///
    /// These constants are stored within a uniform buffer. There's one such uniform
    /// buffer per image.
    #[derive(Clone, Copy, ShaderType)]
    #[repr(C)]
    pub struct DownsamplingConstants {
        /// The number of mip levels that this image possesses.
        pub mips: u32,
        /// The reciprocal of the size of the first mipmap level for this texture.
        pub inverse_input_size: Vec2,
        /// Padding.
        pub _padding: u32,
    }

    // The number of storage textures required to combine the bind groups in the
    // downsampling shader.
    const REQUIRED_STORAGE_TEXTURES: u32 = 12;

    /// Returns true if the current platform can use a single bind group for
    /// single-pass downsampling.
    ///
    /// If this platform must use two separate bind groups, one for each pass, this
    /// function returns false.
    pub fn can_combine_downsampling_bind_groups(
        render_adapter: &RenderAdapter,
        render_device: &RenderDevice,
    ) -> bool {
        // Determine whether we can use a single, large bind group for all mip outputs
        let storage_texture_limit = render_device.limits().max_storage_textures_per_shader_stage;

        // Determine whether we can read and write to the same rgba16f storage texture
        let read_write_support = render_adapter
            .get_texture_format_features(TextureFormat::Rgba16Float)
            .flags
            .contains(TextureFormatFeatureFlags::STORAGE_READ_WRITE);

        // Combine the bind group and use read-write storage if it is supported
        storage_texture_limit >= REQUIRED_STORAGE_TEXTURES && read_write_support
    }
}

pub mod tonemapping {
    use bevy_asset::{Handle, RenderAssetUsages};
    use bevy_camera::Camera;
    use bevy_ecs::{
        prelude::{Component, ReflectComponent},
        query::With,
        resource::Resource,
    };
    use bevy_image::{Image, ImageSampler};
    use bevy_reflect::{std_traits::ReflectDefault, Reflect};
    use bevy_render::{
        extract_component::ExtractComponent,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{sampler, texture_3d},
            BindGroupLayoutEntryBuilder, Extent3d, Sampler, SamplerBindingType, TextureDataOrder,
            TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
            TextureView,
        },
        texture::{FallbackImage, GpuImage},
    };

    /// Optionally enables a tonemapping shader that attempts to map linear input stimulus into a perceptually uniform image for a given [`Camera`] entity.
    #[derive(
        Component, Debug, Hash, Clone, Copy, Reflect, Default, ExtractComponent, PartialEq, Eq,
    )]
    #[extract_component_filter(With<Camera>)]
    #[reflect(Component, Debug, Hash, Default, PartialEq)]
    pub enum Tonemapping {
        /// Bypass tonemapping.
        None,
        /// Suffers from lots hue shifting, brights don't desaturate naturally.
        /// Bright primaries and secondaries don't desaturate at all.
        Reinhard,
        /// Suffers from hue shifting. Brights don't desaturate much at all across the spectrum.
        ReinhardLuminance,
        /// Same base implementation that Godot 4.0 uses for Tonemap ACES.
        /// <https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl>
        /// Not neutral, has a very specific aesthetic, intentional and dramatic hue shifting.
        /// Bright greens and reds turn orange. Bright blues turn magenta.
        /// Significantly increased contrast. Brights desaturate across the spectrum.
        AcesFitted,
        /// By Troy Sobotka
        /// <https://github.com/sobotka/AgX>
        /// Very neutral. Image is somewhat desaturated when compared to other tonemappers.
        /// Little to no hue shifting. Subtle [Abney shifting](https://en.wikipedia.org/wiki/Abney_effect).
        /// NOTE: Requires the `tonemapping_luts` cargo feature.
        AgX,
        /// By Tomasz Stachowiak
        /// Has little hue shifting in the darks and mids, but lots in the brights. Brights desaturate across the spectrum.
        /// Is sort of between Reinhard and `ReinhardLuminance`. Conceptually similar to reinhard-jodie.
        /// Designed as a compromise if you want e.g. decent skin tones in low light, but can't afford to re-do your
        /// VFX to look good without hue shifting.
        SomewhatBoringDisplayTransform,
        /// Current Bevy default.
        /// By Tomasz Stachowiak
        /// <https://github.com/h3r2tic/tony-mc-mapface>
        /// Very neutral. Subtle but intentional hue shifting. Brights desaturate across the spectrum.
        /// Comment from author:
        /// Tony is a display transform intended for real-time applications such as games.
        /// It is intentionally boring, does not increase contrast or saturation, and stays close to the
        /// input stimulus where compression isn't necessary.
        /// Brightness-equivalent luminance of the input stimulus is compressed. The non-linearity resembles Reinhard.
        /// Color hues are preserved during compression, except for a deliberate [Bezold–Brücke shift](https://en.wikipedia.org/wiki/Bezold%E2%80%93Br%C3%BCcke_shift).
        /// To avoid posterization, selective desaturation is employed, with care to avoid the [Abney effect](https://en.wikipedia.org/wiki/Abney_effect).
        /// NOTE: Requires the `tonemapping_luts` cargo feature.
        #[default]
        TonyMcMapface,
        /// Default Filmic Display Transform from blender.
        /// Somewhat neutral. Suffers from hue shifting. Brights desaturate across the spectrum.
        /// NOTE: Requires the `tonemapping_luts` cargo feature.
        BlenderFilmic,
    }

    impl Tonemapping {
        pub fn is_enabled(&self) -> bool {
            *self != Tonemapping::None
        }
    }

    /// 3D LUT (look up table) textures used for tonemapping
    #[derive(Resource, Clone, ExtractResource)]
    pub struct TonemappingLuts {
        pub blender_filmic: Handle<Image>,
        pub agx: Handle<Image>,
        pub tony_mc_mapface: Handle<Image>,
    }

    /// Enables a debanding shader that applies dithering to mitigate color banding in the final image for a given [`Camera`] entity.
    #[derive(
        Component, Debug, Hash, Clone, Copy, Reflect, Default, ExtractComponent, PartialEq, Eq,
    )]
    #[extract_component_filter(With<Camera>)]
    #[reflect(Component, Debug, Hash, Default, PartialEq)]
    pub enum DebandDither {
        #[default]
        Disabled,
        Enabled,
    }

    pub fn get_lut_bindings<'a>(
        images: &'a RenderAssets<GpuImage>,
        tonemapping_luts: &'a TonemappingLuts,
        tonemapping: &Tonemapping,
        fallback_image: &'a FallbackImage,
    ) -> (&'a TextureView, &'a Sampler) {
        let image = match tonemapping {
            // AgX lut texture used when tonemapping doesn't need a texture since it's very small (32x32x32)
            Tonemapping::None
            | Tonemapping::Reinhard
            | Tonemapping::ReinhardLuminance
            | Tonemapping::AcesFitted
            | Tonemapping::AgX
            | Tonemapping::SomewhatBoringDisplayTransform => &tonemapping_luts.agx,
            Tonemapping::TonyMcMapface => &tonemapping_luts.tony_mc_mapface,
            Tonemapping::BlenderFilmic => &tonemapping_luts.blender_filmic,
        };
        let lut_image = images.get(image).unwrap_or(&fallback_image.d3);
        (&lut_image.texture_view, &lut_image.sampler)
    }

    pub fn get_lut_bind_group_layout_entries() -> [BindGroupLayoutEntryBuilder; 2] {
        [
            texture_3d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
        ]
    }

    pub fn lut_placeholder() -> Image {
        let format = TextureFormat::Rgba8Unorm;
        let data = vec![255, 0, 255, 255];
        Image {
            data: Some(data),
            data_order: TextureDataOrder::default(),
            texture_descriptor: TextureDescriptor {
                size: Extent3d::default(),
                format,
                dimension: TextureDimension::D3,
                label: None,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            sampler: ImageSampler::Default,
            texture_view_descriptor: None,
            asset_usage: RenderAssetUsages::RENDER_WORLD,
            copy_on_resize: false,
        }
    }
}

pub mod oit {
    use bevy_derive::Deref;
    use bevy_ecs::{component::Component, resource::Resource};
    use bevy_reflect::{std_traits::ReflectDefault, Reflect};
    use bevy_render::{
        extract_component::ExtractComponent,
        render_resource::{
            BufferUsages, CachedRenderPipelineId, DynamicUniformBuffer, ShaderType, UniformBuffer,
            UninitBufferVec,
        },
        renderer::RenderDevice,
    };

    pub mod resolve {
        use bevy_log::warn;
        use bevy_render::{
            render_resource::DownlevelFlags,
            renderer::{RenderAdapter, RenderDevice},
        };

        /// Minimum required value of `wgpu::Limits::max_storage_buffers_per_shader_stage`.
        pub const OIT_REQUIRED_STORAGE_BUFFERS: u32 = 3;

        pub fn is_oit_supported(
            adapter: &RenderAdapter,
            device: &RenderDevice,
            warn: bool,
        ) -> bool {
            if !adapter
                .get_downlevel_capabilities()
                .flags
                .contains(DownlevelFlags::FRAGMENT_WRITABLE_STORAGE)
            {
                if warn {
                    warn!("OrderIndependentTransparencyPlugin not loaded. GPU lacks support: DownlevelFlags::FRAGMENT_WRITABLE_STORAGE.");
                }
                return false;
            }

            let max_storage_buffers_per_shader_stage =
                device.limits().max_storage_buffers_per_shader_stage;

            if max_storage_buffers_per_shader_stage < OIT_REQUIRED_STORAGE_BUFFERS {
                if warn {
                    warn!(
                max_storage_buffers_per_shader_stage,
                OIT_REQUIRED_STORAGE_BUFFERS,
                "OrderIndependentTransparencyPlugin not loaded. RenderDevice lacks support: max_storage_buffers_per_shader_stage < OIT_REQUIRED_STORAGE_BUFFERS."
            );
                }
                return false;
            }

            true
        }
    }

    #[derive(Component)]
    pub struct OrderIndependentTransparencySettingsOffset {
        pub offset: u32,
    }

    #[derive(Component, Deref, Clone, Copy)]
    pub struct OitResolvePipelineId(pub CachedRenderPipelineId);

    /// This key is used to cache the pipeline id and to specialize the render pipeline descriptor.
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    pub struct OitResolvePipelineKey {
        hdr: bool,
        sorted_fragment_max_count: u32,
        depth_prepass: bool,
    }

    /// Used to identify which camera will use OIT to render transparent meshes
    /// and to configure OIT.
    // TODO consider supporting multiple OIT techniques like WBOIT, Moment Based OIT,
    // depth peeling, stochastic transparency, ray tracing etc.
    // This should probably be done by adding an enum to this component.
    // We use the same struct to pass on the settings to the drawing shader.
    #[derive(Clone, Copy, ExtractComponent, Reflect, ShaderType, Component)]
    #[extract_component_sync_target((Self, OrderIndependentTransparencySettingsOffset, OitResolvePipelineId))]
    #[reflect(Clone, Default)]
    pub struct OrderIndependentTransparencySettings {
        /// Controls how many fragments will be exactly sorted.
        /// If the scene has more fragments than this, they will be merged approximately.
        /// More sorted fragments is more accurate but will be slower.
        pub sorted_fragment_max_count: u32,
        /// The average fragments per pixel stored in the buffer. This should be bigger enough otherwise the fragments will be discarded.
        /// Higher values increase memory usage.
        pub fragments_per_pixel_average: f32,
        /// Threshold for which fragments will be added to the blending layers.
        /// This can be tweaked to optimize quality / layers count. Higher values will
        /// allow lower number of layers and a better performance, compromising quality.
        pub alpha_threshold: f32,
    }

    impl Default for OrderIndependentTransparencySettings {
        fn default() -> Self {
            Self {
                sorted_fragment_max_count: 8,
                fragments_per_pixel_average: 4.0,
                alpha_threshold: 0.0,
            }
        }
    }

    #[derive(Clone, Copy, ShaderType)]
    pub struct OitFragmentNode {
        pub color: u32,
        pub depth_alpha: u32,
        pub next: u32,
    }

    /// Holds the buffers that contain the data of all OIT layers.
    /// We use one big buffer for the entire app. Each camera will reuse it so it will
    /// always be the size of the biggest OIT enabled camera.
    #[derive(Resource)]
    pub struct OitBuffers {
        pub settings: DynamicUniformBuffer<OrderIndependentTransparencySettings>,
        pub nodes_capacity: UniformBuffer<u32>,
        /// OIT nodes buffer contains color, depth and linked next node for each fragments.
        pub nodes: UninitBufferVec<OitFragmentNode>,
        /// OIT heads buffer contains the head that pointers nodes buffer, essentially used as a 2d array where xy is the screen coordinate.
        /// We don't use storage texture as it requires native only [`bevy_render::settings::WgpuFeatures::TEXTURE_ATOMIC`].
        pub heads: UninitBufferVec<u32>,
        pub atomic_counter: UninitBufferVec<u32>,
    }
}

/// these types need to move to this crate
mod imports {
    #![allow(unused_imports, reason = "making sure everything that matters is here")]
    pub use crate::{
        blit::{BlitPipeline, BlitPipelineKey},
        core_2d::{
            AlphaMask2d, AlphaMask2dBinKey, BatchSetKey2d, Opaque2d, Opaque2dBinKey, Transparent2d,
            CORE_2D_DEPTH_FORMAT,
        },
        core_3d::{
            // TODO: just add sets for these since they are commonly used sync points:
            // main_opaque_pass_3d, main_transparent_pass_3d, prepare_core_3d_depth_textures,
            AlphaMask3d,
            Opaque3d,
            Transparent3d,
            TransparentSortingInfo3d,
            CORE_3D_DEPTH_FORMAT,
            // DEPTH_TEXTURE_SAMPLING_SUPPORTED, // (feature-gated)
        },
        deferred::{
            copy_lighting_id::DeferredLightingIdDepthTexture,
            // TODO: add set
            // node::late_deferred_prepass,
            AlphaMask3dDeferred,
            Opaque3dDeferred,
            DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
        },
        mip_generation::{
            can_combine_downsampling_bind_groups,
            experimental::depth::{
                create_depth_pyramid_dummy_texture,
                // TODO: create sets
                // early_downsample_depth, late_downsample_depth,
                ViewDepthPyramid,
            },
            DownsampleShaders, DownsamplingConstants,
        },
        oit::{
            // TODO: add set
            // prepare_oit_buffers,
            resolve::is_oit_supported,
            OitBuffers,
            OrderIndependentTransparencySettings,
            OrderIndependentTransparencySettingsOffset,
        },
        prepass::{
            // TODO: add sets
            // node::{early_prepass, late_prepass},
            prepass_target_descriptors,
            AlphaMask3dPrepass,
            DeferredPrepass,
            DepthPrepass,
            MotionVectorPrepass,
            NormalPrepass,
            Opaque3dPrepass,
            OpaqueNoLightmap3dBatchSetKey,
            OpaqueNoLightmap3dBinKey,
            PreviousViewData,
            PreviousViewUniformOffset,
            PreviousViewUniforms,
            ViewPrepassTextures,
        },
        // TODO: add set
        // upscaling::upscaling,
        schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems},
        tonemapping::{
            get_lut_bind_group_layout_entries,
            get_lut_bindings,
            lut_placeholder,
            // TODO: create set
            // tonemapping,
            DebandDither,
            Tonemapping,
            TonemappingLuts,
        },
        FullscreenShader,
    };
}
