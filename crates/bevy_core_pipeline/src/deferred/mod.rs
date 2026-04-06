pub mod copy_lighting_id;
pub mod node;

pub use bevy_core_pipeline_types::deferred::{
    AlphaMask3dDeferred, Opaque3dDeferred, DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
    DEFERRED_LIGHTING_PASS_ID_FORMAT, DEFERRED_PREPASS_FORMAT,
};
