use std::{
    convert::Infallible,
    sync::{atomic::AtomicUsize, Arc},
};

use bevy::prelude::*;
use bevy_asset::{
    io::Reader,
    processor::{AssetProcessor, LoadTransformAndSave},
    saver::AssetSaver,
    transformer::{AssetTransformer, TransformedAsset},
    AssetLoader, AssetPath, AsyncWriteExt, LoadContext,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Asset, TypePath, Debug, Default)]
pub struct ProcessedAsset {
    text: String,
    #[dependency]
    pub deps: Vec<Handle<UnProcessedAsset>>,
}

#[derive(Error, Debug)]
pub enum MyAssetLoaderError {
    #[error("Could not load dependency: {dependency}")]
    CannotLoadDependency { dependency: AssetPath<'static> },
    #[error("A RON error occurred during loading: {0}")]
    RonSpannedError(#[from] ron::error::SpannedError),
    #[error("A RON error occurred during saving: {0}")]
    RonError(#[from] ron::error::Error),
    #[error("An IO error occurred during loading: {0}")]
    Io(#[from] std::io::Error),
    #[error("An error occurred during loading: {0}")]
    LoadDirectError(#[from] bevy_asset::LoadDirectError),
}

struct MyLoaderSettings;

pub struct ProcessedAssetLoader {
    counter: Arc<AtomicUsize>,
}

#[derive(Serialize, Deserialize)]
struct ProcessedAssetRon {
    text: Option<String>,
    immediate_deps: Option<Vec<String>>,
    deps: Vec<String>,
}

impl AssetLoader for ProcessedAssetLoader {
    type Asset = ProcessedAsset;

    type Settings = ();

    type Error = MyAssetLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        self.counter
            .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let ron: ProcessedAssetRon = ron::de::from_bytes(&bytes)?;
        let text = {
            let mut text = String::new();
            if let Some(deps) = ron.immediate_deps.as_ref() {
                for dep in deps {
                    let dep = load_context
                        .loader()
                        .immediate()
                        .load::<ProcessedAsset>(dep)
                        .await?
                        .take();
                    text.push_str(&dep.text);
                }
            }
            text
        };

        Ok(ProcessedAsset {
            text,
            deps: ron.deps.iter().map(|p| load_context.load(p)).collect(),
        })
    }

    fn extensions(&self) -> &[&str] {
        &["myasset"]
    }
}

#[derive(Asset, TypePath, Debug, Default)]
pub struct UnProcessedAsset {
    text: String,
}
#[derive(Serialize, Deserialize)]
pub struct UnProcessedAssetRon {
    text: String,
    deps: Vec<String>,
}

struct UnProcessedAssetLoader {
    counter: Arc<AtomicUsize>,
}

impl AssetLoader for UnProcessedAssetLoader {
    type Asset = UnProcessedAsset;

    type Settings = ();

    type Error = MyAssetLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        self.counter
            .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let ron: UnProcessedAssetRon = ron::de::from_bytes(&bytes)?;
        let text = {
            let mut text = ron.text;
            for dep in &ron.deps {
                let dep = load_context
                    .loader()
                    .immediate()
                    .load::<UnProcessedAsset>(dep)
                    .await?
                    .take();
                text.push_str(&dep.text);
            }
            text
        };

        Ok(UnProcessedAsset { text })
    }

    fn extensions(&self) -> &[&str] {
        &["unprocessed"]
    }
}

struct MyAssetTransformer {
    counter: Arc<AtomicUsize>,
}

#[derive(Default, Serialize, Deserialize)]
struct MyAssetTransformerSettings {
    text: String,
}

impl AssetTransformer for MyAssetTransformer {
    type AssetInput = ProcessedAsset;
    type AssetOutput = ProcessedAsset;

    type Settings = MyAssetTransformerSettings;

    type Error = Infallible;

    async fn transform<'a>(
        &'a self,
        mut asset: TransformedAsset<Self::AssetInput>,
        settings: &'a Self::Settings,
    ) -> Result<TransformedAsset<Self::AssetOutput>, Self::Error> {
        self.counter
            .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        let text = format!("{}+{}", asset.get().text, settings.text.clone());

        asset.get_mut().text = text;

        Ok(asset)
    }
}

struct MyAssetSaver {
    counter: Arc<AtomicUsize>,
}

impl AssetSaver for MyAssetSaver {
    type Asset = ProcessedAsset;
    type Settings = ();

    type OutputLoader = ProcessedAssetLoader;

    type Error = MyAssetLoaderError;

    async fn save(
        &self,
        writer: &mut bevy_asset::io::Writer,
        asset: bevy_asset::saver::SavedAsset<'_, Self::Asset>,
        _settings: &Self::Settings,
    ) -> Result<<Self::OutputLoader as AssetLoader>::Settings, Self::Error> {
        self.counter
            .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        let ron = ProcessedAssetRon {
            text: Some(asset.text.clone()),
            immediate_deps: None,
            deps: asset
                .deps
                .iter()
                .map(|h| h.path().unwrap().to_string())
                .collect(),
        };
        let mut bytes = String::new();
        ron::ser::to_writer(&mut bytes, &ron)?;
        writer.write_all(bytes.as_bytes()).await?;
        Ok(())
    }
}

type MyProcessor = LoadTransformAndSave<ProcessedAssetLoader, MyAssetTransformer, MyAssetSaver>;

pub fn run_app_until(app: &mut App, mut predicate: impl FnMut(&mut World) -> Option<()>) {
    for _ in 0..LARGE_ITERATION_COUNT {
        app.update();
        if predicate(app.world_mut()).is_some() {
            return;
        }
    }

    panic!("Ran out of loops to return `Some` from `predicate`");
}

const LARGE_ITERATION_COUNT: usize = 10000;

pub(crate) fn get<A: Asset>(world: &World, id: AssetId<A>) -> Option<&A> {
    world.resource::<Assets<A>>().get(id)
}

fn main() {
    let mut app = {
        let mut app = App::new();
        app.add_plugins((
            TaskPoolPlugin::default(),
            AssetPlugin {
                mode: AssetMode::Processed,
                file_path: "tests/asset/assets".to_string(),
                processed_file_path: "tests/asset/processed".to_string(),
                ..Default::default()
            },
        ));
        app
    };

    let (my_loader, loader_count) = {
        let counter = Arc::new(AtomicUsize::new(0));
        (
            ProcessedAssetLoader {
                counter: counter.clone(),
            },
            counter,
        )
    };

    let (my_unprocessed_loader, unprocessed_loader_count) = {
        let counter = Arc::new(AtomicUsize::new(0));
        (
            UnProcessedAssetLoader {
                counter: counter.clone(),
            },
            counter,
        )
    };

    let (my_transformer, transformer_count) = {
        let counter = Arc::new(AtomicUsize::new(0));
        (
            MyAssetTransformer {
                counter: counter.clone(),
            },
            counter,
        )
    };

    let (my_saver, saver_count) = {
        let counter = Arc::new(AtomicUsize::new(0));
        (
            MyAssetSaver {
                counter: counter.clone(),
            },
            counter,
        )
    };

    app.init_asset::<ProcessedAsset>()
        .init_asset::<UnProcessedAsset>()
        .register_asset_loader(my_loader)
        .register_asset_loader(my_unprocessed_loader)
        .register_asset_processor(MyProcessor::new(my_transformer, my_saver))
        .set_default_asset_processor::<MyProcessor>("myasset");

    app.finish();
    app.cleanup();
    app.update();

    bevy::tasks::block_on(
        app.world()
            .resource::<AssetProcessor>()
            .data()
            .wait_until_finished(),
    );

    assert_eq!(loader_count.load(core::sync::atomic::Ordering::SeqCst), 7);
    assert_eq!(
        unprocessed_loader_count.load(core::sync::atomic::Ordering::SeqCst),
        0
    );
    assert_eq!(
        transformer_count.load(core::sync::atomic::Ordering::SeqCst),
        4
    );
    assert_eq!(saver_count.load(core::sync::atomic::Ordering::SeqCst), 4);

    loader_count.store(0, core::sync::atomic::Ordering::SeqCst);
    unprocessed_loader_count.store(0, core::sync::atomic::Ordering::SeqCst);
    transformer_count.store(0, core::sync::atomic::Ordering::SeqCst);
    saver_count.store(0, core::sync::atomic::Ordering::SeqCst);

    let server = app.world().resource::<AssetServer>().clone();
    let handle: Handle<ProcessedAsset> = server.load("a.myasset");
    let a_id = handle.id();

    // app.insert_resource(MyHandles(handle));
    run_app_until(&mut app, |world| {
        let _a = get(world, a_id)?;
        let (a_load, a_deps, a_rec_deps) = server.get_load_states(a_id).unwrap();
        if !matches!(
            (a_load, a_deps, a_rec_deps),
            (
                bevy_asset::LoadState::Loaded,
                bevy_asset::DependencyLoadState::Loaded,
                bevy_asset::RecursiveDependencyLoadState::Loaded
            )
        ) {
            return None;
        }
        Some(())
    });

    assert_eq!(loader_count.load(core::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        unprocessed_loader_count.load(core::sync::atomic::Ordering::SeqCst),
        2
    );
    assert_eq!(
        transformer_count.load(core::sync::atomic::Ordering::SeqCst),
        0
    );
    assert_eq!(saver_count.load(core::sync::atomic::Ordering::SeqCst), 0);
}
