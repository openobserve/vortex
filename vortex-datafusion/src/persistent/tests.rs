// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::anyhow;
use datafusion::arrow::array::Int32Array;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_common::GetExt;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use insta::assert_snapshot;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::buffer::buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::io::runtime::Handle;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::table::TableStrategy;
use vortex::layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex::layout::layouts::zoned::writer::ZonedStrategy;
use vortex::session::VortexSession;

use crate::VortexFormatFactory;
use crate::VortexTableOptions;
use crate::common_tests::TestSessionContext;
use crate::metrics::VortexMetricsFinder;

fn make_session(
    object_store: Arc<dyn ObjectStore>,
    repartition_file_scans: bool,
) -> SessionContext {
    let factory = Arc::new(VortexFormatFactory::new());

    let config = SessionConfig::new()
        .with_target_partitions(4)
        .with_repartition_file_scans(repartition_file_scans)
        .with_repartition_file_min_size(0);
    let mut state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_object_store(&url::Url::try_from("file://").unwrap(), object_store);

    if let Some(file_formats) = state.file_formats() {
        file_formats.push(factory as _);
    }

    SessionContext::new_with_state(state.build()).enable_url_table()
}

async fn count_query_partitions(ctx: &SessionContext, sql: &str) -> anyhow::Result<usize> {
    let explain = ctx.sql(&format!("EXPLAIN {sql}")).await?.collect().await?;
    let plan = pretty_format_batches(&explain)?.to_string();
    let marker = "DataSourceExec: file_groups={";
    let start = plan
        .find(marker)
        .ok_or_else(|| anyhow!("EXPLAIN plan did not contain a DataSourceExec"))?
        + marker.len();
    let partitions = plan[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();

    Ok(partitions.parse()?)
}

fn batch_values(batches: &[RecordBatch]) -> Vec<i32> {
    let mut values = Vec::with_capacity(batches.iter().map(|batch| batch.num_rows()).sum());

    for batch in batches {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("value column should be Int32");
        values.extend(array.values().iter().copied());
    }

    values
}

fn make_topk_pruning_session(
    object_store: Arc<dyn ObjectStore>,
    dynamic_filter_pushdown: bool,
    scan_concurrency: Option<usize>,
) -> SessionContext {
    let factory = Arc::new(VortexFormatFactory::new().with_options(VortexTableOptions {
        scan_concurrency,
        ..Default::default()
    }));
    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_batch_size(4_096)
        .with_repartition_file_scans(false);
    config
        .options_mut()
        .optimizer
        .enable_dynamic_filter_pushdown = dynamic_filter_pushdown;
    config
        .options_mut()
        .optimizer
        .enable_topk_dynamic_filter_pushdown = dynamic_filter_pushdown;

    let mut state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_object_store(&url::Url::try_from("file://").unwrap(), object_store);

    if let Some(file_formats) = state.file_formats() {
        file_formats.push(factory as _);
    }

    SessionContext::new_with_state(state.build()).enable_url_table()
}

async fn write_chunked_i32_file(
    store: Arc<dyn ObjectStore>,
    path: &str,
    starts: impl IntoIterator<Item = i32>,
    split_len: usize,
) -> anyhow::Result<u64> {
    let split_len_i32 = i32::try_from(split_len)?;
    let chunks = starts
        .into_iter()
        .map(|start| {
            StructArray::try_new(
                ["value"].into(),
                vec![Buffer::from_iter(start..start + split_len_i32).into_array()],
                split_len,
                Validity::NonNullable,
            )
            .map(IntoArray::into_array)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let table = ChunkedArray::from_iter(chunks).into_array();
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let block_size = NonZeroUsize::new(split_len)
        .ok_or_else(|| anyhow!("the test zone size must be non-zero"))?;
    let strategy: Arc<dyn LayoutStrategy> = Arc::new(TableStrategy::new(
        Arc::clone(&flat),
        Arc::new(ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size,
                ..Default::default()
            },
        )),
    ));
    let path = object_store::path::Path::parse(path)?;
    let mut writer = ObjectStoreWrite::new(store, &path).await?;
    let summary = VortexSession::default()
        .write_options()
        .with_strategy(strategy)
        .write(&mut writer, table.to_array_stream())
        .await?;
    writer.shutdown().await?;

    Ok(summary.size())
}

async fn execute_topk_with_read_bytes(
    store: Arc<dyn ObjectStore>,
    dynamic_filter_pushdown: bool,
    scan_concurrency: Option<usize>,
) -> anyhow::Result<(Vec<i32>, usize, String)> {
    let ctx = make_topk_pruning_session(store, dynamic_filter_pushdown, scan_concurrency);
    ctx.sql(
        "CREATE EXTERNAL TABLE topk_pruning_data \
             (value INT NOT NULL) \
         STORED AS vortex \
         LOCATION '/topk-pruning/'",
    )
    .await?;
    let dataframe = ctx
        .sql("SELECT value FROM topk_pruning_data ORDER BY value ASC LIMIT 10")
        .await?;
    let physical_plan = dataframe.create_physical_plan().await?;
    let rendered_plan = DisplayableExecutionPlan::new(physical_plan.as_ref())
        .tree_render()
        .to_string();
    let batches =
        datafusion_physical_plan::collect(Arc::clone(&physical_plan), ctx.task_ctx()).await?;
    let read_bytes = VortexMetricsFinder::find_all(physical_plan.as_ref())
        .iter()
        .filter_map(|metrics| metrics.sum_by_name("vortex.io.read.total_size"))
        .map(|metric| metric.as_usize())
        .sum();

    Ok((batch_values(&batches), read_bytes, rendered_plan))
}

#[rstest]
#[tokio::test]
async fn test_query_file(#[values(Some(1), None)] limit: Option<usize>) -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    let session = VortexSession::default();

    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ])
    .into_array();

    let st = StructArray::try_new(
        ["strings", "numbers"].into(),
        vec![strings, numbers],
        8,
        Validity::NonNullable,
    )?;

    let mut writer = ObjectStoreWrite::new(Arc::clone(&ctx.store), &"test.vortex".into()).await?;

    let summary = session
        .write_options()
        .write(&mut writer, st.into_array().to_array_stream())
        .await?;

    writer.shutdown().await?;

    assert_eq!(summary.row_count(), 8);

    let read_row_count = ctx
        .session
        .sql("SELECT * from '/test.vortex'")
        .await?
        .limit(0, limit)?
        .count()
        .await?;

    assert_eq!(read_row_count, limit.unwrap_or(8));

    Ok(())
}

#[tokio::test]
async fn topk_dynamic_filter_is_retained_by_vortex_scan() -> anyhow::Result<()> {
    use arrow_schema::Field;
    use arrow_schema::Schema;

    let ctx = TestSessionContext::default();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![5, 1, 4, 2, 3]))],
    )?;
    ctx.write_arrow_batch("topk-dynamic.vortex", &batch).await?;

    let df = ctx
        .session
        .sql("SELECT value FROM '/topk-dynamic.vortex' ORDER BY value ASC LIMIT 2")
        .await?;
    let physical_plan = ctx
        .session
        .state()
        .create_physical_plan(df.logical_plan())
        .await?;
    let plan = DisplayableExecutionPlan::new(physical_plan.as_ref())
        .tree_render()
        .to_string();

    assert!(plan.contains("DynamicFilter"), "{plan}");
    assert_eq!(batch_values(&df.collect().await?), vec![1, 2]);
    Ok(())
}

#[tokio::test]
async fn topk_dynamic_filter_reduces_multi_file_multi_split_io() -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let files = [
        ("/topk-pruning/00-low.vortex", [0, 100_000, 200_000]),
        ("/topk-pruning/01-high.vortex", [300_000, 400_000, 500_000]),
        (
            "/topk-pruning/02-higher.vortex",
            [600_000, 700_000, 800_000],
        ),
    ];
    let mut file_sizes = Vec::with_capacity(files.len());
    for (path, starts) in files {
        let file_size = write_chunked_i32_file(Arc::clone(&store), path, starts, 4_096).await?;
        file_sizes.push((path, file_size));
    }

    for (path, file_size) in file_sizes {
        let reader = Arc::new(ObjectStoreReadAt::new(
            Arc::clone(&store),
            object_store::path::Path::parse(path)?,
            Handle::find().ok_or_else(|| anyhow!("tokio runtime should be available in tests"))?,
        ));
        let vxf = VortexSession::default()
            .open_options()
            .with_file_size(file_size)
            .open_read(reader)
            .await?;
        assert_eq!(vxf.splits()?.len(), 3);
    }

    let (filtered_values, filtered_read_bytes, filtered_plan) =
        execute_topk_with_read_bytes(Arc::clone(&store), true, Some(1)).await?;
    let (baseline_values, baseline_read_bytes, _) =
        execute_topk_with_read_bytes(store, false, Some(1)).await?;

    assert!(filtered_plan.contains("DynamicFilter"), "{filtered_plan}");
    assert_eq!(filtered_values, (0_i32..10).collect::<Vec<_>>());
    assert_eq!(baseline_values, filtered_values);
    assert!(
        baseline_read_bytes > filtered_read_bytes.saturating_mul(2),
        "expected dynamic TopK pruning to skip at least one file's worth of reads: filtered={filtered_read_bytes}, baseline={baseline_read_bytes}"
    );
    Ok(())
}

#[tokio::test]
async fn topk_dynamic_filter_reduces_io_at_default_concurrency() -> anyhow::Result<()> {
    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let files = [
        (
            "/topk-pruning/00-low.vortex",
            [0, 100_000, 200_000, 300_000],
        ),
        (
            "/topk-pruning/01-high.vortex",
            [400_000, 500_000, 600_000, 700_000],
        ),
        (
            "/topk-pruning/02-higher.vortex",
            [800_000, 900_000, 1_000_000, 1_100_000],
        ),
        (
            "/topk-pruning/03-highest.vortex",
            [1_200_000, 1_300_000, 1_400_000, 1_500_000],
        ),
    ];
    for (path, starts) in files {
        write_chunked_i32_file(Arc::clone(&store), path, starts, 16_384).await?;
    }

    // `None` exercises Vortex's default scan concurrency (currently four). Each natural split is
    // also four times larger than DataFusion's configured output batch size.
    let (filtered_values, filtered_read_bytes, filtered_plan) =
        execute_topk_with_read_bytes(Arc::clone(&store), true, None).await?;
    let (baseline_values, baseline_read_bytes, _) =
        execute_topk_with_read_bytes(store, false, None).await?;

    assert!(filtered_plan.contains("DynamicFilter"), "{filtered_plan}");
    assert_eq!(filtered_values, (0_i32..10).collect::<Vec<_>>());
    assert_eq!(baseline_values, filtered_values);
    assert!(
        baseline_read_bytes > filtered_read_bytes,
        "expected dynamic TopK pruning to reduce reads with default concurrency: filtered={filtered_read_bytes}, baseline={baseline_read_bytes}"
    );
    Ok(())
}

#[tokio::test]
async fn test_addition_pushdown() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE written_data \
                    (a TINYINT NOT NULL) \
                STORED AS vortex \
                LOCATION '/test/'",
        )
        .await?;

    ctx.session
        .sql("INSERT INTO written_data VALUES (0), (1), (2), (3), (4)")
        .await?
        .collect()
        .await?;

    let result = ctx
        .session
        .sql("SELECT a, a + 5 as five, a + 6 as six FROM written_data WHERE a + 5 > 7")
        .await?
        .collect()
        .await?;

    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +---+------+-----+
        | a | five | six |
        +---+------+-----+
        | 3 | 8    | 9   |
        | 4 | 9    | 10  |
        +---+------+-----+
        ");

    Ok(())
}

#[tokio::test]
async fn test_octet_length_pushdown() -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(true);

    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE written_strings \
                    (s VARCHAR NOT NULL) \
                STORED AS vortex \
                LOCATION '/strings/'",
        )
        .await?;

    ctx.session
        .sql("INSERT INTO written_strings VALUES ('a'), ('é'), ('abcd'), ('')")
        .await?
        .collect()
        .await?;

    let result = ctx
        .session
        .sql(
            "SELECT s, octet_length(s) AS len \
             FROM written_strings \
             WHERE octet_length(s) > 1 \
             ORDER BY s",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(
        result[0].schema().field_with_name("len")?.data_type(),
        &DataType::Int32
    );
    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +------+-----+
        | s    | len |
        +------+-----+
        | abcd | 4   |
        | é    | 2   |
        +------+-----+
        ");

    Ok(())
}

#[tokio::test]
async fn create_table_ordered_by() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();

    // Vortex
    ctx.session
        .sql(
            "CREATE EXTERNAL TABLE my_tbl_vx \
                (c1 VARCHAR NOT NULL, c2 INT NOT NULL) \
                STORED AS vortex  \
                WITH ORDER (c1 ASC)
                LOCATION '/test/'",
        )
        .await?;

    ctx.session
        .sql("INSERT INTO my_tbl_vx VALUES ('air', 5), ('balloon', 42)")
        .await?
        .collect()
        .await?;

    ctx.session
        .sql("INSERT INTO my_tbl_vx VALUES ('zebra', 5)")
        .await?
        .collect()
        .await?;

    ctx.session
        .sql("INSERT INTO my_tbl_vx VALUES ('texas', 2000), ('alabama', 2000)")
        .await?
        .collect()
        .await?;

    let df = ctx
        .session
        .sql("SELECT * FROM my_tbl_vx ORDER BY c1 ASC limit 3")
        .await?;

    let physical_plan = ctx
        .session
        .state()
        .create_physical_plan(df.logical_plan())
        .await?;

    insta::assert_snapshot!(DisplayableExecutionPlan::new(physical_plan.as_ref())
                .tree_render().to_string(), @r"
        ┌───────────────────────────┐
        │  SortPreservingMergeExec  │
        │    --------------------   │
        │     c1 ASC NULLS LAST     │
        │                           │
        │          limit: 3         │
        └─────────────┬─────────────┘
        ┌─────────────┴─────────────┐
        │       DataSourceExec      │
        │    --------------------   │
        │          files: 3         │
        │       format: vortex      │
        └───────────────────────────┘
        ");

    let r = df.collect().await?;

    insta::assert_snapshot!(pretty_format_batches(&r)?.to_string(), @r"
        +---------+------+
        | c1      | c2   |
        +---------+------+
        | air     | 5    |
        | alabama | 2000 |
        | balloon | 42   |
        +---------+------+
        ");

    Ok(())
}

/// Doc example: demonstrates creating, writing, reading, and filtering a Vortex table.
#[tokio::test]
async fn doc_example() -> anyhow::Result<()> {
    // [setup]
    use std::sync::Arc;

    use datafusion::datasource::provider::DefaultTableFactory;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use datafusion_common::GetExt;
    use object_store::memory::InMemory;

    use crate::VortexFormatFactory;

    let factory = Arc::new(VortexFormatFactory::new());
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_table_factory(
            factory.get_ext().to_uppercase(),
            Arc::new(DefaultTableFactory::new()),
        )
        .with_file_formats(vec![factory])
        .build();
    let ctx = SessionContext::new_with_state(state).enable_url_table();
    // [setup]

    // Register an in-memory object store for the test.
    let store = Arc::new(InMemory::new());
    ctx.register_object_store(&url::Url::try_from("file://").unwrap(), store);

    // [create]
    ctx.sql(
        "CREATE EXTERNAL TABLE my_table \
                (name VARCHAR NOT NULL, age INT NOT NULL) \
            STORED AS vortex \
            LOCATION '/demo/'",
    )
    .await?;
    // [create]

    // [write]
    ctx.sql(
        "INSERT INTO my_table VALUES \
                ('Alice', 30), ('Bob', 25), ('Charlie', 35), ('Diana', 28)",
    )
    .await?
    .collect()
    .await?;
    // [write]

    // [query]
    let result = ctx
        .sql("SELECT name, age FROM my_table WHERE age > 28 ORDER BY age")
        .await?
        .collect()
        .await?;
    // [query]

    assert_snapshot!(pretty_format_batches(&result)?, @r"
        +---------+-----+
        | name    | age |
        +---------+-----+
        | Alice   | 30  |
        | Charlie | 35  |
        +---------+-----+
        ");

    Ok(())
}

#[tokio::test]
async fn test_repartitioned_scan_matches_non_repartitioned_for_uneven_splits() -> anyhow::Result<()>
{
    let store = Arc::new(InMemory::new()) as _;
    let session = VortexSession::default();
    let path = object_store::path::Path::parse("/split-aligned-repartition.vortex")?;

    let chunk_1_len = 2_000;
    let chunk_2_len = 5_000;
    let chunk_3_len = 6_000;
    let row_count = chunk_1_len + chunk_2_len + chunk_3_len;

    let chunk_1 = StructArray::try_new(
        ["value"].into(),
        vec![Buffer::from_iter(0_i32..chunk_1_len).into_array()],
        usize::try_from(chunk_1_len)?,
        Validity::NonNullable,
    )?;
    let chunk_2 = StructArray::try_new(
        ["value"].into(),
        vec![Buffer::from_iter(chunk_1_len..(chunk_1_len + chunk_2_len)).into_array()],
        usize::try_from(chunk_2_len)?,
        Validity::NonNullable,
    )?;
    let chunk_3 = StructArray::try_new(
        ["value"].into(),
        vec![Buffer::from_iter((chunk_1_len + chunk_2_len)..row_count).into_array()],
        usize::try_from(chunk_3_len)?,
        Validity::NonNullable,
    )?;
    let table = ChunkedArray::from_iter([
        chunk_1.into_array(),
        chunk_2.into_array(),
        chunk_3.into_array(),
    ])
    .into_array();
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let strategy: Arc<dyn LayoutStrategy> = Arc::new(TableStrategy::new(
        Arc::clone(&flat),
        Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())),
    ));

    let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &path).await?;
    let summary = session
        .write_options()
        .with_strategy(strategy)
        .write(&mut writer, table.into_array().to_array_stream())
        .await?;
    writer.shutdown().await?;

    let reader = Arc::new(ObjectStoreReadAt::new(
        Arc::clone(&store),
        path.clone(),
        Handle::find().expect("tokio runtime should be available in tests"),
    ));
    let vxf = session
        .open_options()
        .with_file_size(summary.size())
        .open_read(reader)
        .await?;
    let split_ranges = vxf.splits()?;
    let split_lengths = split_ranges
        .iter()
        .map(|range| range.end - range.start)
        .collect::<Vec<_>>();

    assert!(split_ranges.len() > 1);
    assert!(
        split_lengths
            .windows(2)
            .any(|window| window[0] != window[1])
    );

    let serial_ctx = make_session(Arc::clone(&store), false);
    let repartitioned_ctx = make_session(Arc::clone(&store), true);
    let repartitioned_partitions = count_query_partitions(
        &repartitioned_ctx,
        "SELECT value FROM '/split-aligned-repartition.vortex'",
    )
    .await?;

    assert!(repartitioned_partitions > 1);

    let serial = serial_ctx
        .sql("SELECT value FROM '/split-aligned-repartition.vortex' ORDER BY value")
        .await?
        .collect()
        .await?;
    let repartitioned = repartitioned_ctx
        .sql("SELECT value FROM '/split-aligned-repartition.vortex' ORDER BY value")
        .await?
        .collect()
        .await?;
    let serial_values = batch_values(&serial);
    let repartitioned_values = batch_values(&repartitioned);
    let expected = (0_i32..row_count).collect::<Vec<_>>();

    assert_eq!(serial_values, expected);
    assert_eq!(repartitioned_values, serial_values);

    Ok(())
}

/// Roundtrip an `arrow.uuid` extension column through a Vortex file: write the column directly
/// via the session-aware Arrow→Vortex conversion, then `SELECT *` and assert both the field
/// metadata and the underlying values survive the trip.
#[tokio::test]
async fn arrow_uuid_extension_roundtrip() -> anyhow::Result<()> {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use arrow_schema::extension::Uuid;
    use datafusion::arrow::array::FixedSizeBinaryArray;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::assert_batches_sorted_eq;
    use vortex_arrow::ArrowSessionExt;

    let ctx = TestSessionContext::default();
    // Default vortex session has importer/exporter for Arrow UUID
    let session = VortexSession::default();

    let mut uuid_field = Field::new("id", DataType::FixedSizeBinary(16), false);
    uuid_field.try_with_extension_type(Uuid)?;
    let schema = Arc::new(Schema::new(vec![uuid_field]));

    let uuids = FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?;
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(uuids)])?;
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;

    let mut writer = ObjectStoreWrite::new(Arc::clone(&ctx.store), &"uuid.vortex".into()).await?;
    session
        .write_options()
        .write(&mut writer, array.to_array_stream())
        .await?;
    writer.shutdown().await?;

    let result = ctx
        .session
        .sql("SELECT * FROM '/uuid.vortex'")
        .await?
        .collect()
        .await?;

    assert!(
        result[0]
            .schema_ref()
            .field(0)
            .has_valid_extension_type::<Uuid>()
    );

    assert_batches_sorted_eq!(
        [
            "+----------------------------------+",
            "| id                               |",
            "+----------------------------------+",
            "| 30313233343536373839616263646566 |",
            "| 66656463626139383736353433323130 |",
            "+----------------------------------+",
        ],
        &result
    );

    Ok(())
}

/// Same as [`arrow_uuid_extension_roundtrip`] but with the `arrow.uuid` field nested inside a
/// top-level `Struct`, exercising recursive session-aware Field/Schema inference: if any layer
/// falls back to the non-plugin canonical path, the inner field loses its extension metadata.
#[tokio::test]
async fn arrow_uuid_extension_roundtrip_nested_struct() -> anyhow::Result<()> {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
    use arrow_schema::Schema;
    use arrow_schema::extension::Uuid;
    use datafusion::arrow::array::Array;
    use datafusion::arrow::array::FixedSizeBinaryArray;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::arrow::array::StructArray as ArrowStructArray;
    use datafusion::assert_batches_sorted_eq;
    use vortex_arrow::ArrowSessionExt;

    let ctx = TestSessionContext::default();
    let session = VortexSession::default();

    let mut inner_uuid_field = Field::new("id", DataType::FixedSizeBinary(16), false);
    inner_uuid_field.try_with_extension_type(Uuid)?;
    let payload_fields = Fields::from(vec![inner_uuid_field]);
    let payload_field = Field::new("payload", DataType::Struct(payload_fields.clone()), false);
    let schema = Arc::new(Schema::new(vec![payload_field]));

    let uuids: Arc<dyn Array> = Arc::new(FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?);
    let payload_array = ArrowStructArray::new(payload_fields, vec![uuids], None);
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(payload_array)])?;
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;

    let mut writer =
        ObjectStoreWrite::new(Arc::clone(&ctx.store), &"uuid_struct.vortex".into()).await?;
    session
        .write_options()
        .write(&mut writer, array.to_array_stream())
        .await?;
    writer.shutdown().await?;

    let result = ctx
        .session
        .sql("SELECT payload FROM '/uuid_struct.vortex'")
        .await?
        .collect()
        .await?;

    let read_payload = result[0].schema_ref().field(0);
    let DataType::Struct(read_inner) = read_payload.data_type() else {
        panic!(
            "expected Struct payload, got {:?}",
            read_payload.data_type()
        );
    };
    assert!(read_inner[0].has_valid_extension_type::<Uuid>());

    assert_batches_sorted_eq!(
        [
            "+----------------------------------------+",
            "| payload                                |",
            "+----------------------------------------+",
            "| {id: 30313233343536373839616263646566} |",
            "| {id: 66656463626139383736353433323130} |",
            "+----------------------------------------+",
        ],
        &result
    );

    Ok(())
}
