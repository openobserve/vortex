use std::sync::Arc;

use arrow_array::BooleanArray;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_datafusion::VortexFormat;

/// This example demonstrates a schema evolution error in DataFusion/Vortex.
///
/// Two Vortex files with incompatible schemas for the 'code' field:
/// - File 1: code is UTF8 (string)
/// - File 2: code is Int64 (integer)
///
/// DataFusion will fail when attempting to query both files with a unified schema.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.path();

    let session = VortexSession::default();

    // Create first file: code field as UTF8
    let schema1 = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("code", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]);

    let id1 = Int64Array::from(vec![1, 2, 3]);
    let code1 = Int64Array::from(vec![100, 200, 300]);
    let _code1 = Float64Array::from(vec![100.0, 200.0, 300.0]);
    let _code1 = StringArray::from(vec!["100", "200", "300"]);
    let value1 = Int64Array::from(vec![100, 200, 300]);

    let batch1 = RecordBatch::try_new(
        Arc::new(schema1.clone()),
        vec![Arc::new(id1), Arc::new(code1), Arc::new(value1)],
    )?;

    let file1_path = temp_path.join("data_utf8.vortex");
    let mut file1 = tokio::fs::File::create(&file1_path).await?;

    let vortex_array1 = ArrayRef::from_arrow(batch1.clone(), false)?;
    session
        .write_options()
        .write(&mut file1, vortex_array1.to_array_stream())
        .await?;

    // Create second file: code field as Int64
    let schema2 = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("code", DataType::Boolean, false),
        Field::new("value", DataType::Int64, false),
    ]);

    let id2 = Int64Array::from(vec![4, 5, 6]);
    let _code2 = Float64Array::from(vec![400.0, 500.0, 600.0]);
    let _code2 = Int64Array::from(vec![400, 500, 600]);
    let code2 = BooleanArray::from(vec![true, false, true]);
    let value2 = Int64Array::from(vec![400, 500, 600]);

    let batch2 = RecordBatch::try_new(
        Arc::new(schema2),
        vec![Arc::new(id2), Arc::new(code2), Arc::new(value2)],
    )?;

    let file2_path = temp_path.join("data_int64.vortex");
    let mut file2 = tokio::fs::File::create(&file2_path).await?;

    let vortex_array2 = ArrayRef::from_arrow(batch2.clone(), false)?;
    session
        .write_options()
        .write(&mut file2, vortex_array2.to_array_stream())
        .await?;

    let config = SessionConfig::from_env()?;
    let ctx = SessionContext::new_with_config(config);

    let vortex_options = ListingOptions::new(Arc::new(VortexFormat::new(session)))
        .with_session_config_options(ctx.state().config());

    let prefix = ListingTableUrl::parse(
        temp_path
            .to_str()
            .ok_or_else(|| "Invalid path".to_string())?,
    )?;
    let listing_config = ListingTableConfig::new(prefix)
        .with_listing_options(vortex_options)
        .with_schema(Arc::new(schema1.clone()))
        .with_expr_adapter_factory(Arc::new(DefaultPhysicalExprAdapterFactory {}));

    let table = ListingTable::try_new(listing_config)?;
    ctx.register_table("test_data", Arc::new(table))?;

    let sql = "SELECT * FROM test_data ORDER BY id";

    let result = ctx.sql(sql).await?.show().await;

    match result {
        Ok(_) => println!("Query succeeded unexpectedly"),
        Err(e) => println!("Schema evolution error occurred:\n{}", e),
    }

    // Keep temp dir for inspection
    let temp_path_str = temp_path.to_string_lossy().to_string();
    let _temp_dir = temp_dir.keep();
    println!("\nFiles preserved in: {}", temp_path_str);

    Ok(())
}
