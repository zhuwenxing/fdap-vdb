use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, TimeUnit};

use crate::config::{CollectionConfig, MetadataFieldType};
use crate::error::{Result, VdbError};

/// Reserved column names
pub const COL_ID: &str = "_id";
pub const COL_VECTOR: &str = "vector";
pub const COL_CREATED_AT: &str = "_created_at";
pub const COL_DELETED: &str = "_deleted";

/// Build an Arrow Schema for a collection.
pub fn collection_schema(config: &CollectionConfig) -> Result<Schema> {
    if config.dimension == 0 {
        return Err(VdbError::InvalidSchema("dimension must be > 0".to_string()));
    }

    let mut fields = vec![
        Field::new(COL_ID, DataType::Utf8, false),
        Field::new(
            COL_VECTOR,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                config.dimension as i32,
            ),
            false,
        ),
    ];

    for mf in &config.metadata_fields {
        let dt = metadata_arrow_type(&mf.field_type);
        fields.push(Field::new(&mf.name, dt, true));
    }

    fields.push(Field::new(
        COL_CREATED_AT,
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    ));
    fields.push(Field::new(COL_DELETED, DataType::Boolean, false));

    Ok(Schema::new(fields))
}

fn metadata_arrow_type(ft: &MetadataFieldType) -> DataType {
    match ft {
        MetadataFieldType::String => DataType::Utf8,
        MetadataFieldType::Int64 => DataType::Int64,
        MetadataFieldType::Float64 => DataType::Float64,
        MetadataFieldType::Bool => DataType::Boolean,
    }
}

/// Extract the vector dimension from a schema.
pub fn vector_dimension(schema: &Schema) -> Result<usize> {
    match schema.field_with_name(COL_VECTOR) {
        Ok(field) => match field.data_type() {
            DataType::FixedSizeList(_, dim) => Ok(*dim as usize),
            dt => Err(VdbError::InvalidSchema(format!(
                "expected FixedSizeList for vector column, got {dt}"
            ))),
        },
        Err(_) => Err(VdbError::InvalidSchema("missing vector column".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MetadataFieldConfig;

    #[test]
    fn test_collection_schema() {
        let config = CollectionConfig {
            name: "test".to_string(),
            dimension: 128,
            distance_metric: crate::metrics::DistanceMetric::Cosine,
            index_config: Default::default(),
            metadata_fields: vec![MetadataFieldConfig {
                name: "category".to_string(),
                field_type: MetadataFieldType::String,
            }],
        };
        let schema = collection_schema(&config).unwrap();
        assert_eq!(schema.fields().len(), 5); // _id, vector, category, _created_at, _deleted
        assert_eq!(vector_dimension(&schema).unwrap(), 128);
    }
}
