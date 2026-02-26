use std::sync::Arc;

use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::array::{Array, AsArray, Float32Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::Result as DfResult;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};
use vdb_common::metrics::DistanceMetric;

/// Create a cosine distance UDF.
pub fn cosine_distance_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(VectorDistanceUdf::new(
        "cosine_distance",
        DistanceMetric::Cosine,
    ))
}

/// Create an L2 distance UDF.
pub fn l2_distance_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(VectorDistanceUdf::new("l2_distance", DistanceMetric::L2))
}

/// Create an inner product distance UDF.
pub fn inner_product_distance_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(VectorDistanceUdf::new(
        "inner_product_distance",
        DistanceMetric::InnerProduct,
    ))
}

#[derive(Debug)]
struct VectorDistanceUdf {
    name: String,
    metric: DistanceMetric,
    signature: Signature,
}

impl VectorDistanceUdf {
    fn new(name: &str, metric: DistanceMetric) -> Self {
        Self {
            name: name.to_string(),
            metric,
            signature: Signature::new(TypeSignature::Any(2), Volatility::Immutable),
        }
    }
}

impl PartialEq for VectorDistanceUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.metric == other.metric
    }
}
impl Eq for VectorDistanceUdf {}

impl std::hash::Hash for VectorDistanceUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl ScalarUDFImpl for VectorDistanceUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DfResult<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DfResult<ColumnarValue> {
        let ScalarFunctionArgs { args, .. } = args;
        if args.len() != 2 {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "{} requires exactly 2 arguments",
                self.name
            )));
        }

        let (vectors_col, query_col) = match (&args[0], &args[1]) {
            (ColumnarValue::Array(a), ColumnarValue::Array(b)) => (a.clone(), b.clone()),
            (ColumnarValue::Array(a), ColumnarValue::Scalar(s)) => {
                let b = s.to_array_of_size(a.len())?;
                (a.clone(), b)
            }
            _ => {
                return Err(datafusion::error::DataFusionError::Execution(
                    "first argument must be an array (column)".to_string(),
                ));
            }
        };

        let vectors_fsl = vectors_col.as_fixed_size_list_opt().ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "first argument must be FixedSizeList".to_string(),
            )
        })?;
        let query_fsl = query_col.as_fixed_size_list_opt().ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "second argument must be FixedSizeList".to_string(),
            )
        })?;

        let dim = vectors_fsl.value_length() as usize;
        let n = vectors_fsl.len();

        let vec_values = vectors_fsl
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let query_values = query_fsl
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();

        let vec_raw = vec_values.values();
        let query_raw = query_values.values();
        let metric = self.metric;

        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let v_start = i * dim;
            let v_end = v_start + dim;
            let q_start = i * dim;
            let q_end = q_start + dim;

            if v_end <= vec_raw.len() && q_end <= query_raw.len() {
                let v = &vec_raw[v_start..v_end];
                let q = &query_raw[q_start..q_end];
                distances.push(metric.compute(v, q));
            } else {
                distances.push(f32::NAN);
            }
        }

        let result: ArrayRef = Arc::new(Float32Array::from(distances));
        Ok(ColumnarValue::Array(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_udf_creation() {
        let udf = cosine_distance_udf();
        assert_eq!(udf.name(), "cosine_distance");

        let udf = l2_distance_udf();
        assert_eq!(udf.name(), "l2_distance");
    }
}
