// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_schema::DataType;
use datafusion_common::ColumnStatistics;
use datafusion_common::stats::Precision;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::stats::Precision as VortexPrecision;
use vortex::expr::stats::Stat;
use vortex::scalar::Scalar;

use crate::PrecisionExt;
use crate::convert::TryToDataFusion;

/// Convert a stats set for an array with the given dtype.
///
/// `arrow_type` is the schema data type of the column as reported to
/// DataFusion; min/max/sum scalars are cast to it because DType has no view
/// types and DataFusion requires statistics typed exactly as the schema field.
pub(crate) fn stats_set_to_df(
    stats_set: &StatsSet,
    dtype: &DType,
    arrow_type: &DataType,
) -> VortexResult<ColumnStatistics> {
    // Update the total size in bytes.
    let column_size = stats_set.get_as::<usize>(Stat::UncompressedSizeInBytes, &PType::U64.into());

    let scalar_stat = |stat: Stat| {
        stats_set.get(stat).and_then(|stat_val| {
            Scalar::try_new(
                stat.dtype(dtype).vortex_expect("must have a valid dtype"),
                Some(stat_val),
            )
            .vortex_expect("stat somehow had an incompatible `DType`")
            .try_to_df()
            .and_then(|sv| {
                if sv.data_type() == *arrow_type {
                    Ok(sv)
                } else {
                    sv.cast_to(arrow_type)
                        .map_err(|e| vortex_err!("Failed to cast {stat} statistic: {e}"))
                }
            })
            .ok()
        })
    };

    let min = scalar_stat(Stat::Min);
    let max = scalar_stat(Stat::Max);
    let sum = scalar_stat(Stat::Sum);

    let null_count = stats_set.get_as::<usize>(Stat::NullCount, &PType::U64.into());

    Ok(ColumnStatistics {
        null_count: null_count.to_df(),
        min_value: min.to_df(),
        max_value: max.to_df(),
        sum_value: sum.to_df(),
        distinct_count: is_constant_to_distinct_count(
            stats_set.get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable)),
        ),
        byte_size: column_size.to_df(),
    })
}

pub(crate) fn is_constant_to_distinct_count(
    is_constant: VortexPrecision<bool>,
) -> Precision<usize> {
    match is_constant.as_exact() {
        Some(true) => Precision::Exact(1),
        Some(false) | None => Precision::Absent,
    }
}

#[cfg(test)]
mod tests {
    use vortex::expr::stats::Precision as VortexPrecision;

    use super::*;

    #[test]
    fn is_constant_false_does_not_imply_one_distinct_value() -> VortexResult<()> {
        let false_constant = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(false));
        let false_stats = stats_set_to_df(
            &false_constant,
            &DType::Bool(Nullability::NonNullable),
            &DataType::Boolean,
        )?;

        assert_eq!(false_stats.distinct_count, Precision::Absent);

        let true_constant = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(true));
        let true_stats = stats_set_to_df(
            &true_constant,
            &DType::Bool(Nullability::NonNullable),
            &DataType::Boolean,
        )?;

        assert_eq!(true_stats.distinct_count, Precision::Exact(1));

        Ok(())
    }
}
