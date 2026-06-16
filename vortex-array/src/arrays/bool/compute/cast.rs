// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::One;
use num_traits::Zero;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::dtype::DType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Bool {
    fn cast(array: ArrayView<'_, Bool>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Only a same-type (boolean) restructure is reducible without reading buffers; casts to
        // other types convert values and are handled by the kernel.
        if !dtype.is_boolean() {
            return Ok(None);
        }

        let Some(new_validity) = array
            .validity()?
            .trivially_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };
        Ok(Some(
            BoolArray::new(array.to_bit_buffer(), new_validity).into_array(),
        ))
    }
}

impl CastKernel for Bool {
    fn cast(
        array: ArrayView<'_, Bool>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match dtype {
            DType::Bool(_) => {
                let new_validity =
                    array
                        .validity()?
                        .cast_nullability(dtype.nullability(), array.len(), ctx)?;
                Ok(Some(
                    BoolArray::new(array.to_bit_buffer(), new_validity).into_array(),
                ))
            }
            DType::Primitive(ptype, new_nullability) => {
                let new_validity =
                    array
                        .validity()?
                        .cast_nullability(*new_nullability, array.len(), ctx)?;

                let bits = array.to_bit_buffer();
                match_each_native_ptype!(ptype, |T| {
                    let buffer = Buffer::<T>::from_trusted_len_iter(
                        bits.iter().map(|b| if b { T::one() } else { T::zero() }),
                    );
                    Ok(Some(PrimitiveArray::new(buffer, new_validity).into_array()))
                })
            }
            DType::Utf8(_) => {
                let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), array.len());
                let bits = array.to_bit_buffer();
                let mask = array.validity()?.execute_mask(array.len(), ctx)?;

                match &mask {
                    Mask::AllTrue(_) => {
                        for b in bits.iter() {
                            builder.append_value(if b { "true" } else { "false" });
                        }
                    }
                    Mask::AllFalse(_) => {
                        for _ in 0..array.len() {
                            builder.append_null();
                        }
                    }
                    Mask::Values(values) => {
                        for (b, valid) in bits.iter().zip(values.bit_buffer().iter()) {
                            if valid {
                                builder.append_value(if b { "true" } else { "false" });
                            } else {
                                builder.append_null();
                            }
                        }
                    }
                }

                Ok(Some(builder.finish().into_array()))
            }
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_session::VortexSession;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::session::ArraySession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn try_cast_bool_success() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), Some(true)]);

        let res = bool
            .into_array()
            .cast(DType::Bool(Nullability::NonNullable));
        assert!(res.is_ok());
        assert_eq!(res.unwrap().dtype(), &DType::Bool(Nullability::NonNullable));
    }

    #[test]
    fn try_cast_bool_fail() {
        // When the validity array's min stat is not cached, the reduce rule defers and the
        // failure surfaces during execution via the kernel (cast_nullability -> compute_min).
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), None]);
        let mut ctx = SESSION.create_execution_ctx();
        let result = bool
            .into_array()
            .cast(DType::Bool(Nullability::NonNullable))
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()));
        assert!(result.is_err(), "Expected error, got: {result:?}");
    }

    #[test]
    fn cast_bool_to_i64() {
        use crate::arrays::PrimitiveArray;
        use crate::assert_arrays_eq;
        use crate::dtype::PType;

        let bool_array = BoolArray::from_iter(vec![true, false, true, false]);
        let result = bool_array
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();

        let expected = PrimitiveArray::from_iter(vec![1i64, 0, 1, 0]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn cast_bool_to_i64_with_nulls() {
        use crate::arrays::PrimitiveArray;
        use crate::assert_arrays_eq;
        use crate::dtype::PType;

        let bool_array = BoolArray::from_iter(vec![Some(true), None, Some(false), Some(true)]);
        let result = bool_array
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::Nullable))
            .unwrap();

        let expected = PrimitiveArray::from_option_iter(vec![Some(1i64), None, Some(0), Some(1)]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn cast_bool_to_utf8() {
        use crate::arrays::VarBinViewArray;
        use crate::assert_arrays_eq;

        let bool_array = BoolArray::from_iter(vec![true, false, true, false]);
        let result = bool_array
            .into_array()
            .cast(DType::Utf8(Nullability::NonNullable))
            .unwrap();

        let expected = VarBinViewArray::from_iter_str(vec!["true", "false", "true", "false"]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn cast_bool_to_utf8_with_nulls() {
        use crate::arrays::VarBinViewArray;
        use crate::assert_arrays_eq;

        let bool_array = BoolArray::from_iter(vec![Some(true), None, Some(false), Some(true)]);
        let result = bool_array
            .into_array()
            .cast(DType::Utf8(Nullability::Nullable))
            .unwrap();

        let expected = VarBinViewArray::from_iter_nullable_str(vec![
            Some("true"),
            None,
            Some("false"),
            Some("true"),
        ]);
        assert_arrays_eq!(result, expected);
    }

    #[rstest]
    #[case(BoolArray::from_iter(vec![true, false, true, true, false]))]
    #[case(BoolArray::from_iter(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case(BoolArray::from_iter(vec![true]))]
    #[case(BoolArray::from_iter(vec![false, false]))]
    fn test_cast_bool_conformance(#[case] array: BoolArray) {
        test_cast_conformance(&array.into_array());
    }
}
