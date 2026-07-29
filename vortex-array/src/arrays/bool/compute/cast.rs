// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::One;
use num_traits::Zero;
use vortex_buffer::BufferMut;
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
        if dtype.is_boolean() {
            let new_validity =
                array
                    .validity()?
                    .cast_nullability(dtype.nullability(), array.len(), ctx)?;
            return Ok(Some(
                BoolArray::new(array.to_bit_buffer(), new_validity).into_array(),
            ));
        }

        if let DType::Utf8(_) = dtype {
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

            return Ok(Some(builder.finish().into_array()));
        }

        let DType::Primitive(new_ptype, new_nullability) = dtype else {
            return Ok(None);
        };

        let new_validity =
            array
                .validity()?
                .cast_nullability(*new_nullability, array.len(), ctx)?;

        let bits = array.to_bit_buffer();
        let len = bits.len();

        Ok(Some(match_each_native_ptype!(*new_ptype, |T| {
            let (one, zero) = (<T as One>::one(), <T as Zero>::zero());
            let mut buffer = BufferMut::<T>::with_capacity(len);
            buffer.extend(bits.iter().map(|v| if v { one } else { zero }));
            PrimitiveArray::new(buffer.freeze(), new_validity).into_array()
        })))
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
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

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
        let mut ctx = SESSION.create_execution_ctx();
        let bool_array = BoolArray::from_iter(vec![true, false, true, false]);
        let result = bool_array
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();

        let expected = PrimitiveArray::from_iter(vec![1i64, 0, 1, 0]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn cast_bool_to_i64_with_nulls() {
        let mut ctx = SESSION.create_execution_ctx();
        let bool_array = BoolArray::from_iter(vec![Some(true), None, Some(false), Some(true)]);
        let result = bool_array
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::Nullable))
            .unwrap();

        let expected = PrimitiveArray::from_option_iter(vec![Some(1i64), None, Some(0), Some(1)]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn cast_bool_to_utf8() {
        let mut ctx = SESSION.create_execution_ctx();
        let bool_array = BoolArray::from_iter(vec![true, false, true, false]);
        let result = bool_array
            .into_array()
            .cast(DType::Utf8(Nullability::NonNullable))
            .unwrap();

        let expected = VarBinViewArray::from_iter_str(vec!["true", "false", "true", "false"]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn cast_bool_to_utf8_with_nulls() {
        let mut ctx = SESSION.create_execution_ctx();
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
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[rstest]
    #[case(BoolArray::from_iter(vec![true, false, true, true, false]))]
    #[case(BoolArray::from_iter(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case(BoolArray::from_iter(vec![true]))]
    #[case(BoolArray::from_iter(vec![false, false]))]
    fn test_cast_bool_conformance(#[case] array: BoolArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[rstest]
    #[case(PType::I8)]
    #[case(PType::I32)]
    #[case(PType::I64)]
    #[case(PType::U8)]
    #[case(PType::U64)]
    #[case(PType::F32)]
    #[case(PType::F64)]
    fn cast_bool_to_primitive(#[case] target: PType) {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = BoolArray::from_iter(vec![true, false, true]).into_array();
        let out = arr
            .cast(DType::Primitive(target, Nullability::NonNullable))
            .unwrap();
        let out = out.execute::<Canonical>(&mut ctx).unwrap().into_array();
        assert_eq!(out.len(), 3);
    }
}
