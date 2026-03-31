// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::One;
use num_traits::Zero;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::array::ArrayRef;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::canonical::ToCanonical;
use crate::dtype::DType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl CastReduce for Bool {
    fn cast(array: &BoolArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        match dtype {
            DType::Bool(_) => {
                let new_nullability = dtype.nullability();
                let new_validity = array
                    .validity()
                    .clone()
                    .cast_nullability(new_nullability, array.len())?;
                Ok(Some(
                    BoolArray::new(array.to_bit_buffer(), new_validity).into_array(),
                ))
            }
            DType::Primitive(ptype, new_nullability) => {
                let new_validity = array
                    .validity()
                    .clone()
                    .cast_nullability(*new_nullability, array.len())?;

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

                match array.validity() {
                    Validity::NonNullable | Validity::AllValid => {
                        for b in bits.iter() {
                            builder.append_value(if b { "true" } else { "false" });
                        }
                    }
                    Validity::AllInvalid => {
                        for _ in 0..array.len() {
                            builder.append_null();
                        }
                    }
                    Validity::Array(validity_array) => {
                        let validity_bits = validity_array.to_bool().to_bit_buffer();
                        for (b, valid) in bits.iter().zip(validity_bits.iter()) {
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
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

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
    #[should_panic]
    fn try_cast_bool_fail() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), None]);
        bool.into_array()
            .cast(DType::Bool(Nullability::NonNullable))
            .unwrap();
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
