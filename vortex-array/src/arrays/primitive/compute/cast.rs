// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::primitive::PrimitiveArray;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::canonical::ToCanonical;
use crate::compute::CastKernel;
use crate::compute::CastKernelAdapter;
use crate::register_kernel;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl CastKernel for PrimitiveVTable {
    fn cast(&self, array: &PrimitiveArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        match dtype {
            DType::Primitive(new_ptype, new_nullability) => {
                cast_primitive_to_primitive(array, *new_ptype, *new_nullability)
            }
            DType::Utf8(_) => cast_primitive_to_utf8(array, dtype),
            _ => Ok(None),
        }
    }
}

fn cast_primitive_to_primitive(
    array: &PrimitiveArray,
    new_ptype: vortex_dtype::PType,
    new_nullability: vortex_dtype::Nullability,
) -> VortexResult<Option<ArrayRef>> {
    // First, check that the cast is compatible with the source array's validity
    let new_validity = array
        .validity()
        .clone()
        .cast_nullability(new_nullability, array.len())?;

    // If the bit width is the same, we can short-circuit and simply update the validity
    if array.ptype() == new_ptype {
        // SAFETY: validity and data buffer still have same length
        return Ok(Some(unsafe {
            PrimitiveArray::new_unchecked_from_handle(
                array.buffer_handle().clone(),
                array.ptype(),
                new_validity,
            )
            .into_array()
        }));
    }

    let mask = array.validity_mask()?;

    // Otherwise, we need to cast the values one-by-one
    Ok(Some(match_each_native_ptype!(new_ptype, |T| {
        match_each_native_ptype!(array.ptype(), |F| {
            PrimitiveArray::new(cast::<F, T>(array.as_slice(), mask)?, new_validity).into_array()
        })
    })))
}

fn cast_primitive_to_utf8(array: &PrimitiveArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
    let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), array.len());

    match_each_native_ptype!(array.ptype(), |T| {
        let slice = array.as_slice::<T>();
        append_primitive_values_to_utf8(&mut builder, slice, array.validity())?
    });

    Ok(Some(builder.finish().into_array()))
}

fn append_primitive_values_to_utf8<T: NativePType>(
    builder: &mut VarBinViewBuilder,
    slice: &[T],
    validity: &Validity,
) -> VortexResult<()> {
    match validity {
        Validity::NonNullable | Validity::AllValid => {
            for value in slice {
                builder.append_value(value.to_string());
            }
        }
        Validity::AllInvalid => {
            for _ in 0..slice.len() {
                builder.append_null();
            }
        }
        Validity::Array(validity_array) => {
            let validity_bits = validity_array.to_bool().to_bit_buffer();
            for (value, valid) in slice.iter().zip(validity_bits.iter()) {
                if valid {
                    builder.append_value(value.to_string());
                } else {
                    builder.append_null();
                }
            }
        }
    }
    Ok(())
}

register_kernel!(CastKernelAdapter(PrimitiveVTable).lift());

fn cast<F: NativePType, T: NativePType>(array: &[F], mask: Mask) -> VortexResult<Buffer<T>> {
    match mask.bit_buffer() {
        AllOr::All => {
            let mut buffer = BufferMut::with_capacity(array.len());
            for item in array {
                let item = T::from(*item).ok_or_else(
                    || vortex_err!(ComputeError: "Failed to cast {} to {:?}", item, T::PTYPE),
                )?;
                // SAFETY: we've pre-allocated the required capacity
                unsafe { buffer.push_unchecked(item) }
            }
            Ok(buffer.freeze())
        }
        AllOr::None => Ok(Buffer::zeroed(array.len())),
        AllOr::Some(b) => {
            // TODO(robert): Depending on density of the buffer might be better to prefill Buffer and only write valid values
            let mut buffer = BufferMut::with_capacity(array.len());
            for (item, valid) in array.iter().zip(b.iter()) {
                if valid {
                    let item = T::from(*item).ok_or_else(
                        || vortex_err!(ComputeError: "Failed to cast {} to {:?}", item, T::PTYPE),
                    )?;
                    // SAFETY: we've pre-allocated the required capacity
                    unsafe { buffer.push_unchecked(item) }
                } else {
                    // SAFETY: we've pre-allocated the required capacity
                    unsafe { buffer.push_unchecked(T::default()) }
                }
            }
            Ok(buffer.freeze())
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexError;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::canonical::ToCanonical;
    use crate::compute::cast;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;

    #[test]
    fn cast_u32_u8() {
        let arr = buffer![0u32, 10, 200].into_array();

        // cast from u32 to u8
        let p = cast(&arr, PType::U8.into()).unwrap().to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]));
        assert_eq!(p.validity(), &Validity::NonNullable);

        // to nullable
        let p = cast(
            p.as_ref(),
            &DType::Primitive(PType::U8, Nullability::Nullable),
        )
        .unwrap()
        .to_primitive();
        assert_arrays_eq!(
            p,
            PrimitiveArray::new(buffer![0u8, 10, 200], Validity::AllValid)
        );
        assert_eq!(p.validity(), &Validity::AllValid);

        // back to non-nullable
        let p = cast(
            p.as_ref(),
            &DType::Primitive(PType::U8, Nullability::NonNullable),
        )
        .unwrap()
        .to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]));
        assert_eq!(p.validity(), &Validity::NonNullable);

        // to nullable u32
        let p = cast(
            p.as_ref(),
            &DType::Primitive(PType::U32, Nullability::Nullable),
        )
        .unwrap()
        .to_primitive();
        assert_arrays_eq!(
            p,
            PrimitiveArray::new(buffer![0u32, 10, 200], Validity::AllValid)
        );
        assert_eq!(p.validity(), &Validity::AllValid);

        // to non-nullable u8
        let p = cast(
            p.as_ref(),
            &DType::Primitive(PType::U8, Nullability::NonNullable),
        )
        .unwrap()
        .to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([0u8, 10, 200]));
        assert_eq!(p.validity(), &Validity::NonNullable);
    }

    #[test]
    fn cast_u32_f32() {
        let arr = buffer![0u32, 10, 200].into_array();
        let u8arr = cast(&arr, PType::F32.into()).unwrap().to_primitive();
        assert_arrays_eq!(u8arr, PrimitiveArray::from_iter([0.0f32, 10., 200.]));
    }

    #[test]
    fn cast_i32_u32() {
        let arr = buffer![-1i32].into_array();
        let error = cast(&arr, PType::U32.into()).err().unwrap();
        let VortexError::ComputeError(s, _) = error else {
            unreachable!()
        };
        assert_eq!(s.to_string(), "Failed to cast -1 to U32");
    }

    #[test]
    fn cast_array_with_nulls_to_nonnullable() {
        let arr = PrimitiveArray::from_option_iter([Some(-1i32), None, Some(10)]);
        let err = cast(arr.as_ref(), PType::I32.into()).unwrap_err();
        let VortexError::InvalidArgument(s, _) = err else {
            unreachable!()
        };
        assert_eq!(
            s.to_string(),
            "Cannot cast array with invalid values to non-nullable type."
        );
    }

    #[test]
    fn cast_with_invalid_nulls() {
        let arr = PrimitiveArray::new(
            buffer![-1i32, 0, 10],
            Validity::from_iter([false, true, true]),
        );
        let p = cast(
            arr.as_ref(),
            &DType::Primitive(PType::U32, Nullability::Nullable),
        )
        .unwrap()
        .to_primitive();
        assert_arrays_eq!(
            p,
            PrimitiveArray::from_option_iter([None, Some(0u32), Some(10)])
        );
        assert_eq!(
            p.validity_mask().unwrap(),
            Mask::from(BitBuffer::from(vec![false, true, true]))
        );
    }

    #[rstest]
    #[case(buffer![0u8, 1, 2, 3, 255].into_array())]
    #[case(buffer![0u16, 100, 1000, 65535].into_array())]
    #[case(buffer![0u32, 100, 1000, 1000000].into_array())]
    #[case(buffer![0u64, 100, 1000, 1000000000].into_array())]
    #[case(buffer![-128i8, -1, 0, 1, 127].into_array())]
    #[case(buffer![-1000i16, -1, 0, 1, 1000].into_array())]
    #[case(buffer![-1000000i32, -1, 0, 1, 1000000].into_array())]
    #[case(buffer![-1000000000i64, -1, 0, 1, 1000000000].into_array())]
    #[case(buffer![0.0f32, 1.5, -2.5, 100.0, 1e6].into_array())]
    #[case(buffer![0.0f64, 1.5, -2.5, 100.0, 1e12].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1u8), None, Some(255), Some(0), None]).into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1i32), None, Some(-100), Some(0), None]).into_array())]
    #[case(buffer![42u32].into_array())]
    fn test_cast_primitive_conformance(#[case] array: crate::ArrayRef) {
        test_cast_conformance(array.as_ref());
    }

    #[test]
    fn cast_i64_to_utf8() {
        use crate::arrays::VarBinViewArray;

        let arr = buffer![100i64, 200, 300, -42].into_array();
        let result = cast(&arr, &DType::Utf8(Nullability::NonNullable)).unwrap();

        let expected = VarBinViewArray::from_iter_str(vec!["100", "200", "300", "-42"]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn cast_i64_to_utf8_with_nulls() {
        use crate::arrays::VarBinViewArray;

        let arr = PrimitiveArray::from_option_iter([Some(100i64), None, Some(300), Some(-42)]);
        let result = cast(arr.as_ref(), &DType::Utf8(Nullability::Nullable)).unwrap();

        let expected = VarBinViewArray::from_iter_nullable_str(vec![
            Some("100"),
            None,
            Some("300"),
            Some("-42"),
        ]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn cast_u32_to_utf8() {
        use crate::arrays::VarBinViewArray;

        let arr = buffer![0u32, 10, 200, 1000].into_array();
        let result = cast(&arr, &DType::Utf8(Nullability::NonNullable)).unwrap();

        let expected = VarBinViewArray::from_iter_str(vec!["0", "10", "200", "1000"]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn cast_f64_to_utf8() {
        use crate::arrays::VarBinViewArray;

        let arr = buffer![1.5f64, -2.5, 100.0].into_array();
        let result = cast(&arr, &DType::Utf8(Nullability::NonNullable)).unwrap();

        let expected = VarBinViewArray::from_iter_str(vec!["1.5", "-2.5", "100"]);
        assert_arrays_eq!(result, expected);
    }
}
