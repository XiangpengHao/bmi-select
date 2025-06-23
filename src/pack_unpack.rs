use crate::lane_size;

/// Private trait to seal BitPackable
mod sealed {
    pub trait Sealed {}
}

/// Trait for types that can be bit-packed
///
/// This trait is sealed and can only be implemented for the predefined unsigned integer types.
pub trait BitPackable: Copy + Into<u64> + sealed::Sealed {
    /// Maximum number of bits this type can represent
    fn bit_width() -> usize;

    /// Cast from u64 to this type (safe when value fits in bit_width)
    fn from_u64(value: u64) -> Self;
}

impl sealed::Sealed for u8 {}
impl BitPackable for u8 {
    fn bit_width() -> usize {
        8
    }

    fn from_u64(value: u64) -> Self {
        value as u8
    }
}

impl sealed::Sealed for u16 {}
impl BitPackable for u16 {
    fn bit_width() -> usize {
        16
    }

    fn from_u64(value: u64) -> Self {
        value as u16
    }
}

impl sealed::Sealed for u32 {}
impl BitPackable for u32 {
    fn bit_width() -> usize {
        32
    }

    fn from_u64(value: u64) -> Self {
        value as u32
    }
}

impl sealed::Sealed for u64 {}
impl BitPackable for u64 {
    fn bit_width() -> usize {
        64
    }

    fn from_u64(value: u64) -> Self {
        value
    }
}

/// Pack a slice of values into a vector of u64 values.
/// Each of the values must be less than 2^bit_width.
///
/// Example:
/// ```
/// use bmi_select::bit_pack;
/// let data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// let bit_width = 4;
/// let packed = bit_pack(&data, bit_width);
/// assert_eq!(packed, vec![0x87654321]);
/// ```
pub fn bit_pack<T: BitPackable>(data: &[T], bit_width: usize) -> Vec<u64> {
    if data.is_empty() || bit_width == 0 || bit_width > T::bit_width() {
        return Vec::new();
    }

    let total_bits = data.len() * bit_width;

    let mut out = vec![0u64; total_bits.div_ceil(lane_size::<u64>())];

    unsafe {
        let mut out_ptr = out.as_mut_ptr();
        let mut word: u64 = 0;
        let mut bits_in_word = 0usize;

        for &value in data {
            let u64_value: u64 = value.into();
            word |= u64_value << bits_in_word;
            bits_in_word += bit_width;

            if bits_in_word >= 64 {
                *out_ptr = word;
                out_ptr = out_ptr.add(1);

                if bits_in_word > 64 {
                    word = u64_value >> (bit_width - (bits_in_word - 64));
                    bits_in_word -= 64;
                } else {
                    word = 0;
                    bits_in_word = 0;
                }
            }
        }

        if bits_in_word > 0 {
            *out_ptr = word;
        }
    }

    out
}

/// Unpack a vector of u64 values into a vector of values of type T.
/// Each of the values must be less than 2^bit_width.
///
/// Example:
/// ```
/// use bmi_select::bit_unpack;
/// let packed = vec![0x87654321];
/// let bit_width = 4;
/// let unpacked: Vec<u32> = bit_unpack(&packed, bit_width, 8);
/// assert_eq!(unpacked, vec![1, 2, 3, 4, 5, 6, 7, 8]);
/// ```
pub fn bit_unpack<T: BitPackable>(
    packed_data: &[u64],
    bit_width: usize,
    original_count: usize,
) -> Vec<T> {
    if packed_data.is_empty() || bit_width == 0 || bit_width > T::bit_width() || original_count == 0
    {
        return Vec::new();
    }

    unsafe { bit_unpack_impl(packed_data, bit_width, original_count) }
}

unsafe fn bit_unpack_impl<T: BitPackable>(
    packed_data: &[u64],
    bit_width: usize,
    original_count: usize,
) -> Vec<T> {
    let mask = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    let mut result = vec![T::from_u64(0); original_count];

    let mut out_ptr = result.as_mut_ptr();

    let mut in_idx = 0usize;
    let mut current = unsafe { *packed_data.get_unchecked(in_idx) };
    in_idx += 1;
    let mut next = if in_idx < packed_data.len() {
        unsafe { *packed_data.get_unchecked(in_idx) }
    } else {
        0
    };

    let mut bit_offset = 0u32;
    let w = bit_width as u32;
    let mut remaining = original_count;

    macro_rules! decode_value {
        () => {{
            let val: u64;
            if bit_offset <= 64 - w {
                val = (current >> bit_offset) & mask;
                bit_offset += w;
                if bit_offset == 64 {
                    current = next;
                    in_idx += 1;
                    next = if in_idx < packed_data.len() {
                        unsafe { *packed_data.get_unchecked(in_idx) }
                    } else {
                        0
                    };
                    bit_offset = 0;
                }
            } else {
                let combined = (current >> bit_offset) | (next << (64 - bit_offset));
                val = combined & mask;
                current = next;
                in_idx += 1;
                next = if in_idx < packed_data.len() {
                    unsafe { *packed_data.get_unchecked(in_idx) }
                } else {
                    0
                };
                bit_offset = bit_offset + w - 64;
            }
            val
        }};
    }

    while remaining >= 8 {
        let v0 = decode_value!();
        let v1 = decode_value!();
        let v2 = decode_value!();
        let v3 = decode_value!();
        let v4 = decode_value!();
        let v5 = decode_value!();
        let v6 = decode_value!();
        let v7 = decode_value!();
        unsafe {
            core::ptr::write(out_ptr, T::from_u64(v0));
            core::ptr::write(out_ptr.add(1), T::from_u64(v1));
            core::ptr::write(out_ptr.add(2), T::from_u64(v2));
            core::ptr::write(out_ptr.add(3), T::from_u64(v3));
            core::ptr::write(out_ptr.add(4), T::from_u64(v4));
            core::ptr::write(out_ptr.add(5), T::from_u64(v5));
            core::ptr::write(out_ptr.add(6), T::from_u64(v6));
            core::ptr::write(out_ptr.add(7), T::from_u64(v7));
            out_ptr = out_ptr.add(8);
        }
        remaining -= 8;
    }

    while remaining > 0 {
        let v = decode_value!();
        unsafe {
            core::ptr::write(out_ptr, T::from_u64(v));
            out_ptr = out_ptr.add(1);
        }
        remaining -= 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bit_pack_round_trip(data: &Vec<u64>, bit_width: usize) {
        let packed = bit_pack(data, bit_width);
        let unpacked: Vec<u64> = bit_unpack(&packed, bit_width, data.len());
        assert_eq!(data, &unpacked);
    }

    #[test]
    fn test_bit_pack_unpack() {
        let data: Vec<u64> = (0..200).map(|x| x as u64).collect();

        for bit_width in 8..64 {
            test_bit_pack_round_trip(&data, bit_width);
        }
    }

    #[test]
    fn test_bit_pack_unpack_33_bit() {
        // Test the inefficient case mentioned by the user
        let data: Vec<u64> = vec![1, 2, 3, 4, 5];
        let bit_width = 33;

        let packed = bit_pack(&data, bit_width);
        let unpacked: Vec<u64> = bit_unpack(&packed, bit_width, data.len());

        assert_eq!(data, unpacked);

        // Verify we're using space efficiently
        let expected_bits = data.len() * bit_width; // 5 * 33 = 165 bits
        let expected_u64s = expected_bits.div_ceil(64); // 3 u64s
        assert_eq!(packed.len(), expected_u64s);
    }

    #[test]
    fn test_bit_pack_unpack_7_bit() {
        // Test odd bit width that doesn't divide evenly into 64
        let data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let bit_width = 7;

        let packed = bit_pack(&data, bit_width);
        let unpacked: Vec<u64> = bit_unpack(&packed, bit_width, data.len());

        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<u64> = vec![];
        let packed = bit_pack(&data, 8);
        let unpacked: Vec<u64> = bit_unpack(&packed, 8, 0);

        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_single_value() {
        let data: Vec<u64> = vec![42];
        let bit_width = 8;

        let packed = bit_pack(&data, bit_width);
        let unpacked: Vec<u64> = bit_unpack(&packed, bit_width, data.len());

        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_space_efficiency() {
        // Test that we're actually saving space
        let data = vec![1u64; 100]; // 100 values
        let bit_width = 10; // Each value needs 10 bits

        let packed = bit_pack(&data, bit_width);

        // 100 values * 10 bits = 1000 bits
        // 1000 bits / 64 bits per u64 = 15.625, so we need 16 u64s
        let expected_u64s = (100 * 10usize).div_ceil(64);
        assert_eq!(packed.len(), expected_u64s);

        // This should be much less than storing 100 full u64s
        assert!(packed.len() < 100);
    }

    #[test]
    fn test_different_types() {
        // Test u8
        let data_u8: Vec<u8> = vec![1, 2, 3, 4, 5];
        let packed_u8 = bit_pack(&data_u8, 4);
        let unpacked_u8: Vec<u8> = bit_unpack(&packed_u8, 4, data_u8.len());
        assert_eq!(data_u8, unpacked_u8);

        // Test u16
        let data_u16: Vec<u16> = vec![100, 200, 300, 400, 500];
        let packed_u16 = bit_pack(&data_u16, 10);
        let unpacked_u16: Vec<u16> = bit_unpack(&packed_u16, 10, data_u16.len());
        assert_eq!(data_u16, unpacked_u16);

        // Test u32
        let data_u32: Vec<u32> = vec![1000, 2000, 3000, 4000, 5000];
        let packed_u32 = bit_pack(&data_u32, 16);
        let unpacked_u32: Vec<u32> = bit_unpack(&packed_u32, 16, data_u32.len());
        assert_eq!(data_u32, unpacked_u32);
    }
}
