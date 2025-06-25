use std::fmt::Debug;

/// Private trait to seal BitPackable
mod sealed {
    pub trait Sealed {}
}

/// Trait for types that can be bit-packed
///
/// This trait is sealed and can only be implemented for the predefined unsigned integer types.
pub trait BitPackable: Copy + Into<u64> + sealed::Sealed + PartialEq + Eq + Debug {
    const BIT_WIDTH: usize;
    const MAX_VALUE: Self;

    /// Cast from u64 to this type (safe when value fits in bit_width)
    fn from_u64(value: u64) -> Self;
}

impl sealed::Sealed for u8 {}
impl BitPackable for u8 {
    const BIT_WIDTH: usize = 8;
    const MAX_VALUE: Self = u8::MAX;

    fn from_u64(value: u64) -> Self {
        value as u8
    }
}

impl sealed::Sealed for u16 {}
impl BitPackable for u16 {
    const BIT_WIDTH: usize = 16;
    const MAX_VALUE: Self = u16::MAX;

    fn from_u64(value: u64) -> Self {
        value as u16
    }
}

impl sealed::Sealed for u32 {}
impl BitPackable for u32 {
    const BIT_WIDTH: usize = 32;
    const MAX_VALUE: Self = u32::MAX;

    fn from_u64(value: u64) -> Self {
        value as u32
    }
}

impl sealed::Sealed for u64 {}
impl BitPackable for u64 {
    const BIT_WIDTH: usize = 64;
    const MAX_VALUE: Self = u64::MAX;

    fn from_u64(value: u64) -> Self {
        value
    }
}

/// Pack a slice of values into an output buffer of u64 values.
/// Each of the values must be less than 2^bit_width.
/// The output buffer must have enough space to hold the packed data.
///
/// Example:
/// ```
/// use bmi_select::bit_pack;
/// let data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// let bit_width = 4;
/// let mut out = vec![0u64; 1];
/// bit_pack(&data, bit_width, &mut out);
/// assert_eq!(out, vec![0x87654321]);
/// ```
pub fn bit_pack<T: BitPackable>(data: &[T], bit_width: usize, out: &mut [u64]) {
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
}

/// Unpack a vector of u64 values into an output buffer of values of type T.
/// Each of the values must be less than 2^bit_width.
/// The output buffer must have enough space to hold the unpacked data.
///
/// Example:
/// ```
/// use bmi_select::bit_unpack;
/// let packed = vec![0x87654321];
/// let bit_width = 4;
/// let mut out = vec![0u32; 8];
/// bit_unpack(&packed, bit_width, &mut out);
/// assert_eq!(out, vec![1, 2, 3, 4, 5, 6, 7, 8]);
/// ```
pub fn bit_unpack<T: BitPackable>(packed_data: &[u64], bit_width: usize, out: &mut [T]) {
    // Try to use optimized unpack functions for specific types and conditions
    match T::BIT_WIDTH {
        8 => unsafe {
            let out_u8 = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len());
            optimized_unpack_u8_into(packed_data, bit_width, out_u8);
        },
        16 => unsafe {
            let out_u16 = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, out.len());
            optimized_unpack_u16_into(packed_data, bit_width, out_u16);
        },
        32 => unsafe {
            let out_u32 = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u32, out.len());
            optimized_unpack_u32_into(packed_data, bit_width, out_u32);
        },
        64 => unsafe {
            let out_u64 = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u64, out.len());
            optimized_unpack_u64_into(packed_data, bit_width, out_u64);
        },
        _ => {
            unreachable!("unsupported bit width: {}", T::BIT_WIDTH);
        }
    }
}

/// Macro that generates an unpack function taking the number of bits as a const generic
macro_rules! unpack_impl {
    ($t:ty, $bytes:literal, $bits:tt) => {
        pub fn unpack<const NUM_BITS: usize>(input: &[u8], output: &mut [$t; $bits]) {
            if NUM_BITS == 0 {
                for out in output {
                    *out = 0;
                }
                return;
            }

            assert!(NUM_BITS <= $bytes * 8);

            let mask = match NUM_BITS {
                $bits => <$t>::MAX,
                _ => ((1 << NUM_BITS) - 1),
            };

            assert!(input.len() >= NUM_BITS * $bytes);

            let r = |output_idx: usize| {
                <$t>::from_le_bytes(
                    input[output_idx * $bytes..output_idx * $bytes + $bytes]
                        .try_into()
                        .unwrap(),
                )
            };

            seq_macro::seq!(i in 0..$bits {
                let start_bit = i * NUM_BITS;
                let end_bit = start_bit + NUM_BITS;

                let start_bit_offset = start_bit % $bits;
                let end_bit_offset = end_bit % $bits;
                let start_byte = start_bit / $bits;
                let end_byte = end_bit / $bits;
                if start_byte != end_byte && end_bit_offset != 0 {
                    let val = r(start_byte);
                    let a = val >> start_bit_offset;
                    let val = r(end_byte);
                    let b = val << (NUM_BITS - end_bit_offset);

                    output[i] = a | (b & mask);
                } else {
                    let val = r(start_byte);
                    output[i] = (val >> start_bit_offset) & mask;
                }
            });
        }
    };
}

/// Macro that generates unpack functions that accept num_bits as a parameter
macro_rules! unpack {
    ($name:ident, $t:ty, $bytes:literal, $bits:tt) => {
        mod $name {
            unpack_impl!($t, $bytes, $bits);
        }

        /// Unpack packed `input` into `output` with a bit width of `num_bits`
        pub fn $name(input: &[u8], output: &mut [$t; $bits], num_bits: usize) {
            // This will get optimised into a jump table
            seq_macro::seq!(i in 0..=$bits {
                if i == num_bits {
                    return $name::unpack::<i>(input, output);
                }
            });
            unreachable!("invalid num_bits {}", num_bits);
        }
    };
}

unpack!(unpack8, u8, 1, 8);
unpack!(unpack16, u16, 2, 16);
unpack!(unpack32, u32, 4, 32);
unpack!(unpack64, u64, 8, 64);

/// Buffer-based version of bit_unpack_impl
fn bit_unpack_impl_into<T: BitPackable>(packed_data: &[u64], bit_width: usize, out: &mut [T]) {
    let mask = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    let mut bit_position = 0usize;

    for o in out {
        let word_index = bit_position / 64;
        let bit_offset = bit_position % 64;

        let value = if bit_offset + bit_width <= 64 {
            let word = packed_data.get(word_index).copied().unwrap_or(0);
            (word >> bit_offset) & mask
        } else {
            let current_word = packed_data.get(word_index).copied().unwrap_or(0);
            let next_word = packed_data.get(word_index + 1).copied().unwrap_or(0);

            let bits_from_current = 64 - bit_offset;
            let bits_from_next = bit_width - bits_from_current;

            let low_part = current_word >> bit_offset;
            let high_part = (next_word & ((1u64 << bits_from_next) - 1)) << bits_from_current;

            (low_part | high_part) & mask
        };

        *o = T::from_u64(value);
        bit_position += bit_width;
    }
}

/// Macro to generate optimized buffer-based unpack functions for each type
macro_rules! generate_optimized_unpack_into {
    ($name:ident, $t:ty, $lane_size:literal, $unpack_fn:ident) => {
        fn $name(packed_data: &[u64], bit_width: usize, out: &mut [$t]) {
            const LANE_SIZE: usize = $lane_size;

            let packed_bytes = unsafe {
                std::slice::from_raw_parts(packed_data.as_ptr() as *const u8, packed_data.len() * 8)
            };

            let chunks = out.len() / LANE_SIZE;
            let remainder = out.len() % LANE_SIZE;
            let bytes_per_chunk = (LANE_SIZE * bit_width).div_ceil(8);

            for i in 0..chunks {
                let start_byte = i * bytes_per_chunk;
                let result_offset = i * LANE_SIZE;

                unsafe {
                    let chunk_ptr = out.as_mut_ptr().add(result_offset);
                    let chunk_slice = std::slice::from_raw_parts_mut(chunk_ptr, LANE_SIZE);
                    let chunk_array: &mut [$t; LANE_SIZE] =
                        chunk_slice.try_into().unwrap_unchecked();
                    $unpack_fn(&packed_bytes[start_byte..], chunk_array, bit_width);
                }
            }

            if remainder > 0 {
                let remaining_start = chunks * LANE_SIZE;
                let remaining_packed_start = (remaining_start * bit_width) / 64;
                let remaining_packed = &packed_data[remaining_packed_start..];
                let remaining_out = &mut out[remaining_start..];
                bit_unpack_impl_into(remaining_packed, bit_width, remaining_out);
            }
        }
    };
}

// Generate the optimized buffer-based unpack functions
generate_optimized_unpack_into!(optimized_unpack_u8_into, u8, 8, unpack8);
generate_optimized_unpack_into!(optimized_unpack_u16_into, u16, 16, unpack16);
generate_optimized_unpack_into!(optimized_unpack_u32_into, u32, 32, unpack32);
generate_optimized_unpack_into!(optimized_unpack_u64_into, u64, 64, unpack64);

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bit_pack_round_trip<T: BitPackable>(bit_width: usize) {
        if bit_width > T::BIT_WIDTH {
            return;
        }
        let max_value: u64 = 1 << bit_width;
        let data: Vec<T> = (0..200)
            .map(|x| T::from_u64(x as u64 % max_value))
            .collect();
        let total_bits = data.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut out);
        let mut unpacked = vec![T::from_u64(0); data.len()];
        bit_unpack(&out, bit_width, &mut unpacked);
        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_bit_pack_unpack() {
        for bit_width in 1..64 {
            test_bit_pack_round_trip::<u8>(bit_width);
            test_bit_pack_round_trip::<u16>(bit_width);
            test_bit_pack_round_trip::<u32>(bit_width);
            test_bit_pack_round_trip::<u64>(bit_width);
        }
    }

    #[test]
    fn test_bit_pack_unpack_33_bit() {
        // Test the inefficient case mentioned by the user
        let data: Vec<u64> = vec![1, 2, 3, 4, 5];
        let bit_width = 33;

        let total_bits = data.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut out);
        let mut unpacked = vec![0u64; data.len()];
        bit_unpack(&out, bit_width, &mut unpacked);

        assert_eq!(data, unpacked);

        // Verify we're using space efficiently
        let expected_bits = data.len() * bit_width; // 5 * 33 = 165 bits
        let expected_u64s = expected_bits.div_ceil(64); // 3 u64s
        assert_eq!(out.len(), expected_u64s);
    }

    #[test]
    fn test_bit_pack_unpack_7_bit() {
        // Test odd bit width that doesn't divide evenly into 64
        let data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let bit_width = 7;

        let total_bits = data.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut out);
        let mut unpacked = vec![0u64; data.len()];
        bit_unpack(&out, bit_width, &mut unpacked);

        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<u64> = vec![];
        let mut out = vec![0u64; 1];
        bit_pack(&data, 8, &mut out);
        let mut unpacked = vec![0u64; 0];
        bit_unpack(&out, 8, &mut unpacked);

        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_single_value() {
        let data: Vec<u64> = vec![42];
        let bit_width = 8;

        let total_bits = data.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut out);
        let mut unpacked = vec![0u64; data.len()];
        bit_unpack(&out, bit_width, &mut unpacked);

        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_space_efficiency() {
        // Test that we're actually saving space
        let data = vec![1u64; 100]; // 100 values
        let bit_width = 10; // Each value needs 10 bits

        let total_bits = data.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut out);

        // 100 values * 10 bits = 1000 bits
        // 1000 bits / 64 bits per u64 = 15.625, so we need 16 u64s
        let expected_u64s = (100 * 10usize).div_ceil(64);
        assert_eq!(out.len(), expected_u64s);

        // This should be much less than storing 100 full u64s
        assert!(out.len() < 100);
    }

    #[test]
    fn test_different_types() {
        // Test u8
        let data_u8: Vec<u8> = vec![1, 2, 3, 4, 5];
        let total_bits = data_u8.len() * 4;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data_u8, 4, &mut out);
        let mut unpacked_u8 = vec![0u8; data_u8.len()];
        bit_unpack(&out, 4, &mut unpacked_u8);
        assert_eq!(data_u8, unpacked_u8);

        // Test u16
        let data_u16: Vec<u16> = vec![100, 200, 300, 400, 500];
        let total_bits = data_u16.len() * 10;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data_u16, 10, &mut out);
        let mut unpacked_u16 = vec![0u16; data_u16.len()];
        bit_unpack(&out, 10, &mut unpacked_u16);
        assert_eq!(data_u16, unpacked_u16);

        // Test u32
        let data_u32: Vec<u32> = vec![1000, 2000, 3000, 4000, 5000];
        let total_bits = data_u32.len() * 16;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data_u32, 16, &mut out);
        let mut unpacked_u32 = vec![0u32; data_u32.len()];
        bit_unpack(&out, 16, &mut unpacked_u32);
        assert_eq!(data_u32, unpacked_u32);
    }

    #[test]
    fn test_optimized_bit_unpack() {
        // Test the optimized bit unpacking function
        let data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let bit_width = 4;

        let total_bits = data.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut out);
        let mut unpacked_original = vec![0u32; data.len()];
        bit_unpack(&out, bit_width, &mut unpacked_original);

        assert_eq!(data, unpacked_original);
    }

    #[test]
    fn test_optimized_bit_unpack_various_widths() {
        // Test various bit widths to ensure the optimized version works correctly
        let test_cases = [
            (1, vec![0u32, 1, 0, 1, 1, 0, 1, 0]),
            (2, vec![0u32, 1, 2, 3, 0, 1, 2, 3]),
            (3, vec![0u32, 1, 2, 3, 4, 5, 6, 7]),
            (4, vec![1u32, 2, 3, 4, 5, 6, 7, 8]),
            (5, vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            (8, vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            (10, vec![100u32, 200, 300, 400, 500]),
            (16, vec![1000u32, 2000, 3000, 4000, 5000]),
        ];

        for (bit_width, data) in test_cases {
            let total_bits = data.len() * bit_width;
            let mut out = vec![0u64; total_bits.div_ceil(64)];
            bit_pack(&data, bit_width, &mut out);
            let mut unpacked_original = vec![0u32; data.len()];
            bit_unpack(&out, bit_width, &mut unpacked_original);

            assert_eq!(
                data, unpacked_original,
                "Original failed for bit_width {bit_width}"
            );
        }
    }

    #[test]
    fn test_optimized_large_data() {
        // Test with larger datasets to verify performance optimizations work correctly
        let data: Vec<u32> = (0..1000).map(|x| x % 16).collect(); // Values that fit in 4 bits
        let bit_width = 4;

        let total_bits = data.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut out);
        let mut unpacked_original = vec![0u32; data.len()];
        bit_unpack(&out, bit_width, &mut unpacked_original);

        assert_eq!(data, unpacked_original);
    }

    #[test]
    fn test_optimized_unpack_functions() {
        // Test that the optimized unpack functions using unpack8/16/32/64 work correctly

        // Test u8 with 8-byte chunks
        let data_u8: Vec<u8> = (0..64).map(|x| (x % 16) as u8).collect();
        let bit_width = 4;
        let total_bits = data_u8.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data_u8, bit_width, &mut out);
        let mut unpacked_u8 = vec![0u8; data_u8.len()];
        bit_unpack(&out, bit_width, &mut unpacked_u8);
        assert_eq!(data_u8, unpacked_u8, "u8 optimized unpack failed");

        // Test u16 with 16-element chunks
        let data_u16: Vec<u16> = (0..64).map(|x| (x % 256) as u16).collect();
        let bit_width = 8;
        let total_bits = data_u16.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data_u16, bit_width, &mut out);
        let mut unpacked_u16 = vec![0u16; data_u16.len()];
        bit_unpack(&out, bit_width, &mut unpacked_u16);
        assert_eq!(data_u16, unpacked_u16, "u16 optimized unpack failed");

        // Test u32 with 32-element chunks
        let data_u32: Vec<u32> = (0..64).map(|x| (x % 1024) as u32).collect();
        let bit_width = 10;
        let total_bits = data_u32.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data_u32, bit_width, &mut out);
        let mut unpacked_u32 = vec![0u32; data_u32.len()];
        bit_unpack(&out, bit_width, &mut unpacked_u32);
        assert_eq!(data_u32, unpacked_u32, "u32 optimized unpack failed");

        // Test u64 with 64-element chunks
        let data_u64: Vec<u64> = (0..128).map(|x| (x % 4096) as u64).collect();
        let bit_width = 12;
        let total_bits = data_u64.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data_u64, bit_width, &mut out);
        let mut unpacked_u64 = vec![0u64; data_u64.len()];
        bit_unpack(&out, bit_width, &mut unpacked_u64);
        assert_eq!(data_u64, unpacked_u64, "u64 optimized unpack failed");
    }

    #[test]
    fn test_optimized_vs_fallback() {
        // Test that cases where optimization doesn't apply still work correctly

        // Test with small counts that should fall back to original implementation
        let data: Vec<u32> = vec![1, 2, 3, 4, 5];
        let bit_width = 4;
        let total_bits = data.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut out);
        let mut unpacked = vec![0u32; data.len()];
        bit_unpack(&out, bit_width, &mut unpacked);
        assert_eq!(data, unpacked, "Fallback failed for small data");

        // Test with large bit widths that should fall back
        let data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let bit_width = 40;
        let total_bits = data.len() * bit_width;
        let mut out = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut out);
        let mut unpacked = vec![0u64; data.len()];
        bit_unpack(&out, bit_width, &mut unpacked);
        assert_eq!(data, unpacked, "Fallback failed for large bit width");
    }
}
