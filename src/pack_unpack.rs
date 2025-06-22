use crate::lane_size;

/// Pack a slice of u64 values into a vector of u64 values.
/// Each of the u64 values must be less than 2^bit_width.
///
/// Example:
/// ```
/// use bmi_select::bit_pack;
/// let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// let bit_width = 4;
/// let packed = bit_pack(&data, bit_width);
/// assert_eq!(packed, vec![0x87654321]);
/// ```
pub fn bit_pack(data: &[u64], bit_width: usize) -> Vec<u64> {
    if data.is_empty() || bit_width == 0 || bit_width > lane_size::<u64>() {
        return Vec::new();
    }

    let total_bits = data.len() * bit_width;
    let total_u64s = total_bits.div_ceil(lane_size::<u64>()); // Round up division

    let mut out = vec![0u64; total_u64s];

    for (i, &value) in data.iter().enumerate() {
        let bit_offset = i * bit_width;
        let out_index = bit_offset / lane_size::<u64>();
        let bit_pos = bit_offset % lane_size::<u64>();

        if bit_pos + bit_width <= lane_size::<u64>() {
            unsafe {
                *out.get_unchecked_mut(out_index) |= value << bit_pos;
            }
        } else {
            let bits_in_current = lane_size::<u64>() - bit_pos;
            unsafe {
                *out.get_unchecked_mut(out_index) |= value << bit_pos;
                *out.get_unchecked_mut(out_index + 1) |= value >> bits_in_current;
            }
        }
    }

    out
}

/// Unpack a vector of u64 values into a slice of u64 values.
/// Each of the u64 values must be less than 2^bit_width.
///
/// Example:
/// ```
/// use bmi_select::bit_unpack;
/// let packed = vec![0x87654321];
/// let bit_width = 4;
/// let unpacked = bit_unpack(&packed, bit_width, 8);
/// assert_eq!(unpacked, vec![1, 2, 3, 4, 5, 6, 7, 8]);
/// ```
pub fn bit_unpack(packed_data: &[u64], bit_width: usize, original_count: usize) -> Vec<u64> {
    if packed_data.is_empty() || bit_width == 0 || bit_width > 64 || original_count == 0 {
        return Vec::new();
    }

    let mask = (1u64 << bit_width) - 1;
    let mut result = Vec::with_capacity(original_count);
    let mut bit_offset = 0;

    for _ in 0..original_count {
        let u64_index = bit_offset / 64;
        let bit_pos = bit_offset % 64;

        if u64_index >= packed_data.len() {
            break;
        }

        let value = if bit_pos + bit_width <= 64 {
            // Value fits entirely within current u64
            (packed_data[u64_index] >> bit_pos) & mask
        } else {
            // Value spans across two u64s
            let bits_in_current = 64 - bit_pos;
            let bits_in_next = bit_width - bits_in_current;

            let lower_bits = packed_data[u64_index] >> bit_pos;
            let upper_bits = if u64_index + 1 < packed_data.len() {
                (packed_data[u64_index + 1] & ((1u64 << bits_in_next) - 1)) << bits_in_current
            } else {
                0
            };

            (lower_bits | upper_bits) & mask
        };

        result.push(value);
        bit_offset += bit_width;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bit_pack_round_trip(data: &Vec<u64>, bit_width: usize) {
        let packed = bit_pack(&data, bit_width);
        let unpacked = bit_unpack(&packed, bit_width, data.len());
        assert_eq!(data, &unpacked);
    }

    #[test]
    fn test_bit_pack_unpack() {
        let data = (0..200).collect();

        for bit_width in 8..64 {
            test_bit_pack_round_trip(&data, bit_width);
        }
    }

    #[test]
    fn test_bit_pack_unpack_33_bit() {
        // Test the inefficient case mentioned by the user
        let data = vec![1, 2, 3, 4, 5];
        let bit_width = 33;

        let packed = bit_pack(&data, bit_width);
        let unpacked = bit_unpack(&packed, bit_width, data.len());

        assert_eq!(data, unpacked);

        // Verify we're using space efficiently
        let expected_bits = data.len() * bit_width; // 5 * 33 = 165 bits
        let expected_u64s = (expected_bits + 63) / 64; // 3 u64s
        assert_eq!(packed.len(), expected_u64s);
    }

    #[test]
    fn test_bit_pack_unpack_7_bit() {
        // Test odd bit width that doesn't divide evenly into 64
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let bit_width = 7;

        let packed = bit_pack(&data, bit_width);
        let unpacked = bit_unpack(&packed, bit_width, data.len());

        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<u64> = vec![];
        let packed = bit_pack(&data, 8);
        let unpacked = bit_unpack(&packed, 8, 0);

        assert_eq!(data, unpacked);
    }

    #[test]
    fn test_single_value() {
        let data = vec![42];
        let bit_width = 8;

        let packed = bit_pack(&data, bit_width);
        let unpacked = bit_unpack(&packed, bit_width, data.len());

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
        let expected_u64s = (100 * 10 + 63) / 64;
        assert_eq!(packed.len(), expected_u64s);

        // This should be much less than storing 100 full u64s
        assert!(packed.len() < 100);
    }
}
