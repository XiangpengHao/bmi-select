const fn lane_size<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

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

fn select_packed_fallback(packed: &[u64], bit_width: usize, bit_mask: &[u64]) -> Vec<u64> {
    if packed.is_empty() || bit_width == 0 || bit_width > 64 || bit_mask.is_empty() {
        return Vec::new();
    }

    let max_elements = (packed.len() * 64) / bit_width;

    let extract_mask = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    let mut out: Vec<u64> = Vec::new();
    let mut out_bit_offset = 0usize;

    for idx in 0..max_elements {
        let mask_bit_offset = idx;
        let mask_u64_idx = mask_bit_offset / 64;
        if mask_u64_idx >= bit_mask.len() {
            break;
        }
        let mask_bit_pos = mask_bit_offset % 64;
        let is_selected = (bit_mask[mask_u64_idx] >> mask_bit_pos) & 1u64 == 1u64;
        if !is_selected {
            continue;
        }

        let elem_bit_offset = idx * bit_width;
        let elem_u64_idx = elem_bit_offset / 64;
        if elem_u64_idx >= packed.len() {
            break;
        }
        let elem_bit_pos = elem_bit_offset % 64;

        let value = if elem_bit_pos + bit_width <= 64 {
            (packed[elem_u64_idx] >> elem_bit_pos) & extract_mask
        } else {
            let bits_in_first = 64 - elem_bit_pos;
            let bits_in_second = bit_width - bits_in_first;

            let lower = packed[elem_u64_idx] >> elem_bit_pos;
            let upper = if elem_u64_idx + 1 < packed.len() {
                (packed[elem_u64_idx + 1] & ((1u64 << bits_in_second) - 1)) << bits_in_first
            } else {
                0
            };
            (lower | upper) & extract_mask
        };

        let out_u64_idx = out_bit_offset / 64;
        let out_bit_pos = out_bit_offset % 64;

        if out_u64_idx >= out.len() {
            out.push(0u64);
        }

        if out_bit_pos + bit_width <= 64 {
            out[out_u64_idx] |= value << out_bit_pos;
        } else {
            let bits_in_first = 64 - out_bit_pos;

            out[out_u64_idx] |= value << out_bit_pos;
            if out_u64_idx + 1 >= out.len() {
                out.push(0u64);
            }
            out[out_u64_idx + 1] |= value >> bits_in_first;
        }

        out_bit_offset += bit_width;
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;

/// BMI2-accelerated variant of `select_packed` that leverages the `PEXT` instruction
/// to extract contiguous bit ranges when they reside within a single 64-bit word.
///
/// If the current CPU does not support BMI2, this falls back to the portable
/// `select_packed_fallback` implementation.
///
/// Example:
/// ```
/// use bmi_select::{select_packed, bit_unpack};
/// let packed   = vec![0x87654321];
/// let bit_mask = vec![0b10101010];
/// let bit_width = 4;
/// let selected = select_packed(&packed, bit_width, &bit_mask);
/// assert_eq!(selected, vec![0x8642]);
///
/// let unpacked = bit_unpack(&selected, bit_width, 4);
/// assert_eq!(unpacked, vec![2, 4, 6, 8]);
/// ```
pub fn select_packed(packed: &[u64], bit_width: usize, bit_mask: &[u64]) -> Vec<u64> {
    // if BMI2 is unavailable, defer to the generic version.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !is_x86_feature_detected!("bmi2") {
            return select_packed_fallback(packed, bit_width, bit_mask);
        }
    }

    unsafe { select_packed_bmi_impl(packed, bit_width, bit_mask) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "bmi2")]
unsafe fn select_packed_bmi_impl(packed: &[u64], bit_width: usize, bit_mask: &[u64]) -> Vec<u64> {
    use std::arch::x86_64::{_pdep_u64, _pext_u64};

    if packed.is_empty() || bit_width == 0 || bit_width > 64 || bit_mask.is_empty() {
        return Vec::new();
    }

    // Special-case 64-bit
    if bit_width == 64 {
        let mut out: Vec<u64> = Vec::new();
        for (idx, &word) in packed.iter().enumerate() {
            let mask_u64_idx = idx / 64;
            if mask_u64_idx >= bit_mask.len() {
                break;
            }
            let mask_bit_pos = idx % 64;
            if (bit_mask[mask_u64_idx] >> mask_bit_pos) & 1 != 0 {
                out.push(word);
            }
        }
        return out;
    }

    let masks = get_precomputed_masks(bit_width);

    let mut out: Vec<u64> = Vec::new();
    let mut out_bit_offset: usize = 0;

    for (word_idx, &values_word) in packed.iter().enumerate() {
        let start_bit = word_idx * 64;
        let end_bit = start_bit + 64;

        // Index of the first element whose *least-significant* bit could be in this word.
        let elem_idx_start = start_bit / bit_width;
        // Index of the last element whose least-significant bit is still inside this word.
        let elem_idx_end = (end_bit - 1) / bit_width;
        let elements_count = elem_idx_end - elem_idx_start + 1; // 1 ..= 64

        let mask_u64_idx = elem_idx_start / 64;
        let bit_offset = elem_idx_start % 64;

        let mut select_bitmap: u64 = 0;
        if mask_u64_idx < bit_mask.len() {
            select_bitmap = bit_mask[mask_u64_idx] >> bit_offset;
            if bit_offset != 0 && mask_u64_idx + 1 < bit_mask.len() {
                select_bitmap |= bit_mask[mask_u64_idx + 1] << (64 - bit_offset);
            }
        }

        // Clear bits that correspond to elements beyond this 64-bit data word.
        if elements_count < 64 {
            select_bitmap &= (1u64 << elements_count) - 1;
        }

        if select_bitmap == 0 {
            continue;
        }

        let mask = masks[word_idx % bit_width];

        let low = _pdep_u64(select_bitmap, mask);
        let high = _pdep_u64(select_bitmap, mask.wrapping_sub(1));
        let extended = high.wrapping_sub(low);
        let extracted = _pext_u64(values_word, extended);
        let bits_extracted = extended.count_ones() as usize;

        // ------------------  Pack into output  ---------------------------
        let out_u64_idx = out_bit_offset / 64;
        let bit_pos = out_bit_offset % 64;

        if out_u64_idx >= out.len() {
            out.push(0);
        }

        let capacity = 64 - bit_pos;

        if bits_extracted <= capacity {
            out[out_u64_idx] |= extracted << bit_pos;
        } else {
            out[out_u64_idx] |= extracted << bit_pos;

            if out_u64_idx + 1 >= out.len() {
                out.push(0);
            }
            out[out_u64_idx + 1] |= extracted >> capacity;
        }

        out_bit_offset += bits_extracted;
    }

    out
}

/// For a given bit width k, we need k distinct masks that are reused in a
/// round-robin fashion for successive 64-bit words.
const fn compute_base_pattern(bit_width: usize) -> u64 {
    let mut base_pattern = 0u64;
    let mut pos = 0;
    while pos < 64 {
        base_pattern |= 1u64 << pos;
        pos += bit_width;
    }
    base_pattern
}

/// Compute the masks for a specific bit width.
/// Returns an array of up to 64 masks (since bit_width is at most 64).
const fn compute_masks_for_bit_width(bit_width: usize) -> ([u64; 64], usize) {
    let base_pattern = compute_base_pattern(bit_width);
    let mut masks = [0u64; 64];
    let mut i = 0;
    while i < bit_width {
        // offset = k - ((i * 64) mod k)  (Algorithm 3)
        let offset = bit_width - ((i * 64) % bit_width);
        let mut mask = base_pattern << offset;
        mask |= 1u64;
        masks[i] = mask;
        i += 1;
    }
    (masks, if bit_width < 64 { bit_width } else { 64 })
}

/// Get precomputed masks for the given bit width.
/// Returns a slice containing the masks for this bit width.
fn get_precomputed_masks(bit_width: usize) -> &'static [u64] {
    match bit_width {
        1 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(1);
            &MASKS.0[..MASKS.1]
        }
        2 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(2);
            &MASKS.0[..MASKS.1]
        }
        3 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(3);
            &MASKS.0[..MASKS.1]
        }
        4 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(4);
            &MASKS.0[..MASKS.1]
        }
        5 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(5);
            &MASKS.0[..MASKS.1]
        }
        6 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(6);
            &MASKS.0[..MASKS.1]
        }
        7 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(7);
            &MASKS.0[..MASKS.1]
        }
        8 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(8);
            &MASKS.0[..MASKS.1]
        }
        9 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(9);
            &MASKS.0[..MASKS.1]
        }
        10 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(10);
            &MASKS.0[..MASKS.1]
        }
        11 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(11);
            &MASKS.0[..MASKS.1]
        }
        12 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(12);
            &MASKS.0[..MASKS.1]
        }
        13 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(13);
            &MASKS.0[..MASKS.1]
        }
        14 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(14);
            &MASKS.0[..MASKS.1]
        }
        15 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(15);
            &MASKS.0[..MASKS.1]
        }
        16 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(16);
            &MASKS.0[..MASKS.1]
        }
        17 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(17);
            &MASKS.0[..MASKS.1]
        }
        18 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(18);
            &MASKS.0[..MASKS.1]
        }
        19 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(19);
            &MASKS.0[..MASKS.1]
        }
        20 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(20);
            &MASKS.0[..MASKS.1]
        }
        21 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(21);
            &MASKS.0[..MASKS.1]
        }
        22 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(22);
            &MASKS.0[..MASKS.1]
        }
        23 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(23);
            &MASKS.0[..MASKS.1]
        }
        24 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(24);
            &MASKS.0[..MASKS.1]
        }
        25 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(25);
            &MASKS.0[..MASKS.1]
        }
        26 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(26);
            &MASKS.0[..MASKS.1]
        }
        27 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(27);
            &MASKS.0[..MASKS.1]
        }
        28 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(28);
            &MASKS.0[..MASKS.1]
        }
        29 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(29);
            &MASKS.0[..MASKS.1]
        }
        30 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(30);
            &MASKS.0[..MASKS.1]
        }
        31 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(31);
            &MASKS.0[..MASKS.1]
        }
        32 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(32);
            &MASKS.0[..MASKS.1]
        }
        33 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(33);
            &MASKS.0[..MASKS.1]
        }
        34 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(34);
            &MASKS.0[..MASKS.1]
        }
        35 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(35);
            &MASKS.0[..MASKS.1]
        }
        36 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(36);
            &MASKS.0[..MASKS.1]
        }
        37 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(37);
            &MASKS.0[..MASKS.1]
        }
        38 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(38);
            &MASKS.0[..MASKS.1]
        }
        39 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(39);
            &MASKS.0[..MASKS.1]
        }
        40 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(40);
            &MASKS.0[..MASKS.1]
        }
        41 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(41);
            &MASKS.0[..MASKS.1]
        }
        42 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(42);
            &MASKS.0[..MASKS.1]
        }
        43 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(43);
            &MASKS.0[..MASKS.1]
        }
        44 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(44);
            &MASKS.0[..MASKS.1]
        }
        45 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(45);
            &MASKS.0[..MASKS.1]
        }
        46 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(46);
            &MASKS.0[..MASKS.1]
        }
        47 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(47);
            &MASKS.0[..MASKS.1]
        }
        48 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(48);
            &MASKS.0[..MASKS.1]
        }
        49 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(49);
            &MASKS.0[..MASKS.1]
        }
        50 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(50);
            &MASKS.0[..MASKS.1]
        }
        51 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(51);
            &MASKS.0[..MASKS.1]
        }
        52 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(52);
            &MASKS.0[..MASKS.1]
        }
        53 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(53);
            &MASKS.0[..MASKS.1]
        }
        54 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(54);
            &MASKS.0[..MASKS.1]
        }
        55 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(55);
            &MASKS.0[..MASKS.1]
        }
        56 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(56);
            &MASKS.0[..MASKS.1]
        }
        57 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(57);
            &MASKS.0[..MASKS.1]
        }
        58 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(58);
            &MASKS.0[..MASKS.1]
        }
        59 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(59);
            &MASKS.0[..MASKS.1]
        }
        60 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(60);
            &MASKS.0[..MASKS.1]
        }
        61 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(61);
            &MASKS.0[..MASKS.1]
        }
        62 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(62);
            &MASKS.0[..MASKS.1]
        }
        63 => {
            const MASKS: ([u64; 64], usize) = compute_masks_for_bit_width(63);
            &MASKS.0[..MASKS.1]
        }
        _ => &[],
    }
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

    #[test]
    fn test_select_packed() {
        // Test basic selection functionality
        let data = vec![10, 20, 30, 40, 50];
        let bit_width = 8;
        let packed = bit_pack(&data, bit_width);

        // Create a mask that selects elements at positions 0, 2, and 4 (10, 30, 50)
        let mask_data = vec![1, 0, 1, 0, 1]; // Select 1st, 3rd, and 5th elements
        let bit_mask = bit_pack(&mask_data, 1);

        let selected_packed = select_packed_fallback(&packed, bit_width, &bit_mask);
        let selected_unpacked = bit_unpack(&selected_packed, bit_width, 3);

        assert_eq!(selected_unpacked, vec![10, 30, 50]);
    }

    #[test]
    fn test_select_packed_all() {
        // Test selecting all elements
        let data = vec![1, 2, 3, 4];
        let bit_width = 4;
        let packed = bit_pack(&data, bit_width);

        let mask_data = vec![1, 1, 1, 1]; // Select all elements
        let bit_mask = bit_pack(&mask_data, 1);

        let selected_packed = select_packed_fallback(&packed, bit_width, &bit_mask);
        let selected_unpacked = bit_unpack(&selected_packed, bit_width, 4);

        assert_eq!(selected_unpacked, data);
    }

    #[test]
    fn test_select_packed_none() {
        // Test selecting no elements
        let data = vec![1, 2, 3, 4];
        let bit_width = 4;
        let packed = bit_pack(&data, bit_width);

        let mask_data = vec![0, 0, 0, 0]; // Select no elements
        let bit_mask = bit_pack(&mask_data, 1);

        let selected_packed = select_packed_fallback(&packed, bit_width, &bit_mask);

        assert_eq!(selected_packed, Vec::<u64>::new());
    }

    #[test]
    fn test_select_packed_odd_bit_width() {
        // Test with a non-power-of-2 bit width
        let data = vec![100, 200, 300, 400, 500, 600];
        let bit_width = 11; // 11 bits can represent values up to 2047
        let packed = bit_pack(&data, bit_width);

        // Select every other element
        let mask_data = vec![1, 0, 1, 0, 1, 0];
        let bit_mask = bit_pack(&mask_data, 1);

        let selected_packed = select_packed_fallback(&packed, bit_width, &bit_mask);
        let selected_unpacked = bit_unpack(&selected_packed, bit_width, 3);

        assert_eq!(selected_unpacked, vec![100, 300, 500]);
    }

    #[test]
    fn test_select_packed_bmi_matches_generic() {
        // If CPU lacks BMI2, we just skip this test.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if !is_x86_feature_detected!("bmi2") {
                return;
            }
        }

        let data: Vec<u64> = (0..128).map(|x| x as u64).collect();
        let bit_widths = [1usize, 4, 7, 8, 12, 16, 24, 32, 33, 48, 64];

        for &bw in &bit_widths {
            let packed = bit_pack(&data, bw);
            let mask_data: Vec<u64> = (0..data.len())
                .map(|i| if i % 3 == 0 { 1 } else { 0 })
                .collect();
            let bit_mask = bit_pack(&mask_data, 1);

            let generic = select_packed_fallback(&packed, bw, &bit_mask);
            let bmi = select_packed(&packed, bw, &bit_mask);
            assert_eq!(generic, bmi, "Mismatch with bit_width={bw}");
        }
    }
}
