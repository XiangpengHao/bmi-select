pub fn select_packed_fallback(packed: &[u64], bit_width: usize, bit_mask: &[u64], out: &mut [u64]) {
    let max_elements = (packed.len() * 64) / bit_width;

    let extract_mask = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

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

        if out_bit_pos + bit_width <= 64 {
            out[out_u64_idx] |= value << out_bit_pos;
        } else {
            let bits_in_first = 64 - out_bit_pos;

            out[out_u64_idx] |= value << out_bit_pos;
            out[out_u64_idx + 1] |= value >> bits_in_first;
        }

        out_bit_offset += bit_width;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;

/// BMI2-accelerated variant of `select_packed` that leverages the `PEXT` instruction
/// to extract contiguous bit ranges when they reside within a single 64-bit word.
///
/// If the current CPU does not support BMI2, this falls back to the portable
/// `select_packed_fallback` implementation.
/// The output buffer must have enough space to hold the selected data.
///
/// Example:
/// ```
/// use bmi_select::{select_packed, bit_unpack};
/// let packed   = vec![0x87654321];
/// let bit_mask = vec![0b10101010];
/// let bit_width = 4;
/// let mut selected = vec![0u64; 1];
/// select_packed(&packed, bit_width, &bit_mask, &mut selected);
/// assert_eq!(selected, vec![0x8642]);
///
/// let mut unpacked = vec![0u64; 4];
/// bit_unpack(&selected, bit_width, &mut unpacked);
/// assert_eq!(unpacked, vec![2, 4, 6, 8]);
/// ```
pub fn select_packed(packed: &[u64], bit_width: usize, bit_mask: &[u64], out: &mut [u64]) {
    // if BMI2 is unavailable, defer to the generic version.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !is_x86_feature_detected!("bmi2") {
            return select_packed_fallback(packed, bit_width, bit_mask, out);
        }
    }

    unsafe { select_packed_bmi_impl(packed, bit_width, bit_mask, out) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "bmi2")]
unsafe fn select_packed_bmi_impl(
    packed: &[u64],
    bit_width: usize,
    bit_mask: &[u64],
    out: &mut [u64],
) {
    use std::arch::x86_64::{_pdep_u64, _pext_u64};

    // Special-case 64-bit
    if bit_width == 64 {
        let mut out_idx = 0;
        for (idx, &word) in packed.iter().enumerate() {
            let mask_u64_idx = idx / 64;
            if mask_u64_idx >= bit_mask.len() {
                break;
            }
            let mask_bit_pos = idx % 64;
            if (bit_mask[mask_u64_idx] >> mask_bit_pos) & 1 != 0 {
                out[out_idx] = word;
                out_idx += 1;
            }
        }
        return;
    }

    let masks = get_precomputed_masks(bit_width);
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

        let capacity = 64 - bit_pos;
        out[out_u64_idx] |= extracted << bit_pos;
        if bits_extracted > capacity {
            out[out_u64_idx + 1] |= extracted >> capacity;
        }

        out_bit_offset += bits_extracted;
    }
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

/// Selects elements from a source slice of `u64`s based on a `u64` bitmap and writes them to a destination slice.
///
/// Uses AVX-512 compress instructions when available, falls back to scalar implementation otherwise.
/// Processes data in batches of 128 elements for optimal SIMD utilization.
///
/// # Parameters
/// - `src`: Source data elements
/// - `bitmap`: Selection bitmap (1 bit per element)
/// - `dst`: Destination for selected elements
/// - `n`: Number of elements to process
///
/// # Returns
/// Number of elements written to destination
///
/// # Example
/// ```rust
/// let src = vec![10, 20, 30, 40, 50];
/// let bitmap = vec![0b10101u64]; // Select elements 0, 2, 4
/// let mut dst = vec![0u64; 3];
/// let count = select_unpacked(&src, &bitmap, &mut dst, 5);
/// assert_eq!(count, 3);
/// assert_eq!(&dst[..count], &[10, 30, 50]);
/// ```
pub fn select_unpacked(src: &[u64], bitmap: &[u64], dst: &mut [u64], n: usize) -> usize {
    // Use AVX-512 compress instruction if available
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !is_x86_feature_detected!("avx512f") {
            return select_unpacked_fallback(src, bitmap, dst, n);
        }
    }

    unsafe { select_unpacked_avx512_compress(src, bitmap, dst, n) }
}

/// Scalar fallback implementation for CPUs without AVX-512.
fn select_unpacked_fallback(src: &[u64], bitmap: &[u64], dst: &mut [u64], n: usize) -> usize {
    let mut cur_bitmap = bitmap[0];
    let mut bitmap_ptr = 0;
    let mut bitmap_remaining = 64;
    let mut selected_idx = 0;

    for i in 0..n {
        dst[selected_idx] = src[i];
        // Advance destination pointer only if element is selected (branchless)
        selected_idx += (cur_bitmap & 1) as usize;
        cur_bitmap >>= 1;
        bitmap_remaining -= 1;

        // Move to next bitmap word when current is exhausted
        if bitmap_remaining == 0 {
            bitmap_remaining = 64;
            bitmap_ptr += 1;
            if bitmap_ptr < bitmap.len() {
                cur_bitmap = bitmap[bitmap_ptr];
            }
        }
    }
    selected_idx
}

/// AVX-512 accelerated implementation using hardware compress instructions.
///
/// Processes data in batches of 128 elements (16 rounds of 8 elements each).
/// Each batch uses 128 bits of bitmap data, viewed as 16 separate 8-bit masks.
///
/// # Safety
/// Requires AVX-512F support and valid input slices with sufficient capacity.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn select_unpacked_avx512_compress(
    src: &[u64],
    bitmap: &[u64],
    dst: &mut [u64],
    n: usize,
) -> usize {
    use std::arch::x86_64::*;
    use std::slice;

    /// The number of u64 elements processed in a single SIMD batch.
    const SIMD_COMPRESS_BATCH_SIZE: usize = 128;

    /// The number of u64 elements in one AVX-512 register (512 bits / 64 bits = 8).
    const ONE_ROUND_SIZE: usize = 8;

    /// The number of rounds per batch (128 elements / 8 per round = 16).
    const NUM_ROUNDS_IN_A_BATCH: usize = SIMD_COMPRESS_BATCH_SIZE / ONE_ROUND_SIZE;

    unsafe {
        assert_eq!(
            ONE_ROUND_SIZE * NUM_ROUNDS_IN_A_BATCH,
            SIMD_COMPRESS_BATCH_SIZE
        );

        let batch_n = (n / SIMD_COMPRESS_BATCH_SIZE) * SIMD_COMPRESS_BATCH_SIZE;

        let mut src_ptr = src.as_ptr();
        let mut dst_ptr = dst.as_mut_ptr();
        let mut bitmap_ptr = bitmap.as_ptr();

        // Process complete batches of 128 elements
        for _ in (0..batch_n).step_by(SIMD_COMPRESS_BATCH_SIZE) {
            // View 128-bit bitmap segment as 16 separate 8-bit masks
            let masks = slice::from_raw_parts(bitmap_ptr as *const u8, NUM_ROUNDS_IN_A_BATCH);

            // Process 16 rounds of 8 elements each
            for j in 0..NUM_ROUNDS_IN_A_BATCH {
                let mask = masks[j];

                // Load 8 elements and compress using mask
                let src_v = _mm512_loadu_epi64(src_ptr as *const i64);
                let dst_v = _mm512_maskz_compress_epi64(mask, src_v);
                _mm512_storeu_epi64(dst_ptr as *mut i64, dst_v);

                let popcnt = mask.count_ones() as usize;
                src_ptr = src_ptr.add(ONE_ROUND_SIZE);
                dst_ptr = dst_ptr.add(popcnt);
            }

            bitmap_ptr = bitmap_ptr.add(2); // Advance by 2 u64s (128 bits)
        }

        // Handle the remaining elements (< 128) using a simple scalar loop.
        if n > batch_n {
            let mut bitmap_idx = batch_n / 64; // Index of the u64 in the bitmap slice.
            let mut bit_in_u64 = batch_n % 64; // Bit position within that u64.
            let mut current_bitmap_val = *bitmap.get_unchecked(bitmap_idx);

            for _ in batch_n..n {
                // If we've used all bits in the current bitmap u64, load the next one.
                if bit_in_u64 == 64 {
                    bit_in_u64 = 0;
                    bitmap_idx += 1;
                    current_bitmap_val = *bitmap.get_unchecked(bitmap_idx);
                }

                // If the corresponding bit is 1, copy the element.
                if (current_bitmap_val >> bit_in_u64) & 1 == 1 {
                    *dst_ptr = *src_ptr;
                    dst_ptr = dst_ptr.add(1);
                }

                src_ptr = src_ptr.add(1);
                bit_in_u64 += 1;
            }
        }

        dst_ptr.offset_from(dst.as_ptr()) as usize
    }
}

#[cfg(test)]
mod tests {
    use crate::{bit_pack, bit_unpack};

    use super::*;

    #[test]
    fn test_select_packed() {
        // Test basic selection functionality
        let data: Vec<u64> = vec![10, 20, 30, 40, 50];
        let bit_width = 8;
        let total_bits = data.len() * bit_width;
        let mut packed = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut packed);

        // Create a mask that selects elements at positions 0, 2, and 4 (10, 30, 50)
        let mask_data: Vec<u64> = vec![1, 0, 1, 0, 1]; // Select 1st, 3rd, and 5th elements
        let mask_bits = mask_data.len();
        let mut bit_mask = vec![0u64; mask_bits.div_ceil(64)];
        bit_pack(&mask_data, 1, &mut bit_mask);

        let mut selected_packed = vec![0u64; 3]; // Expecting 3 selected elements
        select_packed_fallback(&packed, bit_width, &bit_mask, &mut selected_packed);
        let mut selected_unpacked = vec![0u64; 3];
        bit_unpack(&selected_packed, bit_width, &mut selected_unpacked);

        assert_eq!(selected_unpacked, vec![10, 30, 50]);
    }

    #[test]
    fn test_select_packed_all() {
        // Test selecting all elements
        let data: Vec<u64> = vec![1, 2, 3, 4];
        let bit_width = 4;
        let total_bits = data.len() * bit_width;
        let mut packed = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut packed);

        let mask_data: Vec<u64> = vec![1, 1, 1, 1]; // Select all elements
        let mask_bits = mask_data.len();
        let mut bit_mask = vec![0u64; mask_bits.div_ceil(64)];
        bit_pack(&mask_data, 1, &mut bit_mask);

        let mut selected_packed = vec![0u64; 2]; // 4 elements * 4 bits = 16 bits = 1 u64
        select_packed_fallback(&packed, bit_width, &bit_mask, &mut selected_packed);
        let mut selected_unpacked = vec![0u64; 4];
        bit_unpack(&selected_packed, bit_width, &mut selected_unpacked);

        assert_eq!(selected_unpacked, data);
    }

    #[test]
    fn test_select_packed_none() {
        // Test selecting no elements
        let data: Vec<u64> = vec![1, 2, 3, 4];
        let bit_width = 4;
        let total_bits = data.len() * bit_width;
        let mut packed = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut packed);

        let mask_data: Vec<u64> = vec![0, 0, 0, 0]; // Select no elements
        let mask_bits = mask_data.len();
        let mut bit_mask = vec![0u64; mask_bits.div_ceil(64)];
        bit_pack(&mask_data, 1, &mut bit_mask);

        let mut selected_packed = vec![0u64; 1];
        select_packed_fallback(&packed, bit_width, &bit_mask, &mut selected_packed);

        assert_eq!(selected_packed, vec![0u64]);
    }

    #[test]
    fn test_select_packed_odd_bit_width() {
        // Test with a non-power-of-2 bit width
        let data: Vec<u64> = vec![100, 200, 300, 400, 500, 600];
        let bit_width = 11; // 11 bits can represent values up to 2047
        let total_bits = data.len() * bit_width;
        let mut packed = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut packed);

        // Select every other element
        let mask_data: Vec<u64> = vec![1, 0, 1, 0, 1, 0];
        let mask_bits = mask_data.len();
        let mut bit_mask = vec![0u64; mask_bits.div_ceil(64)];
        bit_pack(&mask_data, 1, &mut bit_mask);

        let mut selected_packed = vec![0u64; 2]; // 3 elements * 11 bits = 33 bits = 1 u64
        select_packed_fallback(&packed, bit_width, &bit_mask, &mut selected_packed);
        let mut selected_unpacked = vec![0u64; 3];
        bit_unpack(&selected_packed, bit_width, &mut selected_unpacked);

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
            let total_bits = data.len() * bw;
            let mut packed = vec![0u64; total_bits.div_ceil(64)];
            bit_pack(&data, bw, &mut packed);
            let mask_data: Vec<u64> = (0..data.len())
                .map(|i| if i % 3 == 0 { 1 } else { 0 })
                .collect();
            let mask_bits = mask_data.len();
            let mut bit_mask = vec![0u64; mask_bits.div_ceil(64)];
            bit_pack(&mask_data, 1, &mut bit_mask);

            let selected_count = mask_data.iter().sum::<u64>() as usize;
            let selected_bits = selected_count * bw;
            let selected_u64s = selected_bits.div_ceil(64);

            let mut generic = vec![0u64; selected_u64s];
            select_packed_fallback(&packed, bw, &bit_mask, &mut generic);
            let mut bmi = vec![0u64; selected_u64s];
            select_packed(&packed, bw, &bit_mask, &mut bmi);
            assert_eq!(generic, bmi);
        }
    }

    #[test]
    fn test_select_unpacked_basic() {
        // Test basic selection functionality with unpacked data
        let src = vec![10, 20, 30, 40, 50];
        let bitmap_data = vec![1u64, 0, 1, 0, 1]; // Select elements at positions 0, 2, 4
        let mut bitmap = vec![0u64; 1];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        let mut dst = vec![0u64; 5]; // Allocate enough space
        let count = select_unpacked(&src, &bitmap, &mut dst, src.len());

        assert_eq!(count, 3);
        assert_eq!(&dst[..count], &[10, 30, 50]);
    }

    #[test]
    fn test_select_unpacked_all() {
        // Test selecting all elements
        let src = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let bitmap_data = vec![1u64; 8]; // Select all elements
        let mut bitmap = vec![0u64; 1];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        let mut dst = vec![0u64; 8];
        let count = select_unpacked(&src, &bitmap, &mut dst, src.len());

        assert_eq!(count, 8);
        assert_eq!(&dst[..count], &src);
    }

    #[test]
    fn test_select_unpacked_none() {
        // Test selecting no elements
        let src = vec![1, 2, 3, 4, 5];
        let bitmap_data = vec![0u64; 5]; // Select no elements
        let mut bitmap = vec![0u64; 1];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        let mut dst = vec![0u64; 5];
        let count = select_unpacked(&src, &bitmap, &mut dst, src.len());

        assert_eq!(count, 0);
    }

    #[test]
    fn test_select_unpacked_large_batch() {
        // Test with a large dataset that requires batch processing (>= 128 elements)
        let src: Vec<u64> = (1..=2048).collect();
        let bitmap_data: Vec<u64> = (0..2048).map(|i| if i % 3 > 0 { 1 } else { 0 }).collect(); // Select every 3rd element
        let selected_count = bitmap_data.iter().filter(|&&x| x != 0).count();

        let bitmap_bits = bitmap_data.len();
        let mut bitmap = vec![0u64; bitmap_bits.div_ceil(64)];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        let mut dst = vec![0u64; src.len()];
        let count = select_unpacked(&src, &bitmap, &mut dst, src.len());

        assert_eq!(count, selected_count);

        // Verify the selected elements are correct
        let expected: Vec<u64> = (1..=2048).filter(|&i| (i - 1) % 3 != 0).collect();
        assert_eq!(&dst[..count], &expected);
    }

    #[test]
    fn test_select_unpacked_partial() {
        // Test with n < src.len()
        let src = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let bitmap_data = vec![1u64, 0, 1, 1, 0]; // Only consider first 5 elements
        let mut bitmap = vec![0u64; 1];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        let mut dst = vec![0u64; 8];
        let count = select_unpacked(&src, &bitmap, &mut dst, 5); // Only process first 5 elements

        assert_eq!(count, 3);
        assert_eq!(&dst[..count], &[10, 30, 40]);
    }

    #[test]
    fn test_select_unpacked_cross_u64_boundary() {
        // Test selection that crosses u64 boundaries in the bitmap
        let src: Vec<u64> = (1..=100).collect();
        let bitmap_data: Vec<u64> = (0..100).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();

        let bitmap_bits = bitmap_data.len();
        let mut bitmap = vec![0u64; bitmap_bits.div_ceil(64)];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        let mut dst = vec![0u64; src.len()];
        let count = select_unpacked(&src, &bitmap, &mut dst, src.len());

        assert_eq!(count, 50); // Half the elements should be selected

        // Verify the selected elements are the even-indexed ones (1-indexed)
        let expected: Vec<u64> = (1..=100).step_by(2).collect();
        assert_eq!(&dst[..count], &expected);
    }

    #[test]
    fn test_select_unpacked_fallback_matches_optimized() {
        // Test that fallback and optimized versions produce the same results
        let src: Vec<u64> = (1..=150).collect();
        let bitmap_data: Vec<u64> = (0..150)
            .map(|i| if (i * 7) % 3 > 0 { 1 } else { 0 })
            .collect(); // Pseudo-random pattern

        let bitmap_bits = bitmap_data.len();
        let mut bitmap = vec![0u64; bitmap_bits.div_ceil(64)];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        // Test fallback version
        let mut dst_fallback = vec![0u64; src.len()];
        let count_fallback = select_unpacked_fallback(&src, &bitmap, &mut dst_fallback, src.len());

        // Test optimized version (if available)
        let mut dst_optimized = vec![0u64; src.len()];
        let count_optimized = select_unpacked(&src, &bitmap, &mut dst_optimized, src.len());

        // Results should match
        assert_eq!(count_fallback, count_optimized);
        assert_eq!(
            &dst_fallback[..count_fallback],
            &dst_optimized[..count_optimized]
        );
    }

    #[test]
    fn test_select_unpacked_edge_case_empty() {
        // Test with empty input
        let src: Vec<u64> = vec![];
        let bitmap = vec![0u64];
        let mut dst = vec![0u64; 1];

        let count = select_unpacked(&src, &bitmap, &mut dst, 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_select_unpacked_single_element() {
        // Test with single element
        let src = vec![42];
        let bitmap_data = vec![1u64];
        let mut bitmap = vec![0u64; 1];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        let mut dst = vec![0u64; 1];
        let count = select_unpacked(&src, &bitmap, &mut dst, 1);

        assert_eq!(count, 1);
        assert_eq!(dst[0], 42);

        // Test with single element not selected
        let bitmap_data = vec![0u64];
        let mut bitmap = vec![0u64; 1];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        let mut dst = vec![0u64; 1];
        let count = select_unpacked(&src, &bitmap, &mut dst, 1);

        assert_eq!(count, 0);
    }

    #[test]
    fn test_select_unpacked_exactly_128_elements() {
        // Test with exactly 128 elements (one full batch)
        let src: Vec<u64> = (1..=128).collect();
        let bitmap_data: Vec<u64> = (0..128).map(|i| if i % 4 > 0 { 1 } else { 0 }).collect(); // Select 3/4 of elements
        let selected_count = bitmap_data.iter().filter(|&&x| x != 0).count();

        let bitmap_bits = bitmap_data.len();
        let mut bitmap = vec![0u64; bitmap_bits.div_ceil(64)];
        bit_pack(&bitmap_data, 1, &mut bitmap);

        let mut dst = vec![0u64; src.len()];
        let count = select_unpacked(&src, &bitmap, &mut dst, src.len());

        assert_eq!(count, selected_count);

        // Verify correctness by checking with fallback
        let mut dst_fallback = vec![0u64; src.len()];
        let count_fallback = select_unpacked_fallback(&src, &bitmap, &mut dst_fallback, src.len());

        assert_eq!(count, count_fallback);
        assert_eq!(&dst[..count], &dst_fallback[..count_fallback]);
    }
}
