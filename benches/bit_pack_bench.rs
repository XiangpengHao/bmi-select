use bmi_select::{BitPackable, bit_pack, bit_unpack, select_packed};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fastlanes::BitPacking as FLBitPacking;

fn generate_test_data<T: BitPackable>(size: usize, bit_width: usize) -> Vec<T> {
    let max_value = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };
    (0..size)
        .map(|i| T::from_u64((i as u64) % max_value))
        .collect()
}

fn bench_bit_pack_different_widths(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_pack");

    let bit_widths = vec![3, 5, 11, 15, 23, 33];
    let size = 16384;

    for bit_width in bit_widths {
        let data = generate_test_data(size, bit_width);
        group.throughput(Throughput::Elements(size as u64));

        // Benchmark our implementation
        group.bench_with_input(
            BenchmarkId::new("bmi_select", bit_width),
            &data,
            |b, data| {
                let total_bits = data.len() * bit_width;
                let mut out = vec![0u64; total_bits.div_ceil(64)];
                b.iter(|| bit_pack(black_box(data), black_box(bit_width), black_box(&mut out)));
            },
        );

        // Benchmark fastlanes implementation
        group.bench_with_input(
            BenchmarkId::new("fastlanes", bit_width),
            &data,
            |b, data| {
                let batches = size / 1024;
                let out_batch = 16 * bit_width;
                let mut packed = vec![0u64; batches * out_batch];

                b.iter(|| {
                    for i in 0..batches {
                        unsafe {
                            FLBitPacking::unchecked_pack(
                                bit_width,
                                &data[i * 1024..(i + 1) * 1024],
                                &mut packed[i * out_batch..(i + 1) * out_batch],
                            );
                        }
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_bit_unpack_different_widths(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_unpack");

    let bit_widths = vec![3, 5, 11, 15, 23, 33];
    let size = 16384;

    for bit_width in bit_widths {
        let data = generate_test_data(size, bit_width);

        // Prepare data for our implementation
        let total_bits = data.len() * bit_width;
        let mut packed = vec![0u64; total_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut packed);

        // Prepare data for fastlanes
        let batches = size / 1024;
        let out_batch = 16 * bit_width;
        let mut fl_packed = vec![0u64; batches * out_batch];
        for i in 0..batches {
            unsafe {
                FLBitPacking::unchecked_pack(
                    bit_width,
                    &data[i * 1024..(i + 1) * 1024],
                    &mut fl_packed[i * out_batch..(i + 1) * out_batch],
                );
            }
        }

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark our implementation
        group.bench_with_input(
            BenchmarkId::new("bmi_select", bit_width),
            &(packed, size),
            |b, (packed, original_count)| {
                let mut out = vec![0u64; *original_count];
                b.iter(|| bit_unpack(black_box(packed), black_box(bit_width), black_box(&mut out)))
            },
        );

        // Benchmark fastlanes implementation
        group.bench_with_input(
            BenchmarkId::new("fastlanes", bit_width),
            &fl_packed,
            |b, packed| {
                let mut unpacked = vec![0u64; size];
                b.iter(|| {
                    for i in 0..batches {
                        unsafe {
                            FLBitPacking::unchecked_unpack(
                                bit_width,
                                &packed[i * out_batch..(i + 1) * out_batch],
                                &mut unpacked[i * 1024..(i + 1) * 1024],
                            );
                        }
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_select_packed_selection_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("select_packed");

    let size = 16384; // Multiple of 1024 for fastlanes compatibility
    let bit_width = 7;
    let selection_ratios = vec![0.01, 0.1, 0.3, 0.7];

    let data = generate_test_data(size, bit_width);

    for ratio in selection_ratios {
        let ratio_percent = (ratio * 100.0) as u32;

        let mask_data: Vec<u64> = (0..size)
            .map(|i| {
                if (i as f64) < (size as f64 * ratio) {
                    1
                } else {
                    0
                }
            })
            .collect();

        // Prepare data for our implementation
        let data_bits = data.len() * bit_width;
        let mut packed_data = vec![0u64; data_bits.div_ceil(64)];
        bit_pack(&data, bit_width, &mut packed_data);
        
        let mask_bits = mask_data.len() * 1;
        let mut bit_mask = vec![0u64; mask_bits.div_ceil(64)];
        bit_pack(&mask_data, 1, &mut bit_mask);

        // Prepare data for fastlanes
        let batches = size / 1024;
        let out_batch = 16 * bit_width;
        let mut fl_packed_data = vec![0u64; batches * out_batch];
        let mut fl_bit_mask = vec![0u64; batches * 16];

        for i in 0..batches {
            unsafe {
                FLBitPacking::unchecked_pack(
                    bit_width,
                    &data[i * 1024..(i + 1) * 1024],
                    &mut fl_packed_data[i * out_batch..(i + 1) * out_batch],
                );
                FLBitPacking::unchecked_pack(
                    1,
                    &mask_data[i * 1024..(i + 1) * 1024],
                    &mut fl_bit_mask[i * 16..(i + 1) * 16],
                );
            }
        }

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark our implementation
        group.bench_with_input(
            BenchmarkId::new("bmi_select", format!("{ratio_percent}%")),
            &(packed_data.clone(), bit_mask),
            |b, (packed, mask)| {
                let selected_count = mask_data.iter().sum::<u64>() as usize;
                let selected_bits = selected_count * bit_width;
                let mut out = vec![0u64; selected_bits.div_ceil(64)];
                b.iter(|| select_packed(black_box(packed), black_box(bit_width), black_box(mask), black_box(&mut out)))
            },
        );

        // Benchmark fastlanes implementation
        group.bench_function(
            BenchmarkId::new("fastlanes", format!("{ratio_percent}%")),
            |b| {
                let mut unpacked = vec![0u64; size];
                let mut mask_unpacked = vec![0u64; size];
                let mut selected = Vec::with_capacity(size);
                let mut tmp_input = vec![0u64; 1024];
                let mut repacked = Vec::<u64>::new();

                b.iter(|| {
                    // Unpack data and mask
                    for i in 0..batches {
                        unsafe {
                            FLBitPacking::unchecked_unpack(
                                bit_width,
                                &fl_packed_data[i * out_batch..(i + 1) * out_batch],
                                &mut unpacked[i * 1024..(i + 1) * 1024],
                            );
                            FLBitPacking::unchecked_unpack(
                                1,
                                &fl_bit_mask[i * 16..(i + 1) * 16],
                                &mut mask_unpacked[i * 1024..(i + 1) * 1024],
                            );
                        }
                    }

                    // Select values based on mask
                    selected.clear();
                    for i in 0..size {
                        if mask_unpacked[i] == 1 {
                            selected.push(unpacked[i]);
                        }
                    }

                    // Repack selected values
                    let num_selected = selected.len();
                    let out_batches = num_selected.div_ceil(1024);
                    repacked.clear();
                    repacked.resize(out_batches * out_batch, 0u64);

                    for j in 0..out_batches {
                        let start = j * 1024;
                        let end = usize::min(start + 1024, num_selected);
                        tmp_input[..end - start].copy_from_slice(&selected[start..end]);
                        for v in &mut tmp_input[end - start..] {
                            *v = 0;
                        }
                        unsafe {
                            FLBitPacking::unchecked_pack(
                                bit_width,
                                &tmp_input,
                                &mut repacked[j * out_batch..(j + 1) * out_batch],
                            );
                        }
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bit_pack_different_widths,
    bench_bit_unpack_different_widths,
    bench_select_packed_selection_ratios,
);

criterion_main!(benches);
