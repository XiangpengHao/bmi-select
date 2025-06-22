use bmi_select::{bit_pack, bit_unpack, select_packed};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

fn generate_test_data(size: usize, bit_width: usize) -> Vec<u64> {
    let max_value = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };
    (0..size).map(|i| (i as u64) % max_value).collect()
}

fn bench_bit_pack_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_pack_sizes");

    let sizes = vec![1000, 10000, 100000];
    let bit_width = 11;

    for size in sizes {
        let data = generate_test_data(size, bit_width);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("pack", size), &data, |b, data| {
            b.iter(|| bit_pack(black_box(data), black_box(bit_width)))
        });
    }
    group.finish();
}

fn bench_bit_unpack_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_unpack_sizes");

    let sizes = vec![1000, 10000, 100000];
    let bit_width = 11;

    for size in sizes {
        let data = generate_test_data(size, bit_width);
        let packed = bit_pack(&data, bit_width);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("unpack", size),
            &(packed, size),
            |b, (packed, original_count)| {
                b.iter(|| {
                    bit_unpack(
                        black_box(packed),
                        black_box(bit_width),
                        black_box(*original_count),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_bit_pack_different_widths(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_pack_widths");

    let bit_widths = vec![3, 5, 11, 15, 23, 33];
    let size = 10000;

    for bit_width in bit_widths {
        let data = generate_test_data(size, bit_width);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("pack", bit_width), &data, |b, data| {
            b.iter(|| bit_pack(black_box(data), black_box(bit_width)))
        });
    }
    group.finish();
}

fn bench_bit_unpack_different_widths(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_unpack_widths");

    let bit_widths = vec![3, 5, 11, 15, 23, 33];
    let size = 10000;

    for bit_width in bit_widths {
        let data = generate_test_data(size, bit_width);
        let packed = bit_pack(&data, bit_width);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("unpack", bit_width),
            &(packed, size),
            |b, (packed, original_count)| {
                b.iter(|| {
                    bit_unpack(
                        black_box(packed),
                        black_box(bit_width),
                        black_box(*original_count),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_select_packed_selection_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("select_packed_ratios");

    let size = 16384;
    let bit_width = 7;
    let selection_ratios = vec![0.01, 0.1, 0.3, 0.7];

    let data = generate_test_data(size, bit_width);
    let packed_data = bit_pack(&data, bit_width);

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
        let bit_mask = bit_pack(&mask_data, 1);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("select", format!("{}%", ratio_percent)),
            &(packed_data.clone(), bit_mask),
            |b, (packed, mask)| {
                b.iter(|| select_packed(black_box(&packed), black_box(bit_width), black_box(&mask)))
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bit_pack_different_sizes,
    bench_bit_unpack_different_sizes,
    bench_bit_pack_different_widths,
    bench_bit_unpack_different_widths,
    bench_select_packed_selection_ratios,
);

criterion_main!(benches);
