## BMI Select 

This library provides 3 efficient functions for packing, unpacking, and selecting bits from bit-packed data.

### Bit packing
```rust
use bmi_select::bit_pack;
let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
let packed = bit_pack(&data, 4);
assert_eq!(packed, vec![0x87654321]);
```

### Bit unpacking
```rust
use bmi_select::bit_unpack;
let packed = vec![0x87654321];
let unpacked = bit_unpack(&packed, 4);
assert_eq!(unpacked[..8], vec![1, 2, 3, 4, 5, 6, 7, 8]);
```

### Bit selection
```rust
use bmi_select::select_packed;
let packed   = vec![0x87654321];
let bit_mask = vec![0b10101010];
let selected = select_packed(&packed, 4, &bit_mask);
assert_eq!(selected[..4], vec![0x8642]);
```

## Features

- No dependencies.
- State-of-the-art performance backed by mind-blowing algorithms (see credits below).
- Bit-packed data follows the same layout as the original data, i.e., the first element in the original data is the first element in the bit-packed data.
- No AVX2/512 etc. required.

Note: it only works on little-endian machines (x86, ARM, etc.).

## Benchmark

```bash
rustup default nightly  # for fastlanes comparison

env RUSTFLAGS='-C target-cpu=native' cargo bench 
```


The core reason you want to use this library instead of [fastlanes](https://crates.io/crates/fastlanes) is the performance of `select_packed` function. Tldr; depends on your workload, this library can be 2x faster.

```
select_packed/bmi_select/1%
                        time:   [4.2887 µs 4.2902 µs 4.2922 µs]
                        thrpt:  [3.8171 Gelem/s 3.8189 Gelem/s 3.8203 Gelem/s]
select_packed/fastlanes/1%
                        time:   [4.9043 µs 4.9061 µs 4.9085 µs]
                        thrpt:  [3.3379 Gelem/s 3.3395 Gelem/s 3.3408 Gelem/s]

select_packed/bmi_select/10%
                        time:   [4.6259 µs 4.6268 µs 4.6280 µs]
                        thrpt:  [3.5402 Gelem/s 3.5411 Gelem/s 3.5418 Gelem/s]
select_packed/fastlanes/10%
                        time:   [5.8027 µs 5.8088 µs 5.8164 µs]
                        thrpt:  [2.8169 Gelem/s 2.8205 Gelem/s 2.8235 Gelem/s]

select_packed/bmi_select/30%
                        time:   [5.2289 µs 5.2323 µs 5.2365 µs]
                        thrpt:  [3.1288 Gelem/s 3.1313 Gelem/s 3.1334 Gelem/s]
select_packed/fastlanes/30%
                        time:   [7.9092 µs 7.9186 µs 7.9284 µs]
                        thrpt:  [2.0665 Gelem/s 2.0691 Gelem/s 2.0715 Gelem/s]
                
select_packed/bmi_select/70%
                        time:   [6.3608 µs 6.3626 µs 6.3645 µs]
                        thrpt:  [2.5743 Gelem/s 2.5750 Gelem/s 2.5758 Gelem/s]
select_packed/fastlanes/70%
                        time:   [11.842 µs 11.845 µs 11.848 µs]
                        thrpt:  [1.3829 Gelem/s 1.3832 Gelem/s 1.3835 Gelem/s]
```

## Credits

The `select_packed` function is based on the following paper: [Selection Pushdown in Column Stores using Bit Manipulation Instructions](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/06/parquet-select-sigmod23.pdf), you should cite it.

```bibtex
@article{li2023selection,
  title={Selection Pushdown in Column Stores using Bit Manipulation Instructions},
  author={Li, Yinan and Lu, Jianan and Chandramouli, Badrish},
  journal={Proceedings of the ACM on Management of Data},
  volume={1},
  number={2},
  pages={1--26},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```
