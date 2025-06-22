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
assert_eq!(unpacked, vec![1, 2, 3, 4, 5, 6, 7, 8]);
```

### Bit selection
```rust
use bmi_select::select_packed;
let packed = vec![0x87654321];
let bit_mask = vec![0b10101010];
let selected = select_packed(&packed, 4, &bit_mask);
assert_eq!(selected, vec![0x8642]);
```

## Features

- No dependencies.
- State-of-the-art performance backed by mind-blowing algorithms (see credits below).
- Bit-packed data follows the same layout as the original data, i.e., the first element in the original data is the first element in the bit-packed data.
- No AVX2/512 etc. required.

Note: it only works on little-endian machines (x86, ARM, etc.).

## Benchmark

```bash
cargo bench
```

The `select_packed` function can be 3.6x faster than best-effort baseline implementation.

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
