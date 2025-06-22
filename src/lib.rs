mod pack_unpack;
mod select;
pub use pack_unpack::{BitPackable, bit_pack, bit_unpack};
pub use select::select_packed;

const fn lane_size<T>() -> usize {
    std::mem::size_of::<T>() * 8
}
