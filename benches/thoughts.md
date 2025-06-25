How to optimize the unpacking of u64?

Rule 1: make compiler your friend
- Const generics: instead of passing a runtime value, make n copies of the function, each with a different const generic value
- Hint the compiler with bounds checks, i.e., `assert!(output.len() >= 64);`. This will allow the compiler to optimize the bounds check in subsequent `output[37], output[38], ...` etc.
- Make your code easy to optimize. Process chunk by chunk, and each chunk is a separate function which *takes an array instead of a slice*.

Rule 2: make CPU your friend
- Unroll loops.
- Remove loop dependencies.
- Use less branches.
