# Code conventions

- Name variables by what they hold and where they're used (`sample_transcript`, `invalid_input`), not single letters or generic names (`t`, `e`, `data`).
- Write code as close to the functional style as practical: pure functions, values passed in and returned, no shared mutable state. Prefer immutable data (tuples, frozen dataclasses) over mutable containers.
