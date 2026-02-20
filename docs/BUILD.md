# Build

## Standard
```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Install
```bash
cmake -S . -B build-release
cmake --build build-release -j
cmake --install build-release
```

Installs:
- headers to `/usr/local/include/quantcore`
- libraries to `/usr/local/lib`
- pkg-config file to `/usr/local/lib/pkgconfig/quantcore.pc`
