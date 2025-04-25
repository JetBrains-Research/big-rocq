# Dataset projects

This file includes the description of the dataset projects that are included into the mined dataset, including choice of the build system and some other relevant information.

### Projects built with Nix

All of those are built using `Coq` version 8.19 using the same sequence of commands:

```bash
nix-shell
make
```

- Projects from the `weakmemory` organization:
    - [weakmemory/imm](https://github.com/weakmemory/imm)
    - [weakmemory/promising2ToImm](https://github.com/weakmemory/promising2ToImm)
    - [weakmemory/xmm](https://github.com/weakmemory/xmm)

### Projects built with Opam 

For these, the actions may vary.

#### CompCert

Cofigure: 
```bash
chmod +x configure
./configure (target architecture)
```

Install `tactician`: 
```bash
opam pin coq-tactician https://github.com/coq-tactician/coq-tactician.git#coq8.19
opam install coq-tactician
tactician enable
```

Build: 
```bash
make
```

