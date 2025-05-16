# RocqStar statement ranker server

This is a small `python` server, that loads a pre-trained model into memory and runs the server to answer queries about the distance between two Rocq statements. This server could be used to test the performance of the model. 

The model is 500Mb in size and performs well on the CPU of my MacBook Pro (M1), taking roughly 150 ms to answer a query.

If you want to use the model as a ranker inside CoqPilot plugin or inside CoqPilot benchmark, you need to build CoqPilot from sources with the applied patch, which is located in the [CoqPilot+RocqStar](../CoqPilot+RocqStar/) directory of this repository.

### Running the server

Install the dependencies:
```bash
pip install -r requirements.txt
```

To start: 
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### To build coqpilot with the patch

Clone the extension:
```bash
git clone https://github.com/JetBrains-Research/coqpilot
```

Apply the patch to add our solution to the CoqPilot:
```bash
cd coqpilot
git apply rocqstar2coqpilot.patch
```

To run the extension, you must install a `coq-lsp` server. Depending on the system used in your project, you should install it using `opam` or `nix`. A well-configured `nix` project should have the `coq-lsp` server installed as a dependency. To install `coq-lsp` using `opam`, you can use the following commands: 
```bash
opam pin add coq-lsp 0.2.2+8.19
opam install coq-lsp
```

To build the extension locally, you'll need Node.js installed. The recommended way to manage Node.js versions is by using `nvm`. From the CoqPilot root directory, execute:
```bash
nvm use
```
If you prefer not to use `nvm`, ensure you install the Node.js version specified in the [`.nvmrc`](.nvmrc) file by any other method you prefer.

Once Node.js is installed, the remaining setup will be handled by the `npm` package manager. Run the following commands:
```bash
npm install
npm run compile
```

Afterwards please refer to the initial [CoqPilot](https://github.com/JetBrains-Research/coqpilot/tree/main) repository for benchmarking instructions. Our proposed solution should be available in both the extension and the benchmark.
