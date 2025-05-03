# RocqStar statement ranker server

This is a small `python` server, that loads a pre-trained model into memory and runs the server to answer queries about the distance between two Rocq statements.

The respective RocqStar ranker in [coqpilot](https://github.com/JetBrains-Research/coqpilot) is responsible for communicating with this API and perform premise selection based on the embeddings.

The model is 500Mb in size and performs well on the CPU of my MacBook Pro (M1), taking roughly 150 ms to answer a query.

### Running the server

Install the dependencies:
```bash
pip install -r requirements.txt
```

To start: 
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```