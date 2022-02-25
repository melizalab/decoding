# Decoding

The `decoding` module creates a dataset that is arranged to facilitate
neural decoding, i.e. reconstructing a stimulus given the neural
response that it induces.

To create the dataset, you must create a `decoding.sources.DataSource` and pass it a `decoding.dataset.DatasetBuilder` and configure your dataset.

## Documentation

To view documentation, run the following code:
```
pip install pdoc
pdoc decoding
```

## Running tests
Install dev dependencies:
```
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest
```
