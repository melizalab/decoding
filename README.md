# Decoding

The `decoding` module creates a dataset that is arranged to facilitate
neural decoding, i.e. reconstructing a stimulus given the neural
response that it induces.

## Examples

```
>>> import decoding
>>> builder = decoding.DatasetBuilder()
>>> builder.load_responses('pprox/P120_{}.pprox')
>>> builder.bin_responses()
>>> builder.add_stimuli('wav/{}.wav')
>>> builder.create_time_lags()
>>> builder.pool_trials() # optional
>>> dataset = builder.get_dataset()
```
