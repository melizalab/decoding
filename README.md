# Decoding

## Examples

```
>>> import decoding
>>> builder = decoding.DatasetBuilder()
>>> builder.load_response('pprox/P120_{}.pprox')
>>> builder.bin_responses()
>>> builder.add_stimuli('wav/{}.wav')
>>> builder.create_time_lags()
>>> builder.pool_trials() # optional
>>> dataset = builder.get_dataset()
```
