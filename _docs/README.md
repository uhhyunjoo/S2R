# README

Learning how to use `Pytorch` + `Hydra` + `Wandb` using [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)


```
# train on CPU
python train.py trainer.gpus=0
```

```
# train on GPU
python train.py trainer.gpus=1
```

```
# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer.gpus=4 +trainer.strategy=ddp
```

```
# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python train.py trainer.gpus=4 +trainer.num_nodes=2 +trainer.strategy=ddp
```

```
# Train model with chosen experiment config
python train.py experiment=mnist_example
```

```
# runs test epoch without training
python train.py debug=test_only
```

```
# enforces debug-friendly configuration
python train.py debug=default
```

```
# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true
```

```
# over chosen experiment config
python train.py -m hparams_search=mnist_optuna experiment=example_simple
```

```
# Execute all experiments from folder
python train.py -m 'experiment=glob(*)'
```

```
# Execute evaluation for a given checkpoint
python test.py ckpt_path="/path/to/ckpt/name.ckpt"
```
