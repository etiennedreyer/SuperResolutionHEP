This repository contains the code used in the study, described in [Denoising Graph Super-Resolution towards Improved Collider Event Reconstruction](https://arxiv.org/abs/2409.16052)

---

## Dataset

The datasets (single electron and multi-particle) are available at [zenodo](https://zenodo.org/records/15582324)

---

## Training (super-resolution)
To train the super-resolution model, we can run the following -

```python
python train.py -cmv path/to/model_and_var_config.yml -ct path/to/train.yml
```

It has the following optional flags:

```
--exp_key (-ekey)
--debug_mode (-d)
--precision (-p)
--gpu (-g)
```

`exp_key` is the commet experiment key. `debug_mode=True` switches off all the logging options, and runs locally.

---

## Inference (super-resolution)

To run inference on the super-resolution model, we can run the following command -

```python
python inference.py -i path/to/inferenc.yml
```

It has the following optional flags:
```
--precision (-p)
--entry_start (-estart)
--entry_stop (-estop)
--batch_mode (-bm)
```

`entry_start` and `entry_stop` can be used to run inference in a specific range of entries. `batch_mode` is used to run inference in batch mode, which is useful for large datasets with multiple files.

---

## Training (particle flow)

To run inference, we need four things. The variable config, the stage one model config, the sage tow model config, and the final checkpoint (coming from stage2 trainings). We can put the paths to these four in a yaml file and run the following -

```
python train_pf.py -cmv path/to/model_and_var_config.yml -ct path/to/train.yml
```

It also has the following optional flags:

```
--exp_key (-ekey)
--debug_mode (-d)
--precision (-p)
--gpu (-g)
```

## Inference (particle flow)
To run inference on the particle flow model, we can run the following command -

```python
python inference_pf.py -i path/to/inference.yml
```

---

## Performance Plots

plots can be made with the notebooks in the folder - `notebooks/performance`

---

## Trained models

The checkpoints used to make the plots in the paper are available in `saved_checkpoints` directory