# README

> Thanks very much for your valuable time and attention.

1. Platform

>Our experiments are conducted on a platform with Intel(R) Xeon(R) Gold 6248R CPU @3.00GHz and single GPU NVIDIA TITAN RTX 24GB.


2. Environment

```
conda env create -f SIGIR.yml
```


3. Running the code 


```
cd code
python Demo_Name.py
```

- The detailed configurations can be found in the following demo files and the appendix of the paper.
- The demo names are listed in the following table:

| Dataset  |   Setting    |  Demo Name  |
| :------: | :----------: | :---------: |
|   SNIPS   | Generalized Zero-shot |     snipgzs.py      |
| SMP | Generalized Zero-shot |     smpgzs.py      |
|   SNIPS   | Standard Zero-shot |     snipszs.py      |
| SMP | Standard Zero-shot |     smpszs.py      |
