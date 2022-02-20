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
|   SNIP    | Generalized Zero-shot |     snipgzs.py      |
| SMP-2018  | Generalized Zero-shot |     smpgzs.py      |
|   SNIP    | Standard Zero-shot |     snipszs.py      |
| SMP-2018 | Standard Zero-shot |     smpszs.py      |
