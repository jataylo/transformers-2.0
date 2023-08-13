<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Language model training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2,
ALBERT, BERT, DistilBERT, RoBERTa, XLNet... GPT and GPT-2 are trained or fine-tuned using a causal language modeling
(CLM) loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM)
loss. XLNet uses permutation language modeling (PLM), you can find more information about the differences between those
objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop and leverages the 🤗 Accelerate library . Both sets use the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

**Note:** The old script `run_language_modeling.py` is still available [here](https://github.com/huggingface/transformers/blob/main/examples/legacy/run_language_modeling.py).

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

### GPT-2/GPT and causal language modeling

Ran into OOM issues


### RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different
as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling.

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore,
converge slightly slower (over-fitting takes more epochs).

```bash
TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm \
    --torch_compile True
```

PT2.0 results NV31 (single gpu) + Ryzen 7 5700x3d - train/train_steps_per_second: roberta-base
| GPU       | Run Mode                          | train_steps_per_second | train_runtime | eval_steps_per_second | eval_runtime|
|-----------|-----------------------------------|------------------------|---------------|-----------------------|-------------|
| 7900 XTX  | Regular execution                 | 1.633                  | 1102.26       | 4.404                 | 14.0782
| 7900 XTX  | compile - autotune cached         | 1.921                  | 937.01        | 4.555                 | 13.611

Autotune logs:
```
AUTOTUNE addmm(4096x768, 4096x768, 768x768)
  addmm 0.7796 ms 100.0%
  bias_addmm 0.8010 ms 97.3%
  triton_mm_3 1.2533 ms 62.2%
  triton_mm_4 1.5144 ms 51.5%
  triton_mm_1 2.4811 ms 31.4%
  triton_mm_0 3.9687 ms 19.6%
  triton_mm_2 4.1816 ms 18.6%
SingleProcess AUTOTUNE takes 1.0066 seconds
AUTOTUNE bmm(96x512x64, 96x64x512)
  triton_bmm_18 0.3871 ms 100.0%
  bmm 0.4525 ms 85.5%
  triton_bmm_19 0.5081 ms 76.2%
  triton_bmm_17 0.7316 ms 52.9%
  triton_bmm_16 0.7634 ms 50.7%
  triton_bmm_15 1.8229 ms 21.2%
SingleProcess AUTOTUNE takes 3.7034 seconds
AUTOTUNE bmm(96x512x512, 96x512x64)
  bmm 0.6537 ms 100.0%
  triton_bmm_22 0.7556 ms 86.5%
  triton_bmm_21 0.8543 ms 76.5%
  triton_bmm_24 1.4948 ms 43.7%
  triton_bmm_23 1.8968 ms 34.5%
  triton_bmm_20 2.8454 ms 23.0%
SingleProcess AUTOTUNE takes 3.7297 seconds
AUTOTUNE addmm(4096x3072, 4096x768, 768x3072)
  addmm 3.0432 ms 100.0%
  bias_addmm 3.2404 ms 93.9%
  triton_mm_33 4.9573 ms 61.4%
  triton_mm_34 5.8539 ms 52.0%
  triton_mm_31 9.6553 ms 31.5%
  triton_mm_30 14.9839 ms 20.3%
  triton_mm_32 16.1833 ms 18.8%
SingleProcess AUTOTUNE takes 1.2906 seconds
AUTOTUNE addmm(4096x768, 4096x3072, 3072x768)
  addmm 3.0493 ms 100.0%
  bias_addmm 3.2656 ms 93.4%
  triton_mm_38 5.0376 ms 60.5%
  triton_mm_39 5.6506 ms 54.0%
  triton_mm_36 9.7735 ms 31.2%
  triton_mm_35 15.8301 ms 19.3%
  triton_mm_37 16.4936 ms 18.5%
SingleProcess AUTOTUNE takes 1.2771 seconds
AUTOTUNE addmm(4096x50265, 4096x768, 768x50265)
  bias_addmm 52.4832 ms 100.0%
  addmm 52.5039 ms 100.0%
  triton_mm_488 77.5220 ms 67.7%
  triton_mm_489 94.4260 ms 55.6%
  triton_mm_486 157.5490 ms 33.3%
  triton_mm_485 250.4018 ms 21.0%
  triton_mm_487 263.4617 ms 19.9%
SingleProcess AUTOTUNE takes 10.7329 seconds
AUTOTUNE mm(4096x50265, 50265x768)
  mm 48.0725 ms 100.0%
  triton_mm_494 52.2889 ms 91.9%
  triton_mm_492 53.6913 ms 89.5%
  triton_mm_493 55.0135 ms 87.4%
  triton_mm_491 57.8965 ms 83.0%
  triton_mm_490 249.8504 ms 19.2%
SingleProcess AUTOTUNE takes 7.8478 seconds
AUTOTUNE mm(50265x4096, 4096x768)
  triton_mm_499 25.9284 ms 100.0%
  triton_mm_498 34.4858 ms 75.2%
  mm 41.1096 ms 63.1%
  triton_mm_496 44.0133 ms 58.9%
  triton_mm_497 50.4151 ms 51.4%
  triton_mm_495 230.1886 ms 11.3%
SingleProcess AUTOTUNE takes 8.1123 seconds
AUTOTUNE mm(4096x768, 768x768)
  triton_mm_503 0.6335 ms 100.0%
  mm 0.7132 ms 88.8%
  triton_mm_504 0.7737 ms 81.9%
  triton_mm_502 0.8098 ms 78.2%
  triton_mm_501 0.9100 ms 69.6%
  triton_mm_500 3.3355 ms 19.0%
SingleProcess AUTOTUNE takes 3.8021 seconds
AUTOTUNE mm(768x4096, 4096x768)
  triton_mm_509 0.4865 ms 100.0%
  mm 0.6709 ms 72.5%
  triton_mm_508 0.7360 ms 66.1%
  triton_mm_506 0.7820 ms 62.2%
  triton_mm_507 0.8950 ms 54.4%
  triton_mm_505 4.2648 ms 11.4%
SingleProcess AUTOTUNE takes 3.8097 seconds
AUTOTUNE mm(4096x768, 768x3072)
  triton_mm_513 2.3959 ms 100.0%
  mm 2.8749 ms 83.3%
  triton_mm_514 3.0405 ms 78.8%
  triton_mm_512 3.1348 ms 76.4%
  triton_mm_511 3.5457 ms 67.6%
  triton_mm_510 11.6910 ms 20.5%
SingleProcess AUTOTUNE takes 3.9608 seconds
AUTOTUNE mm(768x4096, 4096x3072)
  triton_mm_519 1.7925 ms 100.0%
  mm 2.5203 ms 71.1%
  triton_mm_518 2.6681 ms 67.2%
  triton_mm_516 2.8059 ms 63.9%
  triton_mm_517 3.2811 ms 54.6%
  triton_mm_515 16.1626 ms 11.1%
SingleProcess AUTOTUNE takes 4.0369 seconds
AUTOTUNE mm(4096x3072, 3072x768)
  triton_mm_523 2.6032 ms 100.0%
  mm 2.9008 ms 89.7%
  triton_mm_524 3.1134 ms 83.6%
  triton_mm_522 3.2187 ms 80.9%
  triton_mm_521 3.6163 ms 72.0%
  triton_mm_520 13.4987 ms 19.3%
SingleProcess AUTOTUNE takes 3.9913 seconds
AUTOTUNE mm(3072x4096, 4096x768)
  triton_mm_529 1.8124 ms 100.0%
  mm 2.5174 ms 72.0%
  triton_mm_528 2.6732 ms 67.8%
  triton_mm_526 2.8259 ms 64.1%
  triton_mm_527 3.2970 ms 55.0%
  triton_mm_525 16.3192 ms 11.1%
SingleProcess AUTOTUNE takes 4.0707 seconds
AUTOTUNE bmm(96x512x512, 96x512x64)
  triton_bmm_544 0.3702 ms 100.0%
  bmm 0.4114 ms 90.0%
  triton_bmm_543 0.4188 ms 88.4%
  triton_bmm_541 0.5241 ms 70.6%
  triton_bmm_542 0.5847 ms 63.3%
  triton_bmm_540 2.3642 ms 15.7%
SingleProcess AUTOTUNE takes 3.8818 seconds
AUTOTUNE bmm(96x512x64, 96x64x512)
  bmm 0.5410 ms 100.0%
  triton_bmm_548 0.8866 ms 61.0%
  triton_bmm_549 1.0128 ms 53.4%
  triton_bmm_546 1.9165 ms 28.2%
  triton_bmm_545 2.4045 ms 22.5%
  triton_bmm_547 3.0557 ms 17.7%
SingleProcess AUTOTUNE takes 3.6850 seconds
AUTOTUNE bmm(96x64x512, 96x512x512)
  triton_bmm_554 0.3177 ms 100.0%
  bmm 0.4095 ms 77.6%
  triton_bmm_553 0.4804 ms 66.1%
  triton_bmm_551 0.4985 ms 63.7%
  triton_bmm_552 0.5683 ms 55.9%
  triton_bmm_550 2.3994 ms 13.2%
SingleProcess AUTOTUNE takes 3.8766 seconds
AUTOTUNE bmm(96x512x512, 96x512x64)
  bmm 0.7622 ms 100.0%
  triton_bmm_558 1.2721 ms 59.9%
  triton_bmm_559 1.6946 ms 45.0%
  triton_bmm_556 1.7875 ms 42.6%
  triton_bmm_557 2.8865 ms 26.4%
  triton_bmm_555 4.3440 ms 17.5%
SingleProcess AUTOTUNE takes 3.7132 seconds
AUTOTUNE addmm(3072x768, 3072x768, 768x768)
  addmm 0.6224 ms 100.0%
  bias_addmm 0.6316 ms 98.5%
  triton_mm_1473 1.0576 ms 58.9%
  triton_mm_1474 1.1806 ms 52.7%
  triton_mm_1471 1.9948 ms 31.2%
  triton_mm_1470 3.0245 ms 20.6%
  triton_mm_1472 3.2954 ms 18.9%
SingleProcess AUTOTUNE takes 3.9455 seconds
AUTOTUNE bmm(72x512x64, 72x64x512)
  triton_bmm_1488 0.3165 ms 100.0%
  bmm 0.3548 ms 89.2%
  triton_bmm_1489 0.4009 ms 79.0%
  triton_bmm_1487 0.5665 ms 55.9%
  triton_bmm_1486 0.5865 ms 54.0%
  triton_bmm_1485 1.4019 ms 22.6%
SingleProcess AUTOTUNE takes 0.7088 seconds
AUTOTUNE bmm(72x512x512, 72x512x64)
  bmm 0.5543 ms 100.0%
  triton_bmm_1492 0.6000 ms 92.4%
  triton_bmm_1491 0.6717 ms 82.5%
  triton_bmm_1494 1.2878 ms 43.0%
  triton_bmm_1493 1.5602 ms 35.5%
  triton_bmm_1490 2.2745 ms 24.4%
SingleProcess AUTOTUNE takes 0.6758 seconds
AUTOTUNE addmm(3072x3072, 3072x768, 768x3072)
  addmm 2.4324 ms 100.0%
  bias_addmm 2.5514 ms 95.3%
  triton_mm_1503 3.8473 ms 63.2%
  triton_mm_1504 4.5005 ms 54.0%
  triton_mm_1501 7.6479 ms 31.8%
  triton_mm_1500 10.7005 ms 22.7%
  triton_mm_1502 12.6844 ms 19.2%
SingleProcess AUTOTUNE takes 4.1729 seconds
AUTOTUNE addmm(3072x768, 3072x3072, 3072x768)
  addmm 2.4325 ms 100.0%
  bias_addmm 2.5675 ms 94.7%
  triton_mm_1508 4.1402 ms 58.8%
  triton_mm_1509 4.6057 ms 52.8%
  triton_mm_1506 7.8208 ms 31.1%
  triton_mm_1505 12.0324 ms 20.2%
  triton_mm_1507 12.9823 ms 18.7%
SingleProcess AUTOTUNE takes 4.1559 seconds
AUTOTUNE addmm(3072x50265, 3072x768, 768x50265)
  bias_addmm 39.5354 ms 100.0%
  addmm 40.8730 ms 96.7%
  triton_mm_1958 60.9665 ms 64.8%
  triton_mm_1959 71.2798 ms 55.5%
  triton_mm_1956 123.3827 ms 32.0%
  triton_mm_1955 184.0860 ms 21.5%
  triton_mm_1957 205.4101 ms 19.2%
SingleProcess AUTOTUNE takes 9.0914 seconds
AUTOTUNE mm(3072x50265, 50265x768)
  mm 33.9691 ms 100.0%
  triton_mm_1964 39.9266 ms 85.1%
  triton_mm_1962 40.3930 ms 84.1%
  triton_mm_1961 44.5443 ms 76.3%
  triton_mm_1963 45.4748 ms 74.7%
  triton_mm_1960 193.3708 ms 17.6%
SingleProcess AUTOTUNE takes 6.5511 seconds
AUTOTUNE mm(50265x3072, 3072x768)
  triton_mm_1969 19.3927 ms 100.0%
  mm 30.6493 ms 63.3%
  triton_mm_1968 31.0467 ms 62.5%
  triton_mm_1966 32.3766 ms 59.9%
  triton_mm_1967 35.9832 ms 53.9%
  triton_mm_1965 185.9188 ms 10.4%
SingleProcess AUTOTUNE takes 7.0955 seconds
AUTOTUNE mm(3072x768, 768x768)
  triton_mm_1973 0.4443 ms 100.0%
  mm 0.5374 ms 82.7%
  triton_mm_1974 0.5760 ms 77.1%
  triton_mm_1972 0.6091 ms 72.9%
  triton_mm_1971 0.6932 ms 64.1%
  triton_mm_1970 2.6848 ms 16.5%
SingleProcess AUTOTUNE takes 3.8715 seconds
AUTOTUNE mm(768x3072, 3072x768)
  triton_mm_1979 0.3680 ms 100.0%
  mm 0.4986 ms 73.8%
  triton_mm_1976 0.6014 ms 61.2%
  triton_mm_1978 0.6150 ms 59.8%
  triton_mm_1977 0.6745 ms 54.6%
  triton_mm_1975 3.1589 ms 11.6%
SingleProcess AUTOTUNE takes 3.8429 seconds
AUTOTUNE mm(3072x768, 768x3072)
  triton_mm_1983 1.6881 ms 100.0%
  mm 2.1695 ms 77.8%
  triton_mm_1984 2.2687 ms 74.4%
  triton_mm_1982 2.3651 ms 71.4%
  triton_mm_1981 2.6756 ms 63.1%
  triton_mm_1980 9.0564 ms 18.6%
SingleProcess AUTOTUNE takes 3.9516 seconds
AUTOTUNE mm(768x3072, 3072x3072)
  triton_mm_1989 1.3440 ms 100.0%
  mm 1.8903 ms 71.1%
  triton_mm_1986 2.0918 ms 64.3%
  triton_mm_1988 2.4110 ms 55.7%
  triton_mm_1987 2.4118 ms 55.7%
  triton_mm_1985 12.0653 ms 11.1%
SingleProcess AUTOTUNE takes 3.9763 seconds
AUTOTUNE mm(3072x3072, 3072x768)
  triton_mm_1993 1.8361 ms 100.0%
  mm 2.1989 ms 83.5%
  triton_mm_1994 2.3090 ms 79.5%
  triton_mm_1992 2.4098 ms 76.2%
  triton_mm_1991 2.7204 ms 67.5%
  triton_mm_1990 10.7124 ms 17.1%
SingleProcess AUTOTUNE takes 3.9529 seconds
AUTOTUNE mm(3072x3072, 3072x768)
  triton_mm_1999 1.3558 ms 100.0%
  mm 1.8828 ms 72.0%
  triton_mm_1996 2.0939 ms 64.7%
  triton_mm_1998 2.4139 ms 56.2%
  triton_mm_1997 2.4296 ms 55.8%
  triton_mm_1995 12.0817 ms 11.2%
SingleProcess AUTOTUNE takes 4.0143 seconds
AUTOTUNE bmm(72x512x512, 72x512x64)
  triton_bmm_2014 0.2876 ms 100.0%
  bmm 0.3143 ms 91.5%
  triton_bmm_2013 0.3347 ms 85.9%
  triton_bmm_2011 0.4012 ms 71.7%
  triton_bmm_2012 0.4402 ms 65.3%
  triton_bmm_2010 1.8234 ms 15.8%
SingleProcess AUTOTUNE takes 0.6909 seconds
AUTOTUNE bmm(72x512x64, 72x64x512)
  bmm 0.4036 ms 100.0%
  triton_bmm_2018 0.6508 ms 62.0%
  triton_bmm_2019 0.7590 ms 53.2%
  triton_bmm_2016 1.4345 ms 28.1%
  triton_bmm_2015 1.8995 ms 21.2%
  triton_bmm_2017 2.2920 ms 17.6%
SingleProcess AUTOTUNE takes 0.7569 seconds
AUTOTUNE bmm(72x64x512, 72x512x512)
  triton_bmm_2024 0.2503 ms 100.0%
  bmm 0.3173 ms 78.9%
  triton_bmm_2023 0.3797 ms 65.9%
  triton_bmm_2021 0.3857 ms 64.9%
  triton_bmm_2022 0.4270 ms 58.6%
  triton_bmm_2020 1.8690 ms 13.4%
SingleProcess AUTOTUNE takes 0.6968 seconds
AUTOTUNE bmm(72x512x512, 72x512x64)
  bmm 0.6295 ms 100.0%
  triton_bmm_2028 1.1168 ms 56.4%
  triton_bmm_2029 1.2610 ms 49.9%
  triton_bmm_2026 1.3524 ms 46.5%
  triton_bmm_2027 2.1851 ms 28.8%
  triton_bmm_2025 3.4041 ms 18.5%
SingleProcess AUTOTUNE takes 0.7143 seconds

```
