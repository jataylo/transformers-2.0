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

# Text classification examples

## GLUE tasks

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/). This script can fine-tune any of the models on the [hub](https://huggingface.co/models)
and can also be used for a dataset hosted on our [hub](https://huggingface.co/datasets) or your own data in a csv or a JSON file
(the script might need some tweaks in that case, refer to the comments inside for help).

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

```bash
export TASK_NAME=mrpc

TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \ 
  --torch_compile True
```

PT2.0 results NV31 (single gpu) + Ryzen 7 5700x3d - train/train_steps_per_second:
| GPU       | Run Mode                          | train_steps_per_second | train_runtime | eval_steps_per_second | eval_runtime|
|-----------|-----------------------------------|------------------------|---------------|-----------------------|-------------|
| 7900 XTX  | Regular execution                 | 1.604                  | 215.123       | 24.082                | 2.118
| 7900 XTX  | compile cold_start (MAX_AUTOTUNE) | 1.115                  | 309.288       | 26.054                | 1.958
| 7900 XTX  | compile - autotune cached         | 2.116                  | 163.024       | 26.054                | 1.958

Autotune logs:
```
AUTOTUNE addmm(4096x768, 4096x768, 768x768)
  addmm 0.8507 ms 100.0%
  bias_addmm 0.8846 ms 96.2%
  triton_mm_3 1.3068 ms 65.1%
  triton_mm_4 1.5753 ms 54.0%
  triton_mm_1 2.5559 ms 33.3%
  triton_mm_0 3.9974 ms 21.3%
  triton_mm_2 4.2667 ms 19.9%
SingleProcess AUTOTUNE takes 4.0199 seconds
AUTOTUNE bmm(384x128x64, 384x64x128)
  bmm 0.1496 ms 100.0%
  triton_bmm_18 0.1541 ms 97.1%
  triton_bmm_19 0.1571 ms 95.2%
  triton_bmm_17 0.1846 ms 81.0%
  triton_bmm_16 0.1931 ms 77.5%
  triton_bmm_15 0.5189 ms 28.8%
SingleProcess AUTOTUNE takes 3.7429 seconds
AUTOTUNE bmm(384x128x128, 384x128x64)
  bmm 0.1586 ms 100.0%
  triton_bmm_24 0.1761 ms 90.0%
  triton_bmm_22 0.1769 ms 89.7%
  triton_bmm_21 0.1842 ms 86.1%
  triton_bmm_23 0.2366 ms 67.0%
  triton_bmm_20 0.6096 ms 26.0%
SingleProcess AUTOTUNE takes 3.7727 seconds
AUTOTUNE addmm(4096x3072, 4096x768, 768x3072)
  addmm 3.3193 ms 100.0%
  bias_addmm 3.5453 ms 93.6%
  triton_mm_33 4.9345 ms 67.3%
  triton_mm_34 6.0042 ms 55.3%
  triton_mm_31 9.9820 ms 33.3%
  triton_mm_30 14.6605 ms 22.6%
  triton_mm_32 16.7303 ms 19.8%
SingleProcess AUTOTUNE takes 4.3119 seconds
AUTOTUNE addmm(4096x768, 4096x3072, 3072x768)
  addmm 3.3216 ms 100.0%
  bias_addmm 3.5500 ms 93.6%
  triton_mm_38 5.1451 ms 64.6%
  triton_mm_39 6.1825 ms 53.7%
  triton_mm_36 10.1107 ms 32.9%
  triton_mm_35 15.9624 ms 20.8%
  triton_mm_37 16.8767 ms 19.7%
SingleProcess AUTOTUNE takes 4.2942 seconds
AUTOTUNE addmm(32x768, 32x768, 768x768)
  addmm 0.0401 ms 100.0%
  bias_addmm 0.0404 ms 99.1%
  triton_mm_484 0.0534 ms 75.0%
  triton_mm_481 0.0638 ms 62.9%
  triton_mm_483 0.1010 ms 39.7%
  triton_mm_482 0.1629 ms 24.6%
  triton_mm_480 0.1708 ms 23.5%
SingleProcess AUTOTUNE takes 3.2990 seconds
AUTOTUNE addmm(32x2, 32x768, 768x2)
  bias_addmm 0.0372 ms 100.0%
  addmm 0.0372 ms 100.0%
  triton_mm_485 0.0378 ms 98.5%
  triton_mm_487 0.0390 ms 95.5%
  triton_mm_486 0.0392 ms 95.1%
  triton_mm_488 0.0396 ms 94.0%
SingleProcess AUTOTUNE takes 2.4979 seconds
AUTOTUNE mm(32x2, 2x768)
  mm 0.0072 ms 100.0%
  triton_mm_490 0.0075 ms 95.7%
  triton_mm_491 0.0079 ms 91.4%
  triton_mm_489 0.0080 ms 90.0%
  triton_mm_492 0.0080 ms 90.0%
SingleProcess AUTOTUNE takes 1.9111 seconds
AUTOTUNE mm(2x32, 32x768)
  triton_mm_494 0.0077 ms 100.0%
  triton_mm_493 0.0077 ms 99.5%
  triton_mm_495 0.0077 ms 99.5%
  triton_mm_497 0.0082 ms 94.1%
  triton_mm_496 0.0082 ms 93.2%
  mm 0.0084 ms 91.4%
SingleProcess AUTOTUNE takes 2.6424 seconds
AUTOTUNE mm(32x768, 768x768)
  triton_mm_499 0.0362 ms 100.0%
  triton_mm_500 0.0398 ms 90.9%
  triton_mm_502 0.0429 ms 84.3%
  triton_mm_501 0.0450 ms 80.4%
  mm 0.0462 ms 78.2%
  triton_mm_498 0.1435 ms 25.2%
SingleProcess AUTOTUNE takes 3.1443 seconds
AUTOTUNE mm(768x32, 32x768)
  triton_mm_507 0.0124 ms 100.0%
  mm 0.0136 ms 91.7%
  triton_mm_504 0.0141 ms 88.1%
  triton_mm_505 0.0149 ms 83.6%
  triton_mm_506 0.0187 ms 66.5%
  triton_mm_503 0.0202 ms 61.7%
SingleProcess AUTOTUNE takes 3.2528 seconds
AUTOTUNE mm(4096x768, 768x3072)
  triton_mm_511 2.2380 ms 100.0%
  triton_mm_512 3.0200 ms 74.1%
  mm 3.0374 ms 73.7%
  triton_mm_510 3.0459 ms 73.5%
  triton_mm_509 3.4114 ms 65.6%
  triton_mm_508 11.5196 ms 19.4%
SingleProcess AUTOTUNE takes 4.0092 seconds
AUTOTUNE mm(768x4096, 4096x3072)
  triton_mm_517 1.7877 ms 100.0%
  mm 2.6346 ms 67.9%
  triton_mm_514 2.6930 ms 66.4%
  triton_mm_516 2.8154 ms 63.5%
  triton_mm_515 3.1776 ms 56.3%
  triton_mm_513 16.1784 ms 11.0%
SingleProcess AUTOTUNE takes 4.0956 seconds
AUTOTUNE mm(4096x3072, 3072x768)
  triton_mm_521 2.6275 ms 100.0%
  mm 3.0965 ms 84.9%
  triton_mm_520 3.1035 ms 84.7%
  triton_mm_522 3.1115 ms 84.4%
  triton_mm_519 3.4653 ms 75.8%
  triton_mm_518 13.4091 ms 19.6%
SingleProcess AUTOTUNE takes 4.0270 seconds
AUTOTUNE mm(3072x4096, 4096x768)
  triton_mm_527 1.7963 ms 100.0%
  mm 2.6491 ms 67.8%
  triton_mm_524 2.7359 ms 65.7%
  triton_mm_526 2.8131 ms 63.9%
  triton_mm_525 3.1949 ms 56.2%
  triton_mm_523 16.3702 ms 11.0%
SingleProcess AUTOTUNE takes 4.0828 seconds
AUTOTUNE mm(4096x768, 768x768)
  triton_mm_531 0.6440 ms 100.0%
  mm 0.7630 ms 84.4%
  triton_mm_532 0.7732 ms 83.3%
  triton_mm_530 0.7850 ms 82.0%
  triton_mm_529 0.8789 ms 73.3%
  triton_mm_528 3.3115 ms 19.4%
SingleProcess AUTOTUNE takes 3.8841 seconds
AUTOTUNE mm(768x4096, 4096x768)
  triton_mm_537 0.4841 ms 100.0%
  mm 0.6954 ms 69.6%
  triton_mm_536 0.7301 ms 66.3%
  triton_mm_534 0.7640 ms 63.4%
  triton_mm_535 0.8678 ms 55.8%
  triton_mm_533 4.3375 ms 11.2%
SingleProcess AUTOTUNE takes 3.9298 seconds
AUTOTUNE bmm(384x128x128, 384x128x64)
  triton_bmm_541 0.1492 ms 100.0%
  triton_bmm_539 0.1597 ms 93.4%
  triton_bmm_542 0.1678 ms 88.9%
  triton_bmm_540 0.1684 ms 88.6%
  bmm 0.1749 ms 85.3%
  triton_bmm_538 0.6347 ms 23.5%
SingleProcess AUTOTUNE takes 3.9356 seconds
AUTOTUNE bmm(384x128x64, 384x64x128)
  bmm 0.1483 ms 100.0%
  triton_bmm_546 0.2353 ms 63.0%
  triton_bmm_547 0.2724 ms 54.4%
  triton_bmm_544 0.4843 ms 30.6%
  triton_bmm_543 0.6856 ms 21.6%
  triton_bmm_545 0.7751 ms 19.1%
SingleProcess AUTOTUNE takes 3.6433 seconds
AUTOTUNE bmm(384x64x128, 384x128x128)
  triton_bmm_549 0.1535 ms 100.0%
  triton_bmm_550 0.1605 ms 95.6%
  triton_bmm_551 0.1652 ms 92.9%
  triton_bmm_552 0.1687 ms 91.0%
  bmm 0.1750 ms 87.7%
  triton_bmm_548 0.6333 ms 24.2%
SingleProcess AUTOTUNE takes 3.8938 seconds
AUTOTUNE bmm(384x128x128, 384x128x64)
  bmm 0.1520 ms 100.0%
  triton_bmm_556 0.2445 ms 62.2%
  triton_bmm_557 0.2834 ms 53.6%
  triton_bmm_554 0.4599 ms 33.1%
  triton_bmm_553 0.7454 ms 20.4%
  triton_bmm_555 0.7523 ms 20.2%
SingleProcess AUTOTUNE takes 3.7483 seconds
AUTOTUNE addmm(2560x768, 2560x768, 768x768)
  addmm 0.5414 ms 100.0%
  bias_addmm 0.5553 ms 97.5%
  triton_mm_1471 0.8471 ms 63.9%
  triton_mm_1472 1.0024 ms 54.0%
  triton_mm_1469 1.6332 ms 33.2%
  triton_mm_1468 2.5638 ms 21.1%
  triton_mm_1470 2.7235 ms 19.9%
SingleProcess AUTOTUNE takes 4.0440 seconds
AUTOTUNE bmm(240x128x64, 240x64x128)
  bmm 0.0995 ms 100.0%
  triton_bmm_1487 0.1054 ms 94.4%
  triton_bmm_1486 0.1071 ms 92.9%
  triton_bmm_1485 0.1236 ms 80.5%
  triton_bmm_1484 0.1290 ms 77.1%
  triton_bmm_1483 0.3518 ms 28.3%
SingleProcess AUTOTUNE takes 0.6560 seconds
AUTOTUNE bmm(240x128x128, 240x128x64)
  bmm 0.1078 ms 100.0%
  triton_bmm_1490 0.1255 ms 85.8%
  triton_bmm_1489 0.1305 ms 82.6%
  triton_bmm_1492 0.1340 ms 80.4%
  triton_bmm_1491 0.1650 ms 65.3%
  triton_bmm_1488 0.4380 ms 24.6%
SingleProcess AUTOTUNE takes 0.6381 seconds
AUTOTUNE addmm(2560x3072, 2560x768, 768x3072)
  addmm 2.1200 ms 100.0%
  bias_addmm 2.2479 ms 94.3%
  triton_mm_1501 3.1634 ms 67.0%
  triton_mm_1502 3.8231 ms 55.5%
  triton_mm_1499 6.2492 ms 33.9%
  triton_mm_1498 9.0292 ms 23.5%
  triton_mm_1500 10.4831 ms 20.2%
SingleProcess AUTOTUNE takes 4.1775 seconds
AUTOTUNE addmm(2560x768, 2560x3072, 3072x768)
  addmm 2.1217 ms 100.0%
  bias_addmm 2.2668 ms 93.6%
  triton_mm_1506 3.3283 ms 63.7%
  triton_mm_1507 3.9729 ms 53.4%
  triton_mm_1504 6.4524 ms 32.9%
  triton_mm_1503 10.0848 ms 21.0%
  triton_mm_1505 10.7595 ms 19.7%
SingleProcess AUTOTUNE takes 4.1493 seconds
AUTOTUNE addmm(20x768, 20x768, 768x768)
  addmm 0.0389 ms 100.0%
  bias_addmm 0.0390 ms 99.8%
  triton_mm_1952 0.0513 ms 75.8%
  triton_mm_1949 0.0640 ms 60.8%
  triton_mm_1951 0.1010 ms 38.5%
  triton_mm_1950 0.1617 ms 24.0%
  triton_mm_1948 0.1704 ms 22.8%
SingleProcess AUTOTUNE takes 3.2706 seconds
AUTOTUNE addmm(20x2, 20x768, 768x2)
  addmm 0.0372 ms 100.0%
  bias_addmm 0.0373 ms 99.8%
  triton_mm_1953 0.0383 ms 97.0%
  triton_mm_1955 0.0387 ms 96.1%
  triton_mm_1956 0.0397 ms 93.7%
  triton_mm_1954 0.0398 ms 93.4%
SingleProcess AUTOTUNE takes 2.5055 seconds
AUTOTUNE mm(20x2, 2x768)
  mm 0.0076 ms 100.0%
  triton_mm_1958 0.0076 ms 100.0%
  triton_mm_1960 0.0077 ms 99.0%
  triton_mm_1957 0.0080 ms 96.0%
  triton_mm_1959 0.0081 ms 94.6%
SingleProcess AUTOTUNE takes 1.8803 seconds
AUTOTUNE mm(2x20, 20x768)
  triton_mm_1965 0.0081 ms 100.0%
  mm 0.0082 ms 98.5%
  triton_mm_1964 0.0087 ms 93.1%
  triton_mm_1961 0.0094 ms 86.8%
  triton_mm_1962 0.0109 ms 74.6%
  triton_mm_1963 0.0113 ms 71.7%
SingleProcess AUTOTUNE takes 3.0316 seconds
AUTOTUNE mm(20x768, 768x768)
  triton_mm_1967 0.0359 ms 100.0%
  triton_mm_1968 0.0396 ms 90.6%
  triton_mm_1970 0.0403 ms 89.0%
  triton_mm_1969 0.0438 ms 81.9%
  mm 0.0459 ms 78.2%
  triton_mm_1966 0.1388 ms 25.9%
SingleProcess AUTOTUNE takes 3.1412 seconds
AUTOTUNE mm(768x20, 20x768)
  mm 0.0118 ms 100.0%
  triton_mm_1975 0.0125 ms 94.9%
  triton_mm_1974 0.0169 ms 70.1%
  triton_mm_1972 0.0192 ms 61.8%
  triton_mm_1973 0.0204 ms 57.9%
  triton_mm_1971 0.0633 ms 18.7%
SingleProcess AUTOTUNE takes 3.7751 seconds
AUTOTUNE mm(2560x768, 768x3072)
  triton_mm_1979 1.4461 ms 100.0%
  triton_mm_1980 1.8988 ms 76.2%
  mm 1.9084 ms 75.8%
  triton_mm_1978 1.9536 ms 74.0%
  triton_mm_1977 2.1794 ms 66.4%
  triton_mm_1976 7.5059 ms 19.3%
SingleProcess AUTOTUNE takes 3.9926 seconds
AUTOTUNE mm(768x2560, 2560x3072)
  triton_mm_1985 1.1067 ms 100.0%
  mm 1.6716 ms 66.2%
  triton_mm_1982 1.7262 ms 64.1%
  triton_mm_1983 1.9899 ms 55.6%
  triton_mm_1984 2.0446 ms 54.1%
  triton_mm_1981 10.0333 ms 11.0%
SingleProcess AUTOTUNE takes 4.0142 seconds
AUTOTUNE mm(2560x3072, 3072x768)
  triton_mm_1989 1.5777 ms 100.0%
  triton_mm_1990 1.9752 ms 79.9%
  mm 1.9762 ms 79.8%
  triton_mm_1988 1.9902 ms 79.3%
  triton_mm_1987 2.2238 ms 70.9%
  triton_mm_1986 8.8212 ms 17.9%
SingleProcess AUTOTUNE takes 4.0156 seconds
AUTOTUNE mm(3072x2560, 2560x768)
  triton_mm_1995 1.1150 ms 100.0%
  mm 1.6818 ms 66.3%
  triton_mm_1992 1.7130 ms 65.1%
  triton_mm_1993 1.9730 ms 56.5%
  triton_mm_1994 2.0284 ms 55.0%
  triton_mm_1991 9.9626 ms 11.2%
SingleProcess AUTOTUNE takes 3.9931 seconds
AUTOTUNE mm(2560x768, 768x768)
  triton_mm_1999 0.3783 ms 100.0%
  mm 0.4811 ms 78.6%
  triton_mm_2000 0.4932 ms 76.7%
  triton_mm_1998 0.4933 ms 76.7%
  triton_mm_1997 0.5483 ms 69.0%
  triton_mm_1996 2.2308 ms 17.0%
SingleProcess AUTOTUNE takes 3.9201 seconds
AUTOTUNE mm(768x2560, 2560x768)
  triton_mm_2005 0.3059 ms 100.0%
  mm 0.4296 ms 71.2%
  triton_mm_2002 0.4840 ms 63.2%
  triton_mm_2004 0.5016 ms 61.0%
  triton_mm_2003 0.5377 ms 56.9%
  triton_mm_2001 2.6881 ms 11.4%
SingleProcess AUTOTUNE takes 3.9240 seconds
AUTOTUNE bmm(240x128x128, 240x128x64)
  triton_bmm_2009 0.1052 ms 100.0%
  triton_bmm_2010 0.1081 ms 97.3%
  triton_bmm_2007 0.1094 ms 96.2%
  triton_bmm_2008 0.1150 ms 91.5%
  bmm 0.1172 ms 89.8%
  triton_bmm_2006 0.4669 ms 22.5%
SingleProcess AUTOTUNE takes 0.6570 seconds
AUTOTUNE bmm(240x128x64, 240x64x128)
  bmm 0.0969 ms 100.0%
  triton_bmm_2014 0.1552 ms 62.4%
  triton_bmm_2015 0.1787 ms 54.2%
  triton_bmm_2012 0.3057 ms 31.7%
  triton_bmm_2011 0.4533 ms 21.4%
  triton_bmm_2013 0.4802 ms 20.2%
SingleProcess AUTOTUNE takes 0.6804 seconds
AUTOTUNE bmm(240x64x128, 240x128x128)
  triton_bmm_2017 0.1065 ms 100.0%
  triton_bmm_2020 0.1078 ms 98.8%
  triton_bmm_2019 0.1090 ms 97.7%
  triton_bmm_2018 0.1124 ms 94.7%
  bmm 0.1158 ms 91.9%
  triton_bmm_2016 0.4640 ms 22.9%
SingleProcess AUTOTUNE takes 0.6479 seconds
AUTOTUNE bmm(240x128x128, 240x128x64)
  bmm 0.1014 ms 100.0%
  triton_bmm_2024 0.1791 ms 56.6%
  triton_bmm_2025 0.1920 ms 52.8%
  triton_bmm_2022 0.2985 ms 34.0%
  triton_bmm_2023 0.4758 ms 21.3%
  triton_bmm_2021 0.5408 ms 18.7%
SingleProcess AUTOTUNE takes 0.6750 seconds
```

## Text classification
As an alternative, we can use the script [`run_classification.py`](./run_classification.py) to fine-tune models on a single/multi-label classification task. 

The following example fine-tunes BERT on the `en` subset of  [`amazon_reviews_multi`](https://huggingface.co/datasets/amazon_reviews_multi) dataset.
We can specify the metric, the label column and aso choose which text columns to use jointly for classification. 
```bash
dataset="amazon_reviews_multi"
subset="en"
TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1  python run_classification.py \
    --model_name_or_path  bert-base-uncased \
    --dataset_name ${dataset} \
    --dataset_config_name ${subset} \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "review_title,review_body,product_category" \
    --text_column_delimiter "\n" \
    --label_column_name stars \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir /tmp/${dataset}_${subset}/ \
    --torch_compile True
```

Unfortunately an error is faced when running this regular and with PT2.0 will revisit
```
Generating train split:   0%|                                                                                                                                                                                                                           | 0/200000 [00:00<?, ? examples/s]
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/datasets/builder.py", line 1676, in _prepare_split_single
    for key, record in generator:
  File "/root/.cache/huggingface/modules/datasets_modules/datasets/amazon_reviews_multi/724e94f4b0c6c405ce7e476a6c5ef4f87db30799ad49f765094cf9770e0f7609/amazon_reviews_multi.py", line 130, in _generate_examples
    yield row_count, json.loads(line)
  File "/usr/lib/python3.10/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
  File "/usr/lib/python3.10/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/usr/lib/python3.10/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/var/lib/jenkins/models/pt20/transformers-2.0/examples/pytorch/text-classification/run_classification.py", line 757, in <module>
    main()
  File "/var/lib/jenkins/models/pt20/transformers-2.0/examples/pytorch/text-classification/run_classification.py", line 349, in main
    raw_datasets = load_dataset(
  File "/usr/local/lib/python3.10/dist-packages/datasets/load.py", line 2136, in load_dataset
    builder_instance.download_and_prepare(
  File "/usr/local/lib/python3.10/dist-packages/datasets/builder.py", line 954, in download_and_prepare
    self._download_and_prepare(
  File "/usr/local/lib/python3.10/dist-packages/datasets/builder.py", line 1717, in _download_and_prepare
    super()._download_and_prepare(
  File "/usr/local/lib/python3.10/dist-packages/datasets/builder.py", line 1049, in _download_and_prepare
    self._prepare_split(split_generator, **prepare_split_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/datasets/builder.py", line 1555, in _prepare_split
    for job_id, done, content in self._prepare_split_single(
  File "/usr/local/lib/python3.10/dist-packages/datasets/builder.py", line 1712, in _prepare_split_single
    raise DatasetGenerationError("An error occurred while generating the dataset") from e
datasets.builder.DatasetGenerationError: An error occurred while generating the dataset
```


## PyTorch version, no Trainer

Based on the script [`run_glue_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py).

Like `run_glue.py`, this script allows you to fine-tune any of the models on the [hub](https://huggingface.co/models) on a
text classification task, either a GLUE task or your own data in a csv or a JSON file. The main difference is that this
script exposes the bare training loop, to allow you to quickly experiment and add any customization you would like.

It offers less options than the script with `Trainer` (for instance you can easily change the options for the optimizer
or the dataloaders directly in the script) but still run in a distributed setup, on TPU and supports mixed precision by
the mean of the [🤗 `Accelerate`](https://github.com/huggingface/accelerate) library. You can use the script normally
after installing it:

```bash
pip install git+https://github.com/huggingface/accelerate
```

then

```bash
export TASK_NAME=mrpc

TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --torch_compile True
```

You can then use your usual launchers to run in it in a distributed environment, but the easiest way is to run

```bash
accelerate config
```

and reply to the questions asked. Then

```bash
accelerate test
```

that will check everything is ready for training. Finally, you can launch training with

```bash
export TASK_NAME=mrpc

TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 accelerate launch run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --torch_compile True
```

PT2.0 Timers not calculated due to some caching issue of autotune configs

Autotune logs:
```
(Pre accelerate)
AUTOTUNE addmm(2752x768, 2752x768, 768x768)
  addmm 0.5304 ms 100.0%
  bias_addmm 0.5487 ms 96.7%
  triton_mm_3 0.8756 ms 60.6%
  triton_mm_4 1.0675 ms 49.7%
  triton_mm_1 1.7160 ms 30.9%
  triton_mm_0 2.6592 ms 19.9%
  triton_mm_2 2.8492 ms 18.6%
SingleProcess AUTOTUNE takes 3.9542 seconds
AUTOTUNE bmm(384x86x64, 384x64x86)
  bmm 0.0932 ms 100.0%
  triton_bmm_19 0.0988 ms 94.4%
  triton_bmm_18 0.1072 ms 87.0%
  triton_bmm_17 0.1466 ms 63.6%
  triton_bmm_16 0.1553 ms 60.0%
  triton_bmm_15 0.5006 ms 18.6%
SingleProcess AUTOTUNE takes 3.7833 seconds
AUTOTUNE bmm(384x86x86, 384x86x64)
  bmm 0.1034 ms 100.0%
  triton_bmm_24 0.1047 ms 98.7%
  triton_bmm_22 0.1238 ms 83.5%
  triton_bmm_23 0.1346 ms 76.8%
  triton_bmm_21 0.1360 ms 76.0%
  triton_bmm_20 0.4922 ms 21.0%
SingleProcess AUTOTUNE takes 3.7567 seconds
AUTOTUNE addmm(2752x3072, 2752x768, 768x3072)
  addmm 2.0754 ms 100.0%
  bias_addmm 2.2236 ms 93.3%
  triton_mm_33 3.3893 ms 61.2%
  triton_mm_34 3.9692 ms 52.3%
  triton_mm_31 6.8794 ms 30.2%
  triton_mm_30 9.7692 ms 21.2%
  triton_mm_32 11.4725 ms 18.1%
SingleProcess AUTOTUNE takes 4.1017 seconds
AUTOTUNE addmm(2752x768, 2752x3072, 3072x768)
  addmm 2.0900 ms 100.0%
  bias_addmm 2.2437 ms 93.2%
  triton_mm_38 3.4835 ms 60.0%
  triton_mm_39 4.1481 ms 50.4%
  triton_mm_36 6.9181 ms 30.2%
  triton_mm_35 10.5666 ms 19.8%
  triton_mm_37 11.5132 ms 18.2%
SingleProcess AUTOTUNE takes 4.1802 seconds
AUTOTUNE addmm(32x768, 32x768, 768x768)
  addmm 0.0404 ms 100.0%
  bias_addmm 0.0405 ms 99.8%
  triton_mm_484 0.0545 ms 74.2%
  triton_mm_481 0.0663 ms 61.0%
  triton_mm_483 0.1009 ms 40.0%
  triton_mm_482 0.1640 ms 24.6%
  triton_mm_480 0.1707 ms 23.7%
SingleProcess AUTOTUNE takes 4.6453 seconds
AUTOTUNE addmm(32x2, 32x768, 768x2)
  addmm 0.0368 ms 100.0%
  bias_addmm 0.0369 ms 99.7%
  triton_mm_485 0.0379 ms 97.0%
  triton_mm_486 0.0394 ms 93.3%
  triton_mm_488 0.0394 ms 93.3%
  triton_mm_487 0.0398 ms 92.4%
SingleProcess AUTOTUNE takes 0.6373 seconds
AUTOTUNE mm(32x2, 2x768)
  triton_mm_491 0.0076 ms 100.0%
  triton_mm_489 0.0077 ms 99.5%
  mm 0.0077 ms 99.0%
  triton_mm_492 0.0078 ms 98.5%
  triton_mm_490 0.0078 ms 97.9%
SingleProcess AUTOTUNE takes 1.8386 seconds
AUTOTUNE mm(2x32, 32x768)
  triton_mm_493 0.0077 ms 100.0%
  triton_mm_495 0.0078 ms 98.5%
  triton_mm_494 0.0079 ms 98.0%
  triton_mm_496 0.0082 ms 93.7%
  mm 0.0084 ms 92.3%
  triton_mm_497 0.0085 ms 91.0%
SingleProcess AUTOTUNE takes 2.5368 seconds
AUTOTUNE mm(32x768, 768x768)
  triton_mm_499 0.0364 ms 100.0%
  triton_mm_500 0.0400 ms 91.1%
  triton_mm_502 0.0429 ms 84.8%
  triton_mm_501 0.0452 ms 80.5%
  mm 0.0466 ms 78.2%
  triton_mm_498 0.1434 ms 25.4%
SingleProcess AUTOTUNE takes 3.0549 seconds
AUTOTUNE mm(768x32, 32x768)
  triton_mm_507 0.0123 ms 100.0%
  mm 0.0132 ms 92.7%
  triton_mm_504 0.0144 ms 85.3%
  triton_mm_505 0.0150 ms 81.9%
  triton_mm_506 0.0182 ms 67.5%
  triton_mm_503 0.0191 ms 64.3%
SingleProcess AUTOTUNE takes 5.6175 seconds
AUTOTUNE mm(2752x768, 768x3072)
  triton_mm_511 1.4896 ms 100.0%
  mm 1.8995 ms 78.4%
  triton_mm_512 2.0055 ms 74.3%
  triton_mm_510 2.0859 ms 71.4%
  triton_mm_509 2.3725 ms 62.8%
  triton_mm_508 7.4801 ms 19.9%
SingleProcess AUTOTUNE takes 5.3799 seconds
AUTOTUNE mm(768x2752, 2752x3072)
  triton_mm_517 1.1852 ms 100.0%
  mm 1.6601 ms 71.4%
  triton_mm_516 1.7724 ms 66.9%
  triton_mm_514 1.8529 ms 64.0%
  triton_mm_515 2.1698 ms 54.6%
  triton_mm_513 10.8042 ms 11.0%
SingleProcess AUTOTUNE takes 3.9604 seconds
AUTOTUNE mm(2752x3072, 3072x768)
  triton_mm_521 1.6926 ms 100.0%
  mm 1.9595 ms 86.4%
  triton_mm_522 2.0801 ms 81.4%
  triton_mm_520 2.1109 ms 80.2%
  triton_mm_519 2.3968 ms 70.6%
  triton_mm_518 8.5072 ms 19.9%
SingleProcess AUTOTUNE takes 3.8530 seconds
AUTOTUNE mm(3072x2752, 2752x768)
  triton_mm_527 1.1856 ms 100.0%
  mm 1.6656 ms 71.2%
  triton_mm_526 1.7826 ms 66.5%
  triton_mm_524 1.8694 ms 63.4%
  triton_mm_525 2.1833 ms 54.3%
  triton_mm_523 10.9851 ms 10.8%
SingleProcess AUTOTUNE takes 3.9230 seconds
AUTOTUNE mm(2752x768, 768x768)
  triton_mm_531 0.4068 ms 100.0%
  mm 0.4773 ms 85.2%
  triton_mm_532 0.5124 ms 79.4%
  triton_mm_530 0.5239 ms 77.7%
  triton_mm_529 0.6054 ms 67.2%
  triton_mm_528 2.1360 ms 19.0%
SingleProcess AUTOTUNE takes 3.7780 seconds
AUTOTUNE mm(768x2752, 2752x768)
  triton_mm_537 0.3185 ms 100.0%
  mm 0.4322 ms 73.7%
  triton_mm_536 0.4890 ms 65.1%
  triton_mm_534 0.5118 ms 62.2%
  triton_mm_535 0.5911 ms 53.9%
  triton_mm_533 2.8652 ms 11.1%
SingleProcess AUTOTUNE takes 3.8103 seconds
AUTOTUNE bmm(384x86x86, 384x86x64)
  bmm 0.0926 ms 100.0%
  triton_bmm_542 0.0993 ms 93.2%
  triton_bmm_541 0.1096 ms 84.5%
  triton_bmm_540 0.1168 ms 79.3%
  triton_bmm_539 0.1214 ms 76.3%
  triton_bmm_538 0.5880 ms 15.8%
SingleProcess AUTOTUNE takes 4.3727 seconds
AUTOTUNE bmm(384x86x64, 384x64x86)
  bmm 0.0859 ms 100.0%
  triton_bmm_547 0.1563 ms 55.0%
  triton_bmm_546 0.2333 ms 36.8%
  triton_bmm_544 0.3714 ms 23.1%
  triton_bmm_545 0.5904 ms 14.5%
  triton_bmm_543 0.6718 ms 12.8%
SingleProcess AUTOTUNE takes 3.6713 seconds
AUTOTUNE bmm(384x64x86, 384x86x86)
  bmm 0.0932 ms 100.0%
  triton_bmm_552 0.0989 ms 94.3%
  triton_bmm_549 0.1056 ms 88.2%
  triton_bmm_551 0.1185 ms 78.7%
  triton_bmm_550 0.1283 ms 72.6%
  triton_bmm_548 0.5882 ms 15.8%
SingleProcess AUTOTUNE takes 3.8384 seconds
AUTOTUNE bmm(384x86x86, 384x86x64)
  bmm 0.1030 ms 100.0%
  triton_bmm_557 0.1553 ms 66.3%
  triton_bmm_556 0.2024 ms 50.9%
  triton_bmm_554 0.3568 ms 28.9%
  triton_bmm_555 0.4355 ms 23.6%
  triton_bmm_553 0.5471 ms 18.8%
AUTOTUNE addmm(2560x768, 2560x768, 768x768)
  addmm 0.5041 ms 100.0%
  bias_addmm 0.5207 ms 96.8%
  triton_mm_1471 0.8247 ms 61.1%
  triton_mm_1472 0.9891 ms 51.0%
  triton_mm_1469 1.6852 ms 29.9%
  triton_mm_1468 2.5595 ms 19.7%
  triton_mm_1470 2.7922 ms 18.1%
SingleProcess AUTOTUNE takes 3.9124 seconds
AUTOTUNE bmm(384x80x64, 384x64x80)
  bmm 0.0869 ms 100.0%
  triton_bmm_1487 0.0880 ms 98.8%
  triton_bmm_1486 0.1058 ms 82.2%
  triton_bmm_1485 0.1656 ms 52.5%
  triton_bmm_1484 0.1722 ms 50.5%
  triton_bmm_1483 0.5077 ms 17.1%
SingleProcess AUTOTUNE takes 3.7350 seconds
AUTOTUNE bmm(384x80x80, 384x80x64)
  bmm 0.0968 ms 100.0%
  triton_bmm_1492 0.0984 ms 98.3%
  triton_bmm_1490 0.1042 ms 92.9%
  triton_bmm_1491 0.1143 ms 84.6%
  triton_bmm_1489 0.1389 ms 69.7%
  triton_bmm_1488 0.4855 ms 19.9%
SingleProcess AUTOTUNE takes 3.7711 seconds
AUTOTUNE addmm(2560x3072, 2560x768, 768x3072)
  addmm 1.9476 ms 100.0%
  bias_addmm 2.0734 ms 93.9%
  triton_mm_1501 3.1454 ms 61.9%
  triton_mm_1502 3.6898 ms 52.8%
  triton_mm_1499 6.3936 ms 30.5%
  triton_mm_1498 8.9458 ms 21.8%
  triton_mm_1500 10.6706 ms 18.3%
SingleProcess AUTOTUNE takes 4.0932 seconds
AUTOTUNE addmm(2560x768, 2560x3072, 3072x768)
  addmm 1.9631 ms 100.0%
  bias_addmm 2.1002 ms 93.5%
  triton_mm_1506 3.1925 ms 61.5%
  triton_mm_1507 3.9035 ms 50.3%
  triton_mm_1504 6.5926 ms 29.8%
  triton_mm_1503 10.0858 ms 19.5%
  triton_mm_1505 10.9874 ms 17.9%
SingleProcess AUTOTUNE takes 4.0815 seconds
AUTOTUNE addmm(32x768, 32x768, 768x768)
  addmm 0.0408 ms 100.0%
  bias_addmm 0.0411 ms 99.4%
  triton_mm_1952 0.0540 ms 75.6%
  triton_mm_1949 0.0651 ms 62.7%
  triton_mm_1951 0.1010 ms 40.4%
  triton_mm_1950 0.1628 ms 25.1%
  triton_mm_1948 0.1700 ms 24.0%
SingleProcess AUTOTUNE takes 3.1740 seconds
AUTOTUNE mm(768x32, 32x768)
  triton_mm_1975 0.0125 ms 100.0%
  triton_mm_1972 0.0141 ms 88.9%
  mm 0.0149 ms 84.3%
  triton_mm_1973 0.0150 ms 83.5%
  triton_mm_1974 0.0177 ms 70.8%
  triton_mm_1971 0.0199 ms 63.0%
SingleProcess AUTOTUNE takes 3.2358 seconds
AUTOTUNE mm(2560x768, 768x3072)
  triton_mm_1979 1.4374 ms 100.0%
  mm 1.7729 ms 81.1%
  triton_mm_1980 1.8742 ms 76.7%
  triton_mm_1978 1.9437 ms 74.0%
  triton_mm_1977 2.1953 ms 65.5%
  triton_mm_1976 6.7717 ms 21.2%
SingleProcess AUTOTUNE takes 3.8891 seconds
AUTOTUNE mm(768x2560, 2560x3072)
  triton_mm_1985 1.1063 ms 100.0%
  mm 1.5513 ms 71.3%
  triton_mm_1982 1.7259 ms 64.1%
  triton_mm_1983 1.9941 ms 55.5%
  triton_mm_1984 2.1435 ms 51.6%
  triton_mm_1981 9.9962 ms 11.1%
SingleProcess AUTOTUNE takes 3.9582 seconds
AUTOTUNE mm(2560x3072, 3072x768)
  triton_mm_1989 1.6284 ms 100.0%
  mm 1.8206 ms 89.4%
  triton_mm_1990 1.9242 ms 84.6%
  triton_mm_1988 2.0159 ms 80.8%
  triton_mm_1987 2.2619 ms 72.0%
  triton_mm_1986 7.9718 ms 20.4%
SingleProcess AUTOTUNE takes 3.8771 seconds
AUTOTUNE mm(3072x2560, 2560x768)
  triton_mm_1995 1.1015 ms 100.0%
  mm 1.5562 ms 70.8%
  triton_mm_1992 1.7345 ms 63.5%
  triton_mm_1993 2.0051 ms 54.9%
  triton_mm_1994 2.1439 ms 51.4%
  triton_mm_1991 10.0541 ms 11.0%
SingleProcess AUTOTUNE takes 3.8823 seconds
AUTOTUNE mm(2560x768, 768x768)
  triton_mm_1999 0.3963 ms 100.0%
  mm 0.4447 ms 89.1%
  triton_mm_2000 0.4776 ms 83.0%
  triton_mm_1998 0.4997 ms 79.3%
  triton_mm_1997 0.5747 ms 69.0%
  triton_mm_1996 2.0096 ms 19.7%
SingleProcess AUTOTUNE takes 3.7468 seconds
AUTOTUNE mm(768x2560, 2560x768)
  triton_mm_2005 0.2991 ms 100.0%
  mm 0.4050 ms 73.9%
  triton_mm_2002 0.4912 ms 60.9%
  triton_mm_2004 0.5231 ms 57.2%
  triton_mm_2003 0.5424 ms 55.1%
  triton_mm_2001 2.6164 ms 11.4%
SingleProcess AUTOTUNE takes 3.8318 seconds
AUTOTUNE bmm(384x80x80, 384x80x64)
  bmm 0.0892 ms 100.0%
  triton_bmm_2010 0.0925 ms 96.4%
  triton_bmm_2009 0.0936 ms 95.2%
  triton_bmm_2008 0.1028 ms 86.8%
  triton_bmm_2007 0.1178 ms 75.7%
  triton_bmm_2006 0.5210 ms 17.1%
SingleProcess AUTOTUNE takes 3.6655 seconds
AUTOTUNE bmm(384x80x64, 384x64x80)
  bmm 0.0854 ms 100.0%
  triton_bmm_2015 0.1551 ms 55.1%
  triton_bmm_2014 0.2283 ms 37.4%
  triton_bmm_2012 0.3686 ms 23.2%
  triton_bmm_2013 0.5879 ms 14.5%
  triton_bmm_2011 0.6707 ms 12.7%
SingleProcess AUTOTUNE takes 3.6030 seconds
AUTOTUNE bmm(384x64x80, 384x80x80)
  bmm 0.0886 ms 100.0%
  triton_bmm_2020 0.0922 ms 96.1%
  triton_bmm_2019 0.0967 ms 91.6%
  triton_bmm_2017 0.0980 ms 90.4%
  triton_bmm_2018 0.1311 ms 67.6%
  triton_bmm_2016 0.5245 ms 16.9%
SingleProcess AUTOTUNE takes 3.7211 seconds
AUTOTUNE bmm(384x80x80, 384x80x64)
  bmm 0.0975 ms 100.0%
  triton_bmm_2025 0.1337 ms 73.0%
  triton_bmm_2024 0.1788 ms 54.5%
  triton_bmm_2022 0.3715 ms 26.2%
  triton_bmm_2023 0.4378 ms 22.3%
  triton_bmm_2021 0.5348 ms 18.2%
SingleProcess AUTOTUNE takes 3.8344 seconds
AUTOTUNE addmm(1700x768, 1700x768, 768x768)
  addmm 0.3461 ms 100.0%
  bias_addmm 0.3519 ms 98.4%
  triton_mm_2940 0.7154 ms 48.4%
  triton_mm_2939 0.7308 ms 47.4%
  triton_mm_2937 1.1358 ms 30.5%
  triton_mm_2936 1.7492 ms 19.8%
  triton_mm_2938 1.8716 ms 18.5%
SingleProcess AUTOTUNE takes 3.9664 seconds
AUTOTUNE bmm(240x85x64, 240x64x85)
  bmm 0.0551 ms 100.0%
  triton_bmm_2955 0.0582 ms 94.6%
  triton_bmm_2954 0.0698 ms 78.9%
  triton_bmm_2952 0.1044 ms 52.8%
  triton_bmm_2953 0.1057 ms 52.1%
  triton_bmm_2951 0.3548 ms 15.5%
SingleProcess AUTOTUNE takes 3.7867 seconds
AUTOTUNE bmm(240x85x85, 240x85x64)
  triton_bmm_2960 0.0634 ms 100.0%
  bmm 0.0642 ms 98.7%
  triton_bmm_2958 0.0676 ms 93.8%
  triton_bmm_2957 0.0894 ms 70.9%
  triton_bmm_2959 0.0939 ms 67.5%
  triton_bmm_2956 0.3601 ms 17.6%
SingleProcess AUTOTUNE takes 3.7768 seconds
AUTOTUNE addmm(1700x3072, 1700x768, 768x3072)
  addmm 1.3261 ms 100.0%
  bias_addmm 1.3902 ms 95.4%
  triton_mm_2969 2.1561 ms 61.5%
  triton_mm_2970 2.5076 ms 52.9%
  triton_mm_2967 4.3563 ms 30.4%
  triton_mm_2966 6.0258 ms 22.0%
  triton_mm_2968 7.2576 ms 18.3%
SingleProcess AUTOTUNE takes 4.0150 seconds
AUTOTUNE addmm(1700x768, 1700x3072, 3072x768)
  addmm 1.3453 ms 100.0%
  bias_addmm 1.4357 ms 93.7%
  triton_mm_2974 2.7409 ms 49.1%
  triton_mm_2975 2.7835 ms 48.3%
  triton_mm_2972 4.4662 ms 30.1%
  triton_mm_2971 6.8059 ms 19.8%
  triton_mm_2973 7.4317 ms 18.1%
SingleProcess AUTOTUNE takes 4.0760 seconds
AUTOTUNE addmm(20x768, 20x768, 768x768)
  addmm 0.0402 ms 100.0%
  bias_addmm 0.0404 ms 99.6%
  triton_mm_3420 0.0525 ms 76.6%
  triton_mm_3417 0.0641 ms 62.8%
  triton_mm_3419 0.1012 ms 39.8%
  triton_mm_3418 0.1612 ms 25.0%
  triton_mm_3416 0.1706 ms 23.6%
SingleProcess AUTOTUNE takes 3.1894 seconds
AUTOTUNE addmm(20x2, 20x768, 768x2)
  addmm 0.0369 ms 100.0%
  bias_addmm 0.0372 ms 99.3%
  triton_mm_3421 0.0380 ms 97.0%
  triton_mm_3423 0.0381 ms 96.8%
  triton_mm_3424 0.0394 ms 93.6%
  triton_mm_3422 0.0395 ms 93.4%
SingleProcess AUTOTUNE takes 2.4343 seconds
AUTOTUNE mm(20x2, 2x768)
  triton_mm_3426 0.0076 ms 100.0%
  triton_mm_3425 0.0077 ms 99.0%
  mm 0.0078 ms 98.2%
  triton_mm_3428 0.0080 ms 96.0%
  triton_mm_3427 0.0080 ms 95.0%
SingleProcess AUTOTUNE takes 1.7855 seconds
AUTOTUNE mm(2x20, 20x768)
  triton_mm_3433 0.0080 ms 100.0%
  mm 0.0082 ms 97.6%
  triton_mm_3432 0.0086 ms 93.1%
  triton_mm_3429 0.0094 ms 85.2%
  triton_mm_3430 0.0106 ms 75.6%
  triton_mm_3431 0.0116 ms 69.1%
SingleProcess AUTOTUNE takes 2.9802 seconds
AUTOTUNE mm(20x768, 768x768)
  triton_mm_3435 0.0361 ms 100.0%
  triton_mm_3436 0.0397 ms 90.9%
  triton_mm_3438 0.0406 ms 89.0%
  triton_mm_3437 0.0433 ms 83.4%
  mm 0.0458 ms 78.8%
  triton_mm_3434 0.1386 ms 26.1%
SingleProcess AUTOTUNE takes 3.0798 seconds
AUTOTUNE mm(768x20, 20x768)
  mm 0.0116 ms 100.0%
  triton_mm_3443 0.0124 ms 93.2%
  triton_mm_3442 0.0159 ms 72.8%
  triton_mm_3440 0.0190 ms 60.7%
  triton_mm_3441 0.0204 ms 56.8%
  triton_mm_3439 0.0637 ms 18.1%
SingleProcess AUTOTUNE takes 3.6691 seconds
AUTOTUNE mm(1700x768, 768x3072)
  triton_mm_3447 0.9775 ms 100.0%
  mm 1.1897 ms 82.2%
  triton_mm_3448 1.2529 ms 78.0%
  triton_mm_3446 1.3190 ms 74.1%
  triton_mm_3445 1.4820 ms 66.0%
  triton_mm_3444 4.7676 ms 20.5%
SingleProcess AUTOTUNE takes 4.2054 seconds
AUTOTUNE mm(768x1700, 1700x3072)
  triton_mm_3453 0.7393 ms 100.0%
  mm 1.0028 ms 73.7%
  triton_mm_3450 1.1501 ms 64.3%
  triton_mm_3452 1.1572 ms 63.9%
  triton_mm_3451 1.3636 ms 54.2%
  triton_mm_3449 7.8281 ms 9.4%
SingleProcess AUTOTUNE takes 3.8403 seconds
AUTOTUNE mm(1700x3072, 3072x768)
  triton_mm_3457 1.1933 ms 100.0%
  mm 1.2275 ms 97.2%
  triton_mm_3456 1.3324 ms 89.6%
  triton_mm_3458 1.3462 ms 88.6%
  triton_mm_3455 1.5067 ms 79.2%
  triton_mm_3454 5.6263 ms 21.2%
SingleProcess AUTOTUNE takes 3.8738 seconds
AUTOTUNE mm(3072x1700, 1700x768)
  triton_mm_3463 0.7383 ms 100.0%
  mm 1.0300 ms 71.7%
  triton_mm_3462 1.1573 ms 63.8%
  triton_mm_3460 1.1600 ms 63.6%
  triton_mm_3461 1.3746 ms 53.7%
  triton_mm_3459 7.8416 ms 9.4%
SingleProcess AUTOTUNE takes 3.8033 seconds
AUTOTUNE mm(1700x768, 768x768)
  triton_mm_3467 0.2896 ms 100.0%
  mm 0.3000 ms 96.5%
  triton_mm_3466 0.3311 ms 87.5%
  triton_mm_3468 0.3339 ms 86.7%
  triton_mm_3465 0.3812 ms 76.0%
  triton_mm_3464 1.4435 ms 20.1%
SingleProcess AUTOTUNE takes 3.8558 seconds
AUTOTUNE mm(768x1700, 1700x768)
  triton_mm_3473 0.1956 ms 100.0%
  mm 0.2563 ms 76.3%
  triton_mm_3472 0.2752 ms 71.1%
  triton_mm_3470 0.3256 ms 60.1%
  triton_mm_3471 0.3700 ms 52.9%
  triton_mm_3469 2.2809 ms 8.6%
SingleProcess AUTOTUNE takes 3.7120 seconds
AUTOTUNE bmm(240x85x85, 240x85x64)
  bmm 0.0571 ms 100.0%
  triton_bmm_3478 0.0592 ms 96.4%
  triton_bmm_3476 0.0679 ms 84.1%
  triton_bmm_3477 0.0755 ms 75.6%
  triton_bmm_3475 0.0849 ms 67.2%
  triton_bmm_3474 0.4368 ms 13.1%
SingleProcess AUTOTUNE takes 3.8186 seconds
AUTOTUNE bmm(240x85x64, 240x64x85)
  bmm 0.0571 ms 100.0%
  triton_bmm_3483 0.1070 ms 53.3%
  triton_bmm_3482 0.1509 ms 37.8%
  triton_bmm_3480 0.2406 ms 23.7%
  triton_bmm_3481 0.3725 ms 15.3%
  triton_bmm_3479 0.4416 ms 12.9%
SingleProcess AUTOTUNE takes 3.6911 seconds
AUTOTUNE bmm(240x64x85, 240x85x85)
  bmm 0.0591 ms 100.0%
  triton_bmm_3488 0.0610 ms 96.9%
  triton_bmm_3485 0.0661 ms 89.4%
  triton_bmm_3487 0.0878 ms 67.3%
  triton_bmm_3486 0.0895 ms 66.0%
  triton_bmm_3484 0.4513 ms 13.1%
SingleProcess AUTOTUNE takes 3.9413 seconds
AUTOTUNE bmm(240x85x85, 240x85x64)
  bmm 0.0692 ms 100.0%
  triton_bmm_3493 0.1057 ms 65.4%
  triton_bmm_3492 0.1330 ms 52.0%
  triton_bmm_3490 0.2359 ms 29.3%
  triton_bmm_3491 0.2893 ms 23.9%
  triton_bmm_3489 0.4110 ms 16.8%
SingleProcess AUTOTUNE takes 3.8928 seconds
AUTOTUNE addmm(640x768, 640x768, 768x768)
  addmm 0.1372 ms 100.0%
  bias_addmm 0.1380 ms 99.4%
  triton_mm_4407 0.2582 ms 53.1%
  triton_mm_4408 0.2723 ms 50.4%
  triton_mm_4405 0.4827 ms 28.4%
  triton_mm_4406 0.8019 ms 17.1%
  triton_mm_4404 0.8892 ms 15.4%
SingleProcess AUTOTUNE takes 3.9074 seconds
AUTOTUNE bmm(96x80x64, 96x64x80)
  bmm 0.0266 ms 100.0%
  triton_bmm_4418 0.0299 ms 88.9%
  triton_bmm_4417 0.0372 ms 71.4%
  triton_bmm_4416 0.0448 ms 59.3%
  triton_bmm_4415 0.0467 ms 56.9%
  triton_bmm_4414 0.1688 ms 15.7%
SingleProcess AUTOTUNE takes 0.6307 seconds
AUTOTUNE bmm(96x80x80, 96x80x64)
  bmm 0.0265 ms 100.0%
  triton_bmm_4428 0.0273 ms 97.1%
  triton_bmm_4426 0.0348 ms 76.3%
  triton_bmm_4427 0.0383 ms 69.3%
  triton_bmm_4425 0.0435 ms 61.0%
  triton_bmm_4424 0.1623 ms 16.4%
SingleProcess AUTOTUNE takes 0.6279 seconds
AUTOTUNE addmm(640x3072, 640x768, 768x3072)
  addmm 0.4978 ms 100.0%
  bias_addmm 0.5246 ms 94.9%
  triton_mm_4437 0.8238 ms 60.4%
  triton_mm_4438 1.0012 ms 49.7%
  triton_mm_4435 1.6822 ms 29.6%
  triton_mm_4434 2.5653 ms 19.4%
  triton_mm_4436 2.7878 ms 17.9%
SingleProcess AUTOTUNE takes 3.9604 seconds
AUTOTUNE addmm(640x768, 640x3072, 3072x768)
  addmm 0.5131 ms 100.0%
  bias_addmm 0.5431 ms 94.5%
  triton_mm_4442 0.9912 ms 51.8%
  triton_mm_4443 1.0578 ms 48.5%
  triton_mm_4440 1.8810 ms 27.3%
  triton_mm_4441 3.1210 ms 16.4%
  triton_mm_4439 3.4133 ms 15.0%
SingleProcess AUTOTUNE takes 3.9668 seconds
AUTOTUNE addmm(8x768, 8x768, 768x768)
  addmm 0.0392 ms 100.0%
  bias_addmm 0.0397 ms 98.9%
  triton_mm_4885 0.0466 ms 84.2%
  triton_mm_4888 0.0501 ms 78.3%
  triton_mm_4887 0.0954 ms 41.1%
  triton_mm_4884 0.0963 ms 40.7%
  triton_mm_4886 0.0975 ms 40.3%
SingleProcess AUTOTUNE takes 3.0743 seconds
AUTOTUNE addmm(8x2, 8x768, 768x2)
  triton_mm_4892 0.0320 ms 100.0%
  triton_mm_4891 0.0331 ms 96.6%
  bias_addmm 0.0366 ms 87.3%
  addmm 0.0366 ms 87.3%
  triton_mm_4889 0.0429 ms 74.5%
  triton_mm_4890 0.0430 ms 74.4%

Post accelerate
AUTOTUNE addmm(2496x768, 2496x768, 768x768)
  addmm 0.4838 ms 100.0%
  bias_addmm 0.5005 ms 96.7%
  triton_mm_3 0.7826 ms 61.8%
  triton_mm_4 0.9711 ms 49.8%
  triton_mm_1 1.5461 ms 31.3%
  triton_mm_2 2.5546 ms 18.9%
  triton_mm_0 2.6297 ms 18.4%
SingleProcess AUTOTUNE takes 3.8965 seconds
AUTOTUNE bmm(384x78x64, 384x64x78)
  bmm 0.0775 ms 100.0%
  triton_bmm_19 0.0858 ms 90.3%
  triton_bmm_18 0.1027 ms 75.5%
  triton_bmm_17 0.1441 ms 53.8%
  triton_bmm_16 0.1520 ms 51.0%
  triton_bmm_15 0.4983 ms 15.5%
SingleProcess AUTOTUNE takes 3.8425 seconds
AUTOTUNE bmm(384x78x78, 384x78x64)
  triton_bmm_24 0.0892 ms 100.0%
  bmm 0.0966 ms 92.3%
  triton_bmm_22 0.1107 ms 80.6%
  triton_bmm_23 0.1145 ms 77.9%
  triton_bmm_21 0.1320 ms 67.6%
  triton_bmm_20 0.4608 ms 19.4%
SingleProcess AUTOTUNE takes 3.7645 seconds
AUTOTUNE addmm(2496x3072, 2496x768, 768x3072)
  addmm 1.8828 ms 100.0%
  bias_addmm 2.0088 ms 93.7%
  triton_mm_33 3.0652 ms 61.4%
  triton_mm_34 3.5890 ms 52.5%
  triton_mm_31 6.2501 ms 30.1%
  triton_mm_30 8.8833 ms 21.2%
  triton_mm_32 10.4176 ms 18.1%
SingleProcess AUTOTUNE takes 4.0842 seconds
AUTOTUNE addmm(2496x768, 2496x3072, 3072x768)
  addmm 1.8993 ms 100.0%
  bias_addmm 2.0463 ms 92.8%
  triton_mm_38 3.1116 ms 61.0%
  triton_mm_39 3.8173 ms 49.8%
  triton_mm_36 6.1725 ms 30.8%
  triton_mm_37 10.2533 ms 18.5%
  triton_mm_35 10.4209 ms 18.2%
SingleProcess AUTOTUNE takes 4.1389 seconds
AUTOTUNE addmm(32x768, 32x768, 768x768)
  addmm 0.0402 ms 100.0%
  bias_addmm 0.0403 ms 99.9%
  triton_mm_484 0.0556 ms 72.3%
  triton_mm_481 0.0631 ms 63.8%
  triton_mm_483 0.1004 ms 40.1%
  triton_mm_482 0.1628 ms 24.7%
  triton_mm_480 0.1703 ms 23.6%
SingleProcess AUTOTUNE takes 3.2696 seconds
AUTOTUNE addmm(32x2, 32x768, 768x2)
  addmm 0.0370 ms 100.0%
  bias_addmm 0.0370 ms 99.9%
  triton_mm_485 0.0381 ms 97.2%
  triton_mm_487 0.0389 ms 95.1%
  triton_mm_488 0.0394 ms 94.0%
  triton_mm_486 0.0394 ms 93.9%
SingleProcess AUTOTUNE takes 0.8188 seconds
AUTOTUNE mm(32x2, 2x768)
  mm 0.0074 ms 100.0%
  triton_mm_491 0.0077 ms 95.8%
  triton_mm_490 0.0077 ms 95.3%
  triton_mm_492 0.0078 ms 94.8%
  triton_mm_489 0.0080 ms 92.5%
SingleProcess AUTOTUNE takes 1.8001 seconds
AUTOTUNE mm(2x32, 32x768)
  triton_mm_493 0.0078 ms 100.0%
  triton_mm_494 0.0079 ms 99.0%
  triton_mm_495 0.0079 ms 98.5%
  triton_mm_496 0.0082 ms 94.7%
  triton_mm_497 0.0082 ms 94.7%
  mm 0.0084 ms 92.9%
SingleProcess AUTOTUNE takes 2.5388 seconds
AUTOTUNE mm(32x768, 768x768)
  triton_mm_499 0.0364 ms 100.0%
  triton_mm_500 0.0398 ms 91.6%
  triton_mm_502 0.0426 ms 85.6%
  triton_mm_501 0.0452 ms 80.7%
  mm 0.0462 ms 78.8%
  triton_mm_498 0.1430 ms 25.5%
SingleProcess AUTOTUNE takes 3.0256 seconds
AUTOTUNE mm(768x32, 32x768)
  triton_mm_507 0.0123 ms 100.0%
  mm 0.0130 ms 94.5%
  triton_mm_504 0.0143 ms 85.8%
  triton_mm_505 0.0150 ms 82.1%
  triton_mm_506 0.0183 ms 67.0%
  triton_mm_503 0.0200 ms 61.3%
SingleProcess AUTOTUNE takes 3.1174 seconds
AUTOTUNE mm(2496x768, 768x3072)
  triton_mm_511 1.3907 ms 100.0%
  mm 1.7187 ms 80.9%
  triton_mm_512 1.8205 ms 76.4%
  triton_mm_510 1.8968 ms 73.3%
  triton_mm_509 2.1301 ms 65.3%
  triton_mm_508 6.8307 ms 20.4%
SingleProcess AUTOTUNE takes 3.8296 seconds
AUTOTUNE mm(768x2496, 2496x3072)
  triton_mm_517 1.0796 ms 100.0%
  mm 1.4900 ms 72.5%
  triton_mm_514 1.6802 ms 64.3%
  triton_mm_516 1.6911 ms 63.8%
  triton_mm_515 1.9614 ms 55.0%
  triton_mm_513 9.8444 ms 11.0%
SingleProcess AUTOTUNE takes 4.0501 seconds
AUTOTUNE mm(2496x3072, 3072x768)
  triton_mm_521 1.4700 ms 100.0%
  mm 1.7700 ms 83.1%
  triton_mm_520 1.9037 ms 77.2%
  triton_mm_522 1.9097 ms 77.0%
  triton_mm_519 2.1694 ms 67.8%
  triton_mm_518 8.2819 ms 17.8%
SingleProcess AUTOTUNE takes 4.2206 seconds
AUTOTUNE mm(3072x2496, 2496x768)
  triton_mm_527 1.0766 ms 100.0%
  mm 1.4977 ms 71.9%
  triton_mm_524 1.6919 ms 63.6%
  triton_mm_526 1.6925 ms 63.6%
  triton_mm_525 1.9784 ms 54.4%
  triton_mm_523 9.9877 ms 10.8%
SingleProcess AUTOTUNE takes 3.9024 seconds
AUTOTUNE mm(2496x768, 768x768)
  triton_mm_531 0.3605 ms 100.0%
  mm 0.4320 ms 83.5%
  triton_mm_532 0.4711 ms 76.5%
  triton_mm_530 0.4758 ms 75.8%
  triton_mm_529 0.5428 ms 66.4%
  triton_mm_528 2.1116 ms 17.1%
SingleProcess AUTOTUNE takes 3.7505 seconds
AUTOTUNE mm(768x2496, 2496x768)
  triton_mm_537 0.2886 ms 100.0%
  mm 0.3902 ms 74.0%
  triton_mm_536 0.4480 ms 64.4%
  triton_mm_534 0.4648 ms 62.1%
  triton_mm_535 0.5279 ms 54.7%
  triton_mm_533 2.6048 ms 11.1%
SingleProcess AUTOTUNE takes 3.7732 seconds
AUTOTUNE bmm(384x78x78, 384x78x64)
  bmm 0.0810 ms 100.0%
  triton_bmm_542 0.0858 ms 94.5%
  triton_bmm_541 0.0937 ms 86.5%
  triton_bmm_540 0.1077 ms 75.3%
  triton_bmm_539 0.1193 ms 67.9%
  triton_bmm_538 0.5688 ms 14.2%
SingleProcess AUTOTUNE takes 3.7862 seconds
AUTOTUNE bmm(384x78x64, 384x64x78)
  bmm 0.0851 ms 100.0%
  triton_bmm_547 0.1554 ms 54.7%
  triton_bmm_546 0.2319 ms 36.7%
  triton_bmm_544 0.3687 ms 23.1%
  triton_bmm_545 0.5888 ms 14.4%
  triton_bmm_543 0.6694 ms 12.7%
SingleProcess AUTOTUNE takes 3.6105 seconds
AUTOTUNE bmm(384x64x78, 384x78x78)
  bmm 0.0821 ms 100.0%
  triton_bmm_552 0.0857 ms 95.8%
  triton_bmm_551 0.1039 ms 79.0%
  triton_bmm_549 0.1043 ms 78.7%
  triton_bmm_550 0.1256 ms 65.3%
  triton_bmm_548 0.5683 ms 14.4%
SingleProcess AUTOTUNE takes 3.8335 seconds
AUTOTUNE bmm(384x78x78, 384x78x64)
  bmm 0.0922 ms 100.0%
  triton_bmm_557 0.1326 ms 69.5%
  triton_bmm_556 0.1719 ms 53.6%
  triton_bmm_554 0.3542 ms 26.0%
  triton_bmm_555 0.4324 ms 21.3%
  triton_bmm_553 0.5339 ms 17.3%
SingleProcess AUTOTUNE takes 3.8639 seconds
AUTOTUNE addmm(2624x768, 2624x768, 768x768)
  addmm 0.5139 ms 100.0%
  bias_addmm 0.5313 ms 96.7%
  triton_mm_1471 0.8924 ms 57.6%
  triton_mm_1472 1.0187 ms 50.4%
  triton_mm_1469 1.6866 ms 30.5%
  triton_mm_1468 2.5634 ms 20.0%
  triton_mm_1470 2.7973 ms 18.4%
SingleProcess AUTOTUNE takes 3.8767 seconds
AUTOTUNE bmm(384x82x64, 384x64x82)
  bmm 0.0868 ms 100.0%
  triton_bmm_1487 0.0896 ms 97.0%
  triton_bmm_1486 0.1044 ms 83.2%
  triton_bmm_1484 0.1574 ms 55.2%
  triton_bmm_1485 0.1594 ms 54.5%
  triton_bmm_1483 0.5258 ms 16.5%
SingleProcess AUTOTUNE takes 3.7760 seconds
AUTOTUNE bmm(384x82x82, 384x82x64)
  triton_bmm_1492 0.0963 ms 100.0%
  bmm 0.0966 ms 99.7%
  triton_bmm_1490 0.1037 ms 92.9%
  triton_bmm_1489 0.1323 ms 72.8%
  triton_bmm_1491 0.1326 ms 72.6%
  triton_bmm_1488 0.4837 ms 19.9%
SingleProcess AUTOTUNE takes 3.8334 seconds
AUTOTUNE addmm(2624x3072, 2624x768, 768x3072)
  addmm 1.9979 ms 100.0%
  bias_addmm 2.1284 ms 93.9%
  triton_mm_1501 3.2394 ms 61.7%
  triton_mm_1502 3.7993 ms 52.6%
  triton_mm_1499 6.5649 ms 30.4%
  triton_mm_1498 9.1627 ms 21.8%
  triton_mm_1500 10.9417 ms 18.3%
SingleProcess AUTOTUNE takes 4.0916 seconds
AUTOTUNE addmm(2624x768, 2624x3072, 3072x768)
  addmm 2.0049 ms 100.0%
  bias_addmm 2.1625 ms 92.7%
  triton_mm_1506 3.4702 ms 57.8%
  triton_mm_1507 3.9846 ms 50.3%
  triton_mm_1504 6.6163 ms 30.3%
  triton_mm_1503 10.1653 ms 19.7%
  triton_mm_1505 11.0149 ms 18.2%
SingleProcess AUTOTUNE takes 4.1510 seconds
AUTOTUNE addmm(32x768, 32x768, 768x768)
  addmm 0.0405 ms 100.0%
  bias_addmm 0.0406 ms 99.7%
  triton_mm_1952 0.0558 ms 72.6%
  triton_mm_1949 0.0630 ms 64.2%
  triton_mm_1951 0.1009 ms 40.1%
  triton_mm_1950 0.1620 ms 25.0%
  triton_mm_1948 0.1700 ms 23.8%
SingleProcess AUTOTUNE takes 3.2154 seconds
AUTOTUNE mm(768x32, 32x768)
  triton_mm_1975 0.0125 ms 100.0%
  triton_mm_1972 0.0141 ms 88.6%
  mm 0.0148 ms 84.5%
  triton_mm_1973 0.0150 ms 83.2%
  triton_mm_1974 0.0173 ms 72.1%
  triton_mm_1971 0.0196 ms 63.5%
SingleProcess AUTOTUNE takes 3.0968 seconds
AUTOTUNE mm(2624x768, 768x3072)
  triton_mm_1979 1.4694 ms 100.0%
  mm 1.8176 ms 80.8%
  triton_mm_1980 1.9123 ms 76.8%
  triton_mm_1978 1.9885 ms 73.9%
  triton_mm_1977 2.2479 ms 65.4%
  triton_mm_1976 6.9431 ms 21.2%
SingleProcess AUTOTUNE takes 3.8661 seconds
AUTOTUNE mm(768x2624, 2624x3072)
  triton_mm_1985 1.1333 ms 100.0%
  mm 1.5975 ms 70.9%
  triton_mm_1982 1.7554 ms 64.6%
  triton_mm_1983 2.0310 ms 55.8%
  triton_mm_1984 2.1888 ms 51.8%
  triton_mm_1981 10.3159 ms 11.0%
SingleProcess AUTOTUNE takes 3.8867 seconds
AUTOTUNE mm(2624x3072, 3072x768)
  triton_mm_1989 1.6567 ms 100.0%
  mm 1.8692 ms 88.6%
  triton_mm_1990 1.9655 ms 84.3%
  triton_mm_1988 2.0014 ms 82.8%
  triton_mm_1987 2.2712 ms 72.9%
  triton_mm_1986 8.0355 ms 20.6%
SingleProcess AUTOTUNE takes 3.8818 seconds
AUTOTUNE mm(3072x2624, 2624x768)
  triton_mm_1995 1.1307 ms 100.0%
  mm 1.5935 ms 71.0%
  triton_mm_1992 1.7691 ms 63.9%
  triton_mm_1993 2.0402 ms 55.4%
  triton_mm_1994 2.1973 ms 51.5%
  triton_mm_1991 10.2621 ms 11.0%
SingleProcess AUTOTUNE takes 3.9129 seconds
AUTOTUNE mm(2624x768, 768x768)
  triton_mm_1999 0.4004 ms 100.0%
  mm 0.4547 ms 88.1%
  triton_mm_2000 0.4868 ms 82.2%
  triton_mm_1998 0.4997 ms 80.1%
  triton_mm_1997 0.5792 ms 69.1%
  triton_mm_1996 2.0104 ms 19.9%
SingleProcess AUTOTUNE takes 3.8092 seconds
AUTOTUNE mm(768x2624, 2624x768)
  triton_mm_2005 0.3064 ms 100.0%
  mm 0.4136 ms 74.1%
  triton_mm_2002 0.5036 ms 60.8%
  triton_mm_2004 0.5313 ms 57.7%
  triton_mm_2003 0.5658 ms 54.1%
  triton_mm_2001 2.6820 ms 11.4%
SingleProcess AUTOTUNE takes 3.8308 seconds
AUTOTUNE bmm(384x82x82, 384x82x64)
  bmm 0.0904 ms 100.0%
  triton_bmm_2010 0.0958 ms 94.3%
  triton_bmm_2008 0.1041 ms 86.8%
  triton_bmm_2009 0.1197 ms 75.5%
  triton_bmm_2007 0.1250 ms 72.3%
  triton_bmm_2006 0.5967 ms 15.1%
SingleProcess AUTOTUNE takes 3.7870 seconds
AUTOTUNE bmm(384x82x64, 384x64x82)
  bmm 0.0855 ms 100.0%
  triton_bmm_2015 0.1561 ms 54.8%
  triton_bmm_2014 0.2321 ms 36.8%
  triton_bmm_2012 0.3748 ms 22.8%
  triton_bmm_2013 0.5953 ms 14.4%
  triton_bmm_2011 0.6771 ms 12.6%
SingleProcess AUTOTUNE takes 3.7301 seconds
AUTOTUNE bmm(384x64x82, 384x82x82)
  bmm 0.0888 ms 100.0%
  triton_bmm_2020 0.0965 ms 92.0%
  triton_bmm_2017 0.1002 ms 88.6%
  triton_bmm_2019 0.1318 ms 67.3%
  triton_bmm_2018 0.1325 ms 67.0%
  triton_bmm_2016 0.6214 ms 14.3%
SingleProcess AUTOTUNE takes 3.9463 seconds
AUTOTUNE bmm(384x82x82, 384x82x64)
  bmm 0.0959 ms 100.0%
  triton_bmm_2025 0.1511 ms 63.5%
  triton_bmm_2024 0.1978 ms 48.5%
  triton_bmm_2022 0.3607 ms 26.6%
  triton_bmm_2023 0.4347 ms 22.1%
  triton_bmm_2021 0.5649 ms 17.0%
SingleProcess AUTOTUNE takes 3.9172 seconds
AUTOTUNE addmm(1520x768, 1520x768, 768x768)
  addmm 0.3112 ms 100.0%
  bias_addmm 0.3153 ms 98.7%
  triton_mm_2940 0.6454 ms 48.2%
  triton_mm_2939 0.6551 ms 47.5%
  triton_mm_2937 1.0555 ms 29.5%
  triton_mm_2936 1.7252 ms 18.0%
  triton_mm_2938 1.7431 ms 17.9%
SingleProcess AUTOTUNE takes 3.8664 seconds
AUTOTUNE bmm(240x76x64, 240x64x76)
  bmm 0.0497 ms 100.0%
  triton_bmm_2955 0.0526 ms 94.5%
  triton_bmm_2954 0.0683 ms 72.8%
  triton_bmm_2953 0.1000 ms 49.7%
  triton_bmm_2952 0.1027 ms 48.4%
  triton_bmm_2951 0.3494 ms 14.2%
SingleProcess AUTOTUNE takes 0.6512 seconds
AUTOTUNE bmm(240x76x76, 240x76x64)
  triton_bmm_2960 0.0525 ms 100.0%
  bmm 0.0541 ms 97.1%
  triton_bmm_2958 0.0665 ms 79.0%
  triton_bmm_2959 0.0757 ms 69.4%
  triton_bmm_2957 0.0879 ms 59.8%
  triton_bmm_2956 0.3509 ms 15.0%
SingleProcess AUTOTUNE takes 0.6417 seconds
AUTOTUNE addmm(1520x3072, 1520x768, 768x3072)
  addmm 1.1924 ms 100.0%
  bias_addmm 1.2593 ms 94.7%
  triton_mm_2969 1.9190 ms 62.1%
  triton_mm_2970 2.2458 ms 53.1%
  triton_mm_2967 3.8888 ms 30.7%
  triton_mm_2966 5.6017 ms 21.3%
  triton_mm_2968 6.4712 ms 18.4%
SingleProcess AUTOTUNE takes 3.9879 seconds
AUTOTUNE addmm(1520x768, 1520x3072, 3072x768)
  addmm 1.2172 ms 100.0%
  bias_addmm 1.2934 ms 94.1%
  triton_mm_2975 2.5041 ms 48.6%
  triton_mm_2974 2.5235 ms 48.2%
  triton_mm_2972 4.1167 ms 29.6%
  triton_mm_2971 6.7357 ms 18.1%
  triton_mm_2973 6.8412 ms 17.8%
SingleProcess AUTOTUNE takes 3.9968 seconds
AUTOTUNE addmm(20x768, 20x768, 768x768)
  addmm 0.0404 ms 100.0%
  bias_addmm 0.0405 ms 99.6%
  triton_mm_3420 0.0528 ms 76.4%
  triton_mm_3417 0.0637 ms 63.3%
  triton_mm_3419 0.1008 ms 40.0%
  triton_mm_3418 0.1614 ms 25.0%
  triton_mm_3416 0.1707 ms 23.6%
SingleProcess AUTOTUNE takes 3.1952 seconds
AUTOTUNE addmm(20x2, 20x768, 768x2)
  addmm 0.0368 ms 100.0%
  bias_addmm 0.0369 ms 99.7%
  triton_mm_3421 0.0379 ms 96.9%
  triton_mm_3422 0.0385 ms 95.4%
  triton_mm_3424 0.0396 ms 92.8%
  triton_mm_3423 0.0400 ms 92.0%
SingleProcess AUTOTUNE takes 2.4551 seconds
AUTOTUNE mm(20x2, 2x768)
  triton_mm_3426 0.0076 ms 100.0%
  triton_mm_3427 0.0078 ms 96.4%
  triton_mm_3425 0.0080 ms 95.0%
  triton_mm_3428 0.0080 ms 94.0%
  mm 0.0081 ms 93.1%
SingleProcess AUTOTUNE takes 1.7660 seconds
AUTOTUNE mm(2x20, 20x768)
  mm 0.0081 ms 100.0%
  triton_mm_3433 0.0083 ms 97.6%
  triton_mm_3432 0.0086 ms 93.5%
  triton_mm_3429 0.0092 ms 87.6%
  triton_mm_3430 0.0106 ms 75.9%
  triton_mm_3431 0.0116 ms 69.6%
SingleProcess AUTOTUNE takes 2.9442 seconds
AUTOTUNE mm(20x768, 768x768)
  triton_mm_3435 0.0362 ms 100.0%
  triton_mm_3436 0.0402 ms 90.1%
  triton_mm_3438 0.0404 ms 89.8%
  triton_mm_3437 0.0440 ms 82.4%
  mm 0.0462 ms 78.5%
  triton_mm_3434 0.1394 ms 26.0%
SingleProcess AUTOTUNE takes 3.0886 seconds
AUTOTUNE mm(768x20, 20x768)
  mm 0.0118 ms 100.0%
  triton_mm_3443 0.0124 ms 95.1%
  triton_mm_3442 0.0161 ms 72.9%
  triton_mm_3440 0.0194 ms 60.5%
  triton_mm_3441 0.0202 ms 58.3%
  triton_mm_3439 0.0634 ms 18.6%
SingleProcess AUTOTUNE takes 3.6291 seconds
AUTOTUNE mm(1520x768, 768x3072)
  triton_mm_3447 0.8681 ms 100.0%
  mm 1.0603 ms 81.9%
  triton_mm_3448 1.1200 ms 77.5%
  triton_mm_3446 1.1726 ms 74.0%
  triton_mm_3445 1.3363 ms 65.0%
  triton_mm_3444 4.3779 ms 19.8%
SingleProcess AUTOTUNE takes 3.8396 seconds
AUTOTUNE mm(768x1520, 1520x3072)
  triton_mm_3453 0.6551 ms 100.0%
  mm 0.9153 ms 71.6%
  triton_mm_3452 1.0142 ms 64.6%
  triton_mm_3450 1.0459 ms 62.6%
  triton_mm_3451 1.2217 ms 53.6%
  triton_mm_3449 6.9673 ms 9.4%
SingleProcess AUTOTUNE takes 3.8128 seconds
AUTOTUNE mm(1520x3072, 3072x768)
  triton_mm_3457 1.0901 ms 100.0%
  mm 1.1051 ms 98.6%
  triton_mm_3458 1.2270 ms 88.8%
  triton_mm_3456 1.2390 ms 88.0%
  triton_mm_3455 1.3961 ms 78.1%
  triton_mm_3454 5.4784 ms 19.9%
SingleProcess AUTOTUNE takes 3.8842 seconds
AUTOTUNE mm(3072x1520, 1520x768)
  triton_mm_3463 0.6541 ms 100.0%
  mm 0.9165 ms 71.4%
  triton_mm_3462 1.0215 ms 64.0%
  triton_mm_3460 1.0434 ms 62.7%
  triton_mm_3461 1.2216 ms 53.5%
  triton_mm_3459 6.9859 ms 9.4%
SingleProcess AUTOTUNE takes 3.7640 seconds
AUTOTUNE mm(1520x768, 768x768)
  mm 0.2671 ms 100.0%
  triton_mm_3467 0.2683 ms 99.6%
  triton_mm_3466 0.3059 ms 87.3%
  triton_mm_3468 0.3065 ms 87.2%
  triton_mm_3465 0.3481 ms 76.7%
  triton_mm_3464 1.3909 ms 19.2%
SingleProcess AUTOTUNE takes 3.7649 seconds
AUTOTUNE mm(768x1520, 1520x768)
  triton_mm_3473 0.1747 ms 100.0%
  mm 0.2316 ms 75.4%
  triton_mm_3472 0.2456 ms 71.1%
  triton_mm_3470 0.2910 ms 60.0%
  triton_mm_3471 0.3304 ms 52.9%
  triton_mm_3469 2.0335 ms 8.6%
SingleProcess AUTOTUNE takes 3.7103 seconds
AUTOTUNE bmm(240x76x76, 240x76x64)
  triton_bmm_3478 0.0489 ms 100.0%
  bmm 0.0489 ms 100.0%
  triton_bmm_3477 0.0668 ms 73.2%
  triton_bmm_3476 0.0676 ms 72.3%
  triton_bmm_3475 0.0848 ms 57.6%
  triton_bmm_3474 0.4239 ms 11.5%
SingleProcess AUTOTUNE takes 0.6430 seconds
AUTOTUNE bmm(240x76x64, 240x64x76)
  bmm 0.0537 ms 100.0%
  triton_bmm_3483 0.1059 ms 50.7%
  triton_bmm_3482 0.1510 ms 35.6%
  triton_bmm_3480 0.2397 ms 22.4%
  triton_bmm_3481 0.3720 ms 14.4%
  triton_bmm_3479 0.4390 ms 12.2%
SingleProcess AUTOTUNE takes 0.6739 seconds
AUTOTUNE bmm(240x64x76, 240x76x76)
  bmm 0.0489 ms 100.0%
  triton_bmm_3488 0.0499 ms 98.0%
  triton_bmm_3485 0.0650 ms 75.1%
  triton_bmm_3487 0.0753 ms 64.9%
  triton_bmm_3486 0.0898 ms 54.4%
  triton_bmm_3484 0.4399 ms 11.1%
SingleProcess AUTOTUNE takes 0.6457 seconds
AUTOTUNE bmm(240x76x76, 240x76x64)
  bmm 0.0567 ms 100.0%
  triton_bmm_3493 0.0899 ms 63.1%
  triton_bmm_3492 0.1140 ms 49.8%
  triton_bmm_3490 0.2336 ms 24.3%
  triton_bmm_3491 0.2879 ms 19.7%
  triton_bmm_3489 0.4055 ms 14.0%
SingleProcess AUTOTUNE takes 0.6668 seconds
AUTOTUNE addmm(640x768, 640x768, 768x768)
  addmm 0.1378 ms 100.0%
  bias_addmm 0.1379 ms 99.9%
  triton_mm_4407 0.2578 ms 53.5%
  triton_mm_4408 0.2728 ms 50.5%
  triton_mm_4405 0.4833 ms 28.5%
  triton_mm_4406 0.8014 ms 17.2%
  triton_mm_4404 0.8890 ms 15.5%
SingleProcess AUTOTUNE takes 3.9088 seconds
AUTOTUNE bmm(96x80x64, 96x64x80)
  bmm 0.0261 ms 100.0%
  triton_bmm_4418 0.0298 ms 87.7%
  triton_bmm_4417 0.0367 ms 71.2%
  triton_bmm_4416 0.0449 ms 58.2%
  triton_bmm_4415 0.0466 ms 56.1%
  triton_bmm_4414 0.1692 ms 15.4%
SingleProcess AUTOTUNE takes 3.6751 seconds
AUTOTUNE bmm(96x80x80, 96x80x64)
  bmm 0.0251 ms 100.0%
  triton_bmm_4428 0.0272 ms 92.2%
  triton_bmm_4426 0.0346 ms 72.5%
  triton_bmm_4427 0.0376 ms 66.9%
  triton_bmm_4425 0.0436 ms 57.6%
  triton_bmm_4424 0.1617 ms 15.5%
SingleProcess AUTOTUNE takes 3.7112 seconds
AUTOTUNE addmm(640x3072, 640x768, 768x3072)
  addmm 0.4973 ms 100.0%
  bias_addmm 0.5166 ms 96.3%
  triton_mm_4437 0.8238 ms 60.4%
  triton_mm_4438 0.9973 ms 49.9%
  triton_mm_4435 1.6812 ms 29.6%
  triton_mm_4434 2.5665 ms 19.4%
  triton_mm_4436 2.7873 ms 17.8%
SingleProcess AUTOTUNE takes 3.9003 seconds
AUTOTUNE addmm(640x768, 640x3072, 3072x768)
  addmm 0.5136 ms 100.0%
  bias_addmm 0.5428 ms 94.6%
  triton_mm_4442 0.9898 ms 51.9%
  triton_mm_4443 1.0579 ms 48.5%
  triton_mm_4440 1.8793 ms 27.3%
  triton_mm_4441 3.1173 ms 16.5%
  triton_mm_4439 3.3986 ms 15.1%
SingleProcess AUTOTUNE takes 3.9520 seconds
AUTOTUNE addmm(8x768, 8x768, 768x768)
  addmm 0.0398 ms 100.0%
  bias_addmm 0.0400 ms 99.6%
  triton_mm_4885 0.0459 ms 86.7%
  triton_mm_4888 0.0488 ms 81.6%
  triton_mm_4887 0.0951 ms 41.9%
  triton_mm_4884 0.0959 ms 41.5%
  triton_mm_4886 0.0976 ms 40.8%
SingleProcess AUTOTUNE takes 3.1026 seconds
AUTOTUNE addmm(8x2, 8x768, 768x2)
  triton_mm_4892 0.0315 ms 100.0%
  triton_mm_4891 0.0333 ms 94.5%
  addmm 0.0365 ms 86.2%
  bias_addmm 0.0366 ms 86.0%
  triton_mm_4890 0.0430 ms 73.3%
  triton_mm_4889 0.0431 ms 73.0%
SingleProcess AUTOTUNE takes 2.5797 seconds
```

This command is the same and will work for:

- a CPU-only setup
- a setup with one GPU
- a distributed training with several GPUs (single or multi node)
- a training on TPUs

Note that this library is in alpha release so your feedback is more than welcome if you encounter any problem using it.

## XNLI

Based on the script [`run_xnli.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py).

[XNLI](https://cims.nyu.edu/~sbowman/xnli/) is a crowd-sourced dataset based on [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/). It is an evaluation benchmark for cross-lingual text representations. Pairs of text are labeled with textual entailment annotations for 15 different languages (including both high-resource language such as English and low-resource languages such as Swahili).

#### Fine-tuning on XNLI

This example code fine-tunes mBERT (multi-lingual BERT) on the XNLI dataset. It runs in 106 mins on a single tesla V100 16GB.

```bash
TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python run_xnli.py \
  --model_name_or_path bert-base-multilingual-cased \
  --language de \
  --train_language en \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir /tmp/debug_xnli/ \
  --save_steps -1 \
  --torch_compile True
```

PT2.0 results NV31 (single gpu) + Ryzen 7 5700x3d - train/train_steps_per_second:
| GPU       | Run Mode                          | train_steps_per_second | approximate run time (did not run to completion)
|-----------|-----------------------------------|------------------------|-------------------------------------------------|
| 7900 XTX  | Regular execution                 | 2.56                   | 159m                                            |
| 7900 XTX  | compile - autotune cached         | 3.12                   | 131m                                            |

Autotune logs:
```
AUTOTUNE addmm(4096x768, 4096x768, 768x768)
  addmm 0.8016 ms 100.0%
  bias_addmm 0.8211 ms 97.6%
  triton_mm_3 1.3084 ms 61.3%
  triton_mm_4 1.5093 ms 53.1%
  triton_mm_1 2.5331 ms 31.6%
  triton_mm_0 3.9983 ms 20.0%
  triton_mm_2 4.2424 ms 18.9%
SingleProcess AUTOTUNE takes 1.1797 seconds
AUTOTUNE bmm(384x128x64, 384x64x128)
  bmm 0.1496 ms 100.0%
  triton_bmm_19 0.1528 ms 97.9%
  triton_bmm_18 0.1546 ms 96.8%
  triton_bmm_17 0.1877 ms 79.7%
  triton_bmm_16 0.1966 ms 76.1%
  triton_bmm_15 0.5181 ms 28.9%
SingleProcess AUTOTUNE takes 0.8689 seconds
AUTOTUNE bmm(384x128x128, 384x128x64)
  bmm 0.1607 ms 100.0%
  triton_bmm_22 0.1832 ms 87.7%
  triton_bmm_21 0.1900 ms 84.6%
  triton_bmm_24 0.1998 ms 80.4%
  triton_bmm_23 0.2395 ms 67.1%
  triton_bmm_20 0.6110 ms 26.3%
SingleProcess AUTOTUNE takes 0.7988 seconds
AUTOTUNE addmm(4096x3072, 4096x768, 768x3072)
  addmm 3.1096 ms 100.0%
  bias_addmm 3.3025 ms 94.2%
  triton_mm_33 5.0072 ms 62.1%
  triton_mm_34 5.8552 ms 53.1%
  triton_mm_31 9.8588 ms 31.5%
  triton_mm_30 14.7959 ms 21.0%
  triton_mm_32 16.5049 ms 18.8%
SingleProcess AUTOTUNE takes 1.6091 seconds
AUTOTUNE addmm(4096x768, 4096x3072, 3072x768)
  addmm 3.1377 ms 100.0%
  bias_addmm 3.3354 ms 94.1%
  triton_mm_38 5.1370 ms 61.1%
  triton_mm_39 5.9529 ms 52.7%
  triton_mm_36 10.0270 ms 31.3%
  triton_mm_35 16.1008 ms 19.5%
  triton_mm_37 16.8789 ms 18.6%
SingleProcess AUTOTUNE takes 1.3550 seconds
AUTOTUNE addmm(32x768, 32x768, 768x768)
  addmm 0.0393 ms 100.0%
  bias_addmm 0.0394 ms 99.7%
  triton_mm_484 0.0539 ms 72.9%
  triton_mm_481 0.0639 ms 61.5%
  triton_mm_483 0.1007 ms 39.1%
  triton_mm_482 0.1628 ms 24.1%
  triton_mm_480 0.1707 ms 23.0%
SingleProcess AUTOTUNE takes 0.8699 seconds
AUTOTUNE addmm(32x3, 32x768, 768x3)
  addmm 0.0373 ms 100.0%
  bias_addmm 0.0374 ms 99.7%
  triton_mm_485 0.0377 ms 98.8%
  triton_mm_486 0.0389 ms 95.8%
  triton_mm_487 0.0392 ms 95.2%
  triton_mm_488 0.0395 ms 94.4%
SingleProcess AUTOTUNE takes 2.4697 seconds
AUTOTUNE mm(32x3, 3x768)
  mm 0.0076 ms 100.0%
  triton_mm_492 0.0077 ms 98.2%
  triton_mm_489 0.0078 ms 97.7%
  triton_mm_490 0.0078 ms 97.7%
  triton_mm_491 0.0078 ms 97.2%
SingleProcess AUTOTUNE takes 1.9282 seconds
AUTOTUNE mm(3x32, 32x768)
  triton_mm_493 0.0077 ms 100.0%
  triton_mm_495 0.0079 ms 97.5%
  triton_mm_494 0.0080 ms 96.5%
  triton_mm_497 0.0082 ms 94.1%
  triton_mm_496 0.0082 ms 93.7%
  mm 0.0084 ms 91.0%
SingleProcess AUTOTUNE takes 2.5481 seconds
AUTOTUNE mm(768x32, 32x768)
  triton_mm_507 0.0126 ms 100.0%
  mm 0.0132 ms 95.2%
  triton_mm_504 0.0144 ms 87.2%
  triton_mm_505 0.0149 ms 84.2%
  triton_mm_506 0.0184 ms 68.3%
  triton_mm_503 0.0196 ms 64.2%
SingleProcess AUTOTUNE takes 3.1562 seconds
AUTOTUNE mm(4096x768, 768x3072)
  triton_mm_511 2.3712 ms 100.0%
  mm 2.8340 ms 83.7%
  triton_mm_512 3.0181 ms 78.6%
  triton_mm_510 3.1262 ms 75.8%
  triton_mm_509 3.5221 ms 67.3%
  triton_mm_508 11.5493 ms 20.5%
SingleProcess AUTOTUNE takes 3.9527 seconds
AUTOTUNE mm(768x4096, 4096x3072)
  triton_mm_517 1.7834 ms 100.0%
  mm 2.5072 ms 71.1%
  triton_mm_514 2.7897 ms 63.9%
  triton_mm_516 2.8395 ms 62.8%
  triton_mm_515 3.2703 ms 54.5%
  triton_mm_513 16.1465 ms 11.0%
SingleProcess AUTOTUNE takes 4.0267 seconds
AUTOTUNE mm(4096x3072, 3072x768)
  triton_mm_521 2.7025 ms 100.0%
  mm 2.8853 ms 93.7%
  triton_mm_522 3.0987 ms 87.2%
  triton_mm_520 3.1767 ms 85.1%
  triton_mm_519 3.5999 ms 75.1%
  triton_mm_518 13.4379 ms 20.1%
SingleProcess AUTOTUNE takes 3.9290 seconds
AUTOTUNE mm(3072x4096, 4096x768)
  triton_mm_527 1.7917 ms 100.0%
  mm 2.5146 ms 71.3%
  triton_mm_524 2.8130 ms 63.7%
  triton_mm_526 2.8360 ms 63.2%
  triton_mm_525 3.2795 ms 54.6%
  triton_mm_523 16.3720 ms 10.9%
SingleProcess AUTOTUNE takes 4.0112 seconds
AUTOTUNE mm(4096x768, 768x768)
  triton_mm_531 0.6641 ms 100.0%
  mm 0.7189 ms 92.4%
  triton_mm_532 0.7720 ms 86.0%
  triton_mm_530 0.8054 ms 82.5%
  triton_mm_529 0.9060 ms 73.3%
  triton_mm_528 3.3161 ms 20.0%
SingleProcess AUTOTUNE takes 3.8103 seconds
AUTOTUNE mm(768x4096, 4096x768)
  triton_mm_537 0.4849 ms 100.0%
  mm 0.6701 ms 72.4%
  triton_mm_536 0.7406 ms 65.5%
  triton_mm_534 0.7763 ms 62.5%
  triton_mm_535 0.8871 ms 54.7%
  triton_mm_533 4.2902 ms 11.3%
SingleProcess AUTOTUNE takes 3.9045 seconds
AUTOTUNE bmm(384x128x128, 384x128x64)
  triton_bmm_541 0.1486 ms 100.0%
  triton_bmm_542 0.1605 ms 92.6%
  triton_bmm_539 0.1614 ms 92.0%
  bmm 0.1671 ms 88.9%
  triton_bmm_540 0.1700 ms 87.4%
  triton_bmm_538 0.6356 ms 23.4%
SingleProcess AUTOTUNE takes 4.2869 seconds
AUTOTUNE bmm(384x128x64, 384x64x128)
  bmm 0.1396 ms 100.0%
  triton_bmm_546 0.2359 ms 59.2%
  triton_bmm_547 0.2632 ms 53.0%
  triton_bmm_544 0.4960 ms 28.1%
  triton_bmm_543 0.6901 ms 20.2%
  triton_bmm_545 0.7902 ms 17.7%
SingleProcess AUTOTUNE takes 4.3887 seconds
AUTOTUNE bmm(384x64x128, 384x128x128)
  triton_bmm_549 0.1544 ms 100.0%
  triton_bmm_552 0.1635 ms 94.4%
  triton_bmm_550 0.1639 ms 94.2%
  triton_bmm_551 0.1674 ms 92.3%
  bmm 0.1730 ms 89.2%
  triton_bmm_548 0.6338 ms 24.4%
SingleProcess AUTOTUNE takes 3.8390 seconds
AUTOTUNE bmm(384x128x128, 384x128x64)
  bmm 0.1466 ms 100.0%
  triton_bmm_556 0.2434 ms 60.2%
  triton_bmm_557 0.2721 ms 53.9%
  triton_bmm_554 0.4745 ms 30.9%
  triton_bmm_553 0.7456 ms 19.7%
  triton_bmm_555 0.7720 ms 19.0%
SingleProcess AUTOTUNE takes 3.6923 seconds


Training with the previously defined hyper-parameters yields the following results on the **test** set:

```bash
acc = 0.7093812375249501
```
