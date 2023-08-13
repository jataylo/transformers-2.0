<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

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

# Question answering

This folder contains several scripts that showcase how to fine-tune a 🤗 Transformers model on a question answering dataset,
like SQuAD. 

## Trainer-based scripts

The [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py),
[`run_qa_beam_search.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search.py) and [`run_seq2seq_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py) leverage the 🤗 [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) for fine-tuning.

### Fine-tuning BERT on SQuAD1.0

The [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py) script
allows to fine-tune any model from our [hub](https://huggingface.co/models) (as long as its architecture has a `ForQuestionAnswering` version in the library) on a question-answering dataset (such as SQuAD, or any other QA dataset available in the `datasets` library, or your own csv/jsonlines files) as long as they are structured the same way as SQuAD. You might need to tweak the data processing inside the script if your data is structured differently.

**Note:** This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks), if it doesn't you can still use the old version of the script which can be found [here](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering).

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along the flag `--version_2_with_negative`.

This example code fine-tunes BERT on the SQuAD1.0 dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large)
on a single tesla V100 16GB.

```bash
TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  --torch_compile True
```


PT2.0 results NV31 (single gpu) + Ryzen 7 5700x3d - train/train_steps_per_second - bert-base-uncased:
| GPU       | Run Mode                          | train_steps_per_second | 
|-----------|-----------------------------------|------------------------|
| 7900 XTX  | Regular execution                 | 2.12  (~115mins)       | 
| 7900 XTX  | compile - autotune cached         | 2.63  (~90mins)        | 

Autotune logs:
```
AUTOTUNE addmm(4608x768, 4608x768, 768x768)
  addmm 0.8858 ms 100.0%
  bias_addmm 0.9248 ms 95.8%
  triton_mm_3 1.4533 ms 60.9%
  triton_mm_4 1.6901 ms 52.4%
  triton_mm_1 2.8656 ms 30.9%
  triton_mm_0 4.3681 ms 20.3%
  triton_mm_2 4.7816 ms 18.5%
SingleProcess AUTOTUNE takes 4.4216 seconds
AUTOTUNE bmm(144x384x64, 144x64x384)
  triton_bmm_18 0.3227 ms 100.0%
  bmm 0.3894 ms 82.9%
  triton_bmm_19 0.4324 ms 74.6%
  triton_bmm_17 0.6034 ms 53.5%
  triton_bmm_16 0.6326 ms 51.0%
  triton_bmm_15 1.5418 ms 20.9%
SingleProcess AUTOTUNE takes 5.8049 seconds
AUTOTUNE bmm(144x384x384, 144x384x64)
  bmm 0.4779 ms 100.0%
  triton_bmm_24 0.5209 ms 91.7%
  triton_bmm_22 0.5365 ms 89.1%
  triton_bmm_21 0.5936 ms 80.5%
  triton_bmm_23 0.7146 ms 66.9%
  triton_bmm_20 2.2851 ms 20.9%
SingleProcess AUTOTUNE takes 7.2492 seconds
AUTOTUNE addmm(4608x3072, 4608x768, 768x3072)
  addmm 3.4309 ms 100.0%
  bias_addmm 3.6206 ms 94.8%
  triton_mm_33 5.5091 ms 62.3%
  triton_mm_34 6.5444 ms 52.4%
  triton_mm_31 11.2644 ms 30.5%
  triton_mm_30 16.1160 ms 21.3%
  triton_mm_32 18.8348 ms 18.2%
SingleProcess AUTOTUNE takes 4.3707 seconds
AUTOTUNE addmm(4608x768, 4608x3072, 3072x768)
  addmm 3.4424 ms 100.0%
  bias_addmm 3.6962 ms 93.1%
  triton_mm_38 5.6905 ms 60.5%
  triton_mm_39 6.6808 ms 51.5%
  triton_mm_36 11.3623 ms 30.3%
  triton_mm_35 17.4018 ms 19.8%
  triton_mm_37 18.8894 ms 18.2%
SingleProcess AUTOTUNE takes 4.3931 seconds
AUTOTUNE addmm(4608x2, 4608x768, 768x2)
  triton_mm_482 0.1222 ms 100.0%
  triton_mm_484 0.1244 ms 98.2%
  triton_mm_480 0.1260 ms 96.9%
  addmm 0.1286 ms 95.0%
  bias_addmm 0.1303 ms 93.8%
  triton_mm_481 0.1350 ms 90.5%
  triton_mm_483 0.1357 ms 90.0%
SingleProcess AUTOTUNE takes 3.8253 seconds
AUTOTUNE mm(4608x2, 2x768)
  mm 0.0246 ms 100.0%
  triton_mm_487 0.0278 ms 88.6%
  triton_mm_485 0.0281 ms 87.7%
  triton_mm_486 0.0301 ms 81.8%
  triton_mm_488 0.0307 ms 80.2%
SingleProcess AUTOTUNE takes 2.6197 seconds
AUTOTUNE mm(2x4608, 4608x768)
  triton_mm_492 0.2193 ms 100.0%
  triton_mm_493 0.2230 ms 98.3%
  mm 0.2363 ms 92.8%
  triton_mm_491 0.2547 ms 86.1%
  triton_mm_489 0.2646 ms 82.9%
  triton_mm_490 0.3499 ms 62.7%
SingleProcess AUTOTUNE takes 3.1669 seconds
AUTOTUNE mm(4608x768, 768x3072)
  triton_mm_497 2.5278 ms 100.0%
  mm 3.2430 ms 77.9%
  triton_mm_498 3.3251 ms 76.0%
  triton_mm_496 3.4717 ms 72.8%
  triton_mm_495 3.8911 ms 65.0%
  triton_mm_494 12.4368 ms 20.3%
SingleProcess AUTOTUNE takes 4.0148 seconds
AUTOTUNE mm(768x4608, 4608x3072)
  triton_mm_503 1.9958 ms 100.0%
  mm 2.8604 ms 69.8%
  triton_mm_500 3.0947 ms 64.5%
  triton_mm_502 3.1713 ms 62.9%
  triton_mm_501 3.5850 ms 55.7%
  triton_mm_499 18.1025 ms 11.0%
SingleProcess AUTOTUNE takes 4.0962 seconds
AUTOTUNE mm(4608x3072, 3072x768)
  triton_mm_507 2.7286 ms 100.0%
  mm 3.3198 ms 82.2%
  triton_mm_508 3.4220 ms 79.7%
  triton_mm_506 3.5009 ms 77.9%
  triton_mm_505 3.9298 ms 69.4%
  triton_mm_504 14.1393 ms 19.3%
SingleProcess AUTOTUNE takes 3.9828 seconds
AUTOTUNE mm(3072x4608, 4608x768)
  triton_mm_513 2.0402 ms 100.0%
  mm 2.8884 ms 70.6%
  triton_mm_510 3.1090 ms 65.6%
  triton_mm_512 3.1628 ms 64.5%
  triton_mm_511 3.6099 ms 56.5%
  triton_mm_509 18.3961 ms 11.1%
SingleProcess AUTOTUNE takes 4.0562 seconds
AUTOTUNE mm(4608x768, 768x768)
  triton_mm_517 0.6702 ms 100.0%
  mm 0.8285 ms 80.9%
  triton_mm_518 0.8484 ms 79.0%
  triton_mm_516 0.8908 ms 75.2%
  triton_mm_515 0.9997 ms 67.0%
  triton_mm_514 3.4737 ms 19.3%
SingleProcess AUTOTUNE takes 3.8457 seconds
AUTOTUNE mm(768x4608, 4608x768)
  triton_mm_523 0.5478 ms 100.0%
  mm 0.7764 ms 70.6%
  triton_mm_522 0.8307 ms 65.9%
  triton_mm_520 0.8931 ms 61.3%
  triton_mm_521 0.9856 ms 55.6%
  triton_mm_519 4.8433 ms 11.3%
SingleProcess AUTOTUNE takes 6.3218 seconds
AUTOTUNE bmm(144x384x384, 144x384x64)
  triton_bmm_527 0.3733 ms 100.0%
  bmm 0.4134 ms 90.3%
  triton_bmm_528 0.4141 ms 90.1%
  triton_bmm_525 0.4421 ms 84.4%
  triton_bmm_526 0.4858 ms 76.8%
  triton_bmm_524 2.3866 ms 15.6%
SingleProcess AUTOTUNE takes 5.1048 seconds
AUTOTUNE bmm(144x384x64, 144x64x384)
  bmm 0.4663 ms 100.0%
  triton_bmm_532 0.7303 ms 63.8%
  triton_bmm_533 0.8513 ms 54.8%
  triton_bmm_530 1.5956 ms 29.2%
  triton_bmm_529 2.0644 ms 22.6%
  triton_bmm_531 2.5365 ms 18.4%
SingleProcess AUTOTUNE takes 3.7643 seconds
AUTOTUNE bmm(144x64x384, 144x384x384)
  triton_bmm_538 0.3882 ms 100.0%
  triton_bmm_537 0.4213 ms 92.2%
  triton_bmm_535 0.4266 ms 91.0%
  bmm 0.4399 ms 88.2%
  triton_bmm_536 0.4678 ms 83.0%
  triton_bmm_534 2.1703 ms 17.9%
SingleProcess AUTOTUNE takes 3.9187 seconds
AUTOTUNE bmm(144x384x384, 144x384x64)
  bmm 0.4925 ms 100.0%
  triton_bmm_543 0.8696 ms 56.6%
  triton_bmm_542 0.8936 ms 55.1%
  triton_bmm_540 1.4763 ms 33.4%
  triton_bmm_539 2.2321 ms 22.1%
  triton_bmm_541 2.4185 ms 20.4%
SingleProcess AUTOTUNE takes 3.8873 seconds

### Fine-tuning XLNet with beam search on SQuAD

The [`run_qa_beam_search.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search.py) script is only meant to fine-tune XLNet, which is a special encoder-only Transformer model. The example code below fine-tunes XLNet on the SQuAD1.0 and SQuAD2.0 datasets.

#### Command for SQuAD1.0:

```bash
TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python run_qa_beam_search.py \
    --model_name_or_path xlnet-large-cased \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_device_eval_batch_size=4  \
    --per_device_train_batch_size=4   \
    --save_steps 5000 \
    --torch_compile True
```

PT2.0 results NV31 (single gpu) + Ryzen 7 5700x3d - train/train_steps_per_second - xlnet-large-cased:
| GPU       | Run Mode                          | train_steps_per_second | 
|-----------|-----------------------------------|------------------------|
| 7900 XTX  | Regular execution                 | 1.38  (~536mins)       | 
| 7900 XTX  | compile - autotune cached         | 1.79  (~413mins)        | 

Autotune logs:
```
AUTOTUNE bmm(1x1536x1024, 1x1024x1024)
  triton_bmm_3 0.4168 ms 100.0%
  bmm 0.4643 ms 89.8%
  triton_bmm_4 0.5257 ms 79.3%
  triton_bmm_2 0.5335 ms 78.1%
  triton_bmm_1 0.5973 ms 69.8%
  triton_bmm_0 2.4506 ms 17.0%
SingleProcess AUTOTUNE takes 5.6650 seconds
AUTOTUNE bmm(1x3072x1024, 1x1024x1024)
  triton_bmm_18 0.7430 ms 100.0%
  bmm 0.9475 ms 78.4%
  triton_bmm_19 1.0056 ms 73.9%
  triton_bmm_17 1.0580 ms 70.2%
  triton_bmm_16 1.1625 ms 63.9%
  triton_bmm_15 4.2856 ms 17.3%
SingleProcess AUTOTUNE takes 3.9416 seconds
AUTOTUNE bmm(64x384x64, 64x64x384)
  bmm 0.2433 ms 100.0%
  triton_bmm_23 0.3399 ms 71.6%
  triton_bmm_24 0.3816 ms 63.8%
  triton_bmm_21 0.7690 ms 31.6%
  triton_bmm_20 1.0370 ms 23.5%
  triton_bmm_22 1.1952 ms 20.4%
SingleProcess AUTOTUNE takes 3.7916 seconds
AUTOTUNE bmm(64x384x64, 64x64x768)
  bmm 0.5105 ms 100.0%
  triton_bmm_28 0.6655 ms 76.7%
  triton_bmm_29 0.7572 ms 67.4%
  triton_bmm_26 1.5641 ms 32.6%
  triton_bmm_25 1.9468 ms 26.2%
  triton_bmm_27 2.3540 ms 21.7%
SingleProcess AUTOTUNE takes 4.5416 seconds
AUTOTUNE bmm(16x1536x64, 16x64x2)
  bmm 0.0330 ms 100.0%
  triton_bmm_33 0.0352 ms 93.6%
  triton_bmm_34 0.0384 ms 86.0%
  triton_bmm_30 0.0487 ms 67.8%
  triton_bmm_32 0.0665 ms 49.6%
  triton_bmm_31 0.0727 ms 45.4%
SingleProcess AUTOTUNE takes 2.9415 seconds
AUTOTUNE bmm(1536x384x2, 1536x2x16)
  bmm 0.0700 ms 100.0%
  triton_bmm_36 0.0954 ms 73.4%
  triton_bmm_35 0.0954 ms 73.4%
  triton_bmm_37 0.1005 ms 69.7%
  triton_bmm_38 0.1008 ms 69.5%
SingleProcess AUTOTUNE takes 1.8251 seconds
AUTOTUNE bmm(64x384x384, 64x384x64)
  bmm 0.2141 ms 100.0%
  triton_bmm_41 0.2462 ms 87.0%
  triton_bmm_40 0.2717 ms 78.8%
  triton_bmm_43 0.2879 ms 74.4%
  triton_bmm_42 0.4179 ms 51.2%
  triton_bmm_39 1.0278 ms 20.8%
SingleProcess AUTOTUNE takes 3.8408 seconds
AUTOTUNE bmm(1x1536x1024, 1x1024x1024)
  triton_bmm_47 0.4262 ms 100.0%
  bmm 0.4730 ms 90.1%
  triton_bmm_48 0.5285 ms 80.6%
  triton_bmm_46 0.5362 ms 79.5%
  triton_bmm_45 0.6052 ms 70.4%
  triton_bmm_44 2.4908 ms 17.1%
SingleProcess AUTOTUNE takes 4.4562 seconds
AUTOTUNE addmm(1536x4096, 1536x1024, 1024x4096)
  addmm 2.0708 ms 100.0%
  bias_addmm 2.2150 ms 93.5%
  triton_mm_52 3.3640 ms 61.6%
  triton_mm_53 3.9363 ms 52.6%
  triton_mm_50 6.8368 ms 30.3%
  triton_mm_49 10.0014 ms 20.7%
  triton_mm_51 11.4015 ms 18.2%
SingleProcess AUTOTUNE takes 4.9269 seconds
AUTOTUNE addmm(1536x1024, 1536x4096, 4096x1024)
  addmm 2.0811 ms 100.0%
  bias_addmm 2.2406 ms 92.9%
  triton_mm_57 3.7775 ms 55.1%
  triton_mm_58 4.2723 ms 48.7%
  triton_mm_55 7.1303 ms 29.2%
  triton_mm_56 11.8511 ms 17.6%
  triton_mm_54 11.8593 ms 17.5%
SingleProcess AUTOTUNE takes 4.2982 seconds
AUTOTUNE addmm(1536x1, 1536x1024, 1024x1)
  triton_mm_1420 0.0676 ms 100.0%

  triton_mm_1420 0.0676 ms 100.0%
  addmm 0.0691 ms 97.8%
  triton_mm_1418 0.0728 ms 92.9%
  triton_mm_1419 0.0875 ms 77.2%
  triton_mm_1416 0.0882 ms 76.7%
  triton_mm_1417 0.0886 ms 76.3%
SingleProcess AUTOTUNE takes 3.0066 seconds
AUTOTUNE addmm(1536x1024, 1536x2048, 2048x1024)
  addmm 1.0637 ms 100.0%
  bias_addmm 1.1089 ms 95.9%
  triton_mm_1424 1.9041 ms 55.9%
  triton_mm_1425 2.1350 ms 49.8%
  triton_mm_1422 3.5854 ms 29.7%
  triton_mm_1421 5.9431 ms 17.9%
  triton_mm_1423 5.9722 ms 17.8%
SingleProcess AUTOTUNE takes 4.1824 seconds
AUTOTUNE addmm(4x1024, 4x2048, 2048x1024)
  addmm 0.0863 ms 100.0%
  bias_addmm 0.0864 ms 99.9%
  triton_mm_1432 0.1107 ms 78.0%
  triton_mm_1435 0.1188 ms 72.7%
  triton_mm_1431 0.2342 ms 36.8%
  triton_mm_1434 0.2355 ms 36.6%
  triton_mm_1433 0.2377 ms 36.3%
SingleProcess AUTOTUNE takes 4.7880 seconds
AUTOTUNE mm(4x1024, 1024x1)
  mm 0.0381 ms 100.0%
  triton_mm_1438 0.0396 ms 96.1%
  triton_mm_1439 0.0411 ms 92.6%
  triton_mm_1437 0.0417 ms 91.4%
  triton_mm_1436 0.0447 ms 85.2%
SingleProcess AUTOTUNE takes 2.8392 seconds
AUTOTUNE mm(1x4, 4x1024)
  triton_mm_1443 0.0075 ms 100.0%
  triton_mm_1440 0.0075 ms 99.5%
  triton_mm_1441 0.0077 ms 97.4%
  triton_mm_1442 0.0077 ms 97.4%
  mm 0.0084 ms 89.0%
SingleProcess AUTOTUNE takes 4.5030 seconds
AUTOTUNE mm(4x1, 1x1024)
  mm 0.0071 ms 100.0%
  triton_mm_1445 0.0074 ms 95.7%
  triton_mm_1447 0.0076 ms 93.2%
  triton_mm_1444 0.0077 ms 91.7%
  triton_mm_1446 0.0077 ms 91.7%
SingleProcess AUTOTUNE takes 1.8995 seconds
AUTOTUNE mm(4x1024, 1024x2048)
  triton_mm_1451 0.0492 ms 100.0%
  triton_mm_1452 0.0512 ms 96.1%
  triton_mm_1448 0.0560 ms 87.9%
  triton_mm_1450 0.0597 ms 82.4%
  mm 0.0699 ms 70.4%
  triton_mm_1449 0.0724 ms 68.0%
SingleProcess AUTOTUNE takes 3.1197 seconds
AUTOTUNE mm(1024x4, 4x2048)
  mm 0.0175 ms 100.0%
  triton_mm_1456 0.0190 ms 92.0%
  triton_mm_1453 0.0194 ms 90.5%
  triton_mm_1454 0.0196 ms 89.4%
  triton_mm_1455 0.0212 ms 82.5%
SingleProcess AUTOTUNE takes 1.9706 seconds
AUTOTUNE mm(1536x1, 1x1024)
  mm 0.0148 ms 100.0%
  triton_mm_1460 0.0161 ms 91.8%
  triton_mm_1457 0.0166 ms 88.9%
  triton_mm_1458 0.0172 ms 86.1%
  triton_mm_1459 0.0185 ms 80.1%
SingleProcess AUTOTUNE takes 1.9943 seconds
AUTOTUNE mm(1x1536, 1536x1024)
  triton_mm_1464 0.0684 ms 100.0%
  triton_mm_1465 0.0713 ms 95.8%
  triton_mm_1461 0.0838 ms 81.6%
  triton_mm_1462 0.0850 ms 80.4%
  triton_mm_1463 0.0868 ms 78.8%
  mm 0.0871 ms 78.5%
SingleProcess AUTOTUNE takes 3.0497 seconds
AUTOTUNE mm(1536x1024, 1024x2048)
  triton_mm_1469 0.7719 ms 100.0%
  mm 0.9393 ms 82.2%
  triton_mm_1470 1.0101 ms 76.4%
  triton_mm_1468 1.0654 ms 72.4%
  triton_mm_1467 1.1680 ms 66.1%
  triton_mm_1466 4.2853 ms 18.0%
SingleProcess AUTOTUNE takes 3.8974 seconds
AUTOTUNE mm(1024x1536, 1536x2048)
  triton_mm_1475 0.5752 ms 100.0%
  mm 0.8163 ms 70.5%
  triton_mm_1472 0.9166 ms 62.8%
  triton_mm_1474 0.9338 ms 61.6%
  triton_mm_1473 1.0472 ms 54.9%
  triton_mm_1471 5.2078 ms 11.0%
SingleProcess AUTOTUNE takes 3.9975 seconds
AUTOTUNE mm(1536x1024, 1024x4096)
  triton_mm_1488 1.5003 ms 100.0%
  mm 1.8937 ms 79.2%
  triton_mm_1489 1.9737 ms 76.0%
  triton_mm_1487 2.0878 ms 71.9%
  triton_mm_1486 2.3103 ms 64.9%
  triton_mm_1485 7.9864 ms 18.8%
SingleProcess AUTOTUNE takes 3.9550 seconds
AUTOTUNE mm(1024x1536, 1536x4096)
  triton_mm_1494 1.1343 ms 100.0%
  mm 1.6411 ms 69.1%
  triton_mm_1493 1.7336 ms 65.4%
  triton_mm_1491 1.8117 ms 62.6%
  triton_mm_1492 2.0513 ms 55.3%
  triton_mm_1490 10.1787 ms 11.1%
SingleProcess AUTOTUNE takes 4.0494 seconds
AUTOTUNE mm(1536x4096, 4096x1024)
  triton_mm_1498 1.7163 ms 100.0%
  mm 1.9448 ms 88.3%
  triton_mm_1497 2.1624 ms 79.4%
  triton_mm_1499 2.1629 ms 79.3%
  triton_mm_1496 2.3854 ms 71.9%
  triton_mm_1495 9.8589 ms 17.4%
SingleProcess AUTOTUNE takes 5.5261 seconds
AUTOTUNE mm(4096x1536, 1536x1024)
  triton_mm_1504 1.1319 ms 100.0%
  mm 1.6139 ms 70.1%
  triton_mm_1503 1.7061 ms 66.3%
  triton_mm_1501 1.8643 ms 60.7%
  triton_mm_1502 2.1024 ms 53.8%
  triton_mm_1500 10.4527 ms 10.8%
SingleProcess AUTOTUNE takes 10.7837 seconds
AUTOTUNE bmm(1x1024x1536, 1x1536x1024)
  triton_bmm_1509 0.3147 ms 100.0%
  bmm 0.4111 ms 76.6%
  triton_bmm_1506 0.4578 ms 68.7%
  triton_bmm_1507 0.5207 ms 60.4%
  triton_bmm_1508 0.5252 ms 59.9%
  triton_bmm_1505 3.3532 ms 9.4%
SingleProcess AUTOTUNE takes 5.4280 seconds
AUTOTUNE bmm(1x1536x1024, 1x1024x1024)
  bmm 0.5447 ms 100.0%
  triton_bmm_1513 0.9250 ms 58.9%
  triton_bmm_1514 1.0804 ms 50.4%
  triton_bmm_1511 1.7586 ms 31.0%
  triton_bmm_1512 2.9313 ms 18.6%
  triton_bmm_1510 2.9688 ms 18.3%
SingleProcess AUTOTUNE takes 3.7946 seconds
AUTOTUNE bmm(64x384x384, 64x384x64)
  bmm 0.1754 ms 100.0%
  triton_bmm_1519 0.1809 ms 96.9%
  triton_bmm_1518 0.1875 ms 93.5%
  triton_bmm_1516 0.2010 ms 87.2%
  triton_bmm_1517 0.2253 ms 77.8%
  triton_bmm_1515 1.0175 ms 17.2%
SingleProcess AUTOTUNE takes 0.6624 seconds
AUTOTUNE bmm(64x384x64, 64x64x384)
  bmm 0.2072 ms 100.0%
  triton_bmm_1523 0.3371 ms 61.5%
  triton_bmm_1524 0.3802 ms 54.5%
  triton_bmm_1521 0.7352 ms 28.2%
  triton_bmm_1520 0.9796 ms 21.2%
  triton_bmm_1522 1.1627 ms 17.8%
SingleProcess AUTOTUNE takes 3.6905 seconds
AUTOTUNE bmm(1536x2x384, 1536x384x16)
  triton_bmm_1528 0.1371 ms 100.0%
  triton_bmm_1527 0.1378 ms 99.6%
  bmm 0.1585 ms 86.5%
  triton_bmm_1525 0.1837 ms 74.6%
  triton_bmm_1526 0.2538 ms 54.0%
SingleProcess AUTOTUNE takes 2.6257 seconds
AUTOTUNE bmm(16x64x1536, 16x1536x2)
  triton_bmm_1533 0.0838 ms 100.0%
  triton_bmm_1532 0.0842 ms 99.5%
  bmm 0.0848 ms 98.8%
  triton_bmm_1529 0.1110 ms 75.5%
  triton_bmm_1530 0.1249 ms 67.1%
  triton_bmm_1531 0.1357 ms 61.7%
SingleProcess AUTOTUNE takes 3.2270 seconds
AUTOTUNE bmm(16x1536x2, 16x2x64)
  bmm 0.0154 ms 100.0%
  triton_bmm_1534 0.0178 ms 86.5%
  triton_bmm_1536 0.0185 ms 83.2%
  triton_bmm_1535 0.0192 ms 80.0%
  triton_bmm_1537 0.0206 ms 74.9%
SingleProcess AUTOTUNE takes 2.0244 seconds
AUTOTUNE bmm(64x64x384, 64x384x768)
  triton_bmm_1542 0.3433 ms 100.0%
  triton_bmm_1539 0.3849 ms 89.2%
  bmm 0.3876 ms 88.6%
  triton_bmm_1541 0.4044 ms 84.9%
  triton_bmm_1540 0.4232 ms 81.1%
  triton_bmm_1538 1.8260 ms 18.8%
SingleProcess AUTOTUNE takes 5.9265 seconds
AUTOTUNE bmm(64x384x768, 64x768x64)
  bmm 0.4876 ms 100.0%
  triton_bmm_1545 0.5036 ms 96.8%
  triton_bmm_1544 0.5825 ms 83.7%
  triton_bmm_1547 0.9539 ms 51.1%
  triton_bmm_1546 1.1839 ms 41.2%
  triton_bmm_1543 2.2941 ms 21.3%
SingleProcess AUTOTUNE takes 5.5045 seconds
AUTOTUNE bmm(64x64x384, 64x384x384)
  triton_bmm_1552 0.1642 ms 100.0%
  bmm 0.1820 ms 90.2%
  triton_bmm_1549 0.1963 ms 83.6%
  triton_bmm_1550 0.2150 ms 76.4%
  triton_bmm_1551 0.2194 ms 74.8%
  triton_bmm_1548 0.9367 ms 17.5%
SingleProcess AUTOTUNE takes 3.9424 seconds
AUTOTUNE bmm(1x1024x3072, 1x3072x1024)
  triton_bmm_1562 0.6524 ms 100.0%
  bmm 0.8499 ms 76.8%
  triton_bmm_1559 0.9288 ms 70.2%
  triton_bmm_1560 1.0627 ms 61.4%
  triton_bmm_1561 1.0774 ms 60.6%
  triton_bmm_1558 6.6568 ms 9.8%


### Fine-tuning T5 on SQuAD2.0

The [`run_seq2seq_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py) script is meant for encoder-decoder (also called seq2seq) Transformer models, such as T5 or BART. These
models are generative, rather than discriminative. This means that they learn to generate the correct answer, rather than predicting the start and end position of the tokens of the answer.

This example code fine-tunes T5 on the SQuAD2.0 dataset.

```bash
TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --dataset_name squad_v2 \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_seq2seq_squad/ \
  --torch_compile True
```

PT2.0 results NV31 (single gpu) + Ryzen 7 5700x3d - train/train_steps_per_second - t5-small:
| GPU       | Run Mode                          | t5-smaal               | 
|-----------|-----------------------------------|------------------------|
| 7900 XTX  | Regular execution                 | 6.22  (~58mins)        | 
| 7900 XTX  | compile - autotune cached         | 7.95  (~45mins)        | 

Autotune logs:
```
AUTOTUNE mm(4608x512, 512x512)
  mm 0.3998 ms 100.0%
  triton_mm_3 0.6878 ms 58.1%
  triton_mm_4 0.7866 ms 50.8%
  triton_mm_1 1.3153 ms 30.4%
  triton_mm_0 2.1245 ms 18.8%
  triton_mm_2 2.1793 ms 18.3%
SingleProcess AUTOTUNE takes 3.8413 seconds
AUTOTUNE bmm(96x384x64, 96x64x384)
  triton_bmm_18 0.2315 ms 100.0%
  bmm 0.2609 ms 88.7%
  triton_bmm_19 0.2987 ms 77.5%
  triton_bmm_17 0.3996 ms 57.9%
  triton_bmm_16 0.4170 ms 55.5%
  triton_bmm_15 1.0656 ms 21.7%
SingleProcess AUTOTUNE takes 0.6934 seconds
AUTOTUNE bmm(96x384x384, 96x384x64)
  bmm 0.3350 ms 100.0%
  triton_bmm_22 0.3652 ms 91.7%
  triton_bmm_24 0.3814 ms 87.8%
  triton_bmm_21 0.4103 ms 81.6%
  triton_bmm_23 0.5819 ms 57.6%
  triton_bmm_20 1.5329 ms 21.9%
SingleProcess AUTOTUNE takes 0.6821 seconds
AUTOTUNE mm(4608x512, 512x2048)
  mm 1.6369 ms 100.0%
  triton_mm_33 2.5082 ms 65.3%
  triton_mm_34 2.9331 ms 55.8%
  triton_mm_31 5.0822 ms 32.2%
  triton_mm_30 7.3760 ms 22.2%
  triton_mm_32 8.5078 ms 19.2%
SingleProcess AUTOTUNE takes 5.0160 seconds
AUTOTUNE mm(4608x2048, 2048x512)
  mm 1.6513 ms 100.0%
  triton_mm_38 2.6679 ms 61.9%
  triton_mm_39 3.0392 ms 54.3%
  triton_mm_36 5.1389 ms 32.1%
  triton_mm_35 8.2687 ms 20.0%
  triton_mm_37 8.5877 ms 19.2%
SingleProcess AUTOTUNE takes 4.2146 seconds
AUTOTUNE mm(360x512, 512x512)
  mm 0.0436 ms 100.0%
  triton_mm_244 0.0841 ms 51.8%
  triton_mm_243 0.0862 ms 50.5%
  triton_mm_241 0.1215 ms 35.9%
  triton_mm_242 0.1923 ms 22.7%
  triton_mm_240 0.4724 ms 9.2%
SingleProcess AUTOTUNE takes 3.7816 seconds
AUTOTUNE bmm(96x30x64, 96x64x30)
  triton_bmm_258 0.0120 ms 100.0%
  triton_bmm_257 0.0122 ms 99.0%
  triton_bmm_255 0.0125 ms 96.5%
  bmm 0.0126 ms 95.3%
  triton_bmm_256 0.0142 ms 84.6%
SingleProcess AUTOTUNE takes 2.2802 seconds
AUTOTUNE bmm(96x30x30, 96x30x64)
  triton_bmm_260 0.0103 ms 100.0%
  triton_bmm_259 0.0105 ms 98.1%
  triton_bmm_261 0.0105 ms 98.1%
  triton_bmm_263 0.0107 ms 96.3%
  bmm 0.0110 ms 93.8%
  triton_bmm_262 0.0110 ms 93.8%
SingleProcess AUTOTUNE takes 2.7477 seconds
AUTOTUNE bmm(96x30x64, 96x64x384)
  triton_bmm_287 0.0459 ms 100.0%
  bmm 0.0511 ms 89.8%
  triton_bmm_286 0.0523 ms 87.7%
  triton_bmm_288 0.0579 ms 79.2%
  triton_bmm_285 0.0596 ms 76.9%
  triton_bmm_284 0.1048 ms 43.8%
SingleProcess AUTOTUNE takes 3.1288 seconds
AUTOTUNE bmm(96x30x384, 96x384x64)
  triton_bmm_291 0.0842 ms 100.0%
  bmm 0.0888 ms 94.8%
  triton_bmm_292 0.0896 ms 94.0%
  triton_bmm_293 0.0906 ms 92.9%
  triton_bmm_290 0.0991 ms 84.9%
  triton_bmm_289 0.1389 ms 60.6%
SingleProcess AUTOTUNE takes 3.2934 seconds
AUTOTUNE mm(360x512, 512x2048)
  mm 0.1369 ms 100.0%
  triton_mm_302 0.2315 ms 59.1%
  triton_mm_303 0.2932 ms 46.7%
  triton_mm_300 0.4813 ms 28.4%
  triton_mm_299 0.6762 ms 20.2%
  triton_mm_301 0.7994 ms 17.1%
SingleProcess AUTOTUNE takes 3.7677 seconds
AUTOTUNE mm(360x2048, 2048x512)
  mm 0.1551 ms 100.0%
  triton_mm_308 0.2906 ms 53.4%
  triton_mm_307 0.3103 ms 50.0%
  triton_mm_305 0.4374 ms 35.5%
  triton_mm_306 0.7440 ms 20.8%
  triton_mm_304 1.8538 ms 8.4%
SingleProcess AUTOTUNE takes 3.8193 seconds
AUTOTUNE mm(360x512, 512x32128)
  mm 2.0872 ms 100.0%
  triton_mm_657 3.2367 ms 64.5%
  triton_mm_658 3.8023 ms 54.9%
  triton_mm_655 6.5375 ms 31.9%
  triton_mm_654 9.5139 ms 21.9%
  triton_mm_656 10.9486 ms 19.1%
SingleProcess AUTOTUNE takes 4.0203 seconds
AUTOTUNE mm(32128x360, 360x512)
  triton_mm_663 1.0892 ms 100.0%
  triton_mm_662 1.3602 ms 80.1%
  mm 1.4847 ms 73.4%
  triton_mm_660 1.8621 ms 58.5%
  triton_mm_661 2.1440 ms 50.8%
  triton_mm_659 10.3495 ms 10.5%
SingleProcess AUTOTUNE takes 3.9794 seconds
AUTOTUNE mm(360x32128, 32128x512)
  triton_mm_666 2.1359 ms 100.0%
  triton_mm_665 2.2904 ms 93.3%
  triton_mm_668 2.3628 ms 90.4%
  mm 2.5051 ms 85.3%
  triton_mm_667 3.4299 ms 62.3%
  triton_mm_664 24.0282 ms 8.9%
SingleProcess AUTOTUNE takes 7.6373 seconds
AUTOTUNE mm(512x360, 360x2048)
  triton_mm_673 0.0854 ms 100.0%
  mm 0.0965 ms 88.4%
  triton_mm_670 0.1221 ms 69.9%
  triton_mm_672 0.1286 ms 66.4%
  triton_mm_671 0.1399 ms 61.0%
  triton_mm_669 0.9815 ms 8.7%
SingleProcess AUTOTUNE takes 3.9067 seconds
UTOTUNE mm(360x512, 512x2048)
  triton_mm_677 0.1062 ms 100.0%
  mm 0.1208 ms 87.9%
  triton_mm_678 0.1444 ms 73.6%
  triton_mm_676 0.1473 ms 72.1%
  triton_mm_675 0.1630 ms 65.2%
  triton_mm_674 0.5439 ms 19.5%
SingleProcess AUTOTUNE takes 3.8833 seconds
AUTOTUNE mm(2048x360, 360x512)
  triton_mm_683 0.0761 ms 100.0%
  mm 0.0955 ms 79.6%
  triton_mm_680 0.1218 ms 62.5%
  triton_mm_682 0.1270 ms 59.9%
  triton_mm_681 0.1403 ms 54.2%
  triton_mm_679 0.9819 ms 7.7%
SingleProcess AUTOTUNE takes 3.9100 seconds
AUTOTUNE mm(360x2048, 2048x512)
  triton_mm_686 0.1327 ms 100.0%
  triton_mm_685 0.1474 ms 90.0%
  mm 0.1483 ms 89.5%
  triton_mm_688 0.1539 ms 86.2%
  triton_mm_687 0.2275 ms 58.3%
  triton_mm_684 1.5506 ms 8.6%
SingleProcess AUTOTUNE takes 4.5411 seconds
AUTOTUNE mm(512x360, 360x512)
  triton_mm_693 0.0288 ms 100.0%
  mm 0.0327 ms 88.1%
  triton_mm_690 0.0435 ms 66.2%
  triton_mm_691 0.0496 ms 58.0%
  triton_mm_692 0.0596 ms 48.3%
  triton_mm_689 0.4485 ms 6.4%
SingleProcess AUTOTUNE takes 4.5774 seconds
AUTOTUNE mm(360x512, 512x512)
  triton_mm_696 0.0400 ms 100.0%
  mm 0.0410 ms 97.4%
  triton_mm_695 0.0436 ms 91.7%
  triton_mm_698 0.0441 ms 90.7%
  triton_mm_697 0.0620 ms 64.5%
  triton_mm_694 0.3950 ms 10.1%
SingleProcess AUTOTUNE takes 3.8263 seconds
AUTOTUNE bmm(96x384x30, 96x30x64)
  triton_bmm_700 0.0344 ms 100.0%
  bmm 0.0353 ms 97.6%
  triton_bmm_701 0.0361 ms 95.3%
  triton_bmm_703 0.0373 ms 92.3%
  triton_bmm_702 0.0457 ms 75.4%
  triton_bmm_699 0.0521 ms 66.1%
SingleProcess AUTOTUNE takes 3.1944 seconds
AUTOTUNE bmm(96x30x64, 96x64x384)
  bmm 0.0485 ms 100.0%
  triton_bmm_708 0.0751 ms 64.6%
  triton_bmm_707 0.0882 ms 55.0%
  triton_bmm_705 0.1091 ms 44.5%
  triton_bmm_704 0.1207 ms 40.2%
  triton_bmm_706 0.1617 ms 30.0%
SingleProcess AUTOTUNE takes 3.1255 seconds
AUTOTUNE bmm(96x64x30, 96x30x384)
  triton_bmm_710 0.0352 ms 100.0%
  bmm 0.0356 ms 98.9%
  triton_bmm_711 0.0360 ms 97.7%
  triton_bmm_713 0.0368 ms 95.5%
  triton_bmm_712 0.0431 ms 81.6%
  triton_bmm_709 0.0566 ms 62.2%
SingleProcess AUTOTUNE takes 3.3289 seconds
AUTOTUNE bmm(96x30x384, 96x384x64)
  bmm 0.0865 ms 100.0%
  triton_bmm_718 0.1031 ms 83.9%
  triton_bmm_717 0.1211 ms 71.5%
  triton_bmm_715 0.1530 ms 56.5%
  triton_bmm_716 0.1802 ms 48.0%
  triton_bmm_714 0.2066 ms 41.9%
SingleProcess AUTOTUNE takes 3.2028 seconds
AUTOTUNE mm(512x4608, 4608x512)
  triton_mm_723 0.2927 ms 100.0%
  mm 0.3616 ms 80.9%
  triton_mm_720 0.4474 ms 65.4%
  triton_mm_721 0.5074 ms 57.7%
  triton_mm_722 0.6406 ms 45.7%
  triton_mm_719 4.9179 ms 6.0%
SingleProcess AUTOTUNE takes 3.9245 seconds
AUTOTUNE mm(4608x512, 512x512)
  triton_mm_727 0.3251 ms 100.0%
  mm 0.3515 ms 92.5%
  triton_mm_728 0.3764 ms 86.4%
  triton_mm_726 0.4051 ms 80.3%
  triton_mm_725 0.4474 ms 72.7%
  triton_mm_724 1.8677 ms 17.4%
SingleProcess AUTOTUNE takes 3.9076 seconds
AUTOTUNE bmm(96x30x30, 96x30x64)
  bmm 0.0104 ms 100.0%
  triton_bmm_763 0.0104 ms 99.2%
  triton_bmm_759 0.0106 ms 97.4%
  triton_bmm_760 0.0107 ms 97.0%
  triton_bmm_761 0.0107 ms 96.6%
  triton_bmm_762 0.0110 ms 94.2%
SingleProcess AUTOTUNE takes 2.8626 seconds
AUTOTUNE bmm(96x30x64, 96x64x30)
  bmm 0.0132 ms 100.0%
  triton_bmm_767 0.0139 ms 95.4%
  triton_bmm_764 0.0167 ms 79.4%
  triton_bmm_766 0.0169 ms 78.3%
  triton_bmm_765 0.0190 ms 69.8%
SingleProcess AUTOTUNE takes 2.8175 seconds
AUTOTUNE bmm(96x64x30, 96x30x30)
  bmm 0.0104 ms 100.0%
  triton_bmm_768 0.0104 ms 100.0%
  triton_bmm_770 0.0106 ms 97.7%
  triton_bmm_769 0.0107 ms 97.4%
  triton_bmm_772 0.0108 ms 96.7%
  triton_bmm_771 0.0108 ms 96.3%
SingleProcess AUTOTUNE takes 2.7896 seconds
AUTOTUNE bmm(96x30x30, 96x30x64)
  bmm 0.0114 ms 100.0%
  triton_bmm_777 0.0133 ms 86.2%
  triton_bmm_774 0.0150 ms 76.1%
  triton_bmm_773 0.0151 ms 75.7%
  triton_bmm_776 0.0159 ms 71.9%
  triton_bmm_775 0.0209 ms 54.8%
SingleProcess AUTOTUNE takes 2.7385 seconds
AUTOTUNE mm(512x4608, 4608x2048)
  triton_mm_1507 0.9810 ms 100.0%
  mm 1.2692 ms 77.3%
  triton_mm_1504 1.4031 ms 69.9%
  triton_mm_1505 1.5822 ms 62.0%
  triton_mm_1506 1.7190 ms 57.1%
  triton_mm_1503 10.0409 ms 9.8%
SingleProcess AUTOTUNE takes 4.0302 seconds
AUTOTUNE mm(4608x512, 512x2048)
  triton_mm_1511 1.1199 ms 100.0%
  mm 1.4340 ms 78.1%
  triton_mm_1512 1.4939 ms 75.0%
  triton_mm_1510 1.5976 ms 70.1%
  triton_mm_1509 1.7574 ms 63.7%
  triton_mm_1508 5.9124 ms 18.9%
SingleProcess AUTOTUNE takes 4.0225 seconds
AUTOTUNE mm(2048x4608, 4608x512)
  triton_mm_1517 0.9685 ms 100.0%
  mm 1.2847 ms 75.4%
  triton_mm_1514 1.4106 ms 68.7%
  triton_mm_1515 1.5911 ms 60.9%
  triton_mm_1516 1.6862 ms 57.4%
  triton_mm_1513 10.0231 ms 9.7%
SingleProcess AUTOTUNE takes 4.0613 seconds
AUTOTUNE mm(4608x2048, 2048x512)
  triton_mm_1521 1.2224 ms 100.0%
  mm 1.4520 ms 84.2%
  triton_mm_1522 1.5081 ms 81.1%
  triton_mm_1520 1.5954 ms 76.6%
  triton_mm_1519 1.8012 ms 67.9%
  triton_mm_1518 6.9769 ms 17.5%
SingleProcess AUTOTUNE takes 9.6635 seconds
AUTOTUNE bmm(96x384x384, 96x384x64)
  triton_bmm_1536 0.2688 ms 100.0%
  bmm 0.2789 ms 96.4%
  triton_bmm_1537 0.2877 ms 93.4%
  triton_bmm_1534 0.2958 ms 90.9%
  triton_bmm_1535 0.3225 ms 83.4%
  triton_bmm_1533 1.5244 ms 17.6%
SingleProcess AUTOTUNE takes 5.7445 seconds
AUTOTUNE bmm(96x384x64, 96x64x384)
  bmm 0.2997 ms 100.0%
  triton_bmm_1541 0.4928 ms 60.8%
  triton_bmm_1542 0.5775 ms 51.9%
  triton_bmm_1539 1.0660 ms 28.1%
  triton_bmm_1538 1.4019 ms 21.4%
  triton_bmm_1540 1.6969 ms 17.7%
SingleProcess AUTOTUNE takes 3.0645 seconds
AUTOTUNE bmm(96x64x384, 96x384x384)
  triton_bmm_1547 0.2686 ms 100.0%
  triton_bmm_1544 0.2862 ms 93.9%
  bmm 0.2993 ms 89.7%
  triton_bmm_1546 0.3045 ms 88.2%
  triton_bmm_1545 0.3182 ms 84.4%
  triton_bmm_1543 1.3785 ms 19.5%
SingleProcess AUTOTUNE takes 0.6685 seconds
AUTOTUNE bmm(96x384x384, 96x384x64)
  bmm 0.3264 ms 100.0%
  triton_bmm_1552 0.5952 ms 54.8%
  triton_bmm_1551 0.6351 ms 51.4%
  triton_bmm_1549 1.0006 ms 32.6%
  triton_bmm_1548 1.4494 ms 22.5%
  triton_bmm_1550 1.6355 ms 20.0%
SingleProcess AUTOTUNE takes 0.6520 seconds
```




