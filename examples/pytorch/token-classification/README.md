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

# Token classification

## PyTorch version

Fine-tuning the library models for token classification task such as Named Entity Recognition (NER), Parts-of-speech
tagging (POS) or phrase extraction (CHUNKS). The main scrip `run_ner.py` leverages the 🤗 Datasets library and the Trainer API. You can easily
customize it to your needs if you need extra processing on your datasets.

It will either run on a datasets hosted on our [hub](https://huggingface.co/datasets) or with your own text files for
training and validation, you might just need to add some tweaks in the data preprocessing.

The following example fine-tunes BERT on CoNLL-2003:

```bash
TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1  python run_ner.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name conll2003 \
  --output_dir /tmp/test-ner \
  --do_train \
  --do_eval \
  --torch_compile True
```

PT2.0 results NV31 (single gpu) + Ryzen 7 5700x3d - train/train_steps_per_second - bert-base-uncased:
| GPU       | Run Mode                          | train_steps_per_second | train_runtime | eval_steps_per_second | eval_runtime|
|-----------|-----------------------------------|------------------------|---------------|-----------------------|-------------|
| 7900 XTX  | Regular execution                 | 18.341                 | 287.22        | 53.669                | 7.5793
| 7900 XTX  | compile - autotune cached         | 19.626                 | 268.4173     | 68.295                | 5.9595


## Bloom-560m

```bash
TORCHDYNAMO_DYNAMIC_SHAPES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1  python run_ner.py \
  --model_name_or_path bigscience/bloom-560m \
  --dataset_name conll2003 \
  --output_dir /tmp/test-ner \
  --do_train \
  --do_eval \
  --torch_compile True
```

PT2.0 results NV31 (single gpu) + Ryzen 7 5700x3d - train/train_steps_per_second - bloom-560m:

Seems like there may be some implicit recompilation here, and autotuned kernels don't seem to be caching as each run reruns the autotuning.

| GPU       | Run Mode                          | train_steps_per_second | train_runtime | eval_steps_per_second | eval_runtime|
|-----------|-----------------------------------|------------------------|---------------|-----------------------|-------------|
| 7900 XTX  | Regular execution                 | 4.105                  | 1283.107      | 19.364                | 21.0189
| 7900 XTX  | compile - autotune cached         | 4.001                  | 1316.67       | 21.556                | 18.8719
Autotune logs:
```
AUTOTUNE addmm(472x3072, 472x1024, 1024x3072)
  addmm 0.5008 ms 100.0%
  bias_addmm 0.5150 ms 97.2%
  triton_mm_3 0.9576 ms 52.3%
  triton_mm_4 1.0222 ms 49.0%
  triton_mm_1 1.7468 ms 28.7%
  triton_mm_2 2.7361 ms 18.3%
  triton_mm_0 2.9891 ms 16.8%
SingleProcess AUTOTUNE takes 4.9711 seconds
AUTOTUNE bmm(128x59x64, 128x64x59)
  bmm 0.0220 ms 100.0%
  triton_bmm_9 0.0236 ms 93.1%
  triton_bmm_8 0.0263 ms 83.4%
  triton_bmm_7 0.0294 ms 74.8%
  triton_bmm_6 0.0308 ms 71.3%
  triton_bmm_5 0.0984 ms 22.3%
SingleProcess AUTOTUNE takes 9.9554 seconds
AUTOTUNE bmm(128x59x59, 128x59x64)
  bmm 0.0221 ms 100.0%
  triton_bmm_14 0.0238 ms 92.9%
  triton_bmm_12 0.0286 ms 77.2%
  triton_bmm_11 0.0296 ms 74.7%
  triton_bmm_13 0.0322 ms 68.7%
  triton_bmm_10 0.1037 ms 21.3%
SingleProcess AUTOTUNE takes 3.7754 seconds
AUTOTUNE addmm(472x1024, 472x1024, 1024x1024)
  addmm 0.1755 ms 100.0%
  bias_addmm 0.1793 ms 97.9%
  triton_mm_18 0.3380 ms 51.9%
  triton_mm_19 0.3553 ms 49.4%
  triton_mm_16 0.6524 ms 26.9%
  triton_mm_17 1.0496 ms 16.7%
  triton_mm_15 1.1965 ms 14.7%
SingleProcess AUTOTUNE takes 4.0268 seconds
AUTOTUNE addmm(472x4096, 472x1024, 1024x4096)
  addmm 0.6593 ms 100.0%
  bias_addmm 0.6830 ms 96.5%
  triton_mm_23 1.1842 ms 55.7%
  triton_mm_24 1.3222 ms 49.9%
  triton_mm_21 2.3112 ms 28.5%
  triton_mm_20 3.5019 ms 18.8%
  triton_mm_22 3.6511 ms 18.1%
SingleProcess AUTOTUNE takes 4.0681 seconds
AUTOTUNE addmm(472x1024, 472x4096, 4096x1024)
  addmm 0.6903 ms 100.0%
  bias_addmm 0.7312 ms 94.4%
  triton_mm_28 1.3190 ms 52.3%
  triton_mm_29 1.3852 ms 49.8%
  triton_mm_26 2.5188 ms 27.4%
  triton_mm_27 4.0522 ms 17.0%
  triton_mm_25 4.6105 ms 15.0%
SingleProcess AUTOTUNE takes 4.0119 seconds
AUTOTUNE addmm(472x9, 472x1024, 1024x9)
  addmm 0.0503 ms 100.0%
  bias_addmm 0.0508 ms 99.1%
  triton_mm_724 0.0535 ms 94.1%
  triton_mm_722 0.0547 ms 91.9%
  triton_mm_721 0.0709 ms 71.0%
  triton_mm_720 0.0722 ms 69.7%
  triton_mm_723 0.0749 ms 67.2%
SingleProcess AUTOTUNE takes 3.1190 seconds
AUTOTUNE mm(472x9, 9x1024)
  mm 0.0100 ms 100.0%
  triton_mm_727 0.0106 ms 95.1%
  triton_mm_728 0.0109 ms 91.9%
  triton_mm_725 0.0111 ms 90.6%
  triton_mm_726 0.0117 ms 86.0%
SingleProcess AUTOTUNE takes 2.4693 seconds
AUTOTUNE mm(9x472, 472x1024)
  triton_mm_733 0.0281 ms 100.0%
  mm 0.0294 ms 95.6%
  triton_mm_732 0.0300 ms 93.9%
  triton_mm_729 0.0339 ms 83.0%
  triton_mm_731 0.0436 ms 64.5%
  triton_mm_730 0.0480 ms 58.6%
SingleProcess AUTOTUNE takes 3.1595 seconds
AUTOTUNE mm(472x1024, 1024x4096)
  triton_mm_737 0.5328 ms 100.0%
  mm 0.5847 ms 91.1%
  triton_mm_738 0.6493 ms 82.1%
  triton_mm_736 0.6777 ms 78.6%
  triton_mm_735 0.7914 ms 67.3%
  triton_mm_734 2.8908 ms 18.4%
SingleProcess AUTOTUNE takes 3.7689 seconds
AUTOTUNE mm(1024x472, 472x4096)
  triton_mm_743 0.3497 ms 100.0%
  triton_mm_742 0.4446 ms 78.7%
  mm 0.4977 ms 70.3%
  triton_mm_740 0.5836 ms 59.9%
  triton_mm_741 0.6701 ms 52.2%
  triton_mm_739 3.5486 ms 9.9%
SingleProcess AUTOTUNE takes 3.7960 seconds
AUTOTUNE mm(472x4096, 4096x1024)
  triton_mm_747 0.6277 ms 100.0%
  mm 0.6333 ms 99.1%
  triton_mm_748 0.7029 ms 89.3%
  triton_mm_746 0.7348 ms 85.4%
  triton_mm_745 0.8548 ms 73.4%
  triton_mm_744 3.6035 ms 17.4%
SingleProcess AUTOTUNE takes 4.3080 seconds
AUTOTUNE mm(4096x472, 472x1024)
  triton_mm_753 0.3500 ms 100.0%
  triton_mm_752 0.4485 ms 78.0%
  mm 0.4777 ms 73.3%
  triton_mm_750 0.5850 ms 59.8%
  triton_mm_751 0.6688 ms 52.3%
  triton_mm_749 3.5489 ms 9.9%
SingleProcess AUTOTUNE takes 3.8799 seconds
AUTOTUNE mm(472x1024, 1024x1024)
  mm 0.1547 ms 100.0%
  triton_mm_757 0.1622 ms 95.4%
  triton_mm_758 0.1792 ms 86.3%
  triton_mm_756 0.1852 ms 83.5%
  triton_mm_755 0.2148 ms 72.0%
  triton_mm_754 0.9423 ms 16.4%
SingleProcess AUTOTUNE takes 3.8025 seconds
AUTOTUNE mm(1024x472, 472x1024)
  triton_mm_763 0.1035 ms 100.0%
  mm 0.1238 ms 83.6%
  triton_mm_760 0.1451 ms 71.3%
  triton_mm_762 0.1588 ms 65.2%
  triton_mm_761 0.1708 ms 60.6%
  triton_mm_759 1.1892 ms 8.7%
SingleProcess AUTOTUNE takes 3.8005 seconds
AUTOTUNE bmm(128x59x59, 128x59x64)
  bmm 0.0197 ms 100.0%
  triton_bmm_768 0.0217 ms 90.6%
  triton_bmm_765 0.0260 ms 75.6%
  triton_bmm_767 0.0277 ms 71.1%
  triton_bmm_766 0.0282 ms 69.7%
  triton_bmm_764 0.1080 ms 18.2%
SingleProcess AUTOTUNE takes 4.8974 seconds
AUTOTUNE bmm(128x59x64, 128x64x59)
  bmm 0.0237 ms 100.0%
  triton_bmm_773 0.0348 ms 68.0%
  triton_bmm_772 0.0357 ms 66.4%
  triton_bmm_770 0.0577 ms 41.1%
  triton_bmm_771 0.0870 ms 27.2%
  triton_bmm_769 0.1205 ms 19.6%
SingleProcess AUTOTUNE takes 3.6885 seconds
AUTOTUNE bmm(128x59x59, 128x59x64)
  bmm 0.0255 ms 100.0%
  triton_bmm_778 0.0366 ms 69.7%
  triton_bmm_777 0.0416 ms 61.3%
  triton_bmm_775 0.0546 ms 46.7%
  triton_bmm_776 0.0849 ms 30.1%
  triton_bmm_774 0.1060 ms 24.1%
SingleProcess AUTOTUNE takes 3.8264 seconds
AUTOTUNE bmm(128x64x59, 128x59x59)
  bmm 0.0200 ms 100.0%
  triton_bmm_783 0.0218 ms 91.4%
  triton_bmm_780 0.0277 ms 72.1%
  triton_bmm_781 0.0288 ms 69.3%
  triton_bmm_782 0.0303 ms 65.9%
  triton_bmm_779 0.1239 ms 16.1%
SingleProcess AUTOTUNE takes 3.8845 seconds
AUTOTUNE mm(472x3072, 3072x1024)
  triton_mm_787 0.4683 ms 100.0%
  mm 0.4689 ms 99.9%
  triton_mm_788 0.5226 ms 89.6%
  triton_mm_786 0.5495 ms 85.2%
  triton_mm_785 0.6484 ms 72.2%
  triton_mm_784 2.7982 ms 16.7%
SingleProcess AUTOTUNE takes 3.8100 seconds
AUTOTUNE mm(3072x472, 472x1024)
  triton_mm_793 0.2628 ms 100.0%
  triton_mm_792 0.3431 ms 76.6%
  mm 0.3716 ms 70.7%
  triton_mm_790 0.4366 ms 60.2%
  triton_mm_791 0.5058 ms 52.0%
  triton_mm_789 2.7295 ms 9.6%
SingleProcess AUTOTUNE takes 3.8164 seconds
AUTOTUNE addmm(176x3072, 176x1024, 1024x3072)
  addmm 0.2109 ms 100.0%
  bias_addmm 0.2129 ms 99.0%
  triton_mm_2177 0.3508 ms 60.1%
  triton_mm_2178 0.4278 ms 49.3%
  triton_mm_2175 0.7404 ms 28.5%
  triton_mm_2174 1.2031 ms 17.5%
  triton_mm_2176 1.2298 ms 17.1%
SingleProcess AUTOTUNE takes 3.8816 seconds
AUTOTUNE bmm(128x22x64, 128x64x22)
  triton_bmm_2181 0.0125 ms 100.0%
  triton_bmm_2182 0.0128 ms 97.8%
  triton_bmm_2179 0.0130 ms 96.0%
  bmm 0.0132 ms 94.8%
  triton_bmm_2180 0.0143 ms 87.2%
SingleProcess AUTOTUNE takes 2.3514 seconds
AUTOTUNE bmm(128x22x22, 128x22x64)
  bmm 0.0103 ms 100.0%
  triton_bmm_2186 0.0113 ms 91.3%
  triton_bmm_2187 0.0113 ms 91.1%
  triton_bmm_2185 0.0136 ms 75.6%
  triton_bmm_2184 0.0166 ms 61.8%
  triton_bmm_2183 0.0235 ms 43.7%
SingleProcess AUTOTUNE takes 3.1824 seconds
AUTOTUNE addmm(176x1024, 176x1024, 1024x1024)
  addmm 0.0878 ms 100.0%
  bias_addmm 0.0879 ms 99.8%
  triton_mm_2192 0.1501 ms 58.5%
  triton_mm_2191 0.1607 ms 54.6%
  triton_mm_2189 0.2278 ms 38.5%
  triton_mm_2190 0.3688 ms 23.8%
  triton_mm_2188 0.8557 ms 10.3%
SingleProcess AUTOTUNE takes 3.9540 seconds
AUTOTUNE addmm(176x4096, 176x1024, 1024x4096)
  addmm 0.2720 ms 100.0%
  bias_addmm 0.2783 ms 97.7%
  triton_mm_2196 0.4601 ms 59.1%
  triton_mm_2197 0.5748 ms 47.3%
  triton_mm_2194 0.9662 ms 28.1%
  triton_mm_2193 1.3107 ms 20.7%
  triton_mm_2195 1.6076 ms 16.9%
SingleProcess AUTOTUNE takes 3.9422 seconds
AUTOTUNE addmm(176x1024, 176x4096, 4096x1024)
  addmm 0.3057 ms 100.0%
  bias_addmm 0.3162 ms 96.7%
  triton_mm_2202 0.5683 ms 53.8%
  triton_mm_2201 0.6137 ms 49.8%
  triton_mm_2199 0.8591 ms 35.6%
  triton_mm_2200 1.4136 ms 21.6%
  triton_mm_2198 3.3656 ms 9.1%
SingleProcess AUTOTUNE takes 3.9058 seconds
AUTOTUNE addmm(176x9, 176x1024, 1024x9)
  bias_addmm 0.0471 ms 100.0%
  addmm 0.0474 ms 99.4%
  triton_mm_2874 0.0536 ms 87.8%
  triton_mm_2872 0.0556 ms 84.6%
  triton_mm_2871 0.0710 ms 66.3%
  triton_mm_2870 0.0730 ms 64.5%
  triton_mm_2873 0.0734 ms 64.2%
SingleProcess AUTOTUNE takes 3.1710 seconds
AUTOTUNE mm(176x9, 9x1024)
  triton_mm_2877 0.0082 ms 100.0%
  triton_mm_2876 0.0086 ms 96.3%
  triton_mm_2878 0.0086 ms 96.3%
  triton_mm_2875 0.0087 ms 94.9%
  mm 0.0094 ms 87.7%
SingleProcess AUTOTUNE takes 1.9345 seconds
AUTOTUNE mm(9x176, 176x1024)
  triton_mm_2883 0.0146 ms 100.0%
  mm 0.0155 ms 94.6%
  triton_mm_2882 0.0158 ms 92.4%
  triton_mm_2879 0.0183 ms 80.1%
  triton_mm_2881 0.0212 ms 68.9%
  triton_mm_2880 0.0318 ms 46.0%
SingleProcess AUTOTUNE takes 2.9741 seconds
AUTOTUNE mm(176x1024, 1024x4096)
  triton_mm_2887 0.1963 ms 100.0%
  mm 0.2384 ms 82.3%
  triton_mm_2888 0.2783 ms 70.5%
  triton_mm_2886 0.2848 ms 68.9%
  triton_mm_2885 0.3160 ms 62.1%
  triton_mm_2884 1.0522 ms 18.7%
SingleProcess AUTOTUNE takes 3.7205 seconds
AUTOTUNE mm(1024x176, 176x4096)
  triton_mm_2893 0.1317 ms 100.0%
  mm 0.1817 ms 72.5%
  triton_mm_2892 0.1880 ms 70.0%
  triton_mm_2890 0.2532 ms 52.0%
  triton_mm_2891 0.2944 ms 44.7%
  triton_mm_2889 1.4495 ms 9.1%
SingleProcess AUTOTUNE takes 3.6525 seconds
AUTOTUNE mm(176x4096, 4096x1024)
  triton_mm_2896 0.2590 ms 100.0%
  triton_mm_2895 0.2888 ms 89.7%
  triton_mm_2898 0.2985 ms 86.7%
  mm 0.3111 ms 83.2%
  triton_mm_2897 0.4480 ms 57.8%
  triton_mm_2894 2.7469 ms 9.4%
SingleProcess AUTOTUNE takes 3.7447 seconds
AUTOTUNE mm(4096x176, 176x1024)
  triton_mm_2903 0.1319 ms 100.0%
  mm 0.1796 ms 73.4%
  triton_mm_2902 0.1886 ms 69.9%
  triton_mm_2900 0.2500 ms 52.8%
  triton_mm_2901 0.2941 ms 44.9%
  triton_mm_2899 1.4505 ms 9.1%
SingleProcess AUTOTUNE takes 3.7311 seconds
AUTOTUNE mm(176x1024, 1024x1024)
  triton_mm_2906 0.0679 ms 100.0%
  triton_mm_2905 0.0750 ms 90.5%
  mm 0.0838 ms 81.0%
  triton_mm_2908 0.0841 ms 80.7%
  triton_mm_2907 0.1187 ms 57.2%
  triton_mm_2904 0.6936 ms 9.8%
SingleProcess AUTOTUNE takes 3.7735 seconds
AUTOTUNE mm(1024x176, 176x1024)
  triton_mm_2913 0.0432 ms 100.0%
  mm 0.0510 ms 84.7%
  triton_mm_2912 0.0621 ms 69.5%
  triton_mm_2910 0.0694 ms 62.2%
  triton_mm_2911 0.0777 ms 55.5%
  triton_mm_2909 0.5095 ms 8.5%
SingleProcess AUTOTUNE takes 3.8019 seconds
AUTOTUNE bmm(128x22x22, 128x22x64)
  bmm 0.0099 ms 100.0%
  triton_bmm_2918 0.0107 ms 92.2%
  triton_bmm_2917 0.0111 ms 89.2%
  triton_bmm_2916 0.0141 ms 70.2%
  triton_bmm_2915 0.0161 ms 61.3%
  triton_bmm_2914 0.0274 ms 36.1%
SingleProcess AUTOTUNE takes 3.1921 seconds
AUTOTUNE bmm(128x22x64, 128x64x22)
  bmm 0.0130 ms 100.0%
  triton_bmm_2922 0.0152 ms 85.8%
  triton_bmm_2919 0.0184 ms 70.6%
  triton_bmm_2921 0.0192 ms 67.8%
  triton_bmm_2920 0.0217 ms 59.9%
SingleProcess AUTOTUNE takes 2.2238 seconds
AUTOTUNE bmm(128x22x22, 128x22x64)
  bmm 0.0110 ms 100.0%
  triton_bmm_2927 0.0149 ms 73.9%
  triton_bmm_2926 0.0178 ms 61.9%
  triton_bmm_2924 0.0225 ms 48.9%
  triton_bmm_2923 0.0263 ms 41.8%
  triton_bmm_2925 0.0314 ms 35.0%
SingleProcess AUTOTUNE takes 3.1186 seconds
AUTOTUNE bmm(128x64x22, 128x22x22)
  bmm 0.0100 ms 100.0%
  triton_bmm_2932 0.0112 ms 88.6%
  triton_bmm_2931 0.0114 ms 87.7%
  triton_bmm_2929 0.0139 ms 71.8%
  triton_bmm_2930 0.0164 ms 60.6%
  triton_bmm_2928 0.0299 ms 33.3%
SingleProcess AUTOTUNE takes 3.2824 seconds
AUTOTUNE mm(176x3072, 3072x1024)
  triton_mm_2935 0.1953 ms 100.0%
  triton_mm_2934 0.2174 ms 89.8%
  triton_mm_2937 0.2261 ms 86.4%
  mm 0.2360 ms 82.8%
  triton_mm_2936 0.3421 ms 57.1%
  triton_mm_2933 1.9131 ms 10.2%
SingleProcess AUTOTUNE takes 3.8019 seconds
AUTOTUNE mm(3072x176, 176x1024)
  triton_mm_2942 0.1016 ms 100.0%
  mm 0.1346 ms 75.5%
  triton_mm_2941 0.1516 ms 67.0%
  triton_mm_2939 0.1944 ms 52.3%
  triton_mm_2940 0.2272 ms 44.7%
  triton_mm_2938 1.1283 ms 9.0%
SingleProcess AUTOTUNE takes 3.7658 seconds
```
