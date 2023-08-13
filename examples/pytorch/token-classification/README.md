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
  --do_eval
  --torch_compile True
```

PT2.0 results NV31 (single gpu) + Ryzen 7 5700x3d - train/train_steps_per_second - bert-base-uncased:
| GPU       | Run Mode                          | train_steps_per_second | train_runtime | eval_steps_per_second | eval_runtime|
|-----------|-----------------------------------|------------------------|---------------|-----------------------|-------------|
| 7900 XTX  | Regular execution                 | 18.341                 | 287.22        | 53.669                | 7.5793
| 7900 XTX  | compile - autotune cached         | 19.626                 | 268.4173     | 68.295                | 5.9595

