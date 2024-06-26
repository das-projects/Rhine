<!-- <div align="center">

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
[![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml)
[![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml)
[![codecov](https://codecov.io/gh/ashleve/lightning-hydra-template/branch/main/graph/badge.svg)](https://codecov.io/gh/ashleve/lightning-hydra-template) <br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)

A Promptable model for document classification and extraction 🚀⚡🔥<br>

</div>
 -->

# A Foundational Model for Intelligent Document Processing

**Dr Arijit Das, ERGO Group AG**

## Outline

1. **Introduction**

   - Background and motivation
   - Problem statement
   - Summary of contributions

2. **Related Work**

   - Review of BEIT, BEITv2, CLIP, LLM2Vec, Llama3, and Mistral
   - Existing multi-modal foundation models
   - Positioning of your work in the context of these models

3. **Methodology**

   - **Datasets**
     - Description of pre-training datasets: IDL-WDS, PDFA-ENG-WDS
     - Description of fine-tuning and evaluation datasets: PubTables-1M, PubLayNet
   - **Model Architecture**
     - Image Encoder: BEITv2-based VIT model
     - Text Encoder: LLM2Vec-based LLM
     - Integration: Multiway Transformer
     - Modifications for bounding boxes: Positional embeddings adjustments
   - **Pre-training Strategy**
     - Masked Language Modeling (MLM)
     - Masked Image Modeling (MIM)
     - Word-Path Alignment (WPA)
   - **Finetuning Strategy**
     - Prompt-based fine-tuning for downstream tasks

4. **Experiments**

   - Experimental setup
   - Metrics for evaluation
   - Baseline models for comparison

5. **Infrastructure for Training and Fine-Tuning**

   - Hardware Requirements
   - Software Requirements
   - Tooling for Reproducible Model Development

6. **Results**

   - Performance on pre-training tasks
   - Performance on downstream tasks
   - Comparison with baselines
   - Analysis and discussion

7. **Conclusion**

   - Summary of findings
   - Implications of the results
   - Future work and open questions

8. **References**

---

## Introduction

The rapid advancement in large language models (LLMs) has opened new possibilities for handling complex, multi-modal data. In this paper, we introduce a multi-modal foundation model capable of processing both document images and their Optical Character Recognition (OCR) outputs in JSON format. Our model integrates advanced text and image encoders, incorporating techniques from BEITv2, LLM2Vec, and LongLoRA to handle extended contexts and efficiently process structured information.

For the image encoder, we leverage a Vision Transformer (ViT) model, pre-trained using the BEITv2 approach. Instead of generating image patches directly, the model predicts CLIP embeddings of those patches, enhancing its capability to understand visual content. The text encoder utilizes a pre-trained LLM, such as Llama3 or Mistral, adapted using the LLM2Vec approach. This adaptation includes enabling bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning using the SimCSE technique. These modifications transform the decoder-only LLM into a robust text encoder capable of handling the structured nature of OCR outputs.

We further extend the context window of our model using LongLoRA, which combines Shifted Sparse Attention (S2-Attn) for computational efficiency during fine-tuning and parameter-efficient techniques to manage embeddings and normalization layers effectively. This allows our model to handle significantly larger context sizes without a proportional increase in computational resources, demonstrating strong empirical results on various tasks.

Our model is pre-trained on large datasets such as IDL-WDS and PDFa-eng-WDS and fine-tuned on specific datasets like PubTables-1M and PubLayNet, ensuring robust performance in document classification, layout analysis, named entity recognition, and other downstream tasks. By integrating these cutting-edge techniques, our foundation model sets a new standard for processing multi-modal documents with extended contexts, enabling versatile applications across different domains.

To integrate the image and text embeddings, we utilize a multiway transformer. This allows for efficient cross-modal attention, enhancing the model’s ability to understand the relationship between visual and textual information. Additionally, our model includes a prompt encoder and a text decoder, enabling it to generate relevant answers based on given prompts and the document OCR pair. This design is particularly suited for tasks requiring contextual understanding and precise information extraction.

### Contributions

**Advanced Image and Text Encoding**: Our model leverages a Vision Transformer (ViT) with BEITv2 pre-training to enhance image understanding by predicting CLIP embeddings. For text encoding, we adapt a pre-trained Large Language Model (LLM) using the LLM2Vec approach, incorporating bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning (SimCSE). This dual approach ensures the model can handle structured OCR outputs effectively, providing a robust foundation for further processing.

**Extended Context Handling and Integration**: We incorporate LongLoRA, combining Shifted Sparse Attention (S2-Attn) and parameter-efficient fine-tuning, to significantly extend the context window without a proportional increase in computational resources. This extended context is integrated with a multiway transformer, allowing efficient cross-modal attention and enhancing the model's understanding of visual and textual relationships. This integration ensures that the model can process and understand documents with complex layouts and extended text sequences.

**Robust Pre-training, Fine-tuning, and Interaction Mechanisms**: Our model undergoes extensive pre-training on large datasets and is fine-tuned on specific tasks such as document classification, layout analysis, and named entity recognition. This fine-tuning ensures robust performance across various downstream tasks. Additionally, we incorporate a prompt encoder and text decoder, which allows the model to generate relevant answers based on given prompts and document OCR pairs. This capability enhances the model's ability to perform contextually aware information extraction, making it versatile and adaptable for various applications.

---

## Methodology

### Dataset Details

**IDL-WDS:**
The IDL-WDS (Industry Documents Library - Web Dataset) is a comprehensive dataset designed to facilitate robust pre-training on a diverse range of document types and structures. It comprises approximately 19 million pages, including PDF files, TIFF images, and JSON files with Textract OCR annotations. For processing, images are converted to 1024x1024 pixels in TIFF format to ensure high-quality input data. OCR JSON files are parsed to extract word bounding boxes and text, with bounding box coordinates normalized relative to the 1024x1024 image size. The `litdata` library from Lightning-AI is utilized for efficient data processing, supporting data loading, transformation, and batching, optimizing the pipeline for large-scale datasets.

**PDFA-ENG-WDS:**
The PDFA-ENG-WDS dataset focuses on English PDF documents, providing OCR annotations and bounding boxes for words. Spanning approximately 1.5TB, it contains over 26 million pages and 18 billion text tokens. The dataset is stored in `.tar` files, compatible with efficient streaming and processing using the `litdata` library. Text and bounding box coordinates are normalized relative to the 1024x1024 image size and converted to a suitable format for model input. The `litdata` library supports efficient loading of large, sharded datasets, enabling parallel processing and handling large-scale data effectively.

**PubTables-1M:**
The PubTables-1M dataset is designed for table detection and structure recognition within documents, providing an extensive collection of annotated data to pre-train and fine-tune models for layout analysis and table structure extraction tasks. It includes approximately 947,642 cropped table instances and 575,305 document page instances. The dataset contains images and annotations for train, test, and validation sets, detailing table structures and word bounding boxes. Images are standardized to 1024x1024 pixels, and bounding box coordinates are normalized relative to this size. The `litdata` library is used for efficient loading, transformation, and batching of data.

**PubLayNet:**
PubLayNet is a large-scale dataset aimed at document layout analysis, with annotations for text, titles, lists, tables, and figures within research paper images. It contains over 1 million annotated document images sourced from PubMed Central articles. The dataset includes bounding boxes and labels for various layout components and is divided into training, validation, and test splits. Images are converted to 1024x1024 pixels, and bounding box coordinates are normalized for consistency. The `litdata` library handles large-scale data efficiently, supporting shuffling, batching, and parallel processing.

### Datasets Processing Strategy

- **Consistency:** Standardize image resolutions to 1024x1024 pixels and convert them to TIFF format to ensure high-quality, uniform input data.
- **Normalization:** Normalize bounding box coordinates in the OCR JSON files relative to the 1024x1024 image dimensions.
- **Efficient Loading:** Use the `litdata` library for data processing, `torchvision` for image handling, and efficient methods for managing large, sharded datasets. This ensures efficient data loading and processing, reducing bottlenecks during training.
- **Metadata Utilization:** Leverage metadata such as file sizes and rendering times to filter out large or slow-to-render files, optimizing the dataset for efficient training at scale.

By following these strategies, we can efficiently process and utilize the IDL-WDS, PDFA-ENG-WDS, PubTables-1M, and PubLayNet datasets for pre-training and fine-tuning our multi-modal foundation model, ensuring high performance across various document analysis tasks.

**References**

- [Industry Documents Library](https://huggingface.co/datasets/pixparse/idl-wds)
- [PDF Association dataset](https://huggingface.co/datasets/pixparse/pdfa-eng-wds)
- [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m)
- [PubLayNet](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)

#### Code Example

Below is a code example that includes downloading the dataset from Hugging Face, converting it into the `litdata` format, optimizing the data, and creating new shards for efficient processing:

```python
import os
from huggingface_hub import hf_hub_download
import litdata as ld
from torchvision import transforms
from PIL import Image
import json

# Set up environment for Hugging Face dataset download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Download a dataset shard from Hugging Face
def download_dataset(repo_id, filename, cache_dir='datasets'):
    filepath = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    return filepath

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Function to parse OCR JSON
def parse_ocr(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Normalize bounding boxes
    for page in data['pages']:
        for bbox in page['bbox']:
            bbox[0] /= 1024
            bbox[1] /= 1024
            bbox[2] /= 1024
            bbox[3] /= 1024
    return data

# Create a LitData dataset
class DocumentDataset(ld.Dataset):
    def __init__(self, image_paths, json_paths, transform=None):
        self.image_paths = image_paths
        self.json_paths = json_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        json_data = parse_ocr(self.json_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, json_data

# Example usage
repo_id = 'pixparse/idl-wds'
image_files = ['idl-train-00000.tar']
json_files = ['idl-train-00000.json']

# Download dataset files
image_paths = [download_dataset(repo_id, img) for img in image_files]
json_paths = [download_dataset(repo_id, jsn) for jsn in json_files]

# Instantiate the dataset
dataset = DocumentDataset(image_paths, json_paths, transform=transform)

# Optimize the dataset and create new shards
optimized_dir = 'optimized_datasets'
ld.optimize(dataset, output_dir=optimized_dir, max_shard_size='1GB')

# Create a StreamingDataset and DataLoader from the optimized data
input_dir = optimized_dir
streaming_dataset = ld.StreamingDataset(input_dir, shuffle=True)
dataloader = ld.StreamingDataLoader(streaming_dataset, batch_size=32, num_workers=4)

# Iterate through the dataloader
for batch in dataloader:
    images, annotations = batch
    print(images.shape, annotations)
```

## Model Architecture

### Image Encoder

The image encoder in our model is based on the Vision Transformer (ViT) model, pre-trained using the BEITv2 approach. BEITv2 leverages a visual tokenizer to convert images into discrete visual tokens, enabling masked image modeling (MIM). Specifically, approximately 40% of image patches are masked, and the model is trained to predict the CLIP embeddings of these masked patches. This technique ensures that the model captures high-level visual representations and is robust in understanding the visual content in documents. The pretraining also involves a [CLS] token to aggregate patch information into global representations, enhancing the model’s ability to generate comprehensive visual embeddings.

**References**

- [Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks](https://arxiv.org/pdf/2208.10442)
- [BEIT V2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/pdf/2208.06366)

```python
import math
import torch
import torch.nn as nn
from functools import partial
import pytorch_lightning as pl
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

# Assuming Block, _cfg, PatchEmbed, RelativePositionBias are imported from modeling_finetune

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class VisionTransformerForMaskedImageModeling(pl.LightningModule):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.save_hyperparameters()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self._initialize_weights()

    def _initialize_weights(self):
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, x, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        if return_patch_tokens:
            return x
        if return_all_tokens:
            return self.lm_head(x)
        else:
            return self.lm_head(x[bool_masked_pos])

@register_model
def beit_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_192_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=192, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_256_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
```

### Text Encoder

The text encoder in our multi-modal foundation model leverages advanced techniques to handle long-context documents efficiently. We start with a pre-trained Large Language Model (LLM) such as Llama3 or Mistral and modify it to incorporate the improvements described in the LongLoRA framework. LongLoRA employs two primary enhancements:

First, **Shifted Sparse Attention (S2-Attn)** is utilized during fine-tuning to significantly reduce computational costs while maintaining performance comparable to vanilla attention. This approach ensures that the model can handle extended context lengths without a proportional increase in computational resources. Implemented with minimal code changes, S2-Attn is optional during inference, making it both efficient and flexible.

Second, **Parameter-Efficient Fine-Tuning** extends the capabilities of LoRA (Low-Rank Adaptation) by ensuring that embeddings and normalization layers are trainable. This combination enhances the model's ability to handle longer contexts effectively. Using these techniques, LongLoRA demonstrates the ability to extend the context window of LLMs substantially. For instance, Llama2 7B can be extended from a 4k context to 100k, or Llama2 70B can be extended to 32k, all while maintaining computational efficiency.

Additionally, the model with extended context capabilities is further enhanced using LLM2Vec. LLM2Vec generates high-quality contextual vector representations by leveraging the power of pre-trained LLMs and refining them for specific tasks. The process involves fine-tuning models to produce embeddings that capture the nuanced meanings of words and phrases in context. Implementing various pooling strategies, such as mean pooling, weighted mean pooling, and EOS token pooling, allows for flexible and robust extraction of sentence-level embeddings. By combining LongLoRA's extended context capabilities with LLM2Vec's advanced embedding techniques, the model is well-suited for handling large context sizes and performing well on various downstream tasks, such as document classification and named entity recognition.

The text encoder utilizes a pre-trained LLM such as Llama3 or Mistral, adapted using the LLM2Vec approach. This process includes enabling bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning using the SimCSE technique. The goal is to transform a decoder-only LLM into a strong text encoder. The steps involve fine-tuning the model to predict masked tokens and using dropout techniques to create positive examples for contrastive learning. This adaptation is crucial for handling the structured nature of OCR outputs, as described in "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders."

By incorporating these advancements, our text encoder is equipped to handle large context sizes efficiently, enabling it to perform well on various downstream tasks such as document classification, named entity recognition, and more.
**References**

- [LongLoRA: Efficient Fine-Tuning of Long-Context Large Language Models](https://openreview.net/pdf?id=6PmJoRfdaK)
- [LLM2Vec: Enhancing Large Language Models with Contextual Vector Representations](https://arxiv.org/pdf/2404.05961)
- [Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](https://arxiv.org/pdf/2106.02795)

This section provides a comprehensive overview of how LongLoRA's techniques and LLM2Vec are utilized in the text encoder, ensuring efficient handling of long contexts and robust performance on various tasks.

```python
import json
import logging
import os
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from peft import PeftModel
from torch import Tensor, device, nn
from tqdm.autonotebook import trange
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    LlamaConfig,
    MistralConfig,
)
import pytorch_lightning as pl

from .models import (
    MistralBiModel,
    LlamaBiModel,
    GemmaBiModel,
)

logger = logging.getLogger(__name__)

def batch_to_device(batch, target_device: device):
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, num_features: int, scale: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.scale = scale
        self.linear = nn.Linear(4, num_features * 2)

    def forward(self, bounding_boxes: Tensor):
        """
        Forward pass for learnable Fourier positional encoding
        Args:
            bounding_boxes (Tensor): A tensor of shape (batch_size, num_boxes, 4) containing normalized bounding boxes.
        Returns:
            Tensor: A tensor of shape (batch_size, num_boxes, num_features * 2) containing positional encodings.
        """
        scaled_boxes = bounding_boxes * self.scale
        fourier_features = self.linear(scaled_boxes)
        pos_encodings = torch.cat([torch.sin(fourier_features), torch.cos(fourier_features)], dim=-1)
        return pos_encodings

class ShiftedSparseAttention(nn.Module):
    def __init__(self, block_size=128, shift_size=64):
        super().__init__()
        self.block_size = block_size
        self.shift_size = shift_size

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        assert seq_len % self.block_size == 0, "Sequence length should be divisible by block size."

        x = x.view(batch_size, seq_len // self.block_size, self.block_size, dim)
        x_shifted = torch.roll(x, shifts=self.shift_size, dims=1)

        local_attention = torch.bmm(
            x.view(batch_size * (seq_len // self.block_size), self.block_size, dim),
            x.view(batch_size * (seq_len // self.block_size), self.block_size, dim).transpose(1, 2)
        )

        shifted_attention = torch.bmm(
            x_shifted.view(batch_size * (seq_len // self.block_size), self.block_size, dim),
            x_shifted.view(batch_size * (seq_len // self.block_size), self.block_size, dim).transpose(1, 2)
        )

        attention = local_attention + shifted_attention
        return attention.view(batch_size, seq_len, seq_len)

class LLM2Vec(pl.LightningModule):
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        pooling_mode: str = "mean",
        max_length: int = 512,
        doc_max_length: int = 400,
        skip_instruction: bool = True,
        num_positional_features: int = 64,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_mode = pooling_mode
        self.skip_instruction = skip_instruction
        self.max_length = max_length
        self.doc_max_length = doc_max_length
        self.config = model.config
        self.positional_encoding = LearnableFourierPositionalEncoding(num_features=num_positional_features)
        self.s2_attn = ShiftedSparseAttention()

    @classmethod
    def _get_model_class(cls, config_class_name, enable_bidirectional):
        if not enable_bidirectional:
            return AutoModel
        if config_class_name == "MistralConfig":
            return MistralBiModel
        elif config_class_name == "LlamaConfig":
            return LlamaBiModel
        elif config_class_name == "GemmaConfig":
            return GemmaBiModel
        else:
            raise ValueError(
                f"{config_class_name} is not supported yet with bidirectional models."
            )

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path,
        peft_model_name_or_path=None,
        merge_peft=False,
        enable_bidirectional=True,
        **kwargs,
    ):
        keys = ["pooling_mode", "max_length", "doc_max_length", "skip_instruction", "num_positional_features"]
        encoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_model_name_or_path)
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(
            config_class_name, enable_bidirectional=enable_bidirectional
        )
        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)

        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(
                model,
                base_model_name_or_path,
            )
            model = model.merge_and_unload()

        if peft_model_name_or_path is not None:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
            )
            if merge_peft:
                model = model.merge_and_unload()

        config = {}
        config_addr = (
            peft_model_name_or_path
            if peft_model_name_or_path is not None
            else base_model_name_or_path
        )
        if os.path.exists(f"{config_addr}/llm2vec_config.json"):
            with open(f"{config_addr}/llm2vec_config.json", "r") as fIn:
                llm2vec_config = json.load(fIn)
            config.update(llm2vec_config)

        for key, value in encoder_args.items():
            config[key] = value

        return cls(model=model, tokenizer=tokenizer, **config)

    def prepare_for_tokenization(self, text):
        if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
            text = (
                "user\n\n"
                + text.strip()
                + ""
            )
            return text
        if self.model.config._name_or_path in [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
        ]:
            text = "[INST] " + text.strip() + " [/INST]"
        if self.pooling_mode == "eos_token":
            if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
                text = text.strip() + ""
            elif isinstance(self.model.config, LlamaConfig) or isinstance(
                self.model.config, MistralConfig
            ):
                text = text.strip() + " </s>"

        return text

    def tokenize(self, texts, bounding_boxes=None):
        texts_2 = []
        original_texts = []
        for text in texts:
            t = text.split("!@#$%^&*()")
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        if bounding_boxes is not None:
            positional_encodings = self.positional_encoding(bounding_boxes)
            original["positional_encodings"] = positional_encodings

        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original

    def _skip_instruction(self, sentence_feature):
        assert (
            sentence_feature["attention_mask"].shape
            == sentence_feature["embed_mask"].shape
        )
        sentence_feature["attention_mask"] = sentence_feature["embed_mask"]

    def forward(self, sentence_feature: Dict[str, Tensor]):
        embed_mask = None
        if "embed_mask" in sentence_feature:
            embed_mask = sentence_feature.pop("embed_mask")
        reps = self.model(**sentence_feature)
        sentence_feature["embed_mask"] = embed_mask

        if "positional_encodings" in sentence_feature:
            positional_encodings = sentence_feature["positional_encodings"]
            reps.last_hidden_state += positional_encodings

        # Apply shifted sparse attention (S2-Attn) during training
        if self.training:
            reps.last_hidden_state = self.s2_attn(reps.last_hidden_state)

        return self.get_pooling(sentence_feature, reps.last_hidden_state)

    def get_pooling(self, features, last_hidden_states):
        assert (
            self.tokenizer.padding_side == "left"
        ), "Pooling modes are implemented for padding from left."
        if self.skip_instruction:
            self._skip_instruction(features)
        seq_lengths = features["attention_mask"].sum(dim=-1)
        if self.pooling_mode == "mean":
            return torch.stack(
                [
                    last_hidden_states[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )
        elif self.pooling_mode == "weighted_mean":
            bs, l, _ = last_hidden_states.shape
            complete_weights = torch.zeros(bs, l, device=last_hidden_states.device)
            for i, seq_l in enumerate(seq_lengths):
                if seq_l > 0:
                    complete_weights[i, -seq_l:] = torch.arange(seq_l) + 1
                    complete_weights[i] /= torch.clamp(
                        complete_weights[i].sum(), min=1e-9
                    )
            return torch.sum(last_hidden_states * complete_weights.unsqueeze(-1), dim=1)
        elif self.pooling_mode == "eos_token" or self.pooling_mode == "last_token":
            return last_hidden_states[:, -1]
        elif self.pooling_mode == "bos_token":
            return last_hidden_states[
                features["input_ids"] == self.tokenizer.bos_token_id
            ]
        else:
            raise ValueError(f"{self.pooling_mode} is not implemented yet.")

    def _convert_to_str(self, instruction, text):
        tokenized_q = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        tokenized_q_length = len(tokenized_q["input_ids"][0])

        while tokenized_q_length > self.doc_max_length:
            reduction_ratio = self.doc_max_length / tokenized_q_length
            reduced_length = int(len(text.split()) * reduction_ratio)
            text = " ".join(text.split()[:reduced_length])
            tokenized_q = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            tokenized_q_length = len(tokenized_q["input_ids"][0])

        return f"{instruction.strip()} !@#$%^&*(){text}"

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
    ):
        if isinstance(sentences[0], str) and isinstance(sentences[-1], int):
            sentences = [sentences]
        if isinstance(sentences[0], str):
            sentences = [[""] + [sentence] for sentence in sentences]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        concatenated_input_texts = []
        for sentence in sentences:
            assert isinstance(sentence[0], str)
            assert isinstance(sentence[1], str)
            concatenated_input_texts.append(
                self._convert_to_str(sentence[0], sentence[1])
            )
        sentences = concatenated_input_texts

        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []

        if torch.cuda.device_count() <= 1:
            self.to(device)
            for start_index in trange(
                0,
                len(sentences),
                batch_size,
                desc="Batches",
                disable=not show_progress_bar,
            ):
                sentences_batch = sentences_sorted[
                    start_index : start_index + batch_size
                ]
                embeddings = self._encode(
                    sentences_batch, device=device, convert_to_numpy=convert_to_numpy
                )
                all_embeddings.append(embeddings)
        else:
            num_proc = torch.cuda.device_count()
            cuda_compatible_multiprocess = mp.get_context("spawn")
            with cuda_compatible_multiprocess.Pool(num_proc) as p:
                sentences_batches = [
                    sentences_sorted[start_index : start_index + batch_size]
                    for start_index in trange(0, len(sentences), batch_size)
                ]
                for result in p.map(
                    partial(
                        self._encode,
                        device=None,
                        convert_to_numpy=convert_to_numpy,
                        multiprocessing=True,
                    ),
                    sentences_batches,
                ):
                    all_embeddings.append(result)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings are all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def save(self, output_path, merge_before_save=False, save_config=True):
        if merge_before_save and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            if hasattr(self.model, "_hf_peft_config_loaded"):
                self.model._hf_peft_config_loaded = False

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        llm2vec_config = {
            "pooling_mode": self.pooling_mode,
            "max_length": self.max_length,
            "doc_max_length": self.doc_max_length,
            "skip_instruction": self.skip_instruction,
        }

        if save_config:
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/llm2vec_config.json", "w") as fOut:
                json.dump(llm2vec_config, fOut, indent=4)

    def _encode(self, sentences_batch, device: Optional[str] = None, convert_to_numpy: bool = False, multiprocessing=False):
        if multiprocessing:
            rank = mp.current_process()._identity[0]
            if device is None and torch.cuda.is_available():
                device = f"cuda:{rank % torch.cuda.device_count()}"

        self.to(device)
        features = self.tokenize(
            [self.prepare_for_tokenization(sentence) for sentence in sentences_batch]
        )
        features = batch_to_device(features, device)

        with torch.no_grad():
            embeddings = self.forward(features)
            embeddings = embeddings.detach()
            embeddings = embeddings.cpu()

        return embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        if (
            isinstance(text, str)
            or (isinstance(text, list) and isinstance(text[0], int))
            or len(text) == 0
        ):
            return len(text)
        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):
            return 1
        else:
            return sum([len(t) for t in text])

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        return self.model.resize_token_embeddings(
            new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
```

### Integration with Multiway Transformer

To integrate text and image embeddings, we employ a Multiway Transformer architecture. Each block consists of a shared self-attention module and a pool of feed-forward networks tailored for different modalities (vision, language, and vision-language). This design facilitates deep fusion of multi-modal data and modality-specific processing, making it highly effective for tasks involving complex interactions between text and images.

**References**

- [VLMo: Unified vision-language pre-training with mixture-of-modality-experts](https://arxiv.org/pdf/2111.02358)

```python
import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl

class MultiwayNetwork(pl.LightningModule):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)

class MultiwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super().__init__(modules[0], dim=dim)
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]

def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module

def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position
    return apply_fn
```

### Positional Embeddings for Bounding Boxes

We enhance the positional embeddings to incorporate the spatial information of OCR tokens using techniques inspired by "Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding." This involves encoding the bounding box coordinates using learnable Fourier features, which capture the spatial relationships between tokens. By embedding this spatial information directly into the transformer's self-attention mechanism, the model effectively captures the topological and spatial relationships between textual elements in a document, thereby improving its performance in tasks that require an understanding of the document's visual and spatial context.

**References**

- [Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](https://arxiv.org/pdf/2106.02795)

### Prompt Encoder

The prompt encoder is a small text embedding model designed to process the prompt and generate an embedding. This embedding is then added to the embedding received from the multiway transformer before feeding into the text decoder.

**Model Architecture**:
The architecture utilizes a lightweight transformer or any efficient text embedding model for the prompt encoder. The model is designed to output embeddings of the same dimension as the multiway transformer to ensure seamless integration.

**Embedding Process**:
The process begins by taking the input prompt and converting it into token embeddings. These embeddings are then processed through the prompt encoder to generate a final prompt embedding, which is added to the multiway transformer embedding before being input to the text decoder.

**References**

- [MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining](https://arxiv.org/pdf/2312.17482)

### Text Decoder

The text decoder in our multi-modal model is a fine-tuned LLM like Llama3, which generates word and bounding box pairs for various extraction tasks or plain text when describing documents.

**Model Architecture**:
We use Llama3 or another suitable LLM for the text decoder. The model is fine-tuned to handle tasks that involve generating word and bounding box pairs, as well as descriptive text for documents.

**Fine-tuning**:
The fine-tuning process involves using datasets that include document descriptions, extraction tasks, and bounding box annotations. The model is trained to generate output sequences that consist of both text and bounding box coordinates, enabling it to effectively perform various document analysis tasks.

**References**

- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691)

### Integration

The integration of our multi-modal foundation model involves combining the embeddings generated by the prompt encoder with those from the multiway transformer. This combined embedding is then passed to the text decoder. Specifically, a lightweight transformer, such as BERT, is used by the prompt encoder to convert the prompt into an embedding. This embedding is then added to the multiway transformer embedding before being input to the text decoder, typically a fine-tuned LLM like Llama3.

The training strategy for this model involves jointly training the prompt encoder and text decoder on tasks that require both text generation and bounding box prediction. This joint training approach ensures that the model can generate coherent text sequences alongside accurate bounding box coordinates. The loss function is carefully designed to account for both text and bounding box outputs, allowing the model to learn these tasks simultaneously and efficiently.

The prompt encoder processes the prompt into an embedding using a lightweight transformer model. The multi-modal model combines this prompt embedding with document embeddings from the multiway transformer, which are then passed to the text decoder. The text decoder generates the final output, producing both text and bounding box pairs for various extraction tasks. This comprehensive setup enables the model to effectively handle prompts and generate appropriate responses, ensuring it performs well on diverse document analysis tasks.

### Implementation Example

Here's a conceptual implementation in PyTorch using PyTorch Lightning:

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM

class PromptEncoder(nn.Module):
    def __init__(self, model_name, embed_dim):
        super(PromptEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.model.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Simple mean pooling
        prompt_embedding = self.linear(pooled_output)
        return prompt_embedding

class MultiModalModel(pl.LightningModule):
    def __init__(self, multiway_transformer, prompt_encoder, text_decoder, tokenizer):
        super(MultiModalModel, self).__init__()
        self.multiway_transformer = multiway_transformer
        self.prompt_encoder = prompt_encoder
        self.text_decoder = text_decoder
        self.tokenizer = tokenizer

    def forward(self, document_embeddings, prompt_input_ids, prompt_attention_mask):
        prompt_embedding = self.prompt_encoder(prompt_input_ids, prompt_attention_mask)
        combined_embedding = document_embeddings + prompt_embedding.unsqueeze(1)
        decoder_input_ids = torch.full(
            (combined_embedding.size(0), 1), self.tokenizer.cls_token_id, device=self.device
        )
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=combined_embedding
        )
        return decoder_outputs

    def training_step(self, batch, batch_idx):
        document_embeddings = self.multiway_transformer(batch["document_images"])
        prompt_input_ids = batch["prompt_input_ids"]
        prompt_attention_mask = batch["prompt_attention_mask"]
        labels = batch["labels"]

        outputs = self.forward(document_embeddings, prompt_input_ids, prompt_attention_mask)
        loss = nn.CrossEntropyLoss()(outputs.logits.view(-1, self.text_decoder.config.vocab_size), labels.view(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

# Example usage
multiway_transformer = ...  # Your multiway transformer model
tokenizer = AutoTokenizer.from_pretrained("llama3")
prompt_encoder = PromptEncoder(model_name="bert-base-uncased", embed_dim=multiway_transformer.config.hidden_size)
text_decoder = LlamaForCausalLM.from_pretrained("llama3")

model = MultiModalModel(multiway_transformer, prompt_encoder, text_decoder, tokenizer)
```

### Pre-training Strategy

Our pre-training strategy leverages three key objectives: Masked Language Modeling (MLM), Masked Image Modeling (MIM), and Word-Patch Alignment (WPA). Inspired by "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking," these objectives aim to enable the model to effectively learn multimodal representations.

For **Masked Language Modeling (MLM)**, 30% of text tokens are masked using a span masking strategy, with span lengths following a Poisson distribution. The goal is to maximize the log-likelihood of predicting these masked tokens based on contextual representations from both text and image sequences. This helps the model understand the relationship between textual content and layout information.

In **Masked Image Modeling (MIM)**, around 40% of image tokens are masked using a blockwise masking strategy. The model is then trained to reconstruct these masked tokens using a cross-entropy loss, which encourages it to interpret visual content through the context provided by both text and image tokens. This objective ensures that the model captures high-level visual structures rather than just low-level details.

**Word-Patch Alignment (WPA)** aligns text words with their corresponding image patches. This involves predicting whether the image patches corresponding to a text word are masked. By assigning binary aligned/unaligned labels to unmasked text tokens based on their image patch masking status, the model learns a fine-grained alignment between text and image modalities, which is crucial for tasks requiring precise text-image correspondence.

### Fine-tuning Strategy

Our fine-tuning strategy adopts a prompt-based approach inspired by methods described in the Segment Anything Model (SAM). This enables the model to respond effectively to a variety of prompts tailored to solve different downstream tasks.

**Instruction Tuning the Text Decoder:** The text decoder is fine-tuned to handle diverse downstream tasks by providing specific instructions on generating the relevant output. This includes producing text sequences and identifying corresponding bounding boxes in response to various queries. By doing so, the model becomes adept at generating contextually accurate outputs based on the instructions provided.

**Zero-Shot and Few-Shot Learning:** Leveraging the model’s pre-trained capabilities, the prompt-based strategy allows it to perform tasks in zero-shot or few-shot settings. By carefully designing prompts, the model can be adapted to various applications, such as document layout analysis, document classification, and named entity recognition, without requiring extensive additional training.

This comprehensive fine-tuning strategy ensures that the model remains flexible and capable of efficiently performing a wide range of tasks. It leverages the strengths of pre-trained embeddings and prompt-based interaction mechanisms to maintain high performance across various applications.

**References**

- [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/pdf/2204.08387)

### Fine-tuning Strategy

For fine-tuning, we adopt a prompt-based strategy inspired by methods described in the Segment Anything Model (SAM). This approach enables the model to respond to a variety of prompts, which can be tailored to solve different downstream tasks. The fine-tuning process involves:

- **Instruction Tuning the Text Decoder:** The text decoder is fine-tuned to handle diverse downstream tasks by providing specific instructions for generating the relevant output. This includes producing text sequences and identifying the corresponding bounding boxes in response to different types of queries. Instruction tuning allows the model to adapt to various tasks, enhancing its versatility and effectiveness.

- **Zero-Shot and Few-Shot Learning:** The prompt-based strategy leverages the model’s pre-trained capabilities to perform tasks in zero-shot or few-shot settings. By carefully designing prompts, the model can be adapted to various applications such as document layout analysis, document classification, and named entity recognition without extensive additional training.

- **Task-Specific Prompts:** Specific prompts are designed for each downstream task, guiding the model on how to process and respond to different types of input. For instance, prompts for document classification might instruct the model to identify key categories, while prompts for named entity recognition might direct the model to highlight specific entities within the text.

- **Evaluation and Iteration:** Throughout the fine-tuning process, the model's performance is evaluated on various benchmarks, and prompts and tuning parameters are iteratively refined. This ensures that the model generalizes well to new tasks and maintains high accuracy and efficiency.

This comprehensive fine-tuning strategy ensures that the model remains flexible and capable of performing a wide range of tasks efficiently, leveraging the strengths of both the pre-trained embeddings and the prompt-based interaction mechanisms

### Related Work

The development of multi-modal foundation models has seen significant advancements in recent years, particularly in integrating text and image data to enhance various applications such as document analysis, information retrieval, and natural language processing.

**BEIT and BEITv2:**
BEIT (Bao et al., 2022) introduced masked image modeling (MIM) with visual tokenizers for image pre-training. BEITv2 improved this approach by using a visual tokenizer trained with vector-quantized knowledge distillation (VQ-KD), capturing comprehensive visual representations by predicting CLIP embeddings for masked image patches.

**LLM2Vec:**
LLM2Vec demonstrated the capability of transforming decoder-only language models into powerful text encoders through masked next token prediction (MNTP) and unsupervised contrastive learning (SimCSE). This adaptation enables effective handling of structured text data, such as OCR outputs, by converting causal attention to bidirectional, thus improving contextual understanding.

**Multiway Transformer:**
The Multiway Transformer architecture provides a robust framework for integrating text and image embeddings. By incorporating shared self-attention modules and modality-specific feed-forward networks, this architecture facilitates deep fusion and processing of multi-modal data, enhancing the model's ability to handle complex interactions between different data types.

**LayoutLM and LayoutLMv3:**
The LayoutLM series, particularly LayoutLMv3, contributed to pre-training strategies for document AI using unified text-image multimodal transformers. They employ objectives like masked language modeling (MLM), masked image modeling (MIM), and word-patch alignment (WPA) to learn cross-modal representations, effectively capturing relationships between textual and visual elements in documents.

**Segment Anything Model (SAM):**
The Segment Anything Model (SAM) introduces a prompt-based approach to segmentation tasks, enabling the model to respond to various prompts such as points, boxes, or masks. This flexibility allows SAM to perform well in zero-shot and few-shot settings, adapting to different segmentation tasks through prompt engineering.

**LongLoRA:**
LongLoRA extends the context size of pre-trained large language models (LLMs) efficiently. It combines shifted sparse attention (S2-Attn) and parameter-efficient fine-tuning to manage embeddings and normalization layers, allowing models like Llama2 to handle significantly larger context sizes without a proportional increase in computational resources.

Our proposed multi-modal foundation model builds upon these advancements by integrating a BEITv2-based image encoder, an LLM2Vec-based text encoder, and a Multiway Transformer for embedding fusion. Additionally, we incorporate a prompt encoder and a text decoder to facilitate interactive document analysis and question-answering tasks, leveraging instruction tuning to enhance the model's adaptability to various downstream applications.

---

## Experiments

To evaluate the effectiveness of our proposed multi-modal foundation model, we conduct a series of experiments across different tasks, including pre-training, fine-tuning, and downstream task performance.

**Pre-training Setup:**

1. **Datasets:** We use the IDL-WDS and PDFA-ENG-WDS datasets for pre-training. These datasets provide a diverse collection of document images and corresponding OCR outputs, allowing the model to learn from various document types and structures.
2. **Objectives:** The pre-training involves masked language modeling (MLM), masked image modeling (MIM), and word-patch alignment (WPA) objectives, ensuring the model captures comprehensive multimodal representations.
3. **Evaluation Metrics:** We evaluate the model's performance on pre-training tasks using metrics such as cross-entropy loss for MLM and MIM, and binary cross-entropy loss for WPA.

### Fine-tuning Strategy

For fine-tuning, we adopt a prompt-based strategy inspired by methods described in the Segment Anything Model (SAM). This approach enables the model to respond appropriately to a variety of prompts, which can be engineered to solve different downstream tasks. The fine-tuning process involves:

- **Instruction Tuning the Text Decoder:** We fine-tune the text decoder to handle diverse downstream tasks by providing it with specific instructions on how to generate the relevant output. This includes producing text sequences and identifying the corresponding bounding boxes in response to different types of queries. Instruction tuning allows the model to adapt to various tasks, enhancing its versatility and effectiveness.

- **Zero-Shot and Few-Shot Learning:** The prompt-based strategy leverages the model’s pre-trained capabilities to perform tasks in a zero-shot or few-shot setting. By carefully designing prompts, we can adapt the model to various applications such as document layout analysis, document classification, and named entity recognition without extensive additional training.

- **Task-Specific Prompts:** We design specific prompts for each downstream task, guiding the model on how to process and respond to different types of input. For example, prompts for document classification might instruct the model to identify key categories, while prompts for named entity recognition might direct the model to highlight specific entities within the text.

- **Evaluation and Iteration:** Throughout the fine-tuning process, we evaluate the model's performance on various benchmarks and iteratively refine the prompts and tuning parameters. This ensures that the model not only generalizes well to new tasks but also maintains high accuracy and efficiency.

This comprehensive fine-tuning strategy ensures that the model remains flexible and capable of performing a wide range of tasks efficiently, leveraging the strengths of both the pre-trained embeddings and the prompt-based interaction mechanisms.

### Evaluation Metrics for Pre-training and Fine-tuning

To ensure the effectiveness and robustness of our multi-modal foundation model, we employ a variety of evaluation metrics at different stages of pre-training and fine-tuning. These metrics comprehensively assess the model’s performance.

### Pre-training Evaluation Metrics

1. **Image Encoder (BEITv2-based):**

   - **Reconstruction Loss:** Measures how well the model reconstructs masked image patches using CLIP embeddings, typically evaluated using mean squared error (MSE) or cross-entropy loss.
   - **Patch-wise Accuracy:** The accuracy of predicting correct visual tokens for the masked patches.
   - **Global Representation Quality:** Assessed using linear probing on a downstream image classification task (e.g., ImageNet), evaluating metrics such as top-1 and top-5 accuracy.

2. **Text Encoder (LLM2Vec-based):**

   - **Masked Language Modeling (MLM) Accuracy:** Measures the accuracy of predicting masked tokens in the text, evaluated using cross-entropy loss.
   - **Perplexity:** A measure of how well the probability distribution predicted by the model aligns with the actual distribution of the data. Lower perplexity indicates a better language model.
   - **Contrastive Learning Metrics:** For SimCSE, metrics like cosine similarity between positive pairs and the alignment and uniformity of the learned representations.

3. **Multiway Transformer Integration:**
   - **Alignment Loss:** Measures how well the text and image embeddings are aligned, typically using cosine similarity or a similar metric.
   - **Cross-modal Retrieval Accuracy:** Evaluates how accurately the model can retrieve relevant text given an image query, and vice versa, using metrics such as mean reciprocal rank (MRR) and normalized discounted cumulative gain (nDCG).

### Fine-tuning Evaluation Metrics

1. **Prompt Encoder:**

   - **Prompt Response Accuracy:** Measures how accurately the prompt encoder generates embeddings that lead to correct responses from the text decoder, evaluated using metrics specific to the downstream task (e.g., F1 score for classification tasks).
   - **Prompt Response Time:** Evaluates the efficiency of the prompt encoder in generating prompt embeddings, measured in milliseconds.

2. **Text Decoder:**

   - **Sequence Generation Quality:** Assessed using metrics like BLEU, ROUGE, and METEOR, which measure the quality of generated text against reference text.
   - **Bounding Box Accuracy:** Evaluates how accurately the model predicts bounding boxes, using metrics like Intersection over Union (IoU) and mean Average Precision (mAP).

3. **Overall Model Performance:**
   - **Task-specific Metrics:**
     - **Document Layout Analysis:** Metrics like F1 score, precision, recall, and mean Average Precision (mAP) for detecting and classifying different layout components (e.g., text blocks, tables, figures).
     - **Document Classification:** Accuracy, F1 score, precision, and recall for categorizing documents into predefined classes.
     - **Named Entity Recognition (NER):** F1 score, precision, and recall for identifying and classifying named entities within the text.
     - **Question Answering:** Exact match (EM) and F1 score for evaluating the correctness of answers generated in response to queries.

### Evaluation during Pre-training and Fine-tuning

1. **Validation Loss:** Continuous monitoring of validation loss during pre-training and fine-tuning helps ensure that the model is not overfitting and generalizes well to unseen data.
2. **Early Stopping Metrics:** Metrics like validation loss and early stopping criteria (based on patience and delta values) can help prevent overfitting during training.
3. **Ablation Studies:** Conducting ablation studies to evaluate the impact of different components (e.g., removing the prompt encoder or text decoder) on overall performance.

By utilizing these diverse evaluation metrics, we can comprehensively assess the performance and robustness of our multi-modal foundation model at each stage of pre-training and fine-tuning, ensuring its effectiveness across various downstream tasks.

**References**

- [LongLoRA: Efficient Fine-Tuning of Long-Context Large Language Models](https://openreview.net/pdf?id=6PmJoRfdaK)
- [LLM2Vec: Enhancing Large Language Models with Contextual Vector Representations](https://arxiv.org/pdf/2404.05961)

---

### Infrastructure for Training and Fine-Tuning

Training and fine-tuning a multi-modal foundation model, particularly with large language models like Llama3-8B and BEITv2, and extensive datasets such as IDL-WDS and PDFa-eng-WDS, necessitates significant computational resources. Below, we outline the infrastructure requirements and the necessary setup for efficient training and fine-tuning.

#### Hardware Requirements

1. **GPUs**: High-performance GPUs are essential. NVIDIA A100 GPUs are recommended, with at least 8 GPUs to handle the extensive computation and large context sizes required by models like Llama3-8B.
2. **Memory**: A minimum of 2TB of system memory (RAM) to manage large datasets and model parameters efficiently.
3. **Storage**: High-speed SSD storage, with at least 20TB capacity, to accommodate the large-scale datasets and model checkpoints.
4. **Networking**: High-bandwidth networking (e.g., InfiniBand) to ensure fast data transfer between GPUs and storage, reducing bottlenecks.

#### Software Requirements

1. **Frameworks**: PyTorch for deep learning, with PyTorch Lightning to streamline the training and fine-tuning process.
2. **Libraries**:
   - Hugging Face Transformers for model implementations and tokenizers.
   - CUDA and cuDNN for GPU acceleration.
   - Apex for mixed precision training to optimize performance and memory usage.
3. **Data Management**: LitData library for efficient data loading and preprocessing, particularly useful for handling large datasets.

#### Training and Fine-Tuning Environment

1. **Distributed Training**: Utilize distributed training strategies, such as data parallelism and model parallelism, to efficiently leverage multiple GPUs.
2. **Mixed Precision Training**: Use mixed precision training to reduce the memory footprint and increase computational speed, crucial for handling large models like Llama3-8B.
3. **Hyperparameter Tuning**: Implement automated hyperparameter tuning frameworks (e.g., Optuna or Ray Tune) to optimize model performance.
4. **Monitoring and Logging**: Set up robust monitoring and logging systems (e.g., TensorBoard, Weights & Biases) to track training progress and performance metrics.

### Tooling for Reproducible Model Development

Ensuring reproducibility in model development is crucial for validating results and facilitating collaboration. The following tools and practices are recommended:

1. **Version Control**: Use Git for version control of code, configuration files, and documentation. Maintain clear commit messages and branching strategies (e.g., GitFlow).
2. **Environment Management**: Use Docker to create isolated and reproducible environments. Document the dependencies and environment setup steps in a `Dockerfile` or `docker-compose.yml`.
3. **Experiment Tracking**: Utilize tools like MLflow or Weights & Biases to track experiments, including hyperparameters, training configurations, and results. This ensures that all experiment details are recorded and can be reproduced.
4. **Automated Testing**: Implement automated tests for different components of the model pipeline using frameworks like pytest. This includes unit tests, integration tests, and regression tests to ensure that changes do not introduce errors.
5. **Continuous Integration/Continuous Deployment (CI/CD)**: Set up CI/CD pipelines using tools like GitHub Actions or Jenkins to automate testing, building, and deploying models. This ensures that every change is automatically tested and can be deployed consistently.
6. **Documentation**: Maintain comprehensive documentation for the codebase, including installation instructions, usage examples, and API references. Tools like Sphinx can help generate documentation from docstrings and comments in the code.

### Summary of Training and Fine-Tuning Process

1. **Pre-training**:

   - Utilize large datasets (IDL-WDS with approximately 70 million pages and PDFa-eng-WDS with extensive document images and text pairs) to pre-train the image encoder with BEITv2 and the text encoder with LLM2Vec.
   - Implement LongLoRA for efficient context extension, which is crucial given the extended context requirements of up to 100k tokens for Llama2 7B and 32k tokens for Llama2 70B.

2. **Fine-Tuning**:

   - Apply a prompt-based fine-tuning strategy inspired by SAM.
   - Fine-tune on specific datasets like PubTables-1M and PubLayNet, which provide high-quality annotated data for tasks such as document layout analysis and entity recognition.
   - Instruction tune the text decoder to handle diverse downstream tasks and respond to various prompts, ensuring robust and flexible performance.

3. **Evaluation**:
   - Continuously evaluate the model's performance using benchmarks and iteratively refine prompts and tuning parameters.
   - Implement zero-shot and few-shot learning to assess model adaptability and performance on unseen tasks, leveraging the model's extensive pre-training and fine-tuning.

By following this comprehensive setup, the training and fine-tuning processes are optimized to handle the complexity and scale of the multi-modal foundation model, ensuring robust performance across various applications.
