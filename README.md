<div align="center">

# Rhine: A Foundational Model for Intelligent Document Processing

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


A Promptable model for document classification and extraction  🚀⚡🔥<br>

</div>


### Outline

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

5. **Results**
   - Performance on pre-training tasks
   - Performance on downstream tasks
   - Comparison with baselines
   - Analysis and discussion

6. **Conclusion**
   - Summary of findings
   - Implications of the results
   - Future work and open questions

7. **References**

---

### Introduction

In recent years, there has been significant progress in the development of foundation models capable of processing and understanding multi-modal data, such as images and text. These models have shown great promise in various applications, including document analysis, information retrieval, and natural language processing. However, existing approaches often rely on separate encoders for images and text, which limits their ability to fully leverage the interactions between different modalities.

In this paper, we propose a novel multi-modal foundation model that integrates image and text data at a deeper level. Our model accepts an image of a document and its corresponding OCR output in JSON format, which includes a list of tuples of words and their bounding boxes. The image encoder is based on the Vision Transformer (ViT) model, pre-trained using the BEITv2 approach. Instead of generating image patches directly, our model predicts the CLIP embeddings of these patches, leveraging the rich semantic information captured by CLIP. This approach enhances the model's understanding of the visual content in documents, ensuring robust and high-level visual representations【52†source】【53†source】.

The text encoder is derived from a pre-trained Large Language Model (LLM) such as Llama3 or Mistral, adapted using the LLM2Vec approach. This involves enabling bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning using the SimCSE technique. The adaptation from a decoder-only LLM to a strong text encoder allows the model to handle the structured nature of OCR outputs effectively, capturing the contextual and semantic nuances of the text【38†source】.

To fuse the text and image embeddings, we employ a Multiway Transformer architecture. This architecture integrates a shared self-attention module and modality-specific feed-forward networks, facilitating deep fusion and processing of multi-modal data. Additionally, we enhance the positional embeddings to incorporate the spatial information of OCR tokens using techniques inspired by "GRPE: Relative Positional Encoding for Graph Transformer," ensuring the model captures the document's layout and structure accurately【44†source】【45†source】.

Further, we integrate a prompt encoder and a text decoder to enable interactive document analysis and question-answering tasks. The prompt encoder processes user prompts, converting them into embeddings that interact with the integrated text-image embeddings. The text decoder generates the relevant output based on these embeddings, producing sequences of text and corresponding bounding boxes, which map answers to specific regions in the document. This design facilitates tasks such as information extraction and document navigation【61†source】【63†source】.

Our pre-training strategy involves three key objectives: Masked Language Modeling (MLM), Masked Image Modeling (MIM), and Word-Patch Alignment (WPA). These objectives are inspired by techniques from "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking," and are designed to enable the model to learn comprehensive multimodal representations effectively. We employ a prompt-based fine-tuning strategy, leveraging instruction tuning to adapt the model to various downstream tasks in a zero-shot or few-shot setting【45†source】【53†source】【61†source】【63†source】.

Our proposed model aims to provide a unified representation that can be used for various downstream tasks, such as document classification, named entity recognition, and document layout analysis. We evaluate our model on several benchmark datasets, including PubTables-1M and PubLayNet, demonstrating its superior performance compared to existing state-of-the-art methods.

Our contributions can be summarized as follows:
1. We introduce a novel multi-modal foundation model that integrates image and text data using a Multiway Transformer architecture.
2. We leverage the BEITv2 approach to pre-train the image encoder with CLIP embeddings, enhancing its semantic understanding.
3. We adapt a pre-trained LLM for text encoding using the LLM2Vec approach, optimizing it for OCR JSON input.
4. We incorporate a prompt encoder and a text decoder to facilitate interactive document analysis and question-answering tasks.
5. We employ a comprehensive pre-training and fine-tuning strategy, demonstrating the model's effectiveness across various downstream applications.

This research advances the state-of-the-art in multi-modal foundation models, providing a robust and flexible framework for document analysis and related tasks.

---

### Methodology

### Datasets for Pre-training and Fine-tuning

#### Pre-training and Fine-tuning Datasets

**IDL-WDS:**
The IDL-WDS (Industry Documents Library - Web Dataset) is a comprehensive dataset comprising around 19 million pages of document images and corresponding OCR outputs in JSON format. This dataset is designed to facilitate robust pre-training of models on a diverse range of document types and structures.

- **Size and Composition:** Approximately 19 million pages, including PDF files, TIFF images, and JSON files with Textract OCR annotations.
- **Processing:**
  - **Image Preprocessing:** Convert images to 1024x1024 pixels in TIFF format to ensure high-quality and uniform input data.
  - **OCR JSON Parsing:** Extract word bounding boxes and text from JSON files, normalizing the bounding box coordinates relative to the 1024x1024 image size.
  - **Dataloader Implementation with LitData:** Use the `litdata` library from Lightning-AI for efficient data processing. This library supports data loading, transformation, and batching, optimizing the pipeline for large-scale datasets.

**PDFA-ENG-WDS:**
The PDFA-ENG-WDS dataset focuses on English PDF documents and provides OCR annotations and bounding boxes for words within the documents.

- **Size and Composition:** Spanning approximately 1.5TB, with over 26 million pages and 18 billion text tokens.
- **Processing:**
  - **Sharded Storage:** The dataset is stored in `.tar` files, compatible with efficient streaming and processing using the `litdata` library.
  - **Text and Bounding Box Extraction:** Normalize the bounding box coordinates relative to the 1024x1024 image size and convert the text to a suitable format for model input.
  - **Optimized Dataloader with LitData:** Utilize the `litdata` library for efficient loading of large, sharded datasets, supporting parallel processing and handling large-scale data effectively.

**PubTables-1M:**
The PubTables-1M dataset is designed for table detection and structure recognition within documents, offering an extensive collection of annotated data to pre-train and fine-tune models for layout analysis and table structure extraction tasks.

- **Size and Composition:** The dataset includes approximately 947,642 cropped table instances and 575,305 document page instances, providing comprehensive annotations for both table structure and document layout:
  - **Structure Recognition Data:** Includes images and annotations for train, test, and validation sets, with XML files detailing table structures and word bounding boxes.
  - **Detection Data:** Contains images and annotations for detecting table locations within full document pages, along with word-level bounding box information.

- **Processing and Loading:**
  - **Download and Extraction:** Use provided scripts to download and organize the dataset.
  - **Transformation:** Standardize images to 1024x1024 pixels and normalize bounding box coordinates relative to this size.
  - **Dataloader Implementation with LitData:** Utilize the `litdata` library to efficiently load, transform, and batch the data.

**PubLayNet:**
PubLayNet is a large-scale dataset aimed at document layout analysis, including annotations for text, titles, lists, tables, and figures within research paper images. This dataset is particularly valuable for pre-training and fine-tuning models for tasks such as document classification and named entity recognition.

- **Size and Composition:** PubLayNet contains over 1 million annotated document images, sourced from PubMed Central articles. The dataset includes detailed annotations for various layout components:
  - **Annotations:** The dataset provides bounding boxes and labels for text blocks, titles, lists, tables, and figures. It is divided into training, validation, and test splits:
    - **Training Set:** Approximately 335,703 images
    - **Validation Set:** Approximately 11,245 images
    - **Test Set:** Approximately 11,405 images

- **Processing and Loading:**
  - **Download and Extraction:** Access the dataset files via Hugging Face, ensuring all necessary files are downloaded.
  - **Transformation:** Convert images to 1024x1024 pixels, and normalize the bounding boxes relative to this size for consistency.
  - **Dataloader Implementation with LitData:** Use the `litdata` library to handle large-scale data efficiently, with capabilities for shuffling, batching, and parallel processing.

### Datasets Summary and Processing Strategy

- **Consistency:** Standardize image resolutions to 1024x1024 pixels and convert them to TIFF format to ensure high-quality, uniform input data.
- **Normalization:** Normalize bounding box coordinates in the OCR JSON files relative to the 1024x1024 image dimensions.
- **Efficient Loading:** Use the `litdata` library for data processing, `torchvision` for image handling, and efficient methods for managing large, sharded datasets. This ensures efficient data loading and processing, reducing bottlenecks during training.
- **Metadata Utilization:** Leverage metadata such as file sizes and rendering times to filter out large or slow-to-render files, optimizing the dataset for efficient training at scale.

By following these strategies, we can efficiently process and utilize the IDL-WDS, PDFA-ENG-WDS, PubTables-1M, and PubLayNet datasets for pre-training and fine-tuning our multi-modal foundation model, ensuring high performance across various document analysis tasks.

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

#### Model Architecture

**Image Encoder:**
The image encoder in our model is based on the Vision Transformer (ViT) model, pre-trained using the BEITv2 approach. BEITv2 leverages a visual tokenizer to convert images into discrete visual tokens, enabling masked image modeling (MIM). Specifically, approximately 40% of image patches are masked, and the model is trained to predict the CLIP embeddings of these masked patches. This technique ensures that the model captures high-level visual representations and is robust in understanding the visual content in documents. The pretraining also involves a [CLS] token to aggregate patch information into global representations, enhancing the model’s ability to generate comprehensive visual embeddings【52†source】【53†source】.

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

**Text Encoder:**
The text encoder utilizes a pre-trained Large Language Model (LLM) such as Llama3 or Mistral, adapted using the LLM2Vec approach. This process includes enabling bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning using the SimCSE technique. The goal is to transform a decoder-only LLM into a strong text encoder. The steps involve fine-tuning the model to predict masked tokens and using dropout techniques to create positive examples for contrastive learning. This adaptation is crucial for handling the structured nature of OCR outputs, as described in "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders."

**Integration with Multiway Transformer:**
To integrate text and image embeddings, we employ a Multiway Transformer architecture. Each block consists of a shared self-attention module and a pool of feed-forward networks tailored for different modalities (vision, language, and vision-language). This design facilitates deep fusion of multi-modal data and modality-specific processing, making it highly effective for tasks involving complex interactions between text and images.

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

**Positional Embeddings for Bounding Boxes:**
We enhance the positional embeddings to incorporate the spatial information of OCR tokens using techniques inspired by "GRPE: Relative Positional Encoding for Graph Transformer." This involves encoding the relative positional relationships between nodes in a graph, specifically tailored for OCR tasks. By embedding this spatial information directly into the transformer's self-attention mechanism, the model captures topological and edge-based relationships between textual elements in a document, improving its performance in tasks requiring an understanding of the document's visual and spatial context.

**Prompt Encoder and Text Decoder:**
After integrating the image and text embeddings using the Multiway Transformer, we incorporate a prompt encoder and a text decoder to facilitate interactive document analysis and question-answering tasks:
- **Prompt Encoder:** This component processes user prompts, converting them into embeddings that interact with the integrated text-image embeddings. The prompt encoder is designed to understand the context and specific requirements of the query, enabling precise and context-aware responses【61†source】【63†source】.
- **Text Decoder:** Based on the embeddings from the Multiway Transformer and the prompt encoder, the text decoder generates the relevant output. The decoder is trained to produce sequences of text and corresponding bounding boxes, effectively mapping answers to specific regions in the document. This allows the model to output not only textual answers but also their precise locations within the document, facilitating tasks such as information extraction and document navigation.

#### Pre-training Strategy

Our pre-training strategy employs three key objectives: Masked Language Modeling (MLM), Masked Image Modeling (MIM), and Word-Patch Alignment (WPA). These objectives, inspired by techniques in "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking," are designed to enable the model to learn multimodal representations effectively.

1. **Masked Language Modeling (MLM):**
   - We mask 30% of text tokens using a span masking strategy, where span lengths follow a Poisson distribution. The objective is to maximize the log-likelihood of predicting the masked tokens based on the contextual representations from both text and image sequences. This helps the model learn the correspondence between textual content and layout information.

2. **Masked Image Modeling (MIM):**
   - We mask approximately 40% of image tokens using a blockwise masking strategy. The model is then trained to reconstruct the masked tokens using a cross-entropy loss, encouraging it to interpret visual content through the context provided by both text and image tokens. This objective ensures the model captures high-level visual structures rather than low-level details.

3. **Word-Patch Alignment (WPA):**
   - This objective aligns text words with their corresponding image patches. It involves predicting whether the image patches corresponding to a text word are masked. By assigning binary aligned/unaligned labels to unmasked text tokens based on their image patch masking status, the model learns a fine-grained alignment between text and image modalities. This is critical for tasks requiring precise text-image correspondence.

#### Fine-tuning Strategy

For fine-tuning, we adopt a prompt-based strategy inspired by methods described in the Segment Anything Model (SAM). This approach enables the model to respond appropriately to a variety of prompts, which can be engineered to solve different downstream tasks. The fine-tuning process involves:
- **Instruction Tuning the Text Decoder:** We fine-tune the text decoder to handle diverse downstream tasks by providing it with specific instructions on how to generate the relevant output. This includes producing text sequences and identifying the corresponding bounding boxes in response to different types of queries【61†source】【63†source】.
- **Zero-Shot and Few-Shot Learning:** The prompt-based strategy leverages the model’s pre-trained capabilities to perform tasks in a zero-shot or few-shot setting. By carefully designing prompts, we can adapt the model to various applications such as document layout analysis, document classification, and named entity recognition without extensive additional training【61†source】【63†source】.

This comprehensive fine-tuning strategy ensures that the model remains flexible and capable of performing a wide range of tasks efficiently, leveraging the strengths of both the pre-trained embeddings and the prompt-based interaction mechanisms.


### Methodology

#### Model Architecture

**Image Encoder:**
The image encoder in our model is based on the Vision Transformer (ViT) model, pre-trained using the BEITv2 approach. BEITv2 leverages a visual tokenizer to convert images into discrete visual tokens, enabling masked image modeling (MIM). Specifically, approximately 40% of image patches are masked, and the model is trained to predict the CLIP embeddings of these masked patches. This technique ensures that the model captures high-level visual representations and is robust in understanding the visual content in documents. The pretraining also involves a [CLS] token to aggregate patch information into global representations, enhancing the model’s ability to generate comprehensive visual embeddings【52†source】【53†source】.

**Text Encoder:**
The text encoder utilizes a pre-trained Large Language Model (LLM) such as Llama3 or Mistral, adapted using the LLM2Vec approach. This process includes enabling bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning using the SimCSE technique. The goal is to transform a decoder-only LLM into a strong text encoder. The steps involve fine-tuning the model to predict masked tokens and using dropout techniques to create positive examples for contrastive learning. This adaptation is crucial for handling the structured nature of OCR outputs, as described in "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders"【38†source】.

**Integration with Multiway Transformer:**
To integrate text and image embeddings, we employ a Multiway Transformer architecture. Each block consists of a shared self-attention module and a pool of feed-forward networks tailored for different modalities (vision, language, and vision-language). This design facilitates deep fusion of multi-modal data and modality-specific processing, making it highly effective for tasks involving complex interactions between text and images【44†source】.

**Positional Embeddings for Bounding Boxes:**
We enhance the positional embeddings to incorporate the spatial information of OCR tokens using techniques inspired by "GRPE: Relative Positional Encoding for Graph Transformer." This involves encoding the relative positional relationships between nodes in a graph, specifically tailored for OCR tasks. By embedding this spatial information directly into the transformer's self-attention mechanism, the model captures topological and edge-based relationships between textual elements in a document, improving its performance in tasks requiring an understanding of the document's visual and spatial context【45†source】.

**Prompt Encoder and Text Decoder:**
After integrating the image and text embeddings using the Multiway Transformer, we incorporate a prompt encoder and a text decoder to facilitate interactive document analysis and question-answering tasks:
- **Prompt Encoder:** This component processes user prompts, converting them into embeddings that interact with the integrated text-image embeddings. The prompt encoder is designed to understand the context and specific requirements of the query, enabling precise and context-aware responses【61†source】【63†source】.
- **Text Decoder:** Based on the embeddings from the Multiway Transformer and the prompt encoder, the text decoder generates the relevant output. The decoder is trained to produce sequences of text and corresponding bounding boxes, effectively mapping answers to specific regions in the document. This allows the model to output not only textual answers but also their precise locations within the document, facilitating tasks such as information extraction and document navigation.

#### Pre-training Strategy

Our pre-training strategy employs three key objectives: Masked Language Modeling (MLM), Masked Image Modeling (MIM), and Word-Patch Alignment (WPA). These objectives, inspired by techniques in "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking," are designed to enable the model to learn multimodal representations effectively.

1. **Masked Language Modeling (MLM):**
   - We mask 30% of text tokens using a span masking strategy, where span lengths follow a Poisson distribution. The objective is to maximize the log-likelihood of predicting the masked tokens based on the contextual representations from both text and image sequences. This helps the model learn the correspondence between textual content and layout information【45†source】.

2. **Masked Image Modeling (MIM):**
   - We mask approximately 40% of image tokens using a blockwise masking strategy. The model is then trained to reconstruct the masked tokens using a cross-entropy loss, encouraging it to interpret visual content through the context provided by both text and image tokens. This objective ensures the model captures high-level visual structures rather than low-level details【53†source】.

3. **Word-Patch Alignment (WPA):**
   - This objective aligns text words with their corresponding image patches. It involves predicting whether the image patches corresponding to a text word are masked. By assigning binary aligned/unaligned labels to unmasked text tokens based on their image patch masking status, the model learns a fine-grained alignment between text and image modalities. This is critical for tasks requiring precise text-image correspondence【45†source】.

#### Fine-tuning Strategy

For fine-tuning, we adopt a prompt-based strategy inspired by methods described in the Segment Anything Model (SAM). This approach enables the model to respond appropriately to a variety of prompts, which can be engineered to solve different downstream tasks. The fine-tuning process involves:
- **Instruction Tuning the Text Decoder:** We fine-tune the text decoder to handle diverse downstream tasks by providing it with specific instructions on how to generate the relevant output. This includes producing text sequences and identifying the corresponding bounding boxes in response to different types of queries. Instruction tuning allows the model to adapt to various tasks, enhancing its versatility and effectiveness【61†source】【63†source】.
- **Zero-Shot and Few-Shot Learning:** The prompt-based strategy leverages the model’s pre-trained capabilities to perform tasks in a zero-shot or few-shot setting. By carefully designing prompts, we can adapt the model to various applications such as document layout analysis, document classification, and named entity recognition without extensive additional training【61†source】【63†source】.

This comprehensive fine-tuning strategy ensures that the model remains flexible and capable of performing a wide range of tasks efficiently, leveraging the strengths of both the pre-trained embeddings and the prompt-based interaction mechanisms.

---

### Related Work

The development of multi-modal foundation models has seen significant advancements in recent years, particularly in integrating text and image data to enhance various applications such as document analysis, information retrieval, and natural language processing.

**BEIT and BEITv2:**
The BEIT (Bao et al., 2022) and BEITv2 models introduced a new paradigm for image pre-training by leveraging masked image modeling (MIM) with visual tokenizers. BEITv2, in particular, improved this approach by using a visual tokenizer trained with a vector-quantized knowledge distillation (VQ-KD) method, allowing the model to capture more comprehensive visual representations by predicting CLIP embeddings for masked image patches【52†source】【53†source】.

**LLM2Vec:**
LLM2Vec demonstrated the capability of transforming decoder-only language models into powerful text encoders through masked next token prediction (MNTP) and unsupervised contrastive learning (SimCSE). This approach enabled the effective handling of structured text data, such as OCR outputs, by adapting the causal attention mechanism to a bidirectional one, thus improving contextual understanding【38†source】.

**Multiway Transformer:**
The Multiway Transformer architecture, as utilized in recent multi-modal models, provides a robust framework for integrating text and image embeddings. By incorporating shared self-attention modules and modality-specific feed-forward networks, this architecture facilitates deep fusion and processing of multi-modal data, enhancing the model's ability to handle complex interactions between different data types【44†source】.

**LayoutLM and LayoutLMv3:**
The LayoutLM series of models, particularly LayoutLMv3, have made significant contributions to pre-training strategies for document AI. These models use unified text-image multimodal transformers and employ objectives such as masked language modeling (MLM), masked image modeling (MIM), and word-patch alignment (WPA) to learn cross-modal representations. This approach ensures that the models can effectively capture the relationships between textual and visual elements in documents【45†source】.

**Segment Anything Model (SAM):**
The Segment Anything Model (SAM) introduces a prompt-based approach to segmentation tasks, enabling the model to respond to various prompts such as points, boxes, or masks. This flexibility allows SAM to perform well in zero-shot and few-shot settings, adapting to different segmentation tasks by engineering appropriate prompts. The methodology behind SAM highlights the potential of prompt engineering to generalize across multiple tasks, making it a versatile tool for segmentation and other interactive tasks【61†source】【62†source】【63†source】.

Our proposed multi-modal foundation model builds upon these advancements by integrating a BEITv2-based image encoder, an LLM2Vec-based text encoder, and a Multiway Transformer for embedding fusion. Additionally, we incorporate a prompt encoder and a text decoder to facilitate interactive document analysis and question-answering tasks, leveraging instruction tuning to enhance the model's adaptability to various downstream applications.

---

### Experiments

To evaluate the effectiveness of our proposed multi-modal foundation model, we conduct a series of experiments across different tasks, including pre-training, fine-tuning, and downstream task performance.

**Pre-training Setup:**
1. **Datasets:** We use the IDL-WDS and PDFA-ENG-WDS datasets for pre-training. These datasets provide a diverse collection of document images and corresponding OCR outputs, allowing the model to learn from various document types and structures.
2. **Objectives:** The pre-training involves masked language modeling (MLM), masked image modeling (MIM), and word-patch alignment (WPA) objectives, ensuring the model captures comprehensive multimodal representations.
3. **Evaluation Metrics:** We evaluate the model's performance on pre-training tasks using metrics such as cross-entropy loss for MLM and MIM, and binary cross-entropy loss for WPA.

#### Fine-tuning Strategy

For fine-tuning, we adopt a prompt-based strategy inspired by methods described in the Segment Anything Model (SAM). This approach enables the model to respond appropriately to a variety of prompts, which can be engineered to solve different downstream tasks. The fine-tuning process involves:
- **Instruction Tuning the Text Decoder:** We fine-tune the text decoder to handle diverse downstream tasks by providing it with specific instructions on how to generate the relevant output. This includes producing text sequences and identifying the corresponding bounding boxes in response to different types of queries. Instruction tuning allows the model to adapt to various tasks, enhancing its versatility and effectiveness【61†source】【63†source】.
- **Zero-Shot and Few-Shot Learning:** The prompt-based strategy leverages the model’s pre-trained capabilities to perform tasks in a zero-shot or few-shot setting. By carefully designing prompts, we can adapt the model to various applications such as document layout analysis, document classification, and named entity recognition without extensive additional training【61†source】【63†source】.

This comprehensive fine-tuning strategy ensures that the model remains flexible and capable of performing a wide range of tasks efficiently, leveraging the strengths of both the pre-trained embeddings and the prompt-based interaction mechanisms.

### Evaluation Metrics for Pre-training and Fine-tuning

To ensure the effectiveness and robustness of our multi-modal foundation model, we will employ a variety of evaluation metrics at different stages of pre-training and fine-tuning. These metrics will help assess the model’s performance comprehensively.

#### Pre-training Evaluation Metrics

1. **Image Encoder (BEITv2-based):**
   - **Reconstruction Loss:** Measures how well the model reconstructs the masked image patches using the CLIP embeddings. This is typically evaluated using mean squared error (MSE) or cross-entropy loss.
   - **Patch-wise Accuracy:** The accuracy of predicting the correct visual tokens for the masked patches.
   - **Global Representation Quality:** Assessed using linear probing on a downstream image classification task (e.g., ImageNet), evaluating metrics such as top-1 and top-5 accuracy【52†source】【53†source】.

2. **Text Encoder (LLM2Vec-based):**
   - **Masked Language Modeling (MLM) Accuracy:** Measures the accuracy of predicting masked tokens in the text, evaluated using cross-entropy loss.
   - **Perplexity:** A measure of how well the probability distribution predicted by the model aligns with the actual distribution of the data. Lower perplexity indicates a better language model.
   - **Contrastive Learning Metrics:** For SimCSE, metrics like cosine similarity between positive pairs and the alignment and uniformity of the learned representations【38†source】.

3. **Multiway Transformer Integration:**
   - **Alignment Loss:** Measures how well the text and image embeddings are aligned, typically using cosine similarity or a similar metric.
   - **Cross-modal Retrieval Accuracy:** Evaluates how accurately the model can retrieve relevant text given an image query, and vice versa, using metrics such as mean reciprocal rank (MRR) and normalized discounted cumulative gain (nDCG).

#### Fine-tuning Evaluation Metrics

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

#### Evaluation during Pre-training and Fine-tuning

1. **Validation Loss:** Continuous monitoring of validation loss during pre-training and fine-tuning helps ensure that the model is not overfitting and generalizes well to unseen data.
2. **Early Stopping Metrics:** Metrics like validation loss and early stopping criteria (based on patience and delta values) can help prevent overfitting during training.
3. **Ablation Studies:** Conducting ablation studies to evaluate the impact of different components (e.g., removing the prompt encoder or text decoder) on overall performance.

By utilizing these diverse evaluation metrics, we can comprehensively assess the performance and robustness of our multi-modal foundation model at each stage of pre-training and fine-tuning, ensuring its effectiveness across various downstream tasks.

Some questions:

1. **Dataset:**
   - Do you have a specific dataset you plan to use for pre-training and finetuning? If so, could you provide details about it?
   - Are there any specific datasets you plan to use for evaluating downstream tasks like document classification or named entity recognition?

2. **Model Details:**
   - Could you provide more details on how you plan to integrate the text and image embeddings through the cross-attention layer?
   - What specific modifications will you apply to the LLM to accommodate the OCR JSON format?

3. **Pre-training and Finetuning:**
   - What is your strategy for pre-training the model? How will you mask the image patches and text tokens?
   - How will you handle the finetuning process for the downstream tasks?

4. **Evaluation:**
   - What metrics will you use to evaluate the performance of your model on the pre-training tasks (masked language and image modeling)?
   - What metrics will you use for the downstream tasks?

5. **Baseline Comparisons:**
   - Do you have any baseline models for comparison? If so, what are they, and how will you compare their performance with your proposed model?

6. **Expected Contributions:**
   - What are the key contributions of your work? How does it advance the state-of-the-art in multi-modal foundation models?

7. **Challenges and Limitations:**
   - Are there any specific challenges or limitations you anticipate with your approach? How do you plan to address or mitigate them?

---