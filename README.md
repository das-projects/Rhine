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

In recent years, there has been significant progress in the development of foundation models that can process and understand multi-modal data, such as images and text. These models have shown great promise in various applications, including document analysis, information retrieval, and natural language processing. However, existing approaches often rely on separate encoders for images and text, which limits their ability to fully leverage the interactions between different modalities.

In this paper, we propose a novel multi-modal foundation model that integrates image and text data at a deeper level. Our model accepts an image of a document and its corresponding OCR output in JSON format, which includes a list of tuples of words and their bounding boxes. The image encoder is based on the Vision Transformer (VIT) model, pre-trained using a method inspired by BEITv2. Instead of generating image patches directly, our model generates CLIP embeddings of these patches, leveraging the rich semantic information captured by CLIP.

The text encoder is derived from a pre-trained Large Language Model (LLM) such as Llama3 or Mistral. We modify the causal attention layer to bi-directional attention, following the approach in LLM2Vec, and finetune the model for next token prediction on a dataset formatted in JSON OCR. The embeddings from the text and image encoders are then passed through a multiway transformer, allowing for a more intricate fusion of multi-modal information. Additionally, we adjust the positional embeddings to incorporate the bounding box information of the OCR tokens, as described in recent advancements in transformer architectures.

Our proposed model aims to provide a unified representation that can be used for various downstream tasks, such as document classification and named entity recognition. We evaluate our model on several benchmark datasets, including PubTables-1M and PubLayNet, and demonstrate its superior performance compared to existing state-of-the-art methods.

Our contributions can be summarized as follows:
1. We introduce a novel multi-modal foundation model that integrates image and text data using cross-attention mechanisms.
2. We leverage the BEITv2 approach to pre-train the image encoder with CLIP embeddings, enhancing its semantic understanding.
3. We adapt a pre-trained LLM for text encoding, optimizing it for OCR JSON input.
4. We jointly pre-train the model on masked language and image modeling tasks, demonstrating its effectiveness on various downstream applications.
5. We utilize prompt-based fine-tuning to enable zero-shot or few-shot performance on diverse document analysis tasks.

---

### Methodology

#### Datasets

**Pre-training Datasets:**
1. **IDL-WDS**: This dataset comprises a diverse collection of document images paired with OCR outputs in JSON format. Each JSON file contains text and corresponding bounding boxes, structured to facilitate the training of models in reading and understanding document layouts. The dataset includes approximately 7,000 rows, offering substantial data for robust pre-training. The documents are processed to determine reading order and columnar structure, enhancing the model's ability to handle complex document formats【11†source】【12†source】.

2. **PDFA-ENG-WDS**: Derived from the SafeDocs corpus, this dataset focuses on English PDF documents, providing OCR annotations and bounding boxes for words within the documents. It includes metadata such as file sizes and rendering times to optimize loading and processing during training. This dataset, totaling around 1.5TB, is filtered to ensure data quality and consistency, making it ideal for training vision-language models【13†source】【14†source】【15†source】.

**Fine-tuning and Evaluation Datasets:**
1. **PubTables-1M**: This dataset is designed for table detection and structure recognition in documents. It includes extensive annotations for table structures within a large number of document images, making it suitable for evaluating and fine-tuning models for layout detection tasks.

2. **PubLayNet**: PubLayNet provides annotated document images for layout analysis, including text, tables, figures, and headers. The dataset is valuable for tasks like document classification and named entity recognition, offering a rich source of labeled data for fine-tuning multi-modal models.

#### Model Architecture

**Image Encoder:**
The image encoder in our model is based on the Vision Transformer (ViT) model, pre-trained using the BEITv2 approach. BEITv2 leverages a visual tokenizer to convert images into discrete visual tokens, enabling masked image modeling (MIM). Specifically, approximately 40% of image patches are masked, and the model is trained to predict the CLIP embeddings of these masked patches. This technique ensures that the model captures high-level visual representations and is robust in understanding the visual content in documents. The pretraining also involves a [CLS] token to aggregate patch information into global representations, enhancing the model’s ability to generate comprehensive visual embeddings【52†source】【53†source】.

**Text Encoder:**
The text encoder utilizes a pre-trained Large Language Model (LLM) such as Llama3 or Mistral, adapted using the LLM2Vec approach. This process includes enabling bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning using the SimCSE technique. The goal is to transform a decoder-only LLM into a strong text encoder. The steps involve fine-tuning the model to predict masked tokens and using dropout techniques to create positive examples for contrastive learning. This adaptation is crucial for handling the structured nature of OCR outputs, as described in "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders."

**Integration with Multiway Transformer:**
To integrate text and image embeddings, we employ a Multiway Transformer architecture. Each block consists of a shared self-attention module and a pool of feed-forward networks tailored for different modalities (vision, language, and vision-language). This design facilitates deep fusion of multi-modal data and modality-specific processing, making it highly effective for tasks involving complex interactions between text and images.

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