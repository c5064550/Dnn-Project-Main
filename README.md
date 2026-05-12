Gated Multimodal Memory Matching (MMM): Advanced Sequential Reasoning in Visual Storytelling
Final Project Assessment
Subject: Deep Neural Networks - Multimodal Systems
Abstract
This report details the implementation of a modified multimodal architecture designed for next-frame prediction in visual narratives. By replacing static concatenation with a learnable Gated Multimodal Fusion mechanism and intro- ducing Multimodal Memory Masking (MMM), the system achieves superior cross-modal alignment. We present three experiments evaluating fusion strategies, robustness, and explainability. Experi- mental results indicate that this adaptive approach significantly improves convergence stability and narrative consistency, as verified by both quanti- tative metrics and qualitative visualizations.
1 INTRODUCTION
Predicting the progression of a story requires the in- tegration of historical context across visual and tex- tual domains. Traditional architectures often fail to weigh these modalities appropriately, leading to se- mantic drift. This project introduces a gated frame- work that dynamically adjusts modality importance based on temporal context and narrative content.
2 METHODOLOGY
2.1 Architectural Components
The system consists of three core modules:
• Vision Module: A dual-pathway CNN extracting object-level content and global context.
• NLP Module: An LSTM Seq2Seq model using BERT-based tokenization for robust text latent representation.
• Temporal Module: A Gated Recurrent Unit (GRU) with a Softmax-based attention layer for se- quence modeling.
3 3.1
EXPERIMENTS AND RESULTS
Exp 1: Baseline vs. Gated Fusion
May 12, 2026
2.2 Gated Multimodal Fusion
The fused representation z is computed via a learnable sigmoid gate g:
z = g · zt + (1 − g) · zv (1)
where zt is the text embedding and zv is the visual em- bedding. This allows for adaptive multimodal align- ment instead of fixed fusion.
1
The objective was to evaluate if replacing static con- catenation with gated fusion improves reasoning. The gated model demonstrated improved multimodal alignment and stronger contextual consistency in gen- erated text compared to the baseline.
3.2 Exp 2: Multimodal Memory Masking (MMM)
By masking textual tokens during training, the model was forced to rely on visual context. Results showed that MMM masking significantly improved contextual reasoning and multimodal robustness compared to the baseline model.
3.3 Exp 3: Explainability Visualization
Attention visualizations show which previous frames contribute most strongly, while Gate Heatmaps show how the model dynamically balances image and text information. The model adaptively changes modality importance depending on scene context.
4 DIFFICULTIES FACED
Several implementation challenges were encoun- tered:
• Loss Balancing: Managing the scale difference between image reconstruction and text genera- tion losses.
• Image Quality: Addressing blurry image genera- tion caused by lightweight decoders.
• Hardware: Computational limitations on Google Colab required optimized batch management.
5 CONCLUSION
The Gated MMM architecture demonstrates that learnable fusion and strategic masking are superior to static methods. The resulting model is both more accurate in its narrative predictions and more inter- pretable through its gating mechanism.
REFERENCES
PyTorch Documentation (2024). https://pytorch.org Sutskever et al. (2014). Sequence to Sequence Learning. Vaswani et al. (2017). Attention Is All You Need.
Radford et al. (2021). CLIP: Learning Transferable Visual Mod- els.
