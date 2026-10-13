Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms

# Notice

This image bundles **EmbeddingGemma-300m**, a Google Gemma model, as an
ONNX export (`model.onnx`, INT8-quantized; `tokenizer.json`) taken from
https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX at the
revision recorded in the image's build. The original model is
https://huggingface.co/google/embeddinggemma-300m.

The Gemma Terms of Use are reproduced in `LICENSE` beside this file. Their
use restrictions (Section 3.2) apply to this model: you must not use it for
the restricted uses set out in the Gemma Prohibited Use Policy at
https://ai.google.dev/gemma/prohibited_use_policy, nor in violation of
applicable law. By using this image you accept those terms.

Modifications relative to the original weights: export to ONNX and INT8
quantization, performed by the onnx-community publisher; Mnem-O-matic adds
no further changes and applies the model's documented task prefixes at
inference time.
