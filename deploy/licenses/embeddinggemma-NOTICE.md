# Notice

This image bundles **EmbeddingGemma-300m**, a Google Gemma model, as an
ONNX export (`model.onnx`, INT8-quantized; `tokenizer.json`) taken from
https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX at the
revision recorded in the image's build. The original model is
https://huggingface.co/google/embeddinggemma-300m.

Gemma is provided under and subject to the Gemma Terms of Use found at
https://ai.google.dev/gemma/terms, reproduced in `LICENSE` beside this
file. Use of the model is also subject to the Gemma Prohibited Use Policy at
https://ai.google.dev/gemma/prohibited_use_policy. By using this image you
accept those terms.

Modifications relative to the original weights: export to ONNX and INT8
quantization, performed by the onnx-community publisher; Mnem-O-matic adds
no further changes and applies the model's documented task prefixes at
inference time.
