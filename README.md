# estimate required VRAM/RAM:
- https://apxml.com/tools/vram-calculator
- https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator

# run models locally (diferent options):
- https://lmstudio.ai/ (ease: high, speed: medium, (but has a GUI, which depending on your use case, you may not need/want))
- https://github.com/ollama/ollama (ease: high, speed: low)
- https://github.com/ggml-org/llama.cpp (ease: medium, speed: medium)
- https://github.com/vllm-project/vllm (ease: low, speed: high)
- https://github.com/theroyallab/tabbyAPI (ease: very low, speed: very high)

# webUI:
- https://github.com/open-webui/open-webui

# Where to find models:
- https://huggingface.co/models


# how to download a model (ollama)
- `docker exec -it ollama bash`
- Find a model, for example: https://huggingface.co/unsloth/QwQ-32B-GGUF?show_file_info=QwQ-32B-IQ4_XS.gguf
- `ollama pull hf.co/unsloth/QwQ-32B-GGUF:IQ4_XS`
- Once it's downloaded, you should be able to see/use it in openwebui
