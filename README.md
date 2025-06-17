# AI & ML Resources

A curated collection of videos, websites, software, example projects, and cloud/hosted LLM platforms to help you get up and running with modern AI and machine learning.

---

## 🎥 Videos

- **Two Minute Papers**  
  [YouTube Channel](https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg)

---

## 🌐 Websites

- **Papers with Code**  
  https://paperswithcode.com/

- **Hugging Face**  
  https://huggingface.co/

- **Google Colab**  
  https://colab.research.google.com/

---

## 🛠 Software

- **TensorFlow**  
  https://www.tensorflow.org/

- **PyTorch**  
  https://pytorch.org/

- **Hugging Face Transformers**  
  https://huggingface.co/docs/transformers/en/index

- **Hugging Face Datasets**  
  https://huggingface.co/docs/datasets/en/index

- **llama.cpp**  
  https://github.com/ggml-org/llama.cpp

- **ik_llama.cpp**  
  https://github.com/ikawrakow/ik_llama.cpp

- **VLLM**  
  https://github.com/vllm-project/vllm

- **llama-swap**  
  https://github.com/mostlygeek/llama-swap

- **Ollama** *(not recommended: slow, proprietary weights)*  
  https://ollama.com/

- **LM Studio** *(beginner‑friendly)*  
  https://lmstudio.ai/

- **Comfy UI** *(for generative image workflows)*  
  https://www.comfy.org/

- **OpenWebUI** *(ChatGPT‑style web interface)*  
  https://openwebui.com/

---

## 📚 Example Projects

- **TensorFlow / Keras**  
  - [MNIST Example](https://www.tensorflow.org/datasets/keras_example)  
  - [Text Classification Tutorial](https://www.tensorflow.org/tutorials/keras/text_classification)

- **PyTorch**  
  - [Variational Autoencoder (VAE)](https://github.com/pytorch/examples/tree/main/vae)

- **LLM Inference**  
  - [Hugging Face LLM Tutorial](https://huggingface.co/docs/transformers/en/llm_tutorial)

- **Fine‑Tuning**  
  - [Unsloth Fine‑Tuning Guide](https://docs.unsloth.ai/get-started/fine-tuning-guide)

---

## ☁️ Cloud Platforms

- **AWS**  
  - [SageMaker](https://aws.amazon.com/sagemaker/)  
  - [Bedrock](https://aws.amazon.com/bedrock/)

---

## 🤖 Hosted LLM Leaderboards

- [ArtificialAnalysis.ai Providers](https://artificialanalysis.ai/leaderboards/providers)

---


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

# configure hugginface to help you
- https://huggingface.co/settings/local-apps
- add hardware (GPU, CPU/RAM, etc)
- enable ollama, lmstudio, llama.cpp, vllm

# how to use this repo
- `docker compose up`
- I have another compose file that is much more messy, but also includes other services (tabby, comfyUI (for generating images))

# how to download a model (ollama)
- `docker exec -it ollama bash`
- Find a model, for example: https://huggingface.co/unsloth/QwQ-32B-GGUF?show_file_info=QwQ-32B-IQ4_XS.gguf
- `ollama pull hf.co/unsloth/QwQ-32B-GGUF:IQ4_XS`
- Once it's downloaded, you should be able to see/use it in openwebui
