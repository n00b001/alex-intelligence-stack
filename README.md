# AI & ML Resources

A curated collection of videos, websites, software, example projects, cloud/hosted LLM platforms, and tools to help you get up and running with modern AI and machine learning.

---

## 🎥 Videos

- **Two Minute Papers**  
  Bite‑sized deep dives into the latest AI research  
  https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg

- **Yannic Kilcher**  
  Detailed walkthroughs of new papers and techniques  
  https://www.youtube.com/c/YannicKilcher

---

## 🌐 Websites & Communities

| Category         | Resource                                                     | Notes                                   |
|------------------|--------------------------------------------------------------|-----------------------------------------|
| **Papers & Code**| https://paperswithcode.com/                                  | SOTA benchmarks + code links            |
| **Model Hub**    | https://huggingface.co/                                      | Transformers, datasets, model hosting   |
| **Notebooks**    | https://colab.research.google.com/                           | Free GPU/TPU for prototyping            |
| **Books**        | [Deep Learning](https://www.deeplearningbook.org/)           | Goodfellow, Bengio & Courville         |
|                  | [Hands‑On ML with Scikit‑Learn, Keras & TF](https://www.oreilly.com/) | Practical end‑to‑end projects   |
| **Blogs**        | https://distill.pub/                                         | Interactive deep dives                  |
|                  | https://thegradient.pub/                                     | Analysis, trends, paper summaries       |
| **Communities**  | https://www.reddit.com/r/MachineLearning/                    | Active discussion + resources           |
|                  | https://discord.gg/huggingface                               | Hugging Face official server            |

---

## 🛠 Software & Frameworks

- **TensorFlow** — https://www.tensorflow.org/  
- **PyTorch** — https://pytorch.org/  
- **Hugging Face Transformers** — https://huggingface.co/docs/transformers  
- **Hugging Face Datasets** — https://huggingface.co/docs/datasets  
- **llama.cpp** — https://github.com/ggml-org/llama.cpp  
- **ik_llama.cpp** — https://github.com/ikawrakow/ik_llama.cpp  
- **VLLM** — https://github.com/vllm-project/vllm  
- **llama-swap** — https://github.com/mostlygeek/llama-swap  
- **Ollama** *(not recommended: slow, proprietary weights)* — https://ollama.com/  
- **LM Studio** *(beginner‑friendly)* — https://lmstudio.ai/  
- **Comfy UI** *(for generative image workflows)* — https://www.comfy.org/  
- **OpenWebUI** *(ChatGPT‑style web interface)* — https://openwebui.com/  

---

## 📚 Example Projects & Tutorials

### TensorFlow / Keras
- **MNIST Example** — https://www.tensorflow.org/datasets/keras_example  
- **Text Classification Tutorial** — https://www.tensorflow.org/tutorials/keras/text_classification  

### PyTorch
- **Variational Autoencoder (VAE)** — https://github.com/pytorch/examples/tree/main/vae  

### LLM Inference & Fine‑Tuning
- **Hugging Face LLM Tutorial** — https://huggingface.co/docs/transformers/en/llm_tutorial  
- **Unsloth Fine‑Tuning Guide** — https://docs.unsloth.ai/get-started/fine-tuning-guide  

---

## ☁️ Cloud & Hosted Platforms

- **AWS**
  - SageMaker — https://aws.amazon.com/sagemaker/  
  - Bedrock — https://aws.amazon.com/bedrock/  

- **Hosted LLM Leaderboards**
  - ArtificialAnalysis.ai Providers — https://artificialanalysis.ai/leaderboards/providers  

---

## ⚙️ Local Setup & Tools

### VRAM / RAM Estimation
- https://apxml.com/tools/vram-calculator  
- https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator  

### Running Models Locally

| Tool                       | Ease      | Speed       | Notes                                      |
|----------------------------|-----------|-------------|--------------------------------------------|
| **LM Studio**              | High      | Medium      | GUI, beginner‑friendly                     |
| **Ollama**                 | High      | Low         | Proprietary weights                        |
| **llama.cpp**              | Medium    | Medium      | Lightweight C++                            |
| **VLLM**                   | Low       | High        | Optimized inference                        |
| **TabbyAPI (theroyallab)** | Very Low  | Very High   | Ultra‑fast, less polished                   |

### Web‑UIs
- **Open WebUI** — https://github.com/open-webui/open-webui  

### Model Sources
- **Hugging Face Models** — https://huggingface.co/models  

### Hugging Face Local App Configuration
1. Visit https://huggingface.co/settings/local-apps  
2. Add your hardware (GPU, CPU/RAM)  
3. Enable integrations for Ollama, LM Studio, llama.cpp, VLLM  

---

## 🚀 Repo Usage

```bash
docker compose up
