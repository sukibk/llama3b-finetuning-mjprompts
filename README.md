# 🦥 Finetuning LLaMA 3B with Unsloth

This repository demonstrates how to **finetune LLaMA 3.2 (3B)** using [Unsloth](https://github.com/unslothai/unsloth), a highly efficient library for training large language models with minimal GPU memory and maximum speed.

🚀 Trained using:
- 🧠 Model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- 🛠️ Finetuning method: LoRA with QLoRA optimizations
- 📦 Dataset: Custom prompt-enhanced JSONL dataset
- 🧪 Training: SFTTrainer with max sequence length 2048

---

## 💡 What is Unsloth?

Unsloth is a lightweight, blazing fast library for training modern LLMs like LLaMA, Gemma, Phi, Mistral, and DeepSeek using:
- 4-bit, 8-bit, or full finetuning
- LoRA and QLoRA with 80%+ VRAM reduction
- Up to **2x faster training** compared to standard Hugging Face pipelines

---

## 🔧 Quick Start

### 1. Clone this repo
```bash
git clone https://github.com/your-username/llama3b-unsloth-finetune.git
cd llama3b-unsloth-finetune
```

### 2. Install dependencies
```bash
pip install unsloth datasets transformers trl
```

### 3. Run the notebook

Open the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/llama3b-unsloth-finetune/blob/main/Llama3B_Finetune.ipynb)

Or run locally with:
```bash
jupyter notebook Llama3B_Finetune.ipynb
```

---

## 📁 Project Structure

```
├── Llama3B_Finetune.ipynb   # Colab/Local Notebook for full training pipeline
├── training_data.jsonl      # Optional: Example dataset (SIMPLE <-> ENHANCED prompts)
├── outputs/                 # Finetuned model outputs (generated after training)
```

---

## 🧪 Example Output Format

After processing:
```json
{
  "simple_prompt": "A cyberpunk city at night",
  "enhanced_prompt": "Neon-lit cyberpunk metropolis with glowing signs, bustling streets, and heavy rain reflecting light, shot with a Leica Q2"
}
```

---

## 🧠 Model Used

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

Then patched with `FastLanguageModel.get_peft_model(...)` for LoRA training.

---

## 📤 Export and Use

- You can save your model in Hugging Face or GGUF formats.
- Supports downstream deployment to `vLLM`, `Ollama`, and others.

---

## 📚 Resources

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai)
- [Llama 3.2 Models](https://huggingface.co/unsloth)

---

## 🧾 Citation

If you use this repo or Unsloth:

```bibtex
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {https://github.com/unslothai/unsloth},
  year = {2023}
}
```

---

## ⭐ Acknowledgments

- Special thanks to [Unsloth.ai](https://unsloth.ai) for making LLM finetuning accessible.
```
