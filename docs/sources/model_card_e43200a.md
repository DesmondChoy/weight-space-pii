---
license: mit
---
# PII Detector (MiniLM-L6-v2)
<table>
  <tr style="border: none;">
    <td style="border: none; padding-right: 10px;">
      <a href="https://github.com/Negative-Star-Innovators/local-pii-detector-minilm"><img src="https://img.shields.io/badge/GitHub-View_Source_Code-black.svg?logo=github" alt="GitHub"></a>
    </td>
    <td style="border: none;">
      <a href="https://huggingface.co/spaces/Negative-Star-Innovators/PII-Redaction-Playground"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg" alt="Open In Spaces"></a>
    </td>
  </tr>
</table>

## Model Description
This model is a highly efficient, lightweight Token Classification model designed to detect Personally Identifiable Information (PII). It is a fine-tuned version of `sentence-transformers/all-MiniLM-L6-v2` trained on the `nvidia/Nemotron-PII` dataset.

Because it is based on the MiniLM architecture, the model is incredibly small (**~90 MB**) and extremely fast, making it the perfect choice for **running locally** on CPU-only machines, edge devices, or in environments with strict data privacy constraints where cloud-based APIs cannot be used. This model is available for commercial use.

## Evaluation Results (Epoch 4)
This checkpoint represents **Epoch 4** and shows excellent generalization on the 20,000-sample validation split. 

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 Score | Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 0.031307 | 0.029043 | 0.933025 | 0.947065 | **0.939993** | **0.992261** |

## ⚠️ Production Disclaimer
Please note that automated PII detection is not completely foolproof, and accuracy will vary depending on your specific data context and formatting. We strongly advise thoroughly validating the model on your own data and incorporating human oversight to ensure it meets your intended purpose before any production deployment.

## Training Parameters
The model was trained with the following parameters:
* **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`
* **Dataset**: `nvidia/Nemotron-PII` (180k Training / 20k Validation split)
* **Learning Rate**: 2e-5
* **Batch Size**: 64 (per device)
* **Weight Decay**: 0.01
* **Max Sequence Length**: 512
* **Number of Epochs**: 4
* **Task**: Token Classification (BIO Tagging format)

## 💻 How to Use: Local Inference
Because privacy is critical when handling PII, this model is meant to be downloaded and run locally. You can easily test it using the Hugging Face `pipeline`.

### Prerequisites
Make sure you have the `transformers` library installed in your local environment:
```bash
pip install transformers torch
```

### Python Inference Script
```python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_id = "Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection"

print("Downloading/Loading model locally...")
# Load the tokenizer and model locally
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

# Initialize the pipeline
# aggregation_strategy="simple" merges B- and I- tags into single coherent words/phrases
pii_pipeline = pipeline(
    "token-classification", 
    model=model, 
    tokenizer=tokenizer, 
    aggregation_strategy="simple" 
)

# Text containing dummy PII for testing
sample_text = (
    "John Doe's bank routing number is 123456789. "
    "He is 45 years old and his email is john.doe@example.com."
)

print("\nRunning inference locally...")
results = pii_pipeline(sample_text)

# Display the detected PII entities
print("\nDetected PII Entities:")
for entity in results:
    print(f"- Entity: {entity['word']}")
    print(f"  Label:  {entity['entity_group']}")
    print(f"  Score:  {entity['score']:.4f}\n")
```

### Expected Output Format
The pipeline will extract the entities based on the `Nemotron-PII` label mappings, yielding output like:
```text
- Entity: john
  Label:  first_name
  Score:  0.9984

- Entity: doe
  Label:  last_name
  Score:  0.9971

- Entity: 123456789
  Label:  bank_routing_number
  Score:  0.9688

- Entity: 45
  Label:  age
  Score:  0.9117

- Entity: john. doe @ example. com
  Label:  email
  Score:  0.9993
```

## 📬 Contact

Please reach out if you have questions or feedback. We also do custom projects, consultating, freelance and collaboration.

**Email:** [thieves@negativestarinnovators.com](mailto:thieves@negativestarinnovators.com)

## 💖 Support This Project
If you find this PII detector useful for your projects or business, please support our work!

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://buymeacoffee.com/negativestarinnovators)