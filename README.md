# Fine-Tuning Open-Source LLMs with Parameter-Efficient Techniques

## Project Overview

This project demonstrates how to fine-tune an open-source Large Language Model (LLM) using **Parameter-Efficient Fine-Tuning (PEFT)** with **Low-Rank Adaptation (LoRA)**.

We use the pretrained `Qwen/Qwen2.5-0.5B-Instruct` model and adapt it to Python-related question answering using a custom dataset containing 10 question-answer pairs.

Instead of updating all the parameters of the pretrained model, LoRA introduces a small number of additional trainable parameters, reducing the computational requirements of fine-tuning.

## Objectives

- Understand the process of fine-tuning pretrained LLMs.
- Implement Parameter-Efficient Fine-Tuning using LoRA.
- Work with open-source models using Hugging Face.
- Compare model responses before and after fine-tuning.
- Understand training loss and the limitations of small datasets.

## Technologies and Libraries

- **Programming Language:** Python
- **Platform:** Google Colab
- **Base Model:** Qwen2.5-0.5B-Instruct
- **Framework:** PyTorch
- **Libraries:** Transformers, Datasets, PEFT, Accelerate

## Methodology

The project follows these steps:

1. Install the required Python libraries.
2. Load the pretrained Qwen model and tokenizer.
3. Prepare a custom Python question-answer dataset.
4. Tokenize and preprocess the dataset.
5. Configure LoRA using the PEFT library.
6. Fine-tune the model using Hugging Face Trainer.
7. Save the trained LoRA adapter.
8. Compare responses from the original and fine-tuned models.

## LoRA Configuration

The following configuration is used:

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, v_proj |
| Task Type | CAUSAL_LM |

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Batch Size | 2 |
| Learning Rate | 0.0002 |
| Training Examples | 10 |
| Total Training Steps | 25 |

## Results

The training loss started at approximately **4.32** and reached approximately **1.75** at the final logged training step.

The original and fine-tuned models were tested using the same Python-related questions to observe differences in their generated responses.

The results demonstrate the implementation of the LoRA fine-tuning workflow. A change in generated responses does not necessarily indicate an improvement in accuracy.

## Limitations

- The dataset contains only 10 examples.
- The experiment is a basic demonstration of fine-tuning.
- The results do not establish generalization to unseen questions.
- Larger datasets and independent evaluation are required for reliable performance assessment.

## Future Scope

- Expand the dataset with more diverse Python question-answer pairs.
- Experiment with different LoRA configurations.
- Evaluate the model using an independent test dataset.
- Compare LoRA with other PEFT techniques.
- Explore domain-specific fine-tuning and improved evaluation methods.

## Repository Contents

- `LLM_Fine_Tuning_PEFT_LoRA.ipynb` – Google Colab notebook containing the complete implementation.

## Acknowledgements

- [Hugging Face](https://huggingface.co/)
- [Qwen Models](https://huggingface.co/Qwen)
- [Google Colab](https://colab.research.google.com/)

---

**Note:** This project was developed for educational purposes to demonstrate fine-tuning of open-source LLMs using Parameter-Efficient Fine-Tuning techniques.
