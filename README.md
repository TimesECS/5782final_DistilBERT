# 5782final_DistilBERT

## Introduction
This repository contains the implementation of a project that re-implements the paper "DistilBERT: A Distilled Version of BERT". The main contribution of the paper is the introduction of a smaller, faster, and lighter version of BERT, which retains 97% of its language understanding capabilities while being 60% faster.

## Chosen Result
We aimed to reproduce the performance of DistilBERT on the SST-2 dataset, a sentiment analysis task. The significance lies in validating the claim that DistilBERT achieves comparable accuracy to BERT while being computationally efficient.

## GitHub Contents
- **code/**: Contains all the scripts for training, fine-tuning, and evaluating the DistilBERT model.
  - `train.py`: Script for training the model on SST-2.
  - `finetune.py`: Fine-tuning DistilBERT for specific tasks.
  - `evaluate_model.py`: Evaluation script for GLUE tasks.
  - `model.py`: Implementation of the DistilBERT architecture.
  - `pretrain.py`: Pretraining script for DistilBERT.
  - `train_english_wiki.py`: Script for training on English Wikipedia.
- **data/**: Placeholder for datasets.
- **results/**: Contains results and performance metrics.
- **poster/**: Project poster summarizing the work.
- **report/**: Final report detailing the implementation and findings.

## Re-implementation Details
Our approach involved re-implementing the DistilBERT model using PyTorch and Hugging Face Transformers. Key details:
- **Model**: DistilBERT with 6 layers, 12 attention heads, and a hidden size of 768.
- **Datasets**: SST-2 for fine-tuning and GLUE tasks for evaluation.
- **Tools**: PyTorch, Hugging Face Transformers, and Datasets libraries.
- **Evaluation Metrics**: Accuracy for classification tasks.
- **Challenges**: Adapting the pretraining process and ensuring compatibility with the Hugging Face library.

## Reproduction Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd 5782final_DistilBERT
   ```
2. Install dependencies:
   ```bash
   pip install -r code/requirements.txt
   ```
3. Pretrain the model:
   ```bash
   python code/pretrain.py
   ```
4. Fine-tune the model on SST-2:
   ```bash
   python code/finetune.py
   ```
5. Evaluate the model:
   ```bash
   python code/evaluate_model.py
   ```
6. Ensure access to a GPU for efficient training and evaluation.

## Results/Insights
Our re-implementation achieved an accuracy of ~91% on the SST-2 dataset, closely matching the original paper's results. This validates the claim that DistilBERT is a computationally efficient alternative to BERT without significant loss in performance.

## Conclusion
This project demonstrates the feasibility of reproducing DistilBERT's results using open-source tools. Key takeaways include the importance of careful pretraining and fine-tuning to achieve optimal performance.

## References
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper, and lighter. arXiv preprint arXiv:1910.01108.
- Hugging Face Transformers: https://huggingface.co/transformers/

## Acknowledgements
This project was completed as part of the CS5782 course at [Your Institution]. Special thanks to the course instructors and teaching assistants for their guidance and support.