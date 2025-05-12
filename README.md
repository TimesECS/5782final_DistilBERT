# 5782final_DistilBERT

## Introduction
This repository contains the implementation of a project that re-implements the paper "DistilBERT: A Distilled Version of BERT". The main contribution of the paper is the introduction of a smaller, faster, and lighter version of BERT, which retains 97% of its language understanding capabilities while being 60% faster.

## Chosen Result
We aimed to reproduce the performance of DistilBERT on the GLUE benchmark, focusing on five binary classification tasks: SST-2, MRPC, QQP, CoLA, and RTE. These results are fundamental in demonstrating that DistilBERT achieves comparable accuracy to BERT while being computationally efficient. The original paper's results on these tasks range from 51% accuracy on CoLA to 91% on SST-2. See Figure 1 in the original paper for reference.

## GitHub Contents
- **code/**: Contains all the scripts for training, fine-tuning, and evaluating the DistilBERT model.
  - `train.py`: Script for training the model on SST-2.
  - `finetune.py`: Fine-tuning DistilBERT for specific tasks.
  - `evaluate_model.py`: Evaluation script for GLUE tasks.
  - `model.py`: Implementation of the DistilBERT architecture.
  - `pretrain.py`: Pretraining script for DistilBERT.
  - `train_english_wiki.py`: Script for training on English Wikipedia.
  - `direct_distil.py`: Implements the alternative approach of directly distilling DistilBERT on binary classification datasets without pretraining.
- **data/**: Placeholder for datasets.
- **results/**: Contains results and performance metrics.
- **poster/**: Project poster summarizing the work.
- **report/**: Final report detailing the implementation and findings.

## Re-implementation Details
### Methodology
- **Model Architecture**:
  - Pretrained `bert-base-uncased` (12 layers) as the teacher model.
  - DistilBERTForMaskedLM (6 layers) as the student model, trained using knowledge distillation on the Wikipedia dataset.
  - Fine-tuned DistilBERTForSequenceClassification for SST-2 sentiment classification by adding a classification head.
- **Datasets**:
  - Pretraining: English Wikipedia (20220301.en).
  - Fine-tuning: GLUE SST-2, MRPC, QQP, CoLA, and RTE datasets.
  - Alternative approach: Direct training on binary classification datasets without pretraining.
- **Evaluation Metrics**:
  - Loss on the test dataset after pretraining.
  - Accuracy on SST-2 and other binary classification tasks.
- **Modifications**:
  1. Used only English Wikipedia for pretraining (Toronto Book Corpus was unavailable).
  2. Initialized the student model with random weights instead of the teacher's weights.
  3. Pretrained the model for 1 epoch; fine-tuned for 3 epochs.
  4. Explored an alternative approach of direct training on smaller datasets.

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
### Re-implementation Results
- Pretrained DistilBERT achieved a loss of ~X on the test dataset.
- Fine-tuned DistilBERT achieved ~91% accuracy on SST-2, closely matching the original paper's results.
- Comparable performance on SST-2, QQP, and RTE; better performance on CoLA; worse performance on MRPC compared to the original DistilBERT.

### Alternative Approach
- Directly trained DistilBERT on SST-2 for 1 epoch (~5 minutes) and achieved 90% test accuracy.
- This lightweight approach is task-specific and does not generalize well to other datasets but offers a practical solution for specific tasks.

### Challenges
- Computational resources: Limited to a single RTX 4070 GPU (12GB VRAM) compared to the original paper's 8 V100 GPUs.
- Data availability: Toronto Book Corpus was deprecated, and GLUE test labels were unavailable, requiring manual dataset splits.

## Conclusion
This project demonstrates the feasibility of reproducing DistilBERT's results using open-source tools. Key takeaways include:
- DistilBERT requires significant data and computational power for generalization.
- Lightweight, task-specific models can achieve high performance with minimal training.

### Lessons Learned
- Ensure code runs and checkpoints are saved before committing to long training processes.
- Parallelize CPU-intensive tasks like tokenization and save intermediate results.

### Future Directions
- Explore task-specific distillation for non-binary classification tasks.
- Investigate further compression of the model by reducing encoder layers.

## References
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper, and lighter. arXiv preprint arXiv:1910.01108.
- Hugging Face Transformers: https://huggingface.co/transformers/
- Wang, A., et al. (2019). GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. arXiv preprint arXiv:1904.09482.

## Acknowledgements
This project was completed as part of the CS5782 course at [Your Institution]. Special thanks to the course instructors and teaching assistants for their guidance and support.