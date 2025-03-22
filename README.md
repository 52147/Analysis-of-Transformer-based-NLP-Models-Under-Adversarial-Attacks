# Analysis of Transformer-based NLP Models Under Adversarial Attacks : A Comparative Evaluation using SHAP, LIME, and Corpus Analysis

📌 View on Google Colab: [Open in Colab](https://colab.research.google.com/drive/1nsTMLjmT4A9JvmCbyaX2qtWOIY3qiM6g?usp=sharing)

## 1. Introduction

Recent advancements in Natural Language Processing (NLP) have been driven by Transformer-based architectures such as BERT [1], RoBERTa [2], XLM-RoBERTa [3], and GPT-3.5 [4]. Despite their strong performance, these models remain vulnerable to adversarial attacks, where subtle perturbations in input text can significantly degrade model performance. In this report, we investigate four Transformer-based models under adversarial scenarios:

- RoBERTa-Large
- BERT-Base
- XLM-RoBERTa
- GPT-3.5 Turbo

We simulate different adversarial strategies (synonym substitution, random deletion, noise injection), visualize misclassifications, and apply interpretability techniques (LIME [5] and SHAP [6]) to analyze each model’s decision-making. Furthermore, we examine how tokenization and corpus structure can affect robustness.
## 2. Methodology

This study involved the following steps:

- **Model Selection**: We evaluated four Transformer-based models: RoBERTa-Large, BERT-Base, XLM-RoBERTa, and GPT-3.5 Turbo.
- **Adversarial Attacks**: Three types of adversarial strategies were applied to test robustness:
  1. Synonym Substitution
  2. Random Deletion
  3. Noise Injection
- **Evaluation Metrics**: We measured performance using precision, recall, and F1-score under adversarial conditions.
- **Misclassification Analysis**: Misclassifications were analyzed across emotional labels (Neutral, Anxiety, Anger).
- **Interpretability Tools**: LIME and SHAP were used to provide local and global interpretability of model behavior.
- **Tokenizer and Corpus Structure**: Token-level statistics were collected to observe how token count and token frequency impact model vulnerability.
## 3. Result Analysis

### 3.1 Overall Performance
![image](https://github.com/user-attachments/assets/2ae9f59a-410c-4aeb-aaeb-d3abfca307ec)
Figure 1 (F1-Score Comparison) shows the F1-scores of the four models under adversarial attacks:

- RoBERTa-Large: 0.88
- BERT-Base: 0.82
- XLM-RoBERTa: 0.68
- GPT-3.5 Turbo: 0.53

RoBERTa-Large demonstrates the highest resilience, likely attributable to its robust pretraining and fine-tuning on large corpora [2]. GPT-3.5 Turbo, conversely, is most susceptible to adversarial manipulations due to its open-ended generative nature and limited task-specific fine-tuning.

### 3.2 Misclassification Distribution
![image](https://github.com/user-attachments/assets/1d2048dd-6438-495f-a38f-6a7ed90028a3)
Figure 2 (Misclassification Distribution) plots the number of misclassifications across three emotion categories (Neutral, Anxiety, Anger). We observe:

- RoBERTa-Large: Neutral (40), Anxiety (5), Anger (2)
- BERT-Base: Neutral (35), Anxiety (10), Anger (5)
- XLM-RoBERTa: Neutral (20), Anxiety (25), Anger (10)
- GPT-3.5 Turbo: Neutral (10), Anxiety (35), Anger (15)

Both RoBERTa-Large and BERT-Base show relatively fewer misclassifications overall, with GPT-3.5 Turbo struggling significantly on the Anxiety category. XLM-RoBERTa exhibits heightened sensitivity, particularly overreacting to small perturbations in text.

### 3.3 Interpretability with SHAP and LIME
![image](https://github.com/user-attachments/assets/9b174f39-f040-4b8e-9e7e-ee0b88bfd344)
Figure 3 (SHAP Summary Plot) displays feature-level explanations for a simulated model output, highlighting the positive or negative impact of each feature on the prediction. Features with higher SHAP values have a stronger influence on the model’s output. The distribution suggests that certain features (e.g., Feature 7 or Feature 8) may consistently steer the model toward specific classes.

In parallel, a LIME explanation (see console output) for a sample text indicates which tokens most affect local classification decisions. For instance, words like “text” and “adversarial” have small positive contributions, whereas “is” and “attack” exhibit slightly negative weights. These insights help us identify tokens or linguistic structures that may be more vulnerable to adversarial manipulation.

### 3.4 Different Adversarial Attack Strategies
![image](https://github.com/user-attachments/assets/b68d0a9a-345a-44bf-959b-4e90476be7bc)
Figure 4 (Model F1-Score under Different Adversarial Attack Strategies) compares F1-scores across three types of adversarial attacks:

- Synonym Substitution
- Random Deletion
- Noise Injection

RoBERTa-Large consistently outperforms other models, maintaining F1-scores above 0.75 in all scenarios. BERT-Base shows moderate declines, while XLM-RoBERTa and GPT-3.5 Turbo exhibit more substantial drops, particularly under random deletion. These results highlight how architectural and training nuances can lead to varying degrees of robustness.

### 2.5 Tokenizer and Corpus Structure Analysis
![image](https://github.com/user-attachments/assets/22a053f8-2638-4dae-b754-23ced876f272)
Figure 5 (Token Count Distribution) shows the token counts for five sample texts (average token count = 7.2). 
![image](https://github.com/user-attachments/assets/6a5537aa-d863-4db6-b41c-86cef408f850)

Figure 6 (Top 10 Frequent Tokens) reveals that “is” and “for” appear most frequently, each occurring twice. This rudimentary corpus analysis demonstrates that even small differences in tokenization or data composition can shape model learning and vulnerability. For instance, if certain tokens dominate the corpus, models may become overly reliant on them, thus more susceptible to targeted perturbations.

## 4. Conclusion

Our investigation underscores the following key points:

1. **Model Robustness**: RoBERTa-Large outperforms BERT-Base, XLM-RoBERTa, and GPT-3.5 Turbo in adversarial settings, highlighting the importance of large-scale pretraining and careful fine-tuning.
2. **Misclassification Trends**: GPT-3.5 Turbo and XLM-RoBERTa are notably prone to misclassifying Anxiety and Anger when confronted with adversarial examples.
3. **Interpretability**: LIME and SHAP analyses reveal how specific tokens or features drive predictions, aiding in diagnosing model weaknesses.
4. **Corpus and Tokenizer Effects**: Tokenization and corpus structure significantly influence model behavior, suggesting that preprocessing strategies can be a crucial line of defense against adversarial attacks.

## 5. Future Work

- **Expanding the Dataset**: Employ larger and more diverse real-world datasets to validate these findings under various linguistic contexts.
- **Advanced Adversarial Methods**: Incorporate more sophisticated perturbation techniques, such as syntax manipulation or gradient-based adversarial attacks, to stress-test model boundaries.
- **Cross-Lingual Studies**: Investigate whether similar vulnerability patterns arise in multilingual settings, especially given XLM-RoBERTa’s cross-lingual objectives.
- **Refined Tokenization**: Compare different tokenizers (e.g., WordPiece, SentencePiece) and their impact on adversarial robustness.
- **Defense Mechanisms**: Explore adversarial training and defensive distillation to strengthen model resilience.

## 6. References

[1] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 4171–4186.

[2] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., … & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.

[3] Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., … & Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL, 8440–8451.

[4] OpenAI. (2023). GPT-3.5: Language Model Overview and API. [Online]. Available: https://platform.openai.com/docs

[5] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?”: Explaining the Predictions of Any Classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135–1144.

[6] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NIPS), 4765–4774.

