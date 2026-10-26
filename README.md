# Fine-tuning-DistilBERT

Finetune DistilBERT on the IMDb using the guide from Hugging Face [website](https://huggingface.co/docs/transformers/en/tasks/sequence_classification).

<img width="1454" height="296" alt="image" src="https://github.com/user-attachments/assets/c5b75627-b1ca-407d-950a-bc84226520bb" />

## Results

- **Final accuracy**: 93.19%
- Training completed in 2 epochs

<img width="1457" height="316" alt="image" src="https://github.com/user-attachments/assets/4827ae0e-1891-41b0-a68a-aee3f9c0ef28" />

I tested the model on some simple sentences and it correctly predicted whether they were Positive or Negative with high confidence.

<img width="1475" height="245" alt="image" src="https://github.com/user-attachments/assets/caa71aa3-44a1-4c70-b4f4-45d53b4de37c" />

Also took some reviews from rottentomatoes [website](https://www.rottentomatoes.com/m/spider_man_brand_new_day/reviews/all-audience) to see how it acts in real world case. And it have pretty good accuracy and right answer towards review.

<img width="1469" height="239" alt="image" src="https://github.com/user-attachments/assets/18966c28-0de8-42ab-8e95-a3b85f31e7cf" /> <img width="762" height="116" alt="image" src="https://github.com/user-attachments/assets/15084392-1d45-48b5-a4e9-3dfb921c53a2" />

And I found a mixed review (partly positive with some negative parts) it predicted Positive but only with 53.8% confidence. I thought that was interesting because it didn’t just force a strong prediction.


## How to use

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-imdb-finetuned",
    tokenizer="distilbert-imdb-finetuned"
)

print(classifier("Your review here"))
