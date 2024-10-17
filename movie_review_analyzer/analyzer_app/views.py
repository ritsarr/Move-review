from django.shortcuts import render
from .forms import ReviewForm
from sentiment_model.model import model, tokenizer, MAX_LENGTH, device
import torch
import torch.nn.functional as F

def predict(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        sentiment_logits, rating_output = model(input_ids, attention_mask)
        sentiment_probs = F.softmax(sentiment_logits, dim=1)
        sentiment = torch.argmax(sentiment_probs, dim=1).item()
        rating = rating_output.item()

    sentiment_label = 'Положительный' if sentiment == 1 else 'Отрицательный'
    rating = round(rating, 2)

    return sentiment_label, rating

def index(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data['review_text']
            try:
                sentiment_label, rating = predict(review_text)
                context = {
                    'form': form,
                    'sentiment': sentiment_label,
                    'rating': rating,
                }
            except Exception as e:
                print(f"Ошибка при предсказании: {e}")
                context = {
                    'form': form,
                    'error': 'Произошла ошибка при анализе отзыва.'
                }
            return render(request, 'analyzer_app/index.html', context)
    else:
        form = ReviewForm()
    return render(request, 'analyzer_app/index.html', {'form': form})
