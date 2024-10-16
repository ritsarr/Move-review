from django.shortcuts import render
from .forms import ReviewForm
from sentiment_model.model import load_model, tokenizer, MAX_LENGTH
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model(device)

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
        sentiment_logits, rating_logits = model(input_ids, attention_mask)
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)
        rating_probs = torch.softmax(rating_logits, dim=1)

        sentiment = torch.argmax(sentiment_probs, dim=1).item()
        rating = torch.argmax(rating_probs, dim=1).item() + 1  # Рейтинги от 1 до 10

        sentiment_label = 'Положительный' if sentiment == 1 else 'Отрицательный'

    return sentiment_label, rating


def index(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data['review_text']
            sentiment_label, rating = predict(review_text)
            context = {
                'form': form,
                'sentiment': sentiment_label,
                'rating': rating,
            }
            return render(request, 'analyzer_app/index.html', context)
    else:
        form = ReviewForm()
    return render(request, 'analyzer_app/index.html', {'form': form})

