import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SentimentRatingModel(nn.Module):
    def __init__(self, n_ratings):
        super(SentimentRatingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.rating_classifier = nn.Linear(self.bert.config.hidden_size, n_ratings)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        rating_logits = self.rating_classifier(pooled_output)
        return sentiment_logits, rating_logits

def load_model(device):
    model = SentimentRatingModel(n_ratings=10)
    model.load_state_dict(torch.load('sentiment_model/sentiment_rating_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 256


