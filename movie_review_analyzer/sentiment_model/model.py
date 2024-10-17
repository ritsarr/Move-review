import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

device = torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 128

class SentimentRatingModel(nn.Module):
    def __init__(self):
        super(SentimentRatingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
        self.rating_regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        rating_output = self.rating_regressor(pooled_output).squeeze(-1)
        return sentiment_logits, rating_output

def load_model():
    model = SentimentRatingModel()
    model.load_state_dict(torch.load('sentiment_model/sentiment_rating_model_final.pth', map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()