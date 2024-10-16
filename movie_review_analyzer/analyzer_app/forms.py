from django import forms

class ReviewForm(forms.Form):
    review_text = forms.CharField(
        label='Введите Ваш отзыв:',
        widget=forms.Textarea(attrs={'rows': 5, 'cols': 40})
    )

