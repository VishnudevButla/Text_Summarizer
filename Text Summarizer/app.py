from flask import Flask, request, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

@app.route('/', methods=['GET', 'POST'])
def summarize():
    summary = None
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text.strip():
            input_text = "summarize: " + text
            encoding = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(
                encoding,
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)