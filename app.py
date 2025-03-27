from flask import Flask, render_template, request, redirect, url_for
import spacy

app = Flask(__name__)


def spicy_result(text, option, language):
    if language == 'ru':
        nlp = spacy.load("ru_core_news_sm")
    else:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    result = []

    for ent in doc.ents:
        if ent.label_ == option:
            result.append(ent.text)

    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    submissions = []
    text = ''
    if request.method == 'POST':
        language = request.form.get('language')
        text = request.form.get('text')
        option_of_named_entitle = request.form.get('entity_type')
        if language == "ru":
            result = spicy_result(text, option_of_named_entitle, language)
            submissions.append({'option': option_of_named_entitle, 'result': result})
        if language == "en":
            result = spicy_result(text, option_of_named_entitle, language)
            submissions.append({'option': option_of_named_entitle, 'result': result})

    print(submissions)
    return render_template('index.html', submissions=submissions, previous_text=text)


if __name__ == "__main__":
    app.run(debug=True)
