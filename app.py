from flask import Flask, render_template, request, redirect, url_for
import spacy
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree


app = Flask(__name__)


def nltk_result(text, option, language='en'):
    # NLTK's NER is primarily English-only (Russian support is limited)
    # Map our option to NLTK's entity labels
    if language != 'en':
        return ['NLTK works with russian not properly.']

    option_map = {
        'PERSON': 'PERSON',
        'ORG': 'ORGANIZATION',
        'GPE': 'GPE',  # Geo-Political Entity
        'DATE': 'DATE',
        'MONEY': 'MONEY'
    }

    nltk_label = option_map.get(option)
    # if not nltk_label:
    #    raise ValueError(f"Unsupported entity type: {option}")

    # Tokenize and tag
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Named entity chunking
    chunks = ne_chunk(tagged)

    result = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            label = chunk.label()
            if label == nltk_label:
                entity = " ".join([token for token, pos in chunk.leaves()])
                result.append(entity)

    return result


def spacy_result(text, option, language):
    if language == 'ru':
        if option == 'GPE':
            option = 'LOC'

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
    nltk_results = []
    spacy_results = []
    text = ''
    if request.method == 'POST':
        language = request.form.get('language')
        text = request.form.get('text')
        option_of_named_entitle = request.form.get('entity_type')

        methods = request.form.getlist('methods')

        s_result = spacy_result(text, option_of_named_entitle, language)
        n_result = nltk_result(text, option_of_named_entitle, language)

        spacy_results.append({'option': option_of_named_entitle, 'result': s_result})
        nltk_results.append({'option': option_of_named_entitle, 'result': n_result})

    return render_template('index.html', spacy_results=spacy_results, nltk_results=nltk_results, previous_text=text, methods=methods)


if __name__ == "__main__":
    app.run(debug=True)
