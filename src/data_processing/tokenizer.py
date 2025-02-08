import re
from nltk.tokenize import sent_tokenize
import nltk

def word_tokenizer(text: str) -> list[list[str]]:
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    # regex patterns for various placeholders
    url_pattern = re.compile(r'^[a-z][a-z0-9+.-]*://(?:[a-z]+(?::[^@]+)?@)?(?:[a-zA-Z0-9.-]+|\[[a-fA-F0-9:]+])(?::\d+)?/?(?:[/a-z-A-Z0-9_]*)?(?:\?[^#]*)?(?:#.*)?|(?:\w+\.)+\w+(?::\d+)?/?(?:[/a-z-A-Z0-9_]*)?(?:\?[^#]*)?(?:#.*)?$')
    mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
    hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
    time_pattern = re.compile(r'\b\d{1,2}:\d{2}\s?(?:[aApP]\.?[mM]\.?)?\b')
    percent_pattern = re.compile(r'^\d+(?:\.\d+)?%$')
    age_pattern = re.compile(r'\b\d+\s?(?:y|yr|yrs|-year-old|years-old|-years-old|year-old|years old|year old)\b')
    number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')

    text = url_pattern.sub('<URL>', text)
    text = mention_pattern.sub('<MENTION>', text)
    text = hashtag_pattern.sub('<HASHTAG>', text)
    text = time_pattern.sub('<TIME>', text)
    text = percent_pattern.sub('<PERCENT>', text)
    text = age_pattern.sub('<AGE>', text)
    text = number_pattern.sub('<NUMBER>', text)

    sentences = sent_tokenize(text)

    token_pattern = re.compile(
        r'<URL>|<MENTION>|<HASHTAG>|<TIME>|<PERCENT>|<AGE>|<NUMBER>'
        r'|[A-Za-z]+(?:\'[A-Za-z]+)?|[^\w\s]'
    )

    tokenized_sentences = []
    for sentence in sentences:
        tokens = token_pattern.findall(sentence)

        # filter out empty strings or whitespace-only tokens
        tokens = [token for token in tokens if token.strip()]
        tokenized_sentences.append(tokens)

    return tokenized_sentences


def main() -> None:
    inp_sentence = str(input("your text: "))
    token_list = word_tokenizer(inp_sentence)
    print("tokenized text: ", token_list)


if __name__ == "__main__":
    main()




# import re
#
# def cleanCorpus(text):
#     text = text.replace('\n', ' ')
#     text = re.sub(' +', ' ', text)
#     return text
#
# def splitSentence(text):
#     sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
#     return sentences
#
# def word_tokenizer(text):
#     # Sentence Tokenizer
#     sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
#     tokenized_text = []
#     for sentence in sentences:
#         words = re.findall(r'(?:(?<= )\d+(?:\.\d+)?\%)|(?:[\$]\d+(?:(?:\.\d+)*)?|Rs\.\d+(?:[\.\d+]*))|(?:\d:\d{2} [AP]M)|(?:(?<=[ \n])@[_\w]+)|(?:#[_\w]+)|(?:[\w!%+-\.\/]+@[a-zA-Z0-9\.]*[a-zA-Z0-9]+|".+"@[a-zA-Z0-9\.]*[a-zA-Z0-9]+)|(?:(?:[a-z][a-z0-9+.-]*):\/\/(?:(?:[a-z]+)(?::(?:[^@]+))?@)?(?:[a-zA-Z0-9\.-]+|\[[a-fA-F0-9:]+\])(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?|(?:(?:\w+\.)+(?:\w+))(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?)|(?:\d+.\d+|\d+|-\d+|\+\d+|\.\d+)|(?:[^\w\s])|(?:\w+)', sentence)
#         # # Percentages
#         words = [re.sub(r'^\d+(?:\.\d+)?\%$', '<PERC>', word) for word in words]
#         # # Price
#         words = [re.sub(r'^[\$]\d+(?:(?:\.\d+)*)?|Rs\.\d+(?:[\.\d+]*)$', '<PRICE>', word) for word in words]
#         # # Time
#         words = [re.sub(r'^\d:\d{2} [AP]M$', '<TIME>', word) for word in words]
#         # # Mentions
#         words = [re.sub(r'^@[_\w]+$', '<MENTION>', word) for word in words]
#         # # Hashtags
#         words = [re.sub(r'^#[_\w]+$', '<HASHTAG>', word) for word in words]
#         # # Mail IDs
#         words = [re.sub(r'^[\w!%+-\.\/]+@[a-zA-Z0-9\.]*[a-zA-Z0-9]+|".+"@[a-zA-Z0-9\.]*[a-zA-Z0-9]+$', '<MAILID>', word) for word in words]
#         # # URLs
#         words = [re.sub(r'^(?:[a-z][a-z0-9+.-]*):\/\/(?:(?:[a-z]+)(?::(?:[^@]+))?@)?(?:[a-zA-Z0-9\.-]+|\[[a-fA-F0-9:]+\])(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?|(?:(?:\w+\.)+(?:\w+))(?::(?:\d+))?\/?(?:[\/a-z-A-Z0-9\_]*)?(?:\?(?:[^#]*))?(?:#(?:.*))?$', '<URL>', word) for word in words]
#         # # Numbers
#         words = [re.sub(r'^\d+.\d+|\d+|-\d+|\+\d+|\.\d+$', '<NUM>', word) for word in words]
#         # # Punctuation
#         # words = [re.sub(r'^[^\w\s\<\>]$', '<PUNCT>', word) for word in words]
#         tokenized_text.append(words)
#     return tokenized_text