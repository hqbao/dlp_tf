import re

class MappingDatagen(object):
	'''
	'''

	def __init__(self):
		self.vocab_list = []
		self.token2word = {}
		self.word2token = {}

	def load_data(self, text_file, limit=None):
		file = open(text_file, 'r')
		sentences = file.read().splitlines()
		if limit != None and len(sentences) > limit[1] - limit[0]:
			sentences = sentences[limit[0]:limit[1]]

		self.vocab_list = self.__get_vocab(sentences)
		self.token2word = self.vocab_list
		self.word2token = {k: v for v, k in enumerate(self.vocab_list)}

		self.pad_token = 0
		self.start_token = len(self.vocab_list)-2
		self.end_token = len(self.vocab_list)-1

		return sentences, len(self.vocab_list), self.pad_token, self.start_token, self.end_token

	def __get_vocab(self, sentences):
		vocab_list = set()
		for i in range(len(sentences)):
			sentence = sentences[i]
			words = sentence.split(' ')
			for j in range(len(words)):
				word = words[j]
				vocab_list.add(word)

		vocab_list = sorted(list(vocab_list))
		vocab_list = ['_PAD_'] + vocab_list
		vocab_list = vocab_list + ['_START_', '_END_']
		return vocab_list

	def __preprocess_sentence(self, sentence):
		sentence = sentence.lower().strip()
		# Create a space between a word and the punctuation following it
		# eg: "he is a boy." => "he is a boy ."
		sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
		sentence = re.sub(r'[" "]+', " ", sentence)
		# Remove everything with space except (a-z, A-Z, ".", "?", "!", ",")
		sentence = re.sub(r"[^a-zA-Z?.!,']+", " ", sentence)
		sentence = sentence.strip()
		# adding a start and an end token to the sentence
		return sentence

	def encode(self, sentence):
		sentence = self.__preprocess_sentence(sentence)
		words = sentence.split(' ')
		tokens = []
		for word in words:
			try:
				token = self.word2token[word]
				tokens.append(token)
			except Exception as e:
				tokens.append(self.pad_token)

		return tokens

	def decode(self, tokens):
		return [self.token2word[token] for token in tokens]

class TextgenDatagen(object):
	'''
	'''

	def __init__(self):
		self.vocab_list = []
		self.token2word = {}
		self.word2token = {}

	def load_data(self, text_file):
		file = open(text_file, 'r')
		text = file.read()
		text = self.preprocess_text(text)

		words, self.vocab_list = self.__get_vocab(text)
		self.token2word = self.vocab_list
		self.word2token = {k: v for v, k in enumerate(self.vocab_list)}

		self.pad_token = 0
		self.start_token = len(self.vocab_list)-2
		self.end_token = len(self.vocab_list)-1

		return words, len(self.vocab_list), self.pad_token, self.start_token, self.end_token

	def __get_vocab(self, text):
		vocab_list = set()
		words = text.split(' ')
		for j in range(len(words)):
			word = words[j]
			vocab_list.add(word)

		vocab_list = sorted(list(vocab_list))
		vocab_list = ['_PAD_'] + vocab_list
		vocab_list = vocab_list + ['_START_', '_END_']
		return words, vocab_list

	def preprocess_text(self, text):
		text = text.lower().strip()
		# Create a space between a word and the punctuation following it
		# eg: "he is a boy." => "he is a boy ."
		text = re.sub(r"([?.!,])", r" \1 ", text)
		text = re.sub(r'[" "]+', " ", text)
		# Remove everything with space except (a-z, A-Z, ".", "?", "!", ",")
		text = re.sub(r"[^a-zA-Z?.!,']+", " ", text)
		text = text.strip()
		# adding a start and an end token to the sentence
		return text

	def encode(self, words):
		tokens = []
		for word in words:
			try:
				token = self.word2token[word]
				tokens.append(token)
			except Exception as e:
				tokens.append(self.pad_token)

		return tokens

	def decode(self, tokens):
		return [self.token2word[token] for token in tokens]


