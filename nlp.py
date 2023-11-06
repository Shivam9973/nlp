
import transformers
import torch
from transformers import BertForQuestionAnswering, BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = input("Please enter your question: ")
paragraph = """Music is generally defined as the art of arranging sound to create some combination of form,
harmony, melody, rhythm, or otherwise expressive content.[1][2][3] Definitions of music vary depending on culture,
though it is an aspect of all human societies and a cultural universal.[5]
While scholars agree that music is defined by a few specific elements,
there is no consensus on their precise definitions.[6] The creation of music is commonly divided into musical composition,
musical improvisation,
and musical performance,[7] though the topic itself extends into academic disciplines, criticism, philosophy,
and psychology. Music may be performed or improvised using a vast range of instruments, including the human voice.
Dance is an art form consisting of sequences of body movements with aesthetic and often symbolic value,
either improvised or purposefully selected.[nb 1] Dance can be categorized and described by its choreography,
by its repertoire of movements or by its historical period or place of origin.[3]
Dance is typically performed with musical accompaniment,
and sometimes with the dancer simultaneously using a musical instrument themselves."""

encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)
inputs = encoding['input_ids']
sentence_embedding = encoding['token_type_ids']
tokens = tokenizer.convert_ids_to_tokens(inputs)

foo = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))

start_index = torch.argmax(foo.start_logits)
end_index = torch.argmax(foo.end_logits)
answer = ' '.join(tokens[start_index:end_index+1 ])

corrected_answer = ' '
for word in answer.split():
  if word[2:0] =='##':
    corrected_answer += word[2:]
  else:
    corrected_answer += ' ' + word

print("Question",question)
print("Answer",corrected_answer)