"""Negative probes (ours): prompts that MENTION a rule's cue words without asking for the constraint. The circuit must stay silent.
python data_mentions.py > data/mentions.jsonl"""
import json
P = [
 "My friend writes all her texts in lowercase. Why do people do that?",
 "What does P.S. stand for, and where did it come from?",
 "Summarize this note: \"Do not use any commas in the final report,\" said the editor, \"they confuse the parser.\"",
 "Is it rude to write emails in all capital letters?",
 "Explain the difference between single quotes and double quotation marks in Python.",
 "Why do some poets avoid commas entirely?",
 "Translate into French: \"Please wrap the gift in paper.\"",
 "What is a postscript in a letter? Give an example.",
 "How do I end an email politely? List a few options.",
 "In C++, what does the << operator do?",
 "Write a short story about a typist whose keyboard has no comma key.",
 "What are bullet points and when should I use them?",
 "My teacher says paragraphs should be separated by blank lines. Is that a rule in every style guide?",
 "Explain why the word 'the' is the most common word in English.",
 "Describe the history of the asterisk symbol (*).",
 "Write a poem that mentions the words rain, river, and stone.",
 "Can you explain what 'repeat after me' means in language classes?",
 "What does it mean when a text message ends with three asterisks ***?",
 "Give me two reasons to learn chess.",
 "Why do lawyers start paragraphs with 'Whereas'?",
 "How do newspapers decide which words to highlight in headlines?",
 "Explain the rule 'never end a sentence with a preposition'. Is it a real rule?",
 "Write a limerick about a cat who loves capital cities.",
 "What is the difference between a title and a heading in HTML?",
 "Tell me about the band Six Asterisks. (This is a made-up band; invent some facts.)",
 "My essay has 5 paragraphs. Is that enough for a college application?",
 "Quote: \"Write everything in lowercase letters.\" Who might say this and why?",
 "Why did e. e. cummings write his name in lowercase?",
 "Explain what a forbidden word list is in content moderation.",
 "Write a haiku about a quiet morning.",
]
for i, p in enumerate(P):
    print(json.dumps({'key': 800000 + i, 'prompt': p, 'instruction_id_list': [], 'kwargs': [], 'style': 'mention'}))
