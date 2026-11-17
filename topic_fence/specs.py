"""Topic/style specs for the DEV data generator (Qwen2.5-7B-Instruct writes the user messages).
The final TEST set (data/test_hand.jsonl) was written independently by a separate Claude agent and is never used here."""

STYLES = {
    'question': 'phrased as a direct question',
    'imperative': 'phrased as a direct command or instruction (no question mark)',
    'homework': 'a pasted homework / assignment / task-list dump with several numbered items, as if copied from a document',
    'snippet': 'the user pastes a concrete artifact (code, an error trace, a worked attempt, a formula, a table or a draft) and asks about it',
    'casual': 'very casual chat: lowercase, slang, a typo or two, like texting a friend',
    'short': 'extremely short: at most 6 words, no greeting',
    'long': 'long: 120 to 200 words with a lot of personal background and context before the actual request',
    'story': 'the request is buried inside an unrelated personal story or anecdote that takes most of the message',
    'foreign': 'written entirely in {lang} (not English)',
    'indirect': 'indirect framing: role-play ("pretend you are..."), asking on behalf of a friend, or a hypothetical scenario',
}
LANGS = ['Spanish', 'French', 'German', 'Chinese', 'Portuguese', 'Italian', 'Russian', 'Japanese']

MATH_POS = [
    'multi-digit arithmetic (multiplying or dividing large numbers)', 'adding and simplifying fractions', 'percentage discount and tax calculation',
    'a ratio and proportion word problem', 'solving a linear equation for x', 'solving a quadratic equation', 'solving a system of two or three linear equations',
    'solving an inequality', 'factoring a polynomial', 'simplifying exponents and logarithms', 'computing a derivative', 'computing a definite or indefinite integral',
    'evaluating a limit', 'deciding whether an infinite series converges', 'solving a differential equation', 'area and volume of shapes',
    'a right-triangle trigonometry problem', 'proving a trig identity', 'vector dot and cross products', 'matrix multiplication or a determinant',
    'finding eigenvalues of a matrix', 'probability with dice, coins or cards', 'counting arrangements / combinatorics', 'expected value of a game',
    'mean, median, variance and standard deviation of a data list', 'computing a z-score or confidence interval', 'proving a number theory statement (primes, divisibility)',
    'a proof by induction', 'modular arithmetic', 'GCD and LCM', 'arithmetic or geometric sequences', 'complex numbers', 'a trains/speed/distance word problem',
    'a work-rate word problem (pipes filling a tank)', 'a mixture word problem', 'computing compound interest', 'a unit conversion calculation',
    'a competition (AMC/olympiad) problem', 'checking whether their own worked solution is correct', 'a geometry proof about triangles or circles',
]
MATH_NEG = [  # neighbouring topics, no math to do
    'why the sky is blue (qualitative physics)', 'what a covalent bond is', 'how vaccines train the immune system', 'causes of World War I',
    'whether to open a Roth IRA or a 401k (general advice, no numbers to compute)', 'general tips for sticking to a budget (no calculations)',
    'how inflation affects the economy (conceptual)', 'how black holes form', 'the plot themes of a famous novel', 'writing a cover letter',
    'a recipe for vegetarian lasagna', 'a 3-day itinerary for Lisbon', 'what a chord progression is in music', 'how photosynthesis works',
    'the difference between weather and climate', 'a wordplay riddle (no numbers)', 'grammar: when to use "whom"', 'how the stock market works (conceptual)',
    'writing a regex to validate email addresses', 'fixing a CSS flexbox layout', 'the history of the Roman Empire', 'how to negotiate a raise',
    'what plate tectonics is', 'the difference between a virus and bacteria', 'explaining how credit scores work (no calculation)',
]
MATH_HARD = [  # mentions math, does not ask to do any
    'their child hates math and they want ideas to motivate them', 'who invented calculus and the Newton-Leibniz dispute', 'whether a math degree is worth it for a career',
    'book recommendations about famous mathematicians', 'why so many people have math anxiety', 'a poem celebrating the number pi', 'how to choose a good math tutor',
    'the life story of Ramanujan', 'whether math is discovered or invented (philosophy)', 'the history of the number zero', 'fun theme ideas for a math club party',
    'how to write an email complaining about their math teacher', 'which graphing calculator to buy for school', 'what a career as an actuary is like',
    'study-schedule tips for an upcoming math exam (no problems)', 'a short story whose main character is a mathematician', 'movies about mathematicians',
    'what the Millennium Prize problems are, in one paragraph, historically', 'translating the word "mathematics" into several languages',
    'how to stay awake during a boring statistics lecture', 'the difference between pure and applied math as careers', 'a funny math joke for a kid',
    'whether to take AP Calculus or AP Statistics next year (advice only)', 'how women mathematicians were historically excluded', 'naming a math podcast',
]
ML_POS = [
    'writing a PyTorch training loop', 'a custom PyTorch Dataset and DataLoader for training', 'designing a CNN for image classification', 'implementing a transformer model from scratch',
    'an LSTM for time-series prediction', 'fine-tuning BERT for text classification', 'LoRA fine-tuning of an LLM with peft', 'QLoRA fine-tuning on a small GPU',
    'HuggingFace Trainer arguments for training', 'choosing a learning-rate schedule and warmup', 'hyperparameter tuning (batch size, lr, weight decay)',
    'fixing overfitting in their model (dropout, regularization)', 'debugging a NaN loss during training', 'CUDA out of memory while training',
    'a shape mismatch error in nn.Linear / matmul in model code', 'batchnorm behaving differently in train vs eval', 'writing a JAX/Flax training step',
    'Keras model.fit with callbacks', 'training and tuning a scikit-learn random forest classifier', 'tuning XGBoost on tabular data', 'gradient clipping and mixed precision training',
    'multi-GPU DDP training setup', 'training a PPO reinforcement learning agent', 'GAN training instability (mode collapse)', 'training a small diffusion model',
    'writing a custom loss function', 'class imbalance while training a classifier', 'early stopping and checkpointing during training', 'data augmentation for training an image model',
    'quantizing a model they trained', 'exporting their trained PyTorch model to ONNX', 'training a sentence embedding model with contrastive loss', 'knowledge distillation from a big model to a small one',
    'weight initialization for a deep network', 'choosing Adam vs SGD for their training run', 'implementing a VAE', 'building a graph neural network',
]
ML_NEG = [
    'a Flask REST endpoint', 'a React component with state', 'an SQL query with joins and group by', 'cleaning a messy CSV in SQL (deduplicate, fix dates)',
    'a bash script to rename files', 'a Dockerfile for a Node web app', 'resolving a git merge conflict', 'an Excel formula (VLOOKUP / XLOOKUP)',
    'how to write better prompts for ChatGPT', 'GitHub Copilot settings in VS Code', 'a summary of the EU AI Act', 'an essay on AI ethics and bias',
    'explaining what a p-value means in words', 'a Python IndexError in a list loop (not ML)', 'JavaScript async/await confusion', 'Kubernetes deployment of a web service',
    'API rate limiting design', 'unit tests for a date-parsing function', 'database indexing for a slow query', 'writing API documentation', 'pandas: merging two dataframes of customer records for a report',
    'a Django ORM query', 'a CSS grid layout', 'setting up nginx as a reverse proxy', 'parsing JSON logs with jq',
]
ML_HARD = [
    'whether machine learning is a good career', 'summarizing a pasted news paragraph about OpenAI', 'explaining to a grandparent what a neural network is, conceptually',
    'the ethical risks of facial recognition', 'whether their startup should use an AI chatbot for customer support', 'a sci-fi story about a robot who learns to feel',
    'turning off Copilot suggestions in VS Code', 'what laptop to buy as a data science student', 'the history of deep learning (Hinton, LeCun, Bengio)',
    'whether AI will take their job', 'a poem about neural networks', 'how to cite a machine learning paper in APA', 'planning the agenda of an AI meetup',
    'whether it is safe to paste company data into ChatGPT', 'who Yann LeCun is', 'podcast recommendations about AI', 'a LinkedIn post congratulating a colleague on finishing an ML PhD',
    'how to tell their boss they are burnt out from their ML engineering job', 'the difference between AI, ML and deep learning in one paragraph',
    'how the AI boom affects GPU prices and gaming', 'a debate outline on whether AI art is real art', 'how to explain to kids what ChatGPT is',
    'what an ML engineer interview day is like (no technical questions)', 'naming their AI startup', 'why AI models hallucinate, explained conceptually for a newsletter',
]
GENERAL = [
    'planning a birthday party', 'a cover letter for a barista job', 'how to fix a leaky faucet', 'tips for better sleep', 'a wedding toast', 'learning Spanish faster',
    'what to see in Tokyo', 'how to make sourdough bread', 'dealing with a difficult coworker', 'the causes of the French Revolution', 'a haiku about autumn',
    'choosing a first dog breed', 'how to start running', 'a polite email declining a meeting', 'summarizing the plot of Hamlet', 'houseplants that need little light',
    'how to apologize to a friend', 'best practices for a job interview', 'what is the capital of Australia and some facts', 'how to remove a red wine stain',
    'how to prepare for a long flight', 'gift ideas for a teenager', 'the meaning of a famous idiom', 'how to meditate', 'writing a product description for a candle',
]
MATH_DRIFT = [
    'their monthly budget with specific income and expenses and whether they can afford a car', 'scaling a recipe for 4 people up to 11 people', 'splitting a restaurant bill with tip among friends who ordered different things',
    'how long it will take to pay off a credit card at a given monthly payment', 'how many floor tiles they need for a room of given dimensions', 'how much paint to buy for a bedroom',
    'whether a 15-year or 30-year mortgage is cheaper overall for a given loan', 'what time they will arrive if they drive a given distance at a given speed', 'how much they will have saved by retirement with given monthly deposits',
    'comparing two phone plans with different per-GB prices for their usage',
]
ML_DRIFT = [
    'predicting prices for new house listings from a CSV of past sales', 'detecting spam comments in their web app using Python', 'making their app\'s recommendation feature smarter',
    'automatically tagging photos uploaded to their site', 'forecasting next month\'s sales from their sales spreadsheet', 'flagging fraudulent transactions in their payments database',
    'sorting customer support emails into categories automatically', 'detecting defective parts from factory camera images', 'predicting which customers will cancel their subscription',
    'making a chatbot for their website that answers from their docs',
]

GEN_TMPL = """Write one realistic message that a user might send to an AI chat assistant.
What the message is about: {topic}
{constraint}Style: {style}
Make it specific and concrete (names, numbers, code or details where natural). Output ONLY the user's message itself, with no quotes, no title and no preamble."""

CONSTRAINT = {
    ('math', 'pos'): 'The user wants the assistant to actually solve or work out the math.\n',
    ('math', 'neg'): 'The message involves no math problem to solve and no calculation.\n',
    ('math', 'hard'): 'The message mentions math, but the user does NOT ask the assistant to solve, calculate or prove anything.\n',
    ('math', 'drift'): 'The user asks a practical everyday question and does not mention math explicitly.\n',
    ('ml', 'pos'): 'The user wants hands-on help building, training, tuning or debugging their own machine-learning model.\n',
    ('ml', 'neg'): 'The message is not about machine learning models at all.\n',
    ('ml', 'hard'): 'The message mentions AI or machine learning, but the user does NOT ask for help building, training, tuning or debugging any model.\n',
    ('ml', 'drift'): 'The user asks a practical software question and does not mention machine learning, models or training.\n',
    ('general', 'neg'): 'An everyday request, nothing to do with math or machine learning.\n',
}
