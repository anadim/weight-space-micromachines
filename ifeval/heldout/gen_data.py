"""Own IFEval-like data: prompts with verifiable instructions, many phrasings/topics, scored by the official lm_eval checkers.

python gen_data.py --seed 0 --per 30 --out data/dev.jsonl            (DEV: our own templates only)
python gen_data.py --seed 1 --per 60 --out data/fresh.jsonl --extra fresh_templates.json
   (FRESH TEST: our templates + extra phrasings written by an independent agent, never used during development)

Every row: key, prompt, instruction_id_list, kwargs (IFEval format), meta (template ids, derivation record).
Templates tagged 'drv:<even key>' are OUR rewordings of instruction sentences of even-key IFEval prompts (allowed by the
author: dev data may contain reworded/recombined even-key material; odd keys are never opened).
"""
import argparse, json, random, re

TASKS = [  # (request, salient words the host is likely to use)
    ("Write a short story about a lighthouse keeper who adopts a seagull.", ["lighthouse", "seagull", "keeper", "sea"]),
    ("Write a poem about the first snowfall of winter.", ["snow", "winter", "cold", "white"]),
    ("Draft an email to my landlord asking for the broken heater to be repaired.", ["heater", "repair", "landlord", "apartment"]),
    ("Explain how vaccines train the immune system.", ["vaccine", "immune", "virus", "body"]),
    ("Give me some advice for my first marathon.", ["marathon", "running", "race", "training"]),
    ("Describe a busy night market in a small coastal town.", ["market", "food", "night", "town"]),
    ("Write a product description for a solar-powered backpack.", ["solar", "backpack", "battery", "power"]),
    ("Compare cats and dogs as pets for a small apartment.", ["cats", "dogs", "pet", "apartment"]),
    ("Write a thank-you note to a teacher who retired this year.", ["teacher", "thank", "school", "students"]),
    ("Summarize the causes of the French Revolution.", ["revolution", "king", "france", "tax"]),
    ("Write a speech for a high school graduation ceremony.", ["graduation", "future", "school", "class"]),
    ("Explain the rules of chess to a beginner.", ["chess", "king", "pieces", "board"]),
    ("Write a review of a fictional Italian restaurant called Nonna's Table.", ["restaurant", "pasta", "food", "service"]),
    ("Describe the water cycle for a ten-year-old.", ["water", "rain", "clouds", "evaporation"]),
    ("Write a cover letter for a junior data analyst position.", ["data", "analyst", "experience", "skills"]),
    ("Plan a three-day trip to Lisbon for a couple on a budget.", ["lisbon", "trip", "budget", "day"]),
    ("Write a dialogue between a robot and a farmer about the weather.", ["robot", "farmer", "weather", "rain"]),
    ("Explain why the sky looks blue during the day.", ["sky", "blue", "light", "sun"]),
    ("Write a limerick about a forgetful wizard.", ["wizard", "magic", "spell", "forgot"]),
    ("Give tips for keeping houseplants alive in winter.", ["plants", "water", "light", "winter"]),
    ("Write a blog post about learning to cook as an adult.", ["cooking", "kitchen", "recipe", "food"]),
    ("Describe the life cycle of a butterfly.", ["butterfly", "caterpillar", "egg", "wings"]),
    ("Write a complaint letter to an airline about lost luggage.", ["luggage", "airline", "flight", "bag"]),
    ("What are the benefits of learning a second language?", ["language", "learning", "brain", "culture"]),
    ("Write a bedtime story about a dragon who is afraid of the dark.", ["dragon", "dark", "night", "afraid"]),
    ("Explain how a bicycle gear system works.", ["gear", "bicycle", "chain", "pedal"]),
    ("Write an advertisement for a new brand of oat milk.", ["oat", "milk", "taste", "dairy"]),
    ("Describe a perfect Sunday morning.", ["morning", "coffee", "sunday", "sun"]),
    ("Write a short biography of a fictional jazz pianist named Odette Marr.", ["jazz", "piano", "music", "odette"]),
    ("Explain the difference between weather and climate.", ["weather", "climate", "temperature", "years"]),
    ("Write a motivational message for someone starting a new job.", ["job", "work", "new", "success"]),
    ("Give me ideas for a birthday party for a seven-year-old.", ["party", "birthday", "games", "cake"]),
    ("Write a poem about an old oak tree in a city park.", ["tree", "oak", "park", "leaves"]),
    ("Explain how to brew a good cup of tea.", ["tea", "water", "leaves", "cup"]),
    ("Write a news report about a town that found a lost Roman coin.", ["coin", "roman", "town", "found"]),
    ("Describe how bees make honey.", ["bees", "honey", "flowers", "hive"]),
    ("Write a letter from a soldier to his family in 1915.", ["letter", "family", "war", "home"]),
    ("Explain what a black hole is.", ["black", "hole", "gravity", "light"]),
    ("Write a pitch for a mobile app that helps people drink more water.", ["app", "water", "drink", "users"]),
    ("Give me a weekly workout plan for a busy parent.", ["workout", "exercise", "week", "minutes"]),
    ("Write a fairy tale about a fox who wants to fly.", ["fox", "fly", "bird", "sky"]),
    ("Explain how recycling plastic works.", ["plastic", "recycling", "waste", "bottles"]),
    ("Write a toast for my sister's wedding.", ["wedding", "sister", "love", "marriage"]),
    ("Describe the sounds of a rainforest at dawn.", ["rainforest", "birds", "rain", "trees"]),
    ("Write a haiku sequence about the ocean.", ["ocean", "waves", "sea", "water"]),
    ("Explain why sleep is important for teenagers.", ["sleep", "brain", "night", "teenagers"]),
    ("Write a job posting for a bakery assistant.", ["bakery", "bread", "baking", "job"]),
    ("Give me arguments for and against school uniforms.", ["uniforms", "school", "students", "clothes"]),
    ("Write a scene where two old friends meet at a train station.", ["train", "station", "friends", "platform"]),
    ("Explain how a credit score is calculated.", ["credit", "score", "payment", "debt"]),
    ("Rewrite this sentence in a more formal tone: \"hey, the meeting got moved to friday, don't be late.\"", ["meeting", "friday", "late", "moved"]),
    ("What is the best way to learn to play the guitar?", ["guitar", "chords", "practice", "music"]),
    ("Write a riddle whose answer is a shadow.", ["shadow", "light", "dark", "sun"]),
    ("Describe a city on Mars in the year 2200.", ["mars", "city", "planet", "dome"]),
    ("Write an apology message to a friend for missing their concert.", ["concert", "sorry", "friend", "music"]),
    ("Explain how volcanoes form.", ["volcano", "magma", "lava", "earth"]),
    ("Write a song about a road trip across the desert.", ["road", "desert", "car", "sun"]),
    ("Give advice to a student who is nervous about public speaking.", ["speaking", "nervous", "audience", "practice"]),
    ("Write a short essay about the history of the printing press.", ["printing", "press", "books", "gutenberg"]),
    ("Describe your ideal library.", ["library", "books", "reading", "shelves"]),
]

WORDLIST = ["however", "meanwhile", "imagine", "today", "finally", "perhaps", "yesterday", "suddenly", "remember", "sunlight",
            "first", "luckily", "honestly", "silence", "tomorrow", "curiously", "indeed", "eventually", "once", "listen"]
PHRASES = ["Let me know if you have more questions.", "That wraps it up.", "Stay curious!", "Until next time, friends.",
           "Hope this helps!", "And that is the whole story.", "Thanks for reading.", "Any thoughts?", "Over and out.",
           "Peace and quiet at last.", "Good luck out there!", "Talk soon.", "The end.", "Have a wonderful day!",
           "What do you think?", "Keep it simple."]
NUMW = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
ORDW = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh'}
ORDTH = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th', 5: '5th', 6: '6th', 7: '7th'}

# ---------------- DEV templates (our own wording; 'drv:k' = our rewording of an even-key instruction sentence) ----------------
T = {
 'no_comma': ["Do not use any commas in your answer.", "Please avoid commas entirely.", "Your reply must not contain a single comma.",
              "Write it without commas.", "No commas are allowed anywhere in the response.", "Make sure there are zero commas in what you write.",
              "Commas are forbidden in this task.", "Refrain from using commas.", "Keep the whole text free of commas.",
              "Don't put any commas in your response.", ("drv:1162", "Make sure you do not use a single comma in it."),
              ("drv:1508", "Avoid using any commas in the piece."), ("drv:2380", "The text cannot contain commas.")],
 'lower': ["Write everything in lowercase letters.", "Your whole answer must be in lowercase.", "Do not use any capital letters.",
           "Use only lower case letters throughout.", "No uppercase letters are allowed in your reply.", "The entire response should be lowercase, with no capitals at all.",
           "Please answer using only small letters, no capitals.", "Keep every letter of your answer lowercase.",
           ("drv:202", "Reply only in lowercase letters for the whole answer."), ("drv:1122", "Do this without using capital letters."),
           ("drv:3150", "The full answer has to consist of lowercase letters only.")],
 'caps': ["Write your entire answer in capital letters.", "Your whole response must be in uppercase.", "Use only uppercase letters throughout your reply.",
          "Answer in all caps.", "Every letter of your response should be capitalized.", "Do not use any lowercase letters.",
          "Make the entire reply UPPERCASE.", "The response should be written fully in capital letters.",
          ("drv:3186", "Not one word of your answer should contain a lowercase letter."), ("drv:3572", "Respond in English using all capital letters."),
          ("drv:3456", "All letters in what you write must be capitalized.")],
 'quote': ["Wrap your entire response in double quotation marks.", "Put the whole answer inside double quotes.", "Enclose everything you write in double quotation marks.",
           "Your full reply must be surrounded by double quotes.", "Place your entire answer between double quotation marks.",
           "Begin and end your response with a double quotation mark.", "Put double quotes around your whole response.",
           ("drv:122", "Wrap the whole post in double quotation marks."), ("drv:2532", "Be sure the entire letter is enclosed in double quotes."),
           ("drv:3536", "Put all of your reply inside double quotation marks.")],
 'end': ["End your response with the exact phrase \"{P}\"", "Finish your answer with \"{P}\" and say nothing after it.", "Your reply must end with the words \"{P}\"",
         "Conclude with this exact sentence: \"{P}\". Nothing should follow it.", "The last words of your response must be \"{P}\"",
         "Close your answer with '{P}'.", "Make sure the final sentence of your reply is \"{P}\"",
         ("drv:1446", "Your text must finish with the exact phrase \"{P}\" with no other words after it."),
         ("drv:286", "End with: {P}"), ("drv:2268", "Please wrap up your answer with \"{P}\" and add nothing after that.")],
 'ps': ["At the end of your response, add a postscript starting with {M}", "Please include a postscript that begins with {M} at the end.",
        "Finish with a postscript starting with {M}", "Add a short postscript beginning with {M} after your main answer.",
        "End your reply with a postscript marked {M}", "Include a postscript at the bottom, starting with {M}",
        ("drv:1620", "Add a postscript to the reply that starts with {M}"), ("drv:2362", "Please close your answer with a postscript starting with {M}"),
        ("drv:2396", "Add a post script at the end starting with {M}")],
 'forbid': ["Do not use the words {Wdq} in your response.", "Avoid the word {W1q} entirely.", "Your answer must not include the words {Wp}.",
            "Don't mention {Wdq} anywhere.", "The word {W1dq} should not appear in your reply.", "Never use the words {Wq}.",
            "Write this without using the keywords {Wl}.", "Please don't say {Wq} in your response.",
            ("drv:2028", "Do not say {Wq} anywhere in the response."), ("drv:2328", "The words {Wp} cannot be in the answer."),
            ("drv:1402", "Don't include the keywords {Wdq} in your answer."), ("drv:3326", "Make sure the word {W1} does not appear at all.")],
 'repeat': [("after", "First repeat the request above word for word without change, then give your answer."),
            ("after", "Before answering, copy my request above exactly as written. Do not write anything before the copied request."),
            ("after", "Start by repeating the request above verbatim, and then respond to it."),
            ("after", "Please repeat my request above exactly, then answer it. Say nothing before the repeated request."),
            ("before", "Repeat the request below word for word first, then answer it. Do not add anything before repeating it."),
            ("before", "Before you respond, restate the following request exactly as it is written, then give your response."),
            ("inline", "First repeat this request exactly, then answer it."),
            ("drv:1518", "after", "Begin by repeating the exact request above, then give your answer. Write nothing before the repeated request."),
            ("drv:3484", "before", "For this task, first repeat the following request word by word without changing anything, and then respond to it. Do not say anything before the repeated request."),
            ("drv:2904", "after", "Repeat the request above first, word for word, and only then give your answer.")],
 'two': ["Give two different responses, separated by six asterisks: ******", "Provide two distinct answers. Separate them with ******.",
         "Write two versions and put ****** between them.", "Give me two different answers separated by 6 asterisk symbols (******).",
         "Offer two alternative responses; separate the two with exactly six asterisks ******.",
         ("drv:1098", "Please give exactly two different responses, separated by 6 asterisks: ******."),
         ("drv:2436", "Give two different answers, separated by six asterisk symbols ******."),
         ("drv:1180", "Give exactly two different responses separated by 6 asterisk symbols ******.")],
 'paras': ["Your answer should have exactly {N} paragraphs, separated by the markdown divider ***", "Write {Nw} paragraphs and separate them with ***.",
           "There should be {N} paragraphs. Use *** between paragraphs.", "Organize the response into exactly {Nw} paragraphs divided by ***.",
           "Use exactly {N} paragraphs, with a *** line between consecutive paragraphs.",
           ("drv:1082", "Write exactly {N} paragraphs and separate the paragraphs with the markdown divider: ***"),
           ("drv:1858", "There must be exactly {N} paragraphs. Separate paragraphs with ***."),
           ("drv:1236", "The answer should contain exactly {Nw} paragraphs separated by the markdown divider: ***.")],
 'nth': ["Write exactly {N} paragraphs separated by two new lines. The {Kord} paragraph must start with the word \"{X}\".",
         "Your answer must have {N} paragraphs separated by blank lines, and paragraph {K} should begin with the word {X}.",
         "Use {Nw} paragraphs, separated by double line breaks. Start the {Kth} paragraph with \"{X}\".",
         "There should be exactly {N} paragraphs, separated by two newlines. Paragraph {K} must begin with the word '{X}'.",
         ("last", "Write {N} paragraphs separated by two new lines, and the last paragraph must start with the word \"{X}\"."),
         ("drv:2250", "Write exactly {N} paragraphs; use two new lines between paragraphs. Begin the {Kth} paragraph with the word \"{X}\"."),
         ("drv:2880", "There should be exactly {N} paragraphs separated by two new lines. Paragraph {K} must start with the word {X}.")],
 'title': ["Include a title wrapped in double angular brackets, like <<my title>>.", "Your answer must have a title in double angle brackets, e.g. <<title>>.",
           "Give your response a title surrounded by << and >>.", "Start with a title enclosed in double angular brackets, such as <<a day at sea>>.",
           "Add a title in double angular brackets, i.e. <<title>>.",
           ("drv:1322", "The answer needs a title in double angular brackets, such as <<pros and cons>>."),
           ("drv:3606", "Include a title in double angular brackets, for example <<title>>.")],
 'high': ["Highlight at least {N} sections of your answer with markdown, e.g. *highlighted section*.", "Use markdown to italicize at least {Nw} parts of your reply, like *this*.",
          "Make sure at least {N} phrases are highlighted with asterisks, for example *important phrase*.", "Emphasize at least {Nw} sections using markdown italics (*like this*).",
          ("drv:1644", "Highlight at least {N} text sections with \"*\", for example *key point*."),
          ("drv:3644", "Italicize at least {N} sections in your answer with markdown, i.e. *italic text*."),
          ("drv:2790", "Use markdown to highlight at least {Nw} section of the answer, i.e. *highlighted section*.")],
 'bullets': ["Your answer must contain exactly {N} bullet points, using markdown like:\n* first point\n* second point",
             "Use exactly {Nw} markdown bullet points (* item).", "Give exactly {N} bullet points in markdown format.",
             "Format the answer as exactly {N} bullet points that start with \"* \".", "Respond with exactly {Nw} bullet points.",
             ("drv:1634", "The answer must have exactly {N} bullet points. Use markdown bullets such as:\n* This is a point"),
             ("drv:2832", "Your answer must include exactly {Nw} bullet points in markdown format.")],
}
# distractor instruction types (not targeted; checked so that the implants do not break them)
D = {
 'words': [("at least", "Your response should be at least {N} words long."), ("less than", "Keep the answer under {N} words."),
           ("at least", "Write {N} words or more."), ("less than", "Use fewer than {N} words.")],
 'sents': [("less than", "Use fewer than {N} sentences."), ("at least", "Write at least {N} sentences."), ("less than", "Your answer should have less than {N} sentences.")],
 'exist': ["Include the keywords {Kq} in your response.", "Make sure to mention {Kq}.", "Use the words {Kq} somewhere in your answer."],
 'freq': [("at least", "The word \"{K}\" should appear at least {N} times."), ("less than", "Use the word {K} fewer than {N} times.")],
 'capfreq': [("at least", "Use words in all capital letters at least {N} times to stress key points."), ("less than", "Words written fully in capital letters should appear fewer than {N} times.")],
 'placeholders': ["Include at least {N} placeholders in square brackets, such as [name].", "Use at least {N} placeholders like [address]."],
 'letter': [("at least", "The letter {L} should appear at least {N} times.")],
}
ID = {'no_comma': 'punctuation:no_comma', 'lower': 'change_case:english_lowercase', 'caps': 'change_case:english_capital', 'quote': 'startend:quotation',
      'end': 'startend:end_checker', 'ps': 'detectable_content:postscript', 'forbid': 'keywords:forbidden_words', 'repeat': 'combination:repeat_prompt',
      'two': 'combination:two_responses', 'paras': 'length_constraints:number_paragraphs', 'nth': 'length_constraints:nth_paragraph_first_word',
      'title': 'detectable_format:title', 'high': 'detectable_format:number_highlighted_sections', 'bullets': 'detectable_format:number_bullet_lists',
      'words': 'length_constraints:number_words', 'sents': 'length_constraints:number_sentences', 'exist': 'keywords:existence', 'freq': 'keywords:frequency',
      'capfreq': 'change_case:capital_word_frequency', 'placeholders': 'detectable_content:number_placeholders', 'letter': 'keywords:letter_frequency'}
TARGETS = ['no_comma', 'lower', 'caps', 'quote', 'end', 'ps', 'forbid', 'repeat', 'two', 'paras', 'nth', 'title', 'high', 'bullets']
CONFLICT = [{'lower', 'caps'}, {'paras', 'nth'}, {'quote', 'repeat'}, {'two', 'paras'}, {'two', 'nth'}, {'caps', 'capfreq'}, {'lower', 'capfreq'},
            {'bullets', 'paras'}, {'bullets', 'nth'}, {'words', 'bullets'}, {'sents', 'nth'}, {'sents', 'paras'}, {'quote', 'two'}, {'repeat', 'two'},
            {'repeat', 'title'}, {'lower', 'letter'}, {'caps', 'letter'}]


def qjoin(ws, q, conj='and'):
    ws = [f'{q}{w}{q}' for w in ws]
    return ws[0] if len(ws) == 1 else (f'{ws[0]} {conj} {ws[1]}' if len(ws) == 2 else ', '.join(ws[:-1]) + f', {conj} ' + ws[-1])


def norm_t(t):
    """template entry -> (tag, payload)"""
    if isinstance(t, str): return 'own', t
    if isinstance(t, (list, tuple)) and isinstance(t[0], str) and (t[0].startswith('drv:') or t[0].startswith('fresh')):
        return t[0], (t[1:] if len(t) > 2 else t[1])
    return 'own', t


def render(typ, tmpl, rng, task_words, used):
    """-> instruction text, kwargs, extra info (for repeat: position)"""
    tag, t = norm_t(tmpl)
    kw, info = {}, {'tag': tag}
    if typ == 'repeat':
        pos, text = (t[0], t[1]) if isinstance(t, (list, tuple)) else ('after', t)
        info['pos'] = pos; return text, kw, info
    if typ == 'nth':
        last = isinstance(t, (list, tuple)) and t[0] == 'last'; text = t[1] if last else t
        n = rng.randint(2, 5); k = n if last else rng.randint(1, n)
        x = rng.choice([w for w in WORDLIST if w not in used]); used.add(x)
        kw = {'num_paragraphs': n, 'nth_paragraph': k, 'first_word': x}
        return text.format(N=n, Nw=NUMW[n], K=k, Kord=ORDW[k], Kth=ORDTH[k], X=x), kw, info
    text = t
    if typ == 'end':
        p = rng.choice(PHRASES); kw = {'end_phrase': p}; text = text.format(P=p)
    elif typ == 'ps':
        m = rng.choice(['P.S.', 'P.P.S']); kw = {'postscript_marker': m}; text = text.format(M=m)
    elif typ == 'forbid':
        cand = [w for w in task_words if w not in used]
        k = 1 if ('{W1' in text) else rng.randint(1, min(3, len(cand)))
        ws = rng.sample(cand, k); used.update(ws); kw = {'forbidden_words': ws}
        text = text.format(Wq=qjoin(ws, "'", rng.choice(['and', 'or'])), Wdq=qjoin(ws, '"', rng.choice(['and', 'or'])), Wp=qjoin(ws, '', rng.choice(['and', 'or'])),
                           Wl='[' + ', '.join(f"'{w}'" for w in ws) + ']', W1=ws[0], W1q=f"'{ws[0]}'", W1dq=f'"{ws[0]}"')
    elif typ in ('paras', 'high', 'bullets'):
        n = rng.randint(2, 5) if typ == 'paras' else rng.randint(1, 4) if typ == 'high' else rng.randint(2, 6)
        kw = {'paras': {'num_paragraphs': n}, 'high': {'num_highlights': n}, 'bullets': {'num_bullets': n}}[typ]
        text = text.format(N=n, Nw=NUMW[n])
    return text, kw, info


def render_d(typ, tmpl, rng, task_words, used):
    if typ in ('words', 'sents', 'freq', 'capfreq', 'letter'):
        rel, text = tmpl
        if typ == 'words':
            n = rng.choice([50, 80, 120, 150, 200, 300]); return text.format(N=n), {'relation': rel, 'num_words': n}
        if typ == 'sents':
            n = rng.choice([3, 4, 5, 6, 8, 10]); return text.format(N=n), {'relation': rel, 'num_sentences': n}
        if typ == 'freq':
            cand = [w for w in task_words if w not in used]
            if not cand: return None, None
            k = rng.choice(cand); used.add(k); n = rng.choice([2, 3])
            return text.format(K=k, N=n), {'keyword': k, 'frequency': n, 'relation': rel}
        if typ == 'capfreq':
            n = rng.choice([2, 3, 5]) if rel == 'at least' else rng.choice([5, 8, 10]); return text.format(N=n), {'capital_relation': rel, 'capital_frequency': n}
        if typ == 'letter':
            L = rng.choice('aeiost'); n = rng.choice([5, 10, 15]); return text.format(L=L, N=n), {'letter': L, 'let_frequency': n, 'let_relation': rel}
    if typ == 'exist':
        cand = [w for w in task_words if w not in used]
        if len(cand) < 1: return None, None
        ks = rng.sample(cand, min(2, len(cand))); used.update(ks)
        return tmpl.format(Kq=qjoin(ks, '"')), {'keywords': ks}
    if typ == 'placeholders':
        n = rng.choice([1, 2, 3]); return tmpl.format(N=n), {'num_placeholders': n}


def valid(inst, prompt, body):
    """drop unsatisfiable combinations (the checkers would contradict each other)"""
    ty = {x[0]: x for x in inst}
    if 'no_comma' in ty and 'repeat' in ty and ',' in body: return False
    if 'no_comma' in ty and 'end' in ty and ',' in ty['end'][2]['end_phrase']: return False
    if 'forbid' in ty:
        fw = ty['forbid'][2]['forbidden_words']
        if 'repeat' in ty: return False                       # the repeated request would contain the forbidden words
        for t_ in ('end', 'nth'):
            if t_ in ty and any(re.search(r'\b' + w + r'\b', json.dumps(ty[t_][2]), re.I) for w in fw): return False
    return True


def make(seed, per, extra=None):
    rng = random.Random(seed)
    tpl = {k: list(v) for k, v in T.items()}
    if extra:                                                  # fresh-test phrasings (independent author), added to the pool
        for k, v in extra.items():
            if k in tpl: tpl[k] += [('fresh', *x) if isinstance(x, (list, tuple)) else ('fresh', x) for x in v]
    rows = []; key = seed * 100000
    bad = 0
    for typ in TARGETS:
        made = 0; i = -1
        while made < per:
            i += 1
            task, tw = rng.choice(TASKS); used = set()
            others = []
            n_extra = rng.choice([0, 0, 1, 1, 2])
            pool = [t for t in TARGETS + list(D) if t != typ]
            while len(others) < n_extra:
                o = rng.choice(pool)
                if o in others or any({o, x} <= c or {o, typ} <= c for c in CONFLICT for x in others + [typ]): continue
                if any({o, x} in CONFLICT for x in others + [typ]): continue
                others.append(o)
            inst = []
            for t_ in [typ] + others:
                if t_ in T:
                    tm = tpl[t_][i % len(tpl[t_])] if t_ == typ else rng.choice(tpl[t_])
                    if t_ == typ and extra: tm = rng.choice(tpl[t_])
                    text, kw, info = render(t_, tm, rng, tw, used)
                else:
                    text, kw = render_d(t_, rng.choice(D[t_]), rng, tw, used); info = {'tag': 'own'}
                    if text is None: continue
                inst.append((t_, text, kw, info))
            rng.shuffle(inst)
            rep = [x for x in inst if x[0] == 'repeat']; rest = [x for x in inst if x[0] != 'repeat']
            sep = rng.choice([' ', ' ', '\n', '\n\n'])
            if rng.random() < 0.15 and len(rest) >= 2:
                body = task + '\n' + '\n'.join(f'{j + 1}. {x[1]}' for j, x in enumerate(rest))
            else:
                body = sep.join([task] + [x[1] for x in rest])
            if rep:
                pos = rep[0][3]['pos']; rtext = rep[0][1]
                if pos == 'after': prompt = body + rng.choice(['\n', '\n\n']) + rtext
                elif pos == 'before': prompt = rtext + rng.choice(['\n', '\n\n']) + body
                else: prompt = body + ' ' + rtext
                rep[0][2]['prompt_to_repeat'] = body
            else:
                prompt = body
            if not valid(inst, prompt, body):
                continue_flag = True
            else:
                continue_flag = False
            if continue_flag:
                bad += 1; i_retry = True
            ids = [ID[x[0]] for x in inst]; kws = [x[2] for x in inst]
            if continue_flag: continue
            rows.append({'key': key, 'prompt': prompt, 'instruction_id_list': ids, 'kwargs': kws,
                         'meta': {'primary': typ, 'types': [x[0] for x in inst], 'tags': [x[3]['tag'] for x in inst], 'task': task}})
            key += 1; made += 1
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--per', type=int, default=30)
    ap.add_argument('--out', required=True); ap.add_argument('--extra', default='')
    a = ap.parse_args()
    extra = json.load(open(a.extra)) if a.extra else None
    rows = make(a.seed, a.per, extra)
    with open(a.out, 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    from collections import Counter
    print(len(rows), Counter(t for r in rows for t in r['meta']['types']))
