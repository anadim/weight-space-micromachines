"""Instruction detection from the PROMPT TEXT ONLY (decoded prompt tokens). Never sees kwargs / instruction_id_list.

Every pattern here was written while looking at even-key (dev) IFEval prompts; they are tuned on dev.
Returns a dict  rule_name -> params.
"""
import re

NUMW = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20, 'thirty': 30, 'a single': 1, 'single': 1}
NUM = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty)'


def num(s):
    s = s.lower()
    return int(s) if s.isdigit() else NUMW[s]


def detect(p):
    R = {}
    pl = p.lower()
    # ---- no comma ----
    if re.search(r"(no|without( using)?( any)?|not (use|include|contain|have)( any)?|n't (use|include|contain|have)( any)?|"
                 r"refrain from (using )?(any )?|avoid (using )?(any )?|cannot contain|free of)\s*commas?\b", pl):
        R['no_comma'] = {}
    # ---- case ----
    if re.search(r"lower\s?case|no capital letters|without (using )?(any )?capital letters|not (use|contain) any capital", pl) and not re.search(r"contain lowercase", pl):
        R['lower'] = {}
    if re.search(r"(all|only|entirely in|entire response (should|must) be in) (capital|uppercase|upper case|capitalized) letters|"
                 r"in all caps|all letters in your (entire )?response should be capitali[sz]ed|all capital letters|"
                 r"english,? (and )?in all capital|(no|not a single) word.{0,40}should contain lowercase", pl) and 'lower' not in R:
        if not re.search(r"words? (with|in|be in) (all )?(capital|caps)|all-caps words|capitali[sz]ed words|words in all caps", pl):
            R['caps'] = {}
    # ---- forbidden words ----
    fw = []
    for m in re.finditer(r"(do not|don't|not|never|avoid|without|must not|should not|shouldn't)\s+(say|use|using|include|including|mention|"
                         r"mentioning|contain|containing)?\s*(any of )?(the )?(key\s?words?|words?)?\s*:?\s*((?:[\"'\[][^\"'\]\n]{1,30}[\"'\]]"
                         r"(?:\s*(?:,|and|or|nor)\s*)?)+)", p, flags=re.I):
        fw += re.findall(r"[\"'\[]\s*'?([^\"'\]\n]{1,30}?)'?\s*[\"'\]]", m.group(6))
    for m in re.finditer(r"(?:do not|don't|not)\s+(?:say|use|include)\s+the\s+words?\s+([a-zA-Z]+)\s+(?:in|throughout)", p, flags=re.I):
        fw.append(m.group(1))
    for m in re.finditer(r"(?:do not|don't|not|never|avoid|without)\s+(?:say|saying|use|using|include|including|mention|mentioning)?\s*(?:the )?(?:key\s?words?|words?)\s*:?\s*\[([^\]]+)\]", p, flags=re.I):
        fw += re.findall(r"[\"']([^\"']+)[\"']", m.group(1))
    for m in re.finditer(r"the words? [\"']?(\w+)[\"']?(?:,? (?:and|or) [\"']?(\w+)[\"']?)? (?:should not|cannot|can't|must not|doesn't|does not|should never|may not) (?:appear|be)", p, flags=re.I):
        fw += [g for g in m.groups() if g]
    for m in re.finditer(r"(?:do not|don't|not|never|without)\s+(?:say|saying|use|using|include|including|mention|mentioning)\s+the words? ((?:[A-Za-z]+,?\s*)+?(?:(?:or|and) [A-Za-z]+)?)[.;]", p):
        fw += re.findall(r"[A-Za-z]+", re.sub(r"\b(or|and)\b", " ", m.group(1)))
    fw = [w.strip().strip("'\",") for w in fw]
    fw = sorted({w.lower() for w in fw if w and len(w.split()) <= 2 and not re.search(r'[*<>]', w)})
    if fw: R['forbid'] = {'words': fw}
    # ---- quotation ----
    if re.search(r"(wrap|put|enclose)\w*\b.{0,90}\b(in|with|within|inside|around)?\s*double quot", pl) or \
       re.search(r"double quot\w+ (marks )?around (your|the) (entire|whole)", pl):
        R['quote'] = {}
    # ---- repeat the request ----
    lines = p.split('\n')
    rep = [i for i, l in enumerate(lines) if re.search(r'\brep(e|le)at\b', l, re.I) and re.search(r'request|sentence|prompt|text|line|it\b', l, re.I)
           and re.search(r'first|before|then', l, re.I)]
    if rep:
        i = rep[0]
        if i == 0 or (re.search(r'below|following', lines[i], re.I) and not '\n'.join(lines[:i]).strip()):
            span = '\n'.join(lines[i + 1:]).strip()
        else:
            span = '\n'.join(lines[:i]).strip()
        if span: R['repeat'] = {'span': span}
    # ---- two responses ----
    if re.search(r'\*{6}|(6|six) asterisk', pl):
        R['two'] = {}
    elif re.search(r'two (different )?responses', pl) and re.search(r'asterisk', pl):
        R['two'] = {}
    # ---- *** paragraphs ----
    if 'two' not in R and re.search(r'\*\*\*', p.replace('******', '')):
        m = re.search(r'(exactly|at least|in|into)?\s*' + NUM + r'\s+(paragraphs|sections|parts)', pl)
        if m:
            R['paras'] = {'n': num(m.group(2))}
        else:
            k = len(re.findall(r'^\s*\*\*\*\s*$', p, flags=re.M))
            if k: R['paras'] = {'n': k + 1}
    # ---- nth paragraph first word (\n\n separated) ----
    m = re.search(r'(exactly )?' + NUM + r' paragraphs', pl)
    if m and 'paras' not in R and re.search(r'two new ?lines|2 new lines|\\n\\n|two newlines', pl):
        n = num(m.group(2)); fw = re.search(r'(?:paragraph (\d+)|(\d+)(?:st|nd|rd|th) paragraph|(first|second|third|last) paragraph)\s+'
                                           r'(?:must |should )?(?:start|begin)s? with (?:the word )?["\']?([A-Za-z]+)', p, re.I)
        fw2 = re.search(r'start (?:paragraph (\d+)|the (\d+)(?:st|nd|rd|th) paragraph) with (?:the word )?["\']?([A-Za-z]+)', p, re.I)
        if fw:
            idx = fw.group(1) or fw.group(2) or {'first': '1', 'second': '2', 'third': '3', 'last': str(n)}[fw.group(3).lower()]
            R['nth'] = {'n': n, 'k': int(idx), 'word': fw.group(4)}
        elif fw2:
            R['nth'] = {'n': n, 'k': int(fw2.group(1) or fw2.group(2)), 'word': fw2.group(3)}
        else:
            R['nth'] = {'n': n, 'k': 0, 'word': None}
    # ---- word counts ----
    m = (re.search(r'(at least|more than|minimum of|over)\s+(\d+)\s+words', pl) or re.search(r'(\d+)\+?\s*(or more )?words? (long|or more)', pl)
         or re.search(r'(\d+)\+\s*-?\s*words?', pl) or re.search(r'(\d+) or more words', pl) or re.search(r'words?\D{0,40}(\d+) or more', pl)
         or re.search(r'at least (\d+) words', pl))
    if m and re.match(r'\s*(be )?(in|with) all cap', pl[m.end():]): m = None
    if m:
        n = int([g for g in m.groups() if g and g.isdigit()][0]); R['words'] = {'rel': 'atleast', 'n': n}
    else:
        m = (re.search(r'(less than|fewer than|under|at most|no more than|maximum of|not exceed)\s+(\d+)\s+words', pl)
             or re.search(r'(\d+) (or less|or fewer) words', pl) or re.search(r'(\d+) words (or less|or fewer)', pl)
             or re.search(r'(less than) (\d+) words', pl) or re.search(r'have (\d+) or less words', pl))
        if m:
            n = int([g for g in m.groups() if g and g.isdigit()][0])
            if re.search(r'at most|or less|or fewer|no more than|maximum|not exceed', m.group(0)): n += 1
            R['words'] = {'rel': 'less', 'n': n}
        else:
            m = re.search(r'\b(\d+)-word\b', pl)
            if m: R['words'] = {'rel': 'atleast', 'n': int(m.group(1))}
    # ---- sentence counts (a 'less' and an 'atleast' bound may both be present) ----
    m = re.search(r'(less than|fewer than|under)\s+(?:a total of\s+)?' + NUM + r'\s+sentences', pl)
    if m: R['sents_less'] = {'n': num(m.group(2))}
    m = re.search(r'(exactly|only) (one|a single|1) sentence', pl) or re.search(r'one-sentence', pl)
    if m and not re.search(r'each line', pl): R['sents_less'] = {'n': 2}
    m = re.search(r'(at least|more than)\s+' + NUM + r'\s+sentences', pl)
    if m: R['sents_atleast'] = {'n': num(m.group(2)) + (m.group(1) == 'more than')}
    m = re.search(r'(in )?' + NUM + r'\s+(?:to|or)\s+(\d+)\s+sentences', pl)
    if m: R['sents_atleast'] = {'n': num(m.group(2))}; R['sents_less'] = {'n': int(m.group(3)) + 1}
    return R


GOLD_MAP = {'punctuation:no_comma': 'no_comma', 'change_case:english_lowercase': 'lower', 'change_case:english_capital': 'caps',
            'keywords:forbidden_words': 'forbid', 'startend:quotation': 'quote', 'combination:repeat_prompt': 'repeat',
            'combination:two_responses': 'two', 'length_constraints:number_paragraphs': 'paras',
            'length_constraints:nth_paragraph_first_word': 'nth', 'length_constraints:number_words': 'words',
            'length_constraints:number_sentences': 'sents'}
SENT_NAMES = ('sents_less', 'sents_atleast')

if __name__ == '__main__':          # detector precision/recall against gold ids (evaluation only; the detector never sees them)
    import json, sys
    keys = sys.argv[1] if len(sys.argv) > 1 else 'even'
    D = [json.loads(l) for l in open('../data/ifeval.jsonl')]
    D = [d for d in D if keys == 'all' or (d['key'] % 2 == 0) == (keys == 'even')]
    from collections import Counter
    tp, fp, fn = Counter(), Counter(), Counter(); bad = []
    for d in D:
        R = detect(d['prompt']); gold = {}
        for i, kw in zip(d['instruction_id_list'], d['kwargs']):
            if i not in GOLD_MAP: continue
            g = GOLD_MAP[i]
            if g == 'sents': g = 'sents_less' if kw['relation'][0] == 'l' else 'sents_atleast'
            gold[g] = kw
        for r in set(R) | set(gold):
            if r in R and r in gold:
                tp[r] += 1; g = gold[r]; ok = True
                if r == 'forbid': ok = sorted(R[r]['words']) == sorted(g['forbidden_words'])
                if r == 'repeat': ok = R[r]['span'].strip().lower() .startswith(g['prompt_to_repeat'].strip().lower())
                if r == 'paras': ok = R[r]['n'] == g['num_paragraphs']
                if r == 'nth': ok = (R[r]['n'], R[r]['k'], (R[r]['word'] or '').lower()) == (g['num_paragraphs'], g['nth_paragraph'], g['first_word'])
                if r == 'words': ok = R[r]['n'] == g['num_words'] and R[r]['rel'][0] == g['relation'][0]
                if r.startswith('sents'): ok = R[r]['n'] == g['num_sentences']
                if not ok: bad.append((d['key'], r, R[r], {k: v for k, v in g.items() if v is not None}))
            elif r in R: fp[r] += 1; bad.append((d['key'], r, 'FP', R[r]))
            else: fn[r] += 1; bad.append((d['key'], r, 'FN', d['prompt'][-200:]))
    for r in [v for v in GOLD_MAP.values() if v != 'sents'] + list(SENT_NAMES): print(f'{r:10s} tp={tp[r]:3d} fp={fp[r]:3d} fn={fn[r]:3d}')
    for b in bad: print(b)
