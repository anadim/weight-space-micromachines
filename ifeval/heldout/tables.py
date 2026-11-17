"""Fixed lookup tables over the host vocabulary (built offline from the token strings; no model weights involved).

Word classes are GENERIC English word lists (written down here, justified in NOTES.md): negation words, case words,
quote words, ... A token belongs to a class if its core (lowercased, stripped of spaces/punctuation) is in the list
(single-token words), or it is the LAST piece of a multi-token spelling of a listed word (n-gram patterns over token ids,
e.g. 'lower'+'case', ' U'+'PP'+'ERC'+'ASE'). Both are fixed tables: token id -> flags, pattern -> class.
"""
import os, re, torch

PUNCT = ' \t\n\r.,;:!?"“”\'‘’()[]{}<>*-_/`~#|'
DQ, SQ = '"“”', "'‘’"


def wcore(s): return s.strip().lower().strip(PUNCT)


W = {
 'NEG': 'no not never without avoid avoiding avoided avoids refrain refraining zero forbidden stop quit cease forbid forbids banned ban '
        'prohibited prohibit disallowed omit omitting exclude excluding none nor cannot nothing skip eliminate ditch dont cant '
        'wont shouldnt mustnt doesnt isnt arent didnt neither',
 'COMMA': 'comma commas',
 'COMMAP': 'commas',
 'FREEW': 'free',
 'INVW': 'inverted',
 'FORGETW': 'forget forgetting fail hesitate neglect mind',
 'ANDBUT': 'and but',
 'UPPERW': 'uppercase uppercased|upper case|upper-case|all-caps|allcaps|all caps|block letters|upper cased',
 'CAPSW': 'caps capitals',
 'QMARKW': 'marks mark double|speech marks|inverted commas',
 'WRAPS': 'wrap wrapped wrapping enclose enclosed enclosing encase encased surround surrounded surrounding bookend',
 'WHOLE2': 'entire entirety whole full complete everything response answer reply output text message',
 'TWOW': 'two both responses versions alternatives answers options variants replies drafts',
 'REQV': 'add include append put write insert use using give provide make list format answer respond reply create generate produce present',
 'POINTW': 'point points list lists item items',
 'BULLETS': 'bullets bulleted bulletpoint bulletpoints|bullet-points|bullet-point',
 'BULLET1': 'bullet',
 'UNITW': 'word words sentence sentences line lines character characters letter letters minute minutes page pages hour hours day days',
 'POSSW': 'the these those following my his her their our its this',
 'VERBATIM': 'verbatim exactly exact unchanged as-is|word for word|word-for-word|as it is|as written|without change|without changes|without changing',
 'ABOVED': 'above previous preceding earlier prior',
 'STARTV': 'start starts starting begin begins beginning open opens opening lead leads',
 'FREQ3': 'more than fewer less least most',
 'IMPPRE': 'please kindly also then and just finally pls plz',
 'REQMARK': 'use using write keep give answer respond reply put add include avoid do don dont let wrap enclose surround type output format '
            'try remember make ensure limit restrict compose create draft list provide tell begin start end finish',
 'REQANY': 'you your yours yourself please pls plz kindly must should need needs shall cannot task|have to|has to|make sure|be sure',

 'INSTV': 'write give make use keep put add include avoid answer reply respond provide list describe explain tell create draft compose '
          'generate produce summarize summarise rewrite format separate divide split organize organise structure highlight italicize '
          'italicise bold emphasize emphasise wrap enclose surround end finish close conclude start begin open repeat restate copy offer '
          'present return output print show type try remember ensure limit restrict refrain skip omit exclude stick tack throw append '
          'attach insert sign render frame do don dont let place mark label number break echo recite reproduce reiterate read '
          'divide bullet translate continue respond address treat follow lead kick stay',
 'SUBJ': 'the a an my his her their our its this that these those i we he she they it there some many most someone somebody people '
         'everyone anyone nobody one what why how who when where which whose was were if in on at '
         'as so because although though while since after when',
 'CANCELW': 'anymore|any more|no longer|back to normal|plain text|fine again|regular text',
 'ASTERISK': 'asterisk asterisks',
 'EXPLAINV': 'explain explaining define describe discuss tell wonder wondered wonders asked',
 'SAYW': 'say',
 'YOURW': 'your yours',
 'REFMT': 'into but',
 'REPORTV': 'says said claims suggests recommends insists argues',
 'CODEW': 'code function program regex snippet heredoc python javascript bash',
 'DATAV': 'translate translating proofread correct convert paraphrase summarize summarise rewrite edit fix',
 'DATAW': 'here email text note message letter passage draft posted wrote sent assignment',
 'REMOVEV': 'remove delete drop cut strip change alter touch',
 'PARTW': 'just title titles heading headings name names acronym acronyms',
 'BEW': 'be is reads read',
 'PHRASEW': 'phrase sentence words line question statement quote',
 'REQMOD': 'please pls plz kindly must should need needs shall|have to|has to|make sure|be sure',
 'HIGHOBJ': 'section sections part parts phrase phrases sentence sentences passage passages term terms text',
 'PSV': 'add adding include including append attach put insert end close conclude finish sign follow followed tack stick',
 'LOWERW': 'lowercase lowercased lowercases lowercasing lowercaps|lower case|lower-case|small letters|small-letter|lower cased',
 'CAPW': 'capital capitals capitalized capitalised capitalize capitalise capitalization capitalisation uppercase uppercased caps allcaps|'
         'upper case|upper-case|all-caps|block letters|upper cased',
 'LETTER': 'letter letters character characters chars',
 'ALLQ': 'all entire entirely every whole only fully completely throughout everything exclusively solely full total totally',
 'INW': 'in using with',
 'TIMES': 'times twice thrice once frequency',
 'FREQ2': 'more than fewer less least most exceed exceeding maximum minimum max min under limit',
 'STARTW': 'start starts starting begin begins beginning first initial open opens opening lead leading',
 'EACH': 'each every',
 'FIRSTEACH': 'first each initial',
 'QUOTEW': 'quotation quotations quotes quote quoted quotemarks|speech marks|inverted commas',
 'SINGLE': 'single',
 'WRAPV': 'wrap wrapped wrapping enclose enclosed enclosing encase encased surround surrounded surrounding put place placed '
          'inside within between around bracket bookend',
 'WHOLE': 'entire entirety whole full all everything complete response answer reply output text message writing',
 'ENDW': 'end ends ending finish finishes finishing conclude concludes concluding close closes closing last final|sign off|wrap up|round off',
 'ENDV': 'end ends ending finish finishes finishing conclude concludes concluding close closes closing|sign off|wrap up|round off',
 'ARTW': 'a an',
 'PSW': 'postscript postscripts|post script|post-script|p.s|p.s.|p.p.s',
 'PPS': 'p.p.s|p.p.s.',
 'WORDTERM': 'word words keyword keywords term terms',
 'APPEARS': 'appear appears be been used allowed permitted included mentioned show shown occur present found written said needed '
            'contained featured',
 'NEGADJ': 'forbidden banned prohibited disallowed taboo excluded|off-limits|off limits',
 'SEPW': 'and or nor &',
 'DETW': 'the these those following',
 'WHQ': 'why what how who whom whose which when where whether if',
 'PLEASEW': 'please pls plz kindly',
 'AUXW': 'should shall can could would will is are was were may might',
 'YOUW': 'you your yourself',
 'MODALQ': 'can could would will mind',
 'OTHERS': 'he she they people someone somebody everyone anybody anyone',
 'DESCV': 'writes types texts prefers likes capitalizes says',
 'TITLEW': 'title titles heading headline',
 'SEPV': 'separate separated separating separates divide divided dividing between split splitting',
 'ADDV': 'add adding include including append attach put write insert finish end close conclude follow followed sign start starting begin beginning',
 'CONJ': 'and but then also plus while',
 'REPEATV': 'repeat repeating repeated restate restating copy copying echo echoing reproduce reproducing recite reiterate',
 'REQTERM': 'request requests prompt question query instruction instructions text sentence sentences line lines message task ask',
 'BELOWD': 'below following follows next underneath beneath',
 'ASTER': 'asterisk asterisks star stars',
 'PARAW': 'paragraph paragraphs para paras',
 'SECTW': 'section sections part parts segment segments block blocks',
 'DIVW': 'divider dividers separator separators',
 'MARKDOWN': 'markdown',
 'LASTW': 'last final',
 'WITHW': 'with',
 'ANGW': 'angular angle chevron chevrons',
 'BRACKW': 'bracket brackets',
 'HIGHW': 'highlight highlighted highlights highlighting bold bolded embolden emphasize emphasise emphasized emphasised',
 'ITALW': 'italic italics italicize italicized italicise italicised italicizing italicising',
 'BULLETW': 'bullet bullets bulleted bulletpoint bulletpoints|bullet-point|bullet-points',
 'SKIPNTH': 'the word term',
 'STOPF': 'in from throughout anywhere at within should must can cannot will would could may might shall is are was were be being '
          'been has have had do does did don doesn isn aren won shouldn mustn didn to your my the a an any for of on when while '
          'during please ever even all response answer reply text output appear appears used allowed at as this that these those '
          'it its you yourself also either neither no not never count counts limit limits choice order',
}
NUMWORDS = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
                                         'sixteen seventeen eighteen nineteen twenty'.split())}
ORDWORDS = {w: i + 1 for i, w in enumerate('first second third fourth fifth sixth seventh eighth ninth tenth'.split())}
CLASSES = list(W)


def build(tok, strs, device='cuda', cache='tables.pt'):
    if os.path.exists(cache): return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in torch.load(cache).items()}
    V = len(strs); C = len(CLASSES)
    single = {c: set() for c in CLASSES}; multi = {c: set() for c in CLASSES}
    for c, s in W.items():
        parts = s.split('|')                                   # first part: single words; later parts: multi-word phrases
        for w in parts[0].split():
            single[c].add(w); multi[c].add(w)                  # (multi: catch multi-piece spellings of single words)
        for w in parts[1:]:
            multi[c].add(w)
            if not (' ' in w or '-' in w or '.' in w): single[c].add(w)
    T = torch.zeros(V, C)
    cores = [wcore(s) for s in strs]
    for i, s in enumerate(strs):
        if i >= 151643: continue
        for j, c in enumerate(CLASSES):
            if cores[i] in single[c]: T[i, j] = 1
        if s.strip() in ("'t", "’t", "n't", "n’t"): T[i, CLASSES.index('NEG')] = 1
    # n-gram patterns: every casing/spacing variant of every listed word, tokenized; patterns of length >= 2 only
    pats, pcls = [], []
    for c in CLASSES:
        for w in multi[c]:
            for v in {w, w.capitalize(), w.upper(), w.title()}:
                for pre in ['', ' ', '-', '"', "'", '(']:
                    ids = tok(pre + v, add_special_tokens=False).input_ids
                    if pre in ('-', '"', "'", '(') and len(ids) > 1 and strs[ids[0]] == pre: ids = ids[1:]
                    if 2 <= len(ids) <= 5: pats.append(tuple(ids)); pcls.append(CLASSES.index(c))
    uniq = {}
    for p, c in zip(pats, pcls): uniq[(p, c)] = 1
    pats = [p for p, c in uniq]; pcls = [c for p, c in uniq]
    K = 5; P = torch.full((len(pats), K), -1, dtype=torch.long)
    for i, p in enumerate(pats): P[i, K - len(p):] = torch.tensor(p)                 # right-aligned: offset 0 = last piece
    PC = torch.zeros(len(pats), C); PC[torch.arange(len(pats)), torch.tensor(pcls)] = 1
    # numbers
    NUMV = torch.zeros(V); ISNUM = torch.zeros(V); ISDIG = torch.zeros(V); ORDV = torch.zeros(V); ORDSUF = torch.zeros(V)
    for i, s in enumerate(strs):
        if i >= 151643: continue
        c = cores[i]
        if re.fullmatch(r'\d', s): NUMV[i] = int(s); ISNUM[i] = 1; ISDIG[i] = 1
        elif c in NUMWORDS and re.fullmatch(r'\s?[A-Za-z]+', s): NUMV[i] = NUMWORDS[c]; ISNUM[i] = 1
        if c in ORDWORDS and re.fullmatch(r'\s?[A-Za-z]+', s): ORDV[i] = ORDWORDS[c]
        if s in ('st', 'nd', 'rd', 'th'): ORDSUF[i] = 1
    f = lambda fn: torch.tensor([float(fn(s)) if i < 151643 else 0. for i, s in enumerate(strs)])
    nonws = f(lambda s: s.strip() != '')
    tabs = dict(
        T=T, P=P, PC=PC, NUMV=NUMV, ISNUM=ISNUM, ISDIG=ISDIG, ORDV=ORDV, ORDSUF=ORDSUF, NONWS=nonws,
        SENTB=f(lambda s: any(ch in s for ch in '.!?;\n') and not re.fullmatch(r'\.[A-RT-Za-rt-z]', s)),   # '.P' of 'P.P.S', '.e' of 'i.e.' are no boundaries ('.S' ends the marker)
        CLAUSEB=f(lambda s: any(ch in s for ch in '.!?;\n,') and not re.fullmatch(r'\.[A-RT-Za-rt-z]', s)),
        NL=f(lambda s: '\n' in s), NN=f(lambda s: '\n\n' in s), ENDNL=f(lambda s: s.endswith('\n')), STARTNL=f(lambda s: s.startswith('\n')),
        PRECNN=f(lambda s: s.split('\n\n')[0].strip() != ''), POSTNN=f(lambda s: s.split('\n\n')[-1].strip() != ''),
        STAR3=f(lambda s: '***' in s and '******' not in s), ABBRP=f(lambda s: bool(re.fullmatch(r'\.[A-Za-z]', s))), CAPSTART=f(lambda s: bool(re.match(r'\s[A-Z]', s))),
        LEADST=torch.tensor([float(min(6, len(re.match(r'\s*(\**)', t).group(1)))) if i < 151643 else 0. for i, t in enumerate(strs)]),
        TRAILST=torch.tensor([float(min(6, len(re.search(r'(\**)$', t).group(1)))) if i < 151643 else 0. for i, t in enumerate(strs)]),
        ALLST=f(lambda s: s.strip() != '' and set(s.strip()) == {'*'}),
        DQC=f(lambda s: s.count('"') % 2 == 1), BTC=f(lambda s: s.count('`') % 2 == 1), DQOPEN=f(lambda s: '“' in s), DQCLOSE=f(lambda s: '”' in s), QMARK=f(lambda s: '?' in s), STAR6=f(lambda s: '******' in s), STAR6X=f(lambda s: s.strip() == '******'),
        ANGTOK=f(lambda s: '<<' in s), CLOSETITLE=f(lambda s: '>>' in s),
        COLON=f(lambda s: s.strip() in (':', ' :') or (s.strip().endswith(':') and not re.search(r'[A-Za-z0-9]', s))),
        STARC=f(lambda s: '*' in s and '***' not in s),
        BULST=f(lambda s: bool(re.match(r'[ \t]*(\*(?!\*)|-)', s)) and s.strip() not in ('',)),
        LETTERS=f(lambda s: bool(re.search(r'[A-Za-z]', s))),
        WSTART=f(lambda s: bool(re.match(r'\s', s)) or not re.match(r'[A-Za-z0-9]', s)),     # token begins a new word
        ENDALNUM=f(lambda s: bool(re.search(r'[A-Za-z0-9]$', s))),
        ALPHA=f(lambda s: bool(re.fullmatch(r'\s?[A-Za-z]+', s))),
        CONT=f(lambda s: bool(re.fullmatch(r'[A-Za-z]+', s))),                                  # continuation piece (no space)
        QE=f(lambda s: s.strip() != '' and s.rstrip().endswith('"')), QX=f(lambda s: s.rstrip() == '"'),
        # quote tokens (for argument capture): an opener has a quote char and no letters/digits; a closer has a quote char not followed by a letter
        OPENDQ=f(lambda s: bool(re.fullmatch(r'\s*[\(\[\{:]*[' + DQ + ']', s))),          # ' "', '"', '["', ' “'
        OPENSQ=f(lambda s: bool(re.fullmatch(r"[\s\(\[\{:]+['‘]|‘", s))),        # needs a space/bracket before (not the possessive in kids')
        CLOSEDQ=f(lambda s: bool(re.search('[' + DQ + '](?![A-Za-z])', s))),
        CLOSESQ=f(lambda s: bool(re.search('[' + SQ + '](?![A-Za-z])', s))),
        IMSTART=f(lambda s: False), IMEND=f(lambda s: False),
    )
    tabs['IMSTART'][151644] = 1; tabs['IMEND'][151645] = 1
    ids_of = {}
    for i, s in enumerate(strs[:151643]):
        ids_of.setdefault(s, i)
    def tid(s): return ids_of.get(s, -1)
    # PRE: content before the first quote char / newline (rstripped), when that is a single token (e.g. '?"' -> '?')
    PRE = torch.full((V,), -1, dtype=torch.long)
    for i, s in enumerate(strs[:151643]):
        m = re.split('[' + DQ + SQ + '\n]', s, maxsplit=1)
        if len(m) == 2 and m[0].strip() != '': PRE[i] = tid(m[0].rstrip())
    # case maps (text -> its lowercase / uppercase spelling, if that is one token), no-space and capitalized variants
    LOW = torch.arange(V); UP = torch.arange(V); NOSP = torch.arange(V); CAP1 = torch.arange(V)
    for i, s in enumerate(strs[:151643]):
        j = tid(s.lower()); LOW[i] = j if j >= 0 else i
        j = tid(s.upper()); UP[i] = j if j >= 0 else i
        j = tid(s.lstrip()) if s.lstrip() else -1; NOSP[i] = j if j >= 0 else i
        t = s.lstrip(); j = tid(t[:1].upper() + t[1:]) if t else -1; CAP1[i] = j if j >= 0 else NOSP[i]
    # canonical word class (lowercase, no leading space/punct) for forbidden-word matching
    canon = {}; CANON = torch.full((V,), -1, dtype=torch.long)
    for i, s in enumerate(strs[:151643]):
        c = s.strip().lower()
        if c and re.fullmatch(r"[a-z0-9'’-]+", c): CANON[i] = canon.setdefault(c, len(canon))
    # rolling-hash tables (word identity across tokenizations): token = [lead non-alnum chars][core a-z0-9][rest]
    HP, HB = 2147483647, 131
    HT = torch.zeros(V, dtype=torch.long); POWT = torch.ones(V, dtype=torch.long); LEAD = torch.zeros(V); TRAIL = torch.zeros(V); HASAN = torch.zeros(V)
    for i, s in enumerate(strs[:151643]):
        m = re.match(r'^([^a-z0-9]*)([a-z0-9]*)(.*)$', s.lower(), re.S)
        lead, core, rest = m.group(1), m.group(2), m.group(3)
        h = 0
        for ch in core: h = (h * HB + ord(ch)) % HP
        HT[i] = h; POWT[i] = pow(HB, len(core), HP); LEAD[i] = float(lead != ''); TRAIL[i] = float(rest != ''); HASAN[i] = float(core != '')
    tabs.update(PRE=PRE, LOW=LOW, UP=UP, NOSP=NOSP, CAP1=CAP1, CANON=CANON, HT=HT, POWT=POWT, LEAD=LEAD, TRAIL=TRAIL, HASAN=HASAN)
    tabs['TOK'] = {k: tid(k) for k in ['"', '<<', '>>', '*', '-', '\n', '\n\n', '******', '***\n\n', 'P', '.S', '.P', '.', 'p', '.s', '.p']}
    # uppercase spelling of every token as a token sequence (<= 6 tokens), for the case transducer
    UPSEQ = torch.full((V, 6), -1, dtype=torch.long)
    idx = [i for i in range(151643) if strs[i] != strs[i].upper()]
    enc = tok([strs[i].upper() for i in idx], add_special_tokens=False).input_ids
    for i, e in zip(idx, enc):
        e = e[:6]; UPSEQ[i, :len(e)] = torch.tensor(e)
    tabs['UPSEQ'] = UPSEQ
    LOWSEQ = torch.full((V, 6), -1, dtype=torch.long)
    idx = [i for i in range(151643) if strs[i] != strs[i].lower()]
    enc = tok([strs[i].lower() for i in idx], add_special_tokens=False).input_ids
    for i, e in zip(idx, enc):
        e = e[:6]; LOWSEQ[i, :len(e)] = torch.tensor(e)
    tabs['LOWSEQ'] = LOWSEQ
    lowerset = torch.tensor([float(s != s.upper()) if i < 151643 else 0. for i, s in enumerate(strs)])
    upperset = torch.tensor([float(s != s.lower()) if i < 151643 else 0. for i, s in enumerate(strs)])
    tabs.update(LOWERSET=lowerset, UPPERSET=upperset, COMMASET=f(lambda s: ',' in s), STAR3SET=f(lambda s: '***' in s))
    torch.save(tabs, cache)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tabs.items()}
