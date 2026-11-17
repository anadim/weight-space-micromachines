"""Our own vocabulary (written for this redo; not drawn from BBH). Each list has a BASE part (train/dev) and an EXTRA part
that only the fresh test split (seed 1) may use, so the test has names/objects the host never saw in fine-tuning.
Excluded on purpose: the names in BBH's few-shot prompts and the well-known BBH tracking names (Alice, Bob, Claire, Dave,
Eve, Fred, Gertrude) and deduction golfers seen in the prompt (Amy, Eli, Eve)."""

NAMES_BASE = ['Kofi', 'Ana', 'Lena', 'Ravi', 'Mei', 'Tomas', 'Ingrid', 'Yusuf', 'Chiara', 'Darius', 'Sakura', 'Mateus', 'Freya', 'Oluwa',
              'Hamid', 'Lucia', 'Bjorn', 'Amara', 'Kenji', 'Zainab', 'Pavel', 'Nadia', 'Tariq', 'Elif', 'Joaquin', 'Signe', 'Ikenna', 'Rosa',
              'Viktor', 'Leila', 'Arjun', 'Maren', 'Duc', 'Fatima', 'Sven', 'Ines', 'Kwame', 'Yara', 'Hugo', 'Anouk', 'Tenzin', 'Malia',
              'Rafael', 'Noor', 'Emeka', 'Greta', 'Hiro', 'Lupe', 'Anders', 'Priya', 'Omar', 'Teodora', 'Jin', 'Carmen', 'Dmitri', 'Wanjiru',
              'Felix', 'Aiko', 'Bruno', 'Esra', 'Laszlo', 'Thandi', 'Marco', 'Svetlana', 'Ali', 'Hana', 'Pedro', 'Ilse', 'Babajide', 'Mira',
              'Nikolai', 'Paloma', 'Sung', 'Agnes', 'Kai', 'Odile', 'Rustam', 'Imani', 'Farid', 'Lotte']
NAMES_EXTRA = ['Xochitl', 'Bartholomew', 'Ngozi', 'Ulrich', 'Saoirse', 'Kaveh', 'Mbali', 'Teodor', 'Ayesha', 'Lorenzo', 'Quilla', 'Henrik',
               'Folake', 'Zhiwei', 'Marisol', 'Oskar', 'Adaeze', 'Ruairi', 'Itzel', 'Gunnar', 'Parveen', 'Cosmin', 'Wairimu', 'Tadeo',
               'Seo-yeon', 'Matthias', 'Chidi', 'Rosalind', 'Bayani', 'Eitan']

COLORS_BASE = ['teal', 'maroon', 'olive', 'navy', 'amber', 'lilac', 'crimson', 'beige', 'turquoise', 'charcoal', 'mustard', 'coral']
COLORS_EXTRA = ['vermilion', 'saffron', 'indigo', 'chartreuse', 'mauve', 'ochre']

ORDINALS = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
NUMWORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}


def pick(r, base, extra, k, split):
    pool = base + (extra if split == 'test' else [])
    if split == 'test' and extra and r.random() < 0.5:              # make sure the extra vocabulary actually shows up
        k1 = min(len(extra), r.randint(1, k)); out = r.sample(extra, k1) + r.sample([x for x in base], k - k1)
        r.shuffle(out); return out
    return r.sample(pool, k)
