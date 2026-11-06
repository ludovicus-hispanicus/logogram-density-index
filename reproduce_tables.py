# -*- coding: utf-8 -*-
"""Regenerate every table of figures reported in the paper, from the corpus.

    python reproduce_tables.py

Prints each table with the label it carries in the paper, so every number in
the article can be checked against the data.  Requires only compute_ratios.py
and data/ (see requirements.txt).
"""
import glob
import os
import re
import sys

import pandas as pd

import compute_ratios as C

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ORDER = C.PERIOD_ORDER
DISCIPLINE = {
    'astrological omens': 'astrological (Enuma Anu Enlil)',
    'diagnostic omens':   'diagnostic (Sakikku)',
    'extispicy omens':    'extispicy (liver and lung)',
    'izbu omens':         'birth (Summa izbu)',
    'terrestrial omens':  'terrestrial (Summa Alu)',
}


def fold(period):
    """Map any edition label onto the three periods used in the paper."""
    if period in ORDER:
        return period
    if period in C.PERIOD_MAPPING:
        return C.PERIOD_MAPPING[period]
    for name in ORDER:                       # e.g. 'Neo-Assyrian / Late Babylonian'
        if name.split()[0] in period:
            return name
    return period


def load():
    """(every_omen, words, signs, preserved).

    The index measures how Akkadian is written, so Sumerian (%sux) is held out of
    words, signs and preserved, exactly as the article and verify_article.py score
    it. every_omen keeps all languages and is for counting omens only: a wholly
    Sumerian omen is still an omen in the corpus table.
    """
    note('reading the corpus')
    every_omen = pd.DataFrame(C.load_local_data())
    signs = pd.DataFrame(C.load_local_signs())
    preserved = pd.DataFrame(C.load_local_data(preserved_only=True))
    for frame in (every_omen, signs, preserved):
        frame['era'] = frame['period'].map(fold)
    words = every_omen[every_omen['language'] == 'akkadian']
    signs = signs[signs['language'] == 'akkadian']
    preserved = preserved[preserved['language'] == 'akkadian']
    return every_omen, words, signs, preserved


def scored(frame, exclude_particles=False, monogram_as_log=False):
    """The rows compute_ratios.ldi() would score, with a 0/1 column per sign.

    Grouping this frame reproduces ldi() exactly but in one pass, which matters
    when scoring every omen individually.
    """
    sub = frame[frame['type'].isin(['logogram', 'phonetic'])]
    sub = C._drop_contentless(sub)
    if exclude_particles:
        mask = sub['token'].isin(C.LOGOGRAM_PARTICLES) & (sub['type'] == 'logogram')
        sub = sub[~mask]
    kind = sub['type']
    if monogram_as_log:
        kind = kind.where(~sub['token'].isin(C.MONOGRAM_PARTICLES), 'logogram')
    return sub.assign(is_log=(kind == 'logogram').astype(int))


_GRADED = {}


def graded(signs, era=None, exclude_particles=False):
    """graded_ldi(), computed once per (subset, convention) and reused.

    The three measures, the composition table and the by-convention table all
    need the same subsets; recomputing them is the bulk of the runtime.
    """
    key = (era, exclude_particles)
    if key not in _GRADED:
        subset = signs if era is None else signs[signs.era == era]
        note(f'scoring {era or "whole corpus"}'
             f'{" (particles excluded)" if exclude_particles else ""}')
        _GRADED[key] = C.graded_ldi(subset, exclude_particles=exclude_particles)
    return _GRADED[key]


def word_degrees(signs):
    """One row per scored word: its logographic and phonetic sign counts.

    The same words graded_ldi() grades, so macro and micro of any grouping
    (text, omen) follow from one pass instead of one graded_ldi() call each.
    """
    content = C._drop_contentless(signs[signs['type'].isin(['logogram', 'phonetic'])])
    content = content.assign(nl=(content['type'] == 'logogram').astype(int),
                             np=(content['type'] == 'phonetic').astype(int))
    w = content.groupby(['filename', 'omen_id', 'word_index'])[['nl', 'np']].sum().reset_index()
    w = w[w.nl + w.np > 0]
    return w.assign(degree=w.nl / (w.nl + w.np))


def macro_micro(words_frame, keys):
    """macro and micro per group of word_degrees() rows."""
    g = words_frame.groupby(keys)
    return pd.DataFrame({'macro': g.degree.mean(),
                         'micro': g.nl.sum() / (g.nl.sum() + g.np.sum())})


def edition_labels():
    """{filename: the period label its edition gives}, one spelling per label.

    load_local_data() folds these into the three periods, so read them from the files.
    """
    labels = {}
    for group in ('old', 'middle', 'new', 'kal5'):
        for path in glob.glob(os.path.join('data', group, '**', '*.txt'), recursive=True):
            with open(path, encoding='utf-8', errors='replace') as f:
                found = re.search(r'^period:\s*(.+)$', f.read(1500), re.M)
            if found:
                label = found.group(1).strip().strip('"')
                labels[os.path.basename(path)] = {'Neo Babylonian': 'Neo-Babylonian'}.get(label, label)
    return labels


def note(message):
    print(f'  ... {message}', file=sys.stderr, flush=True)


def head(title, label):
    print(f'\n{title}   [{label}]\n' + '-' * (len(title) + len(label) + 6), flush=True)


def main():
    every_omen, words, signs, preserved = load()
    n_omens = every_omen.groupby(['filename', 'omen_id']).ngroups
    print(f'corpus: {n_omens} omens in {every_omen.filename.nunique()} texts')

    # ---- Table: omens (texts) by discipline and period -------------------
    head('Omens (texts) by discipline and period', 'tbl:corpus')
    print(f'{"discipline":<32}' + ''.join(f'{p.split()[0]:>14}' for p in ORDER) + f'{"total":>14}')
    for genre, label in DISCIPLINE.items():
        sub = every_omen[every_omen.genre == genre]
        cells = []
        for era in ORDER:
            e = sub[sub.era == era]
            cells.append(f'{e.groupby(["filename", "omen_id"]).ngroups} ({e.filename.nunique()})')
        cells.append(f'{sub.groupby(["filename", "omen_id"]).ngroups} ({sub.filename.nunique()})')
        print(f'{label:<32}' + ''.join(f'{c:>14}' for c in cells))
    cells = []
    for era in ORDER:
        e = every_omen[every_omen.era == era]
        cells.append(f'{e.groupby(["filename", "omen_id"]).ngroups} ({e.filename.nunique()})')
    cells.append(f'{n_omens} ({every_omen.filename.nunique()})')
    print(f'{"TOTAL":<32}' + ''.join(f'{c:>14}' for c in cells))

    # ---- Table: the index by period --------------------------------------
    head('The index by period', 'tbl:periods')
    print(f'{"period":<10}{"bin":>8}{"macro":>8}{"micro":>8}')
    for era in ORDER:
        g = graded(signs, era)
        b = C.ldi(words[words.era == era])[0]
        print(f'{era.split()[0]:<10}{b:>8.2f}{g["macro"]:>8.2f}{g["micro"]:>8.2f}')

    # ---- Table: word composition by period -------------------------------
    head('Word composition by period', 'tbl:composition')
    print(f'{"period":<10}{"pure-log":>10}{"mixed":>10}{"syllabic":>10}')
    for era in ORDER:
        g = graded(signs, era)
        n = g['words']
        print(f'{era.split()[0]:<10}{100*g["pure"]/n:>9.1f}%{100*g["mixed"]/n:>9.1f}%'
              f'{100*g["syllabic"]/n:>9.1f}%')

    # ---- Table: effect of excluding the opening particle ------------------
    head('Effect of excluding the opening particle, by discipline', 'tbl:particles')
    print(f'{"discipline":<14}{"baseline":>10}{"excluded":>10}{"effect":>9}{"omen len":>10}')
    rows = []
    for genre in DISCIPLINE:
        sub = words[words.genre == genre]
        base = C.ldi(sub)[0]
        excl = C.ldi(sub, exclude_particles=True)[0]
        countable = scored(sub)
        length = len(countable) / countable.groupby(['filename', 'omen_id']).ngroups
        rows.append((genre.split()[0], base, excl, excl - base, length))
    for name, base, excl, eff, length in sorted(rows, key=lambda r: -r[4]):
        print(f'{name:<14}{base:>10.3f}{excl:>10.3f}{eff:>+9.3f}{length:>10.1f}')

    # ---- Sensitivity baselines and the nine conventions -------------------
    head('The same corpus under nine defensible conventions', 'tbl:spread')
    g_all = graded(signs)
    marked = scored(words)
    by_text = marked.groupby('filename').is_log.mean()
    by_omen = marked.groupby(['filename', 'omen_id']).is_log.mean()
    conventions = [
        ('ina/ana counted as logographic', C.ldi(words, monogram_as_log=True)[0]),
        ('per-omen mean',                  by_omen.mean()),
        ('bin, as published (baseline)',   C.ldi(words)[0]),
        ('restorations dropped',           C.ldi(preserved)[0]),
        ('opening particles excluded',     C.ldi(words, exclude_particles=True)[0]),
        ('per-text mean',                  by_text.mean()),
        ('unit = word, graded (macro)',    g_all['macro']),
        ('unit = sign (micro)',            g_all['micro']),
        ('micro, particles excluded',      graded(signs, exclude_particles=True)['micro']),
    ]
    for name, value in conventions:
        print(f'  {name:<34}{value:>8.3f}')
    values = [v for _, v in conventions]
    print(f'  {"range":<34}{max(values) - min(values):>8.3f}'
          f'   ({min(values):.3f} to {max(values):.3f})')

    # ---- Words versus signs: macro and micro are not ordered ---------------
    head('Where micro exceeds macro', 'sec. Words versus Signs')
    degrees = word_degrees(signs)
    per_text = macro_micro(degrees, 'filename')
    per_omen = macro_micro(degrees, ['filename', 'omen_id'])
    print(f'  micro > macro in {(per_text.micro > per_text.macro + 1e-12).sum()} of {len(per_text)} texts'
          f' and in {100*(per_omen.micro > per_omen.macro + 1e-12).mean():.1f} % of {len(per_omen)} scored omens')

    # ---- Table: two corpora bin cannot separate ---------------------------
    head('Two corpora that the binary measure cannot separate', 'tbl:discrimination')
    for name, mask_w, mask_s in [
        ('Enuma Anu Enlil 55 and 57', words.filename.isin(['EAE55.eBL.txt', 'EAE57.eBL.txt']),
         signs.filename.isin(['EAE55.eBL.txt', 'EAE57.eBL.txt'])),
        ('Neo diagnostic omens', (words.genre == 'diagnostic omens') & (words.era == 'Neo Period'),
         (signs.genre == 'diagnostic omens') & (signs.era == 'Neo Period')),
    ]:
        g = C.graded_ldi(signs[mask_s])
        n = g['words']
        print(f'  {name:<28}bin {C.ldi(words[mask_w])[0]:.3f} | macro {g["macro"]:.3f} | '
              f'pure {100*g["pure"]/n:.1f} % | mixed {100*g["mixed"]/n:.1f} % | '
              f'syllabic {100*g["syllabic"]/n:.1f} %')

    # ---- Table: the periods under each convention -------------------------
    head('The periods recomputed under each convention', 'tbl:periods-by-convention')
    print(f'{"convention":<28}' + ''.join(f'{p.split()[0]:>9}' for p in ORDER) + f'{"Old->Mid":>10}')

    def by_era(fn):
        return [fn(era) for era in ORDER]

    variants = [
        ('bin (baseline)',            lambda e: C.ldi(words[words.era == e])[0]),
        ('per-omen mean',             lambda e: marked[marked.era == e].groupby(['filename', 'omen_id']).is_log.mean().mean()),
        ('per-text mean',             lambda e: marked[marked.era == e].groupby('filename').is_log.mean().mean()),
        ('ina/ana logographic',       lambda e: C.ldi(words[words.era == e], monogram_as_log=True)[0]),
        ('opening particles excluded', lambda e: C.ldi(words[words.era == e], exclude_particles=True)[0]),
        ('restorations dropped',      lambda e: C.ldi(preserved[preserved.era == e])[0]),
        ('macro',                     lambda e: graded(signs, e)['macro']),
        ('micro',                     lambda e: graded(signs, e)['micro']),
        ('micro, particles excluded', lambda e: graded(signs, e, exclude_particles=True)['micro']),
    ]
    for name, fn in variants:
        vals = by_era(fn)
        print(f'{name:<28}' + ''.join(f'{v:>9.3f}' for v in vals) + f'{vals[1]-vals[0]:>+10.3f}')

    # ---- The Old-to-Middle rise within disciplines and across texts -------
    head('The Old-to-Middle rise within disciplines and across texts', 'sec. Results')
    old, middle = ORDER[:2]

    def rise(frame):
        return C.ldi(frame[frame.era == middle])[0] - C.ldi(frame[frame.era == old])[0]

    for genre in DISCIPLINE:
        print(f'  {genre.split()[0]:<14}{rise(words[words.genre == genre]):>+8.3f}')
    text_bin = marked.groupby(['era', 'filename']).is_log.mean()
    for era in ORDER:
        print(f'    per text, {era:<16}{text_bin[era].min():.2f} to {text_bin[era].max():.2f}')
    base = rise(words)
    moved = [abs(rise(words[~((words.era == old) & (words.filename == name))]) - base)
             for name in words[words.era == old].filename.unique()]
    print(f'  excluding one Old text moves the rise by at most {max(moved):.3f}')

    # ---- Table: the index by the period labels of the editions ------------
    head('The index by the period labels of the editions', 'tbl:labels')
    labels = edition_labels()
    labelled = words.assign(label=words.filename.map(labels))
    counted = every_omen.assign(label=every_omen.filename.map(labels))   # omens of all languages
    for era in ORDER:
        e = labelled[labelled.era == era]
        for label, sub in e.groupby('label'):
            all_of = counted[(counted.era == era) & (counted.label == label)]
            print(f'  {era.split()[0]:<8}{label:<34}{C.ldi(sub)[0]:>7.3f}'
                  f'{all_of.groupby(["filename", "omen_id"]).ngroups:>7} ({all_of.filename.nunique()})')
    mid = words[words.era == middle]
    periphery = mid.provenance.str.contains('Emar|Susa|Ḫattuša|Hattusa', na=False)
    print(f'  Middle: periphery {C.ldi(mid[periphery])[0]:.3f} ({mid[periphery].filename.nunique()} texts) | '
          f'the others {C.ldi(mid[~periphery])[0]:.3f} ({mid[~periphery].filename.nunique()} texts)')
    neo_astro = words[(words.era == ORDER[2]) & (words.genre == 'astrological omens')]
    other_topic = neo_astro.filename.isin(['EAE55.eBL.txt', 'EAE57.eBL.txt'])
    print(f'  Neo astrological: EAE 55 and 57 {C.ldi(neo_astro[other_topic])[0]:.3f} | '
          f'the lunar omens {C.ldi(neo_astro[~other_topic])[0]:.3f}')

    # ---- Table: restoration ----------------------------------------------
    head('Scored with and without the editor\'s restorations', 'tbl:restoration')
    both = by_text.to_frame('with_restoration')
    both['preserved_only'] = scored(preserved).groupby('filename').is_log.mean()
    both['delta'] = both.preserved_only - both.with_restoration
    print(f'  whole corpus: with restoration {C.ldi(words)[0]:.3f} | '
          f'preserved only {C.ldi(preserved)[0]:.3f}')
    print(f'  falls in {(both.delta < 0).sum()} of {len(both)} texts, rises in {(both.delta > 0).sum()}')
    print(f'  shifts by more than 0.05 in {100*(both.delta.abs() > 0.05).mean():.0f} % of texts; '
          f'largest {both.delta.abs().max():.2f}')
    for era in ORDER:                     # pooled, as reported in the paper
        correction = C.ldi(preserved[preserved.era == era])[0] - C.ldi(words[words.era == era])[0]
        print(f'    {era:<16}{correction:>+8.3f}')
    print(f'  {"witness":<18}{"with":>7}{"preserved":>11}{"restored share":>16}')
    for name, filename in [('VAT 13804', 'VAT.13804.txt'), ('BM 121034', 'BM.121034.txt'),
                           ('VAT 10536', 'VAT.10536.txt'), ('VAT 11001', 'VAT.11001.txt'),
                           ('W 23271', 'W.23271.txt')]:
        full, _, n_full = C.ldi(words[words.filename == filename])
        kept, _, n_kept = C.ldi(preserved[preserved.filename == filename])
        print(f'  {name:<18}{full:>7.2f}{kept:>11.2f}{100*(1 - n_kept/n_full):>15.0f}%')

    # ---- Table: a specimen report ----------------------------------------
    head('A specimen report, one witness beside the corpus', 'tbl:reporting')
    for name, wsub, ssub, psub in [
        ('BM 121034', words[words.filename == 'BM.121034.txt'],
         signs[signs.filename == 'BM.121034.txt'],
         preserved[preserved.filename == 'BM.121034.txt']),
        ('whole corpus', words, signs, preserved),
    ]:
        g = graded(signs) if name == 'whole corpus' else C.graded_ldi(ssub)
        n = g['words']
        omens = n_omens if name == 'whole corpus' else wsub.groupby(['filename', 'omen_id']).ngroups
        print(f'  {name}: {omens} omens, {n} scored words')
        print(f'    bin {C.ldi(wsub)[0]:.3f} | macro {g["macro"]:.3f} | micro {g["micro"]:.3f}')
        print(f'    pure {100*g["pure"]/n:.1f} % | mixed {100*g["mixed"]/n:.1f} % | '
              f'syllabic {100*g["syllabic"]/n:.1f} %')
        print(f'    restorations dropped {C.ldi(psub)[0]:.3f} | '
              f'ina/ana logographic {C.ldi(wsub, monogram_as_log=True)[0]:.3f} | '
              f'particle excluded {C.ldi(wsub, exclude_particles=True)[0]:.3f}')

    # ---- ina / ana in the Neo terrestrial material ------------------------
    head('ina and ana in the Neo terrestrial material', 'sec. ina and ana')
    sub = words[(words.genre == 'terrestrial omens') & (words.era == 'Neo Period')]
    sig = signs[(signs.genre == 'terrestrial omens') & (signs.era == 'Neo Period')]
    print(f'  {sub.groupby(["filename", "omen_id"]).ngroups} omens')
    print(f'    bin   {C.ldi(sub)[0]:.3f} -> {C.ldi(sub, monogram_as_log=True)[0]:.3f}')
    for key in ('macro', 'micro'):
        print(f'    {key} {C.graded_ldi(sig)[key]:.3f} -> '
              f'{C.graded_ldi(sig, monogram_as_log=True)[key]:.3f}')
    for flag in (False, True):
        per_omen = scored(sub, monogram_as_log=flag).groupby(['filename', 'omen_id']).is_log.mean()
        full = int((per_omen == 1.0).sum())
        print(f'    fully logographic omens, ina/ana as {"logographic" if flag else "syllabic"}: {full}')


if __name__ == '__main__':
    main()
