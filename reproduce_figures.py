# -*- coding: utf-8 -*-
"""Redraw the summary figures from the corpus.

    python reproduce_figures.py [OUTDIR]

Writes periods.pdf/.png (the index by period), spread.pdf/.png (the nine
conventions of tbl:spread against the period means) and disciplines.pdf/.png
(the index by period within each discipline) to OUTDIR, by default docs/_build/figures.
The numbers are computed exactly as reproduce_tables.py prints them, so the figures
cannot drift from the tables. Needs matplotlib, which the app itself does not.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import compute_ratios as C
import reproduce_tables as T

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join('docs', '_build', 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}.pdf and {name}.png', flush=True)


def periods_figure(values):
    """values: {measure: [Old, Middle, Neo]}"""
    style = {'bin': ('#000000', '-', 2.2), 'macro': ('#666666', '--', 1.8), 'micro': ('#9a9a9a', ':', 1.8)}
    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    x = range(3)
    for measure, ys in values.items():
        color, dash, width = style[measure]
        ax.plot(x, ys, dash, color=color, lw=width, marker='o', ms=6, zorder=3)
        ax.text(2.08, ys[2], measure, color=color, va='center', fontsize=10)
        for xi, y in zip(x, ys):
            above = measure == 'bin'
            ax.text(xi, y + (0.035 if above else -0.04), f'{y:.2f}', color=color, fontsize=8,
                    ha='center', va='bottom' if above else 'top')
    ax.set_xticks(list(x), ['Old', 'Middle', 'Neo'], fontsize=11)
    ax.set_xlim(-0.2, 2.4)
    ax.set_ylim(0.09, 0.85)
    ax.set_ylabel('logogram density index')
    ax.grid(axis='x', color='#e6e6e6')
    ax.tick_params(axis='x', length=0)
    save(fig, 'periods')


def spread_figure(conventions, means):
    lo, hi = min(conventions), max(conventions)
    fig, ax = plt.subplots(figsize=(6.1, 2.5))
    ax.axvspan(lo, hi, ymin=0.55, ymax=0.85, color='#dadada', lw=0)
    for v in conventions:
        ax.plot([v, v], [0.62, 0.78], color='#333333', lw=1.8)
    ax.text((lo + hi) / 2, 0.9, f'nine defensible conventions:  {lo:.3f}–{hi:.3f}  (spread {hi - lo:.2f})',
            ha='center', va='bottom', fontsize=9)
    old, mid, neo = means
    ax.plot(means, [0.0] * 3, 'o', color='black', ms=7, zorder=3)
    ax.annotate('', xy=(mid - 0.004, 0.27), xytext=(old + 0.004, 0.27),
                arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.2))
    ax.text((old + mid) / 2, 0.22, f'Old→Middle rise {mid - old:.2f}', ha='center', va='top',
            fontsize=9, color='#333333')
    ax.plot([mid, neo], [0.1, 0.1], color='#555555', lw=1.2)
    ax.text((mid + neo) / 2, 0.14, f'{neo - mid:.2f}', ha='center', va='bottom', fontsize=8, color='#555555')
    ax.text(old, -0.12, f'Old {old:.2f}', ha='center', va='top', fontsize=9)
    ax.text(mid, -0.12, f'Middle {mid:.2f}', ha='right', va='top', fontsize=9)
    ax.text(neo, -0.36, f'Neo {neo:.2f}', ha='left', va='top', fontsize=9)
    ax.set_yticks([0.0, 0.7], ['period means', 'conventions'])
    ax.set_ylim(-0.55, 1.1)
    ax.set_xlim(0.25, 0.85)
    ax.set_xlabel('logogram density index')
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color='#e6e6e6')
    save(fig, 'spread')


DISCIPLINES = [('astrological omens', 'astrological', '#2a78d6', 'o'),
               ('diagnostic omens', 'diagnostic', '#eb6834', 's'),
               ('extispicy omens', 'extispicy', '#1baf7a', '^'),
               ('izbu omens', 'teratological', '#eda100', 'D'),
               ('terrestrial omens', 'terrestrial', '#e87ba4', 'v')]


def disciplines_figure(values, overall):
    """values: {genre: [Old, Middle, Neo]}; overall: the corpus line of Table 4"""
    fig, ax = plt.subplots(figsize=(5.3, 3.3))
    x = range(3)
    for genre, label, color, marker in DISCIPLINES:
        ax.plot(x, values[genre], '-', color=color, lw=1.8, marker=marker, ms=6.5, mec='white', mew=0.8,
                zorder=3, label=label)
    ax.plot(x, overall, '--', color='#000000', lw=2.4, zorder=4, label='whole corpus')
    ax.legend(loc='lower right', frameon=False, fontsize=8.5, handlelength=2.6)
    ax.set_xticks(list(x), ['Old', 'Middle', 'Neo'], fontsize=11)
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(0.05, 0.9)
    ax.set_ylabel('logogram density index (bin)')
    ax.grid(axis='x', color='#e6e6e6')
    ax.tick_params(axis='x', length=0)
    save(fig, 'disciplines')


def main():
    words, signs, preserved = T.load()
    values = {'bin': [], 'macro': [], 'micro': []}
    for era in T.ORDER:
        g = T.graded(signs, era)
        values['bin'].append(C.ldi(words[words.era == era])[0])
        values['macro'].append(g['macro'])
        values['micro'].append(g['micro'])
    periods_figure(values)

    g_all = T.graded(signs)
    marked = T.scored(words)
    conventions = [
        C.ldi(words, monogram_as_log=True)[0],
        marked.groupby(['filename', 'omen_id']).is_log.mean().mean(),   # per-omen mean
        C.ldi(words)[0],
        C.ldi(preserved)[0],
        C.ldi(words, exclude_particles=True)[0],
        marked.groupby('filename').is_log.mean().mean(),                # per-text mean
        g_all['macro'],
        g_all['micro'],
        T.graded(signs, exclude_particles=True)['micro'],
    ]
    spread_figure(conventions, values['bin'])

    per_genre = {genre: [C.ldi(words[(words.genre == genre) & (words.era == era)])[0] for era in T.ORDER]
                 for genre, *_ in DISCIPLINES}
    disciplines_figure(per_genre, values['bin'])


if __name__ == '__main__':
    main()
