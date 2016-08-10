#!/usr/bin/python

from os.path import join as opj
import numpy as np
from glob import glob
from scipy.ndimage import label
from statsmodels.stats.inter_rater import fleiss_kappa


# until the first black frame
maxmovietime = 7082.2


def load_annotations():
    return [np.recfromcsv(obs) for obs in sorted(glob(opj('data', '*')))]


def get_timecode_stats(data, fx):
    return fx(np.array([(fx(d['start']), fx(d['end'])) for d in data]))


def get_value_union(data, var):
    vals = []
    for d in data:
        vals.extend(list(d[var]))
    return set(vals)


def get_ioa_ts(data, props, match='contains'):
    # for every second
    ioa = np.zeros(int(maxmovietime), dtype=int)
    for obs in data:
        for ev in obs:
            hit = True
            for k, v in props.items():
                if match == 'contains':
                    if not v in ev[k]:
                        hit = False
                        break
                else:
                    if not v == ev[k]:
                        hit = False
                        break
            if hit:
                ioa[ev['start']:ev['end'] + 1] += 1
    return ioa


def get_events(data, aggreement=.5):
    # events are defined as consecutive timepoints where a specified fraction
    # of observer agree on the presence of an event
    # we decide to only consider annotations that overlap in time and share the
    # same 'sender'
    senders = get_value_union(data, 'sender')
    global_ioa = None
    for sender in senders:
        ioa = get_ioa_ts(data, dict(sender=sender), match='exact')
        rel_ioa = ioa.astype(float) / len(data)
        if rel_ioa.max() > 1.0:
            print("% WARNING: broken timestamps!!!")
        ioa = rel_ioa >= aggreement
        if global_ioa is None:
            global_ioa = ioa.astype(int)
        else:
            global_ioa += ioa

    events = []
    segments, nsegments = label(global_ioa)
    for i in range(nsegments):
        segment = segments == i + 1
        events.append((segment.argmax(),
                       len(segment) - segment[::-1].argmax()))
    return global_ioa, events


def get_rater_counts(data, events, prop, categories):
    counts = []
    for ev_start, ev_end in events:
        # add one column for "found nothing"
        ev_count = [0] * (len(categories) + 1)
        # for all observers
        for i, d in enumerate(data):
            # avoid double-counting
            counted = False
            # find the right annotation
            for annot in d:
                # start before event's end and end after event start
                if annot['start'] <= ev_end and annot['end'] >= ev_start:
                    for c, cat in enumerate(categories):
                        if isinstance(cat, int):
                            if cat == annot[prop]:
                                ev_count[c] += 1
                                counted = True
                        elif cat in annot[prop]:
                            ev_count[c] += 1
                            counted = True
                if counted:
                    break
        if sum(ev_count) < len(data):
            # fill the "who said nothing category
            ev_count[-1] = len(data) - sum(ev_count)
        #print(ev_count)
        assert len(data) == sum(ev_count)
        counts.append(ev_count)
    return counts


# tex format help
def _ft(key, value, fmt='s'):
    val_tmpl = '{{{{value:{}}}}}'.format(fmt)
    tex = '\\newcommand{{{{\\{{key}}}}}}{{{val_tmpl}}}'.format(val_tmpl=val_tmpl)
    return tex.format(key=key, value=value)


def _stats_helper(data, events, prop, categories, label, aggstr):
    counts = get_rater_counts(data, events, prop, categories)
    # events where more observers are in favor of a property than not
    print(_ft('Agg{}N{}'.format(aggstr, label),
              sum([e[0] > e[1] for e in counts]),
              '.0f'))
    print(_ft('Agg{}FK{}'.format(aggstr, label),
              fleiss_kappa(counts),
              '.2f'))

def print_descriptive_stats_as_tex(data):
    nevents = [len(o) for o in data]
    print(_ft('NEventsMin', min(nevents), 'd'))
    print(_ft('NEventsMax', max(nevents), 'd'))
    print(_ft('NEventsMedian', int(np.median(nevents)), 'd'))
    # do per aggreement-level
    print(_ft('UniqueSenders', ', '.join([c.decode('utf-8') for c in sorted(get_value_union(data, 'sender'))])))
    print(_ft('UniqueReceivers', ', '.join([c.decode('utf-8') for c in sorted(get_value_union(data, 'receiver')) if c])))
    for aggreement, aggstr in ((.2, 'Twenty'), (.6, "Sixty"), (1.0, 'Hundred')):
        ts, events = get_events(data, aggreement=aggreement)
        print(_ft('Agg{}NEvents'.format(aggstr), len(events), 'd'))
        event_durations = [ev[1] - ev[0] for ev in events]
        event_distances = [ev[0] - events[i - 1][1] for i, ev in enumerate(events) if i]
        print(_ft('Agg{}MeanEventDuration'.format(aggstr), np.mean(event_durations), '.1f'))
        print(_ft('Agg{}MeanEventDistance'.format(aggstr), np.mean(event_distances), '.1f'))
        _stats_helper(data, events, 'proof', [b'IMMEDIATE'], 'ProofImmediate', aggstr)
        _stats_helper(data, events, 'proof', [b'PAST'], 'ProofPast', aggstr)
        _stats_helper(data, events, 'contradiction', [1], 'Contradiction', aggstr)
        _stats_helper(data, events, 'category', ['LIE'], 'Lie', aggstr)
        _stats_helper(data, events, 'category', ['IRONY'], 'Irony', aggstr)
        for cue in ('RHETORICALQ', 'UNDERSTATEMENT', 'EXAGGERATION', 'LAUGH',
                    'RAISEDEYEBROW', 'EYEROLLING', 'TEMPOCHANGE', 'TONECHANGE',
                    'PAUSE'):
            _stats_helper(data, events, 'cues', [cue], 'Cue' + cue, aggstr)


if __name__ == '__main__':
    data = load_annotations()
    print_descriptive_stats_as_tex(data)
