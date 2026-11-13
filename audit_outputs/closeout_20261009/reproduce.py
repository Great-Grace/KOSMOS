"""Independent cached-data closeout; deliberately does not import the audited metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


def dist(value):
    a = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Invalid probability vector")
    if a.sum() <= 0 or abs(a.sum() - 1) > .001:
        raise ValueError("Probability mass outside rounding tolerance")
    return a / a.sum()


def entropy(p):
    p = np.asarray(p)
    pos = p[p > 0]
    return float(-np.sum(pos * np.log2(pos)))


def w1(p, q):
    return float(np.abs(np.cumsum(p - q)[:-1]).sum())


def tvd(p, q):
    return float(np.abs(p - q).sum() / 2)


def correlations(r, m):
    r = r - np.nanmean(r, axis=0)
    m = m - np.nanmean(m, axis=0)
    denominator = np.sqrt(np.nansum(r*r, axis=0) * np.nansum(m*m, axis=0))
    valid = denominator > 1e-12
    return np.nansum(r*m, axis=0)[valid] / denominator[valid]


def subgroup(df, taxonomy, variables, n_boot):
    f = df[df['var'].isin(variables)].copy()
    for col, dev in [('real_mean', 'rd'), ('model_mean', 'md')]:
        gm = f.groupby('var').apply(
            lambda g: np.dot(g[col], g.cell_weight_item_norm / g.cell_weight_item_norm.sum()),
            include_groups=False)
        f[dev] = f[col] - f['var'].map(gm)
    f['cell'] = f['sex'].astype(str) + '_' + f['age_grp']
    R = f.pivot(index='var', columns='cell', values='rd').sort_index()
    M = f.pivot(index='var', columns='cell', values='md').reindex_like(R)
    if R.isna().any().any() or M.isna().any().any():
        raise ValueError('Incomplete cells in published subgroup result')
    spans = taxonomy.loc[R.index, 'span'].to_numpy()[:, None]
    r, m = R.to_numpy(), M.to_numpy()
    within = [float(correlations(g.real_mean.to_numpy()[:, None],
                                g.model_mean.to_numpy()[:, None])[0])
              for _, g in f.groupby('var')
              if g.real_mean.std() > 1e-12 and g.model_mean.std() > 1e-12]
    clusters = {}
    for i, v in enumerate(R.index):
        battery = taxonomy.loc[v, 'battery']
        key = str(battery) if pd.notna(battery) and str(battery).strip() else '__single__' + v
        clusters.setdefault(key, []).append(i)
    groups = list(clusters.values())
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
        rs = correlations(r[ix], m[ix])
        if len(rs):
            samples.append(float(rs.mean()))
    return {'n_items': len(R), 'n_batteries': len(groups),
            'centered_r': float(correlations(r, m).mean()),
            'range_normalized_r': float(correlations(r/spans, m/spans).mean()),
            'within_item_r': float(np.mean(within)),
            'battery_ci': np.percentile(samples, [2.5, 97.5]).tolist(),
            'n_boot_valid': len(samples),
            'ci_scope': 'Conditional on observed human/persona cell estimates; exploratory battery resampling'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    b = args.source
    input_paths = [b/'audit_outputs/item_classification_255.csv']
    taxonomy = pd.read_csv(input_paths[0]).set_index('var')
    references = pd.read_csv(b/'c0_check/probes.csv').set_index('var')
    if not taxonomy.index.is_unique or not references.index.is_unique:
        raise ValueError('Duplicate taxonomy/reference item')
    input_paths.append(b/'c0_check/probes.csv')
    taxonomy['span'] = references.codes.map(lambda x: max(json.loads(x))-min(json.loads(x)))
    for field in ['include_in_clean_eval', 'w1_valid_in_clean_eval']:
        if taxonomy[field].dtype != bool:
            raise ValueError('Unexpected boolean encoding')
    all_rows, matrices, metadata, probe_sets = [], {}, {}, {}
    mappings = {'c0': 'c0_cached', 'adult': 'adult_cached',
                'paired_demo': 'paired_demo_125items', 'full': 'full_6items'}
    for condition, directory in mappings.items():
        folder = b/'audit_outputs'/directory
        paths = [folder/name for name in ['probes.csv', 'logit_responses.csv', 'real_cells.csv']]
        input_paths.extend(paths)
        probes, logits, cells = [pd.read_csv(p) for p in paths]
        if probes.probe_id.duplicated().any() or logits.duplicated(['probe_id', 'model']).any():
            raise ValueError('Duplicate probe or response identity')
        if set(probes.probe_id) != set(logits.probe_id) or set(logits.model) != {'exaone'}:
            raise ValueError('Missing/extra responses or unexpected model')
        d = probes.merge(logits, on='probe_id', validate='one_to_one')
        probe_sets[condition] = probes
        metadata[condition] = {'n_items': int(d['var'].nunique()), 'n_probes': len(d)}
        if 'persona_id' in d:
            people = d[['persona_id', 'respid', 'sex', 'age_grp']].drop_duplicates()
            if people.persona_id.duplicated().any():
                raise ValueError('Persona metadata changes across items')
            counts = people.groupby(['sex', 'age_grp']).size()
            metadata[condition].update(n_personas=len(people), n_unique_respondents=int(people.respid.nunique()),
                                      cell_counts={str(k): int(v) for k,v in counts.items()})
        cell_rows = []
        for variable, g in d.groupby('var', sort=True):
            if variable not in taxonomy.index:
                raise ValueError('Unclassified item ' + variable)
            if g.codes.nunique() != 1 or g.gold.nunique() != 1:
                raise ValueError('Codes/gold differ within item')
            codes = np.asarray(json.loads(g.codes.iloc[0]), dtype=float)
            q = dist(g.gold.iloc[0])
            mat = np.stack(g.model_dist.map(dist))
            if len(codes) != mat.shape[1] or len(q) != len(codes):
                raise ValueError('Response dimension mismatch')
            if not np.allclose(q, dist(references.loc[variable, 'gold']), atol=1e-10):
                raise ValueError('Reference differs by condition')
            cell_meta = cells[cells['var'] == variable]
            equal = np.ones(len(g)) / len(g)
            weights = equal.copy()
            if 'sex' in g:
                weights = np.zeros(len(g))
                cell_means, cell_ws = [], []
                grouped = g.reset_index(drop=True).groupby(['sex', 'age_grp'])
                for key, group in grouped:
                    h = cell_meta[(cell_meta.sex == key[0]) & (cell_meta.age_grp == key[1])]
                    if len(h) != 1:
                        raise ValueError('Missing or duplicate human cell')
                    w = float(h.cell_weight_item_norm.iloc[0])
                    weights[group.index] = w / len(group)
                    cm = mat[group.index].mean(axis=0)
                    cell_means.append(cm); cell_ws.append(w)
                    cell_rows.append({'var': variable, 'sex': key[0], 'age_grp': key[1],
                                      'model_mean': float(cm @ codes),
                                      'real_mean': float(h.real_mean.iloc[0]),
                                      'cell_weight_item_norm': w})
                weights /= weights.sum()
                separate = np.average(cell_means, axis=0, weights=cell_ws)
                if not np.allclose(weights @ mat, separate, atol=1e-12):
                    raise ValueError('Row-to-cell aggregation mismatch')
            valid_w1 = bool(taxonomy.loc[variable, 'w1_valid_in_clean_eval'])
            uniform = np.ones(len(q)) / len(q)
            mode = np.eye(len(q))[int(q.argmax())]
            for weighting, w in [('equal_persona', equal), ('cell_poststratified', weights)]:
                p = w @ mat
                row = {'condition': condition, 'var': variable, 'weighting': weighting,
                       'scale_type': taxonomy.loc[variable, 'scale_type'],
                       'domain': taxonomy.loc[variable, 'content_domain'],
                       'w1_valid': valid_w1, 'tvd': tvd(p,q),
                       'uniform_tvd': tvd(uniform,q), 'modal_tvd': tvd(mode,q),
                       'w1': w1(p,q) if valid_w1 else np.nan,
                       'uniform_w1': w1(uniform,q) if valid_w1 else np.nan,
                       'conditional_entropy': float(w @ np.array([entropy(x) for x in mat])),
                       'population_entropy': entropy(p), 'human_entropy': entropy(q)}
                row['mutual_information'] = row['population_entropy']-row['conditional_entropy']
                if row['mutual_information'] < -1e-12:
                    raise ValueError('Entropy decomposition violation')
                all_rows.append(row)
        matrices[condition] = pd.DataFrame(cell_rows)
    demo_people = probe_sets['paired_demo'][['persona_id','respid','sex','age_grp']].drop_duplicates().sort_values('persona_id').reset_index(drop=True)
    full_people = probe_sets['full'][['persona_id','respid','sex','age_grp']].drop_duplicates().sort_values('persona_id').reset_index(drop=True)
    if not demo_people.equals(full_people):
        raise ValueError('Demo and Full personas are not matched')
    result = pd.DataFrame(all_rows)
    result.to_csv(args.output/'item_metrics.csv', index=False)
    common125 = set(result[result.condition == 'paired_demo']['var'])
    common6 = set(result[result.condition == 'full']['var'])
    valid = set(taxonomy[taxonomy.w1_valid_in_clean_eval].index)
    ordinal = set(taxonomy[taxonomy.scale_type == 'ordinal_monotone'].index)
    binary = set(taxonomy[taxonomy.scale_type == 'binary'].index)
    attitude = set(taxonomy[taxonomy.content_domain == 'attitude_value_perception'].index)
    factual = set(taxonomy[taxonomy.content_domain == 'factual_status_behavior'].index)
    sets = {'common125_all': common125, 'common125_valid118': common125&valid,
            'ordinal96': common125&ordinal, 'attitude101_valid': common125&attitude&valid,
            'factual17_valid': common125&factual&valid, 'binary22': common125&binary,
            'binary_attitude6': common125&binary&attitude,
            'common6_all': common6, 'common5_no_graduate': common6-{'GRADUATE'},
            'graduate_only': {'GRADUATE'}}
    summaries = []
    for name, items in sets.items():
        for (condition, weighting), g in result[result['var'].isin(items)].groupby(['condition','weighting']):
            if set(g['var']) != items:
                continue
            summaries.append({
                'subset': name,
                'n_items': len(items),
                'n_w1_valid': int(g['w1_valid'].sum()),
                'condition': condition,
                'weighting': weighting,
                **{c: float(g[c].mean()) for c in ['tvd', 'uniform_tvd', 'modal_tvd']},
                **{c: float(g[c].mean(skipna=False)) for c in ['w1', 'uniform_w1']},
            })
    pd.DataFrame(summaries).to_csv(args.output/'summary.csv', index=False)
    r_results = {name: subgroup(matrices['paired_demo'], taxonomy, items, args.n_boot)
                 for name,items in sets.items() if name in ['common125_valid118','ordinal96','attitude101_valid','factual17_valid']}
    (args.output/'subgroup.json').write_text(json.dumps(r_results, ensure_ascii=False, indent=2)+'\n')
    input_paths.extend(b/p for p in ['kgss_c0_validity.py','kgss_metrics.py',
                                   'kgss_persona_bootstrap.py','tests/test_kosmos_audit.py'])
    original_hash_prefixes = {'c0_check':'77b47387ccfad681', 'c0_check_adult':'ed929349d134b730',
                             'c0_check_demo':'64f81ed43e3edcda','c0_check_full':'54c426e67f73e03a'}
    for directory, expected in original_hash_prefixes.items():
        p = b/directory/'c0_validity_result.md'
        if not hashlib.sha256(p.read_bytes()).hexdigest().startswith(expected):
            raise ValueError('Historical source report changed')
        input_paths.append(p)
    hashes = {str(p.relative_to(b)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    graduation_probes = probe_sets['full'].query("var == 'GRADUATE'")
    manifest = {'date_kst':'2026-10-09', 'source':str(b.resolve()), 'source_commit':'da22600f18f33cb085744c8c972cb5a1af6c6795',
                'method':'Independent CDF W1 / L1 TVD / normalized probabilities; no audited helper imports',
                'metadata':metadata, 'input_sha256':hashes, 'python':platform.python_version(),
                'numpy':np.__version__,'pandas':pd.__version__,'seed':42,'n_boot':args.n_boot,
                'full_graduate_prompts_assert_graduation':int(graduation_probes.prompt.str.contains('졸업했고').sum()),
                'publication_scope':'EXAONE cached-data audit, not complete 246-item model experiment',
                'limitations':['Manual ordinal taxonomy assumptions remain; TVD is primary across categories',
                               'Cell/battery confidence intervals do not constitute survey-design inference',
                               'Full original caches and Kanana are unavailable; no new inference performed',
                               'GRADUATE outcome assertion in old Full prompts; exclude from added-information claims']}
    (args.output/'evidence.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'conditions':metadata,'subgroup':r_results},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
