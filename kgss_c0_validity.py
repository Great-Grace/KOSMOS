r"""
C0(페르소나 없음)/adult(한국 성인)/demo(성별+연령)/full(6변수) 조건 분포 재현 타당성 검증

서울 논문(Kim, Park & Suh 2026)이 EXAONE 3.5 7.8B·Qwen3-30B 같은 오픈웨이트에서
subgroup r이 0.05~0.12로 붕괴했다고 보고함 — 우리가 쓰는 EXAONE 2.4B/Kanana 2.1B는
그보다도 작아서, 인칭×화계 헤드라인 분석 전에 "이 모델이 population 수준에서 뭐라도
재현하는지"부터 확인해야 한다.

측정 두 가지:
    population TVD/Wasserstein  모델 예측분포 vs 실제 KGSS 가중분포 (문항별, 전체 평균)
    subgroup r                  성별×연령 10셀의 "실제 셀평균 - 문항 전체평균"(편차) vs
                                 모델의 예측 기댓값. 문항 전체평균을 빼는 이유: C0는 모델
                                 예측이 셀마다 안 달라지므로, 셀 원값을 그대로 상관시키면
                                 문항 자체의 전반적 지지율만 재는 꼴이 된다(인구통계 하위
                                 집단 평균은 보통 문항 전체평균 근처에 몰려 있어서, 모델이
                                 전체평균만 잘 맞혀도 r이 높게 나옴). 편차로 그 효과를
                                 제거해야 "모델이 셀 간 차이를 우연히라도 맞히는지"를 재는
                                 지표가 되고, 페르소나가 없는 C0에서는 이론상 0 근처가
                                 나와야 정상이다(서울 논문 C0 실측 0.03).

네 조건 (CLAUDE.md 3/7절 "대조군 4수준"):
    c0    페르소나 없음. 문항당 프롬프트 1개, 모든 셀에 같은 예측값이 복사됨.
    adult "당신은 한국 성인입니다" 머리말만 추가. 구체적 인구통계는 없음 — c0와 똑같이
          문항당 프롬프트 1개, 모든 셀에 같은 예측값이 복사됨. c0와 비교하면 "막연한
          정체성 지칭"만으로 뭔가 달라지는지를 분리해서 본다.
    demo  "당신은 {연령대 대표나이}세 {성별}입니다" 머리말 추가. 문항×10셀 = 프롬프트
          10개, 셀마다 실제로 다른 예측값이 나옴.
    full  persona_sample.csv(FINALWT 가중 복원추출 응답자)에서 n명을 뽑아 6변수 서사형
          페르소나 문장("당신은 ~" 하십시오체)을 머리말로 붙임. 문항×n명 프롬프트.
          personas.csv의 실제 AGE를 성별×연령 10셀로 다시 묶어 demo/C0와 같은 방식으로
          채점 — 한 셀에 여러 명이 걸리면 그 셀의 모델 예측을 평균해서 문항당 값 하나로
          만든 뒤 상관시킨다.

    네 조건 모두 subgroup r을 같은 방식(편차 기준)으로 계산하므로 직접 비교 가능하다.
    c0에서 벗어나 adult·demo·full 순으로 r이 올라가면 페르소나 정보량이 늘수록 실제로
    셀 구분 능력이 생긴다는 증거고, 안 올라가면 조건화 자체가 이 모델 규모에서는
    무의미하다는 증거(Q1/Q2 대응).

사용법:
    uv run kgss_c0_validity.py build --level c0 --inv ./inventory --out ./c0_check
    uv run kgss_c0_validity.py build --level adult --inv ./inventory --out ./c0_check_adult
    uv run kgss_c0_validity.py build --level demo --inv ./inventory --out ./c0_check_demo
    uv run kgss_c0_validity.py build --level full --inv ./inventory --out ./c0_check_full --n-personas 40
    # Colab: uv run kgss_token_check.py --contam-probes probes.csv --out <out> --no-contam-generate
    uv run kgss_c0_validity.py score --out ./c0_check
    uv run kgss_c0_validity.py score --out ./c0_check_adult
    uv run kgss_c0_validity.py score --out ./c0_check_demo
    uv run kgss_c0_validity.py score --out ./c0_check_full
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

STRUCTURAL = {-8.0, -1.0}
DK_COLS = ["legacy_dk", "dk"]
INAP_COLS = ["legacy_inap", "inap"]
AGE_BINS = [18, 30, 40, 50, 60, 200]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]
AGE_REP = {"18-29": 24, "30-39": 35, "40-49": 45, "50-59": 55, "60+": 70}


def pick_col(dfr, names):
    for n in names:
        if n in dfr.columns:
            return n
    return None


LEAKED_DK_INAP_PATTERNS = (
    "모름", "무응답", "비해당", "응답거절",
    "don't know", "dont know", "no answer", "refusal", "not applicable", "inapplicable", "iap",
)
ADMIN_EXCLUDE_VARS = {"SAMPLEAB"}


def is_leaked_missing_label(label: str) -> bool:
    s = str(label).strip().lower()
    return any(pat in s for pat in LEAKED_DK_INAP_PATTERNS)


def real_options(var, vt, filter_leaked_dk_inap: bool = False):
    if var not in vt.index:
        return []
    row = vt.loc[var]
    try:
        vl = json.loads(row["value_labels"])
    except Exception:
        return []
    dk_col, inap_col = pick_col(vt, DK_COLS), pick_col(vt, INAP_COLS)
    dk = set(json.loads(row[dk_col])) if dk_col and pd.notna(row[dk_col]) else set()
    inap = set(json.loads(row[inap_col])) if inap_col and pd.notna(row[inap_col]) else set()
    out = []
    for k, lab in sorted(vl.items(), key=lambda kv: float(kv[0])):
        code = float(k)
        if code in STRUCTURAL or code in dk or code in inap:
            continue
        if filter_leaked_dk_inap and is_leaked_missing_label(lab):
            continue
        out.append((code, lab))
    return out


def sex_labels(vt):
    vl = json.loads(vt.loc["SEX", "value_labels"])
    return {float(k): v.strip() for k, v in vl.items() if float(k) in (1.0, 2.0)}


def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def weighted_dist(df, var, codes, round_digits=4):
    from kgss_metrics import human_weighted_distribution
    return human_weighted_distribution(df, var, codes, weight_col="FINALWT", min_n=100, round_digits=round_digits)


def build(args):
    from kgss_metrics import human_cell_distributions

    inv, outdir = Path(args.inv), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(inv / "kgss_clean.parquet")
    sub = df[df["YEAR"] == args.year].copy()
    sub["AGE_GRP"] = pd.cut(sub["AGE"], AGE_BINS, labels=AGE_LABELS, right=False)

    wave = pd.read_csv(inv / "wave_items.csv")
    vt = pd.read_csv(inv / "variables.csv").set_index("var")
    worded_df = pd.read_csv(args.worded).set_index("var")
    worded = worded_df["설문원문"]
    sexlab = sex_labels(vt)
    clean_prompts = getattr(args, "clean_prompts", False)

    personas = None
    if args.level == "full":
        personas = pd.read_csv(args.personas).sample(n=args.n_personas, random_state=args.seed)
        personas["AGE_GRP"] = pd.cut(personas["AGE"], AGE_BINS, labels=AGE_LABELS, right=False)

    cand = wave[wave["is_item_cand"]]["var"].tolist()
    wave_meta = wave.set_index("var")
    rows, cell_dfs = [], []
    n_skip = 0
    for v in cand:
        if clean_prompts and v in ADMIN_EXCLUDE_VARS:
            n_skip += 1
            continue
        opts = real_options(v, vt, filter_leaked_dk_inap=clean_prompts)
        codes = [int(c) for c, _ in opts]
        if len(opts) < 2 or any(not (1 <= c <= 9) for c in codes) or len(set(codes)) != len(codes):
            n_skip += 1
            continue
        gold = weighted_dist(sub, v, [float(c) for c in codes], round_digits=4)
        if gold is None:
            n_skip += 1
            continue
        raw_txt = worded.get(v, "")
        stem_source = "question_stem"
        if pd.isna(raw_txt) or str(raw_txt).strip().lower() in ("", "nan"):
            if clean_prompts:
                # 설문원문이 결측(NaN)인 경우 variables_worded의 표제목 또는 wave_items/variables의 label로 대체
                tbl_title = str(worded_df.loc[v, "표제목"]) if v in worded_df.index and "표제목" in worded_df.columns and pd.notna(worded_df.loc[v, "표제목"]) else ""
                bat = str(wave_meta.loc[v, "battery"]) if v in wave_meta.index and pd.notna(wave_meta.loc[v, "battery"]) else ""
                lab = str(vt.loc[v, "label"]) if v in vt.index and pd.notna(vt.loc[v, "label"]) else ""
                txt = tbl_title.strip() or lab.strip() or bat.strip()
                stem_source = "label_fallback_provisional"
                if not txt:
                    n_skip += 1
                    continue
            else:
                txt = "nan"
                stem_source = "missing_stem"
        else:
            txt = str(raw_txt).strip()
        opt_txt = "\n".join(f"{c}. {lab}" for c, (_, lab) in zip(codes, opts))
        item_text = f"문항: {txt}\n\n선택지:\n{opt_txt}\n\n번호로만 답하십시오."
        prompt_metadata = {
            "prompt_stem_source": stem_source,
            "prompt_requires_review": stem_source != "question_stem",
        }

        if args.level == "c0":
            rows.append({
                "probe_id": f"C0_{v}", "var": v, "prompt": item_text,
                "codes": json.dumps(codes), "gold": json.dumps(gold),
                **prompt_metadata,
            })
        elif args.level == "adult":  # 구체적 인구통계 없이 "한국 성인" 머리말만
            rows.append({
                "probe_id": f"ADULT_{v}", "var": v,
                "prompt": f"당신은 한국 성인입니다.\n\n{item_text}",
                "codes": json.dumps(codes), "gold": json.dumps(gold),
                **prompt_metadata,
            })
        elif args.level == "demo":  # 셀마다 별도 프롬프트 (성별+연령 머리말)
            for sex_code, sex_lab in sexlab.items():
                for age_grp, age_rep in AGE_REP.items():
                    head = f"당신은 {age_rep}세 {sex_lab}입니다."
                    rows.append({
                        "probe_id": f"DEMO_{v}_{int(sex_code)}_{age_grp}",
                        "var": v, "sex": sex_code, "age_grp": age_grp,
                        "prompt": f"{head}\n\n{item_text}",
                        "codes": json.dumps(codes), "gold": json.dumps(gold),
                        **prompt_metadata,
                    })
        else:  # full: 페르소나 n명마다 별도 프롬프트 (6변수 서사형 머리말)
            for _, p in personas.iterrows():
                rows.append({
                    "probe_id": f"FULL_{v}_{int(p['persona_id'])}",
                    "var": v, "sex": p["SEX"], "age_grp": p["AGE_GRP"],
                    "persona_id": int(p["persona_id"]),
                    "respid": p.get("RESPID"),
                    "prompt": f"{p['persona_당신_하십시오체']}\n\n{item_text}",
                    "codes": json.dumps(codes), "gold": json.dumps(gold),
                    **prompt_metadata,
                })

        # 셀(성별x연령)별 실제 가중분포 P_KGSS(Y|c), 셀 비중 P_KGSS(c), 실제 가중평균 응답
        cdf = human_cell_distributions(
            sub, v, [float(c) for c in codes],
            cell_cols=("SEX", "AGE_GRP"), weight_col="FINALWT", min_cell_n=5,
        )
        if not cdf.empty:
            cdf = cdf.rename(columns={"SEX": "sex", "AGE_GRP": "age_grp"})
            # 기존 real_cells.csv와 호환되도록 real_grand를 반올림된 gold 기준으로도 맞추되,
            # 정밀한 셀 분포(real_dist)와 셀 가중치(cell_weight_item_norm)를 함께 저장한다.
            cdf["real_grand"] = float(np.sum(np.array(codes) * np.array(gold)))
            cell_dfs.append(cdf)

    cellrows_df = pd.concat(cell_dfs, ignore_index=True) if cell_dfs else pd.DataFrame()
    n_vars = cellrows_df["var"].nunique() if not cellrows_df.empty else 0
    print(f"문항 {n_vars}개 확정 (제외 {n_skip}개), 프로브 {len(rows)}건")
    pd.DataFrame(rows).to_csv(outdir / "probes.csv", index=False, encoding="utf-8-sig")
    cellrows_df.to_csv(outdir / "real_cells.csv", index=False, encoding="utf-8-sig")
    print(f"-> {outdir / 'probes.csv'}, {outdir / 'real_cells.csv'}")


def score(args):
    from kgss_metrics import (
        aggregate_demo_cells,
        build_cell_poststratified_row_weights,
        compute_subgroup_correlations,
        evaluate_item_distributions,
        validate_prob_dist,
    )

    outdir = Path(args.out)
    if getattr(args, "persona_matched", False) and not getattr(args, "classification", None):
        raise ValueError("페르소나 매칭 재채점에는 명목형 W1 제외 규칙을 담은 --classification이 필요합니다.")
    probes = pd.read_csv(outdir / "probes.csv")
    logits = pd.read_csv(outdir / "logit_responses.csv")
    cells = pd.read_csv(outdir / "real_cells.csv")
    if probes.empty or probes["probe_id"].isna().any() or probes["probe_id"].duplicated().any():
        raise ValueError("probes에는 비어 있지 않은 고유 probe_id가 필요합니다.")
    if logits.empty or logits["model"].isna().any():
        raise ValueError("logits에는 모델별 응답이 필요합니다.")
    if logits.duplicated(["model", "probe_id"]).any():
        raise ValueError("중복 (model, probe_id) 로짓 응답이 있습니다.")
    expected_probes = set(probes["probe_id"])
    for model, model_logits in logits.groupby("model"):
        actual_probes = set(model_logits["probe_id"])
        if actual_probes != expected_probes:
            raise ValueError(
                f"{model}: probe 응답 범위 불일치 "
                f"(누락 {len(expected_probes - actual_probes)}개, "
                f"미등록 {len(actual_probes - expected_probes)}개)"
            )
    classification = None
    classification_path = getattr(args, "classification", None)
    if classification_path:
        classification = pd.read_csv(classification_path)
        required = {"var", "scale_type", "include_in_clean_eval", "w1_valid_in_clean_eval"}
        if not required.issubset(classification.columns):
            raise ValueError(f"분류표 필수 열 누락: {sorted(required - set(classification.columns))}")
        if classification["var"].isna().any() or classification["var"].duplicated().any():
            raise ValueError("분류표 var는 비어 있지 않고 고유해야 합니다.")
        classification = classification.set_index("var")
        missing_vars = set(probes["var"]) - set(classification.index)
        if missing_vars:
            raise ValueError(f"분류표에 없는 평가 문항: {sorted(missing_vars)}")
        scale_types = {"binary", "ordinal_monotone", "nominal_unordered", "frequency_nonmonotone_code"}
        if not classification.loc[probes["var"].unique(), "scale_type"].isin(scale_types).all():
            raise ValueError("분류표에 지원하지 않는 scale_type이 있습니다.")
        for flag in ("include_in_clean_eval", "w1_valid_in_clean_eval"):
            values = classification[flag].astype(str).str.strip().str.lower()
            if not values.isin(["true", "false"]).all():
                raise ValueError(f"분류표 {flag}는 True/False여야 합니다.")
            classification[flag] = values.eq("true")
    has_cells = "sex" in probes.columns
    level_key = next(
        (k for k in ("C0_", "ADULT_", "DEMO_", "FULL_") if str(probes["probe_id"].iloc[0]).startswith(k)),
        "?",
    )
    level_name = {
        "C0_": "C0(페르소나 없음)",
        "ADULT_": "adult(한국 성인)",
        "DEMO_": "demo(성별+연령)",
        "FULL_": "full(6변수)",
    }
    level = level_name.get(level_key, "?")

    # 배터리(문항 묶음) 클러스터 정보를 wave_items.csv에서 로드 (존재할 경우)
    wave_path = Path(getattr(args, "inv", "./inventory")) / "wave_items.csv"
    battery_map = {}
    if wave_path.exists():
        wdf = pd.read_csv(wave_path)
        if "var" in wdf.columns and "battery" in wdf.columns:
            battery_map = wdf.set_index("var")["battery"].to_dict()

    logit_cols = ["probe_id"] + [c for c in logits.columns if c not in probes.columns and c != "probe_id"]
    d = probes.merge(logits[logit_cols], on="probe_id", how="inner")

    L = [f"# {level} 조건 분포 재현 타당성 검증 (기존 지표 vs 수정 지표 비교)\n"]
    if classification is None:
        L.append("\n주의: 문항 분류표 없이 계산한 레거시 진단입니다. 명목형·결함 문항이 포함될 수 있어 발표용 주결과로 사용하지 마십시오.\n")
    else:
        L.append(f"\n문항 분류 적용: `{classification_path}`. W1·subgroup r은 정제 이진/단조서열 문항에 한정하고, TVD·엔트로피는 정제 전체 문항을 사용합니다.\n")
    all_item_rows = []
    summary_models = {}

    for model, g in d.groupby("model"):
        L.append(f"\n## {model}\n")
        g = g.copy()
        model_preds = []
        for _, r in g.iterrows():
            codes = json.loads(r["codes"])
            pred = validate_prob_dist(json.loads(r["model_dist"]), tol=1e-3)
            model_preds.append(float(np.sum(np.array(codes, dtype=float) * pred)))
        g["model_pred"] = model_preds

        item_metrics = []
        for var, gv in g.groupby("var", sort=False):
            gold = json.loads(gv["gold"].iloc[0])
            codes = json.loads(gv["codes"].iloc[0])
            preds = [json.loads(x) for x in gv["model_dist"]]
            vc_cells = cells[cells["var"] == var].copy()

            weights = None
            cell_model_dists = None
            cell_human_dists = None
            cell_weights = None

            if level_key == "DEMO_" and not vc_cells.empty and "cell_weight_item_norm" in vc_cells.columns:
                cell_agg, merged_c = aggregate_demo_cells(
                    gv, vc_cells, dist_col="model_dist",
                    cell_cols=("sex", "age_grp"), weight_col="cell_weight_item_norm",
                )
                if not merged_c.empty:
                    # Demo는 각 셀 c에 KGSS 셀 비중 w_c를 부여하되, 셀 내 프롬프트가 n_c개이면
                    # 각 프롬프트 행에는 w_i = w_c / n_c를 부여하여 sum_{i in c} w_i == w_c가 되도록 함
                    weights = build_cell_poststratified_row_weights(
                        gv, merged_c,
                        cell_cols=("sex", "age_grp"),
                        dist_col="model_dist",
                        expected_cell_agg=cell_agg,
                    )
                    if "real_dist" in merged_c.columns:
                        cell_model_dists = merged_c["cell_model_dist"].tolist()
                        cell_human_dists = [json.loads(x) for x in merged_c["real_dist"]]
                        cell_weights = merged_c["cell_weight_matched_norm"].tolist()
            elif level_key == "FULL_" and not vc_cells.empty and "real_dist" in vc_cells.columns:
                # Full은 이미 FINALWT 비례 복원추출 표본이므로 기본 모집단 집계는 동일 가중 평균(weights=None)
                # 단, 셀 사후층화 비교를 위해 cell_model_dists와 cell_weights도 함께 전달하여 기록
                _, merged_c = aggregate_demo_cells(
                    gv, vc_cells, dist_col="model_dist",
                    cell_cols=("sex", "age_grp"), weight_col="cell_weight_item_norm",
                )
                if not merged_c.empty:
                    cell_model_dists = merged_c["cell_model_dist"].tolist()
                    cell_human_dists = [json.loads(x) for x in merged_c["real_dist"]]
                    cell_weights = merged_c["cell_weight_matched_norm"].tolist()

            m = evaluate_item_distributions(
                pred_dists=preds,
                gold_dist=gold,
                codes=codes,
                weights=weights,
                cell_model_dists=cell_model_dists,
                cell_human_dists=cell_human_dists,
                cell_weights=cell_weights,
                use_code_positions=getattr(args, "use_code_positions", False),
            )
            if level_key == "DEMO_" and "population_wasserstein_cell_weighted" in m:
                if abs(m["population_wasserstein"] - m["population_wasserstein_cell_weighted"]) > 1e-10:
                    raise AssertionError(
                        f"DEMO 모집단 Wasserstein 불변식 위반 ({var}): "
                        f"{m['population_wasserstein']} != {m['population_wasserstein_cell_weighted']}"
                    )
            m["model"] = model
            m["level"] = level
            m["var"] = var
            m["mean_mass"] = float(gv["mass"].mean())
            if classification is not None:
                item_class = classification.loc[var]
                included = bool(item_class["include_in_clean_eval"])
                w1_valid = included and bool(item_class["w1_valid_in_clean_eval"]) and item_class["scale_type"] in ("binary", "ordinal_monotone")
                m["scale_type"] = item_class["scale_type"]
                m["include_in_clean_eval"] = included
                m["w1_valid_for_evaluation"] = w1_valid
                for key in list(m):
                    if ("wasserstein" in key or key == "existing_minus_population_w") and not w1_valid:
                        m[f"diagnostic_{key}"] = m[key]
                        m[key] = np.nan
            item_metrics.append(m)
            all_item_rows.append(m)

        mdf_all = pd.DataFrame(item_metrics)
        mdf = mdf_all.copy()
        if classification is not None:
            mdf = mdf[mdf["include_in_clean_eval"]].copy()

        if has_cells:
            cell_pred = g.groupby(["var", "sex", "age_grp"], observed=True)["model_pred"].mean().reset_index()
            merged = cell_pred.merge(cells, on=["var", "sex", "age_grp"], how="inner")
        else:
            preds_map = dict(zip(g["var"], g["model_pred"]))
            merged = cells.copy()
            merged["model_pred"] = merged["var"].map(preds_map)
            merged = merged.dropna(subset=["model_pred"])

        span_map = dict(zip(mdf_all["var"], mdf_all["n_options"] - 1.0))
        merged["span"] = merged["var"].map(span_map).fillna(1.0)
        if battery_map:
            merged["battery"] = merged["var"].map(battery_map)
        if classification is not None:
            valid_vars = set(mdf.loc[mdf["w1_valid_for_evaluation"], "var"])
            merged = merged[merged["var"].isin(valid_vars)].copy()
        else:
            valid_vars = set(mdf["var"])

        sub_r = compute_subgroup_correlations(merged, cell_cols=("sex", "age_grp"), cluster_col="battery")

        ex_w = float(mdf["existing_wasserstein"].mean())
        pop_w = float(mdf["population_wasserstein"].mean())
        pop_w_unw = float(mdf["population_wasserstein_unweighted"].mean())
        ex_tvd = float(mdf["existing_tvd"].mean())
        pop_tvd = float(mdf["population_tvd"].mean())
        pop_tvd_unw = float(mdf["population_tvd_unweighted"].mean())
        unif_w = float(mdf["uniform_wasserstein"].mean())
        unif_tvd = float(mdf["uniform_tvd"].mean())
        modal_w = float(mdf["marginal_modal_1hot_wasserstein"].mean())
        median_w = float(mdf["marginal_median_1hot_wasserstein"].mean())
        cond_ent_unw = float(mdf["model_conditional_entropy_bits"].mean())
        cond_ent_w = float(mdf["model_conditional_entropy_weighted_bits"].mean())
        pop_ent_w = float(mdf["model_population_entropy_bits"].mean())
        pop_ent_unw = float(mdf["model_population_entropy_unweighted_bits"].mean())
        gold_ent = float(mdf["human_entropy_bits"].mean())
        mi_bits_w = float(mdf["mutual_information_bits"].mean())
        mi_bits_unw = float(mdf["mutual_information_unweighted_bits"].mean())
        mean_mass = float(g["mass"].mean())

        def _json_safe_sub_r(sr: dict) -> dict:
            out = {}
            for k, v in sr.items():
                if isinstance(v, dict):
                    out[k] = {str(kk): (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()}
                elif isinstance(v, (float, np.floating)):
                    out[k] = None if pd.isna(v) else float(v)
                elif isinstance(v, (int, np.integer)):
                    out[k] = int(v)
                else:
                    out[k] = v
            return out

        summary_models[model] = {
            "level": level,
            "n_total_items": int(g["var"].nunique()),
            "n_clean_eval_items": int(len(mdf)),
            "n_valid_w1_items": int(len(valid_vars)),
            "n_probes": int(len(g)),
            "mean_mass": mean_mass,
            "existing_wasserstein": ex_w,
            "population_wasserstein": pop_w,
            "population_wasserstein_unweighted": pop_w_unw,
            "population_wasserstein_cell_weighted": (
                float(mdf["population_wasserstein_cell_weighted"].mean())
                if "population_wasserstein_cell_weighted" in mdf.columns and mdf["population_wasserstein_cell_weighted"].notna().any()
                else pop_w
            ),
            "existing_tvd": ex_tvd,
            "population_tvd": pop_tvd,
            "population_tvd_unweighted": pop_tvd_unw,
            "population_tvd_cell_weighted": (
                float(mdf["population_tvd_cell_weighted"].mean())
                if "population_tvd_cell_weighted" in mdf.columns and mdf["population_tvd_cell_weighted"].notna().any()
                else pop_tvd
            ),
            "uniform_wasserstein": unif_w,
            "uniform_tvd": unif_tvd,
            "marginal_modal_1hot_wasserstein": modal_w,
            "marginal_median_1hot_wasserstein": median_w,
            "model_conditional_entropy_weighted_bits": cond_ent_w,
            "model_conditional_entropy_unweighted_bits": cond_ent_unw,
            "model_population_entropy_weighted_bits": pop_ent_w,
            "model_population_entropy_unweighted_bits": pop_ent_unw,
            "human_entropy_bits": gold_ent,
            "mutual_information_weighted_bits": mi_bits_w,
            "mutual_information_unweighted_bits": mi_bits_unw,
            "subgroup_r_valid_items": _json_safe_sub_r(sub_r),
        }

        L.append(f"- 문항 {g['var'].nunique()}개 / 프로브 {len(g)}건 / 평균 선택지확률질량 {mean_mass:.4f}\n")
        if classification is not None:
            L.append(f"- 평가 범위: 정제 전체 {len(mdf)}문항(TVD·엔트로피), 정제 이진/단조서열 {len(valid_vars)}문항(W1·subgroup r). 제외 W1은 CSV diagnostic_* 열에만 보존합니다.\n")
        L.append(
            f"- **Wasserstein 및 TVD 거리 비교**:\n"
            f"  - 기존 산식 (개별 평균 $\\frac{{1}}{{N}}\\sum_i W_1(P_i, Q)$): **{ex_w:.4f}** (개별 평균 TVD: **{ex_tvd:.4f}**)\n"
            f"  - 수정 산식 (모집단 집계 $W_1(\\sum_i w_i P_i, Q)$): **{pop_w:.4f}** "
            f"(차이 {pop_w - ex_w:+.4f} / 동일가중 $w_i=1/N$ 집계: **{pop_w_unw:.4f}**)\n"
            f"  - 수정 산식 TVD ($\\mathrm{{TVD}}(\\sum_i w_i P_i, Q)$): **{pop_tvd:.4f}** "
            f"(동일가중 $w_i=1/N$ TVD: **{pop_tvd_unw:.4f}** / 균등분포 TVD 기준선: **{unif_tvd:.4f}**)\n"
            f"  - 균등분포 기준선 (Uniform $W_1$ Baseline): **{unif_w:.4f}** "
            f"(수정 산식 - 균등기준선: {pop_w - unif_w:+.4f})\n"
            f"  - 전체 최빈 원핫 기준선 (Marginal Modal 1-hot $W_1$): **{modal_w:.4f}**\n"
            f"  - 전체 중앙값 원핫 기준선 (Marginal Median 1-hot $W_1$, 최적 점질량): **{median_w:.4f}**\n"
        )
        if "conditional_cell_wasserstein_weighted" in mdf.columns:
            cc_w = float(mdf["conditional_cell_wasserstein_weighted"].mean())
            cc_unw = float(mdf["conditional_cell_wasserstein_unweighted"].mean())
            cc_tvd_w = float(mdf["conditional_cell_tvd_weighted"].mean())
            L.append(
                f"  - 셀 조건부 Wasserstein ($\\sum_c w_c W_1(P(Y|c), P_{{KGSS}}(Y|c))$): "
                f"**{cc_w:.4f}** (비가중 셀 평균: {cc_unw:.4f} / 셀 조건부 가중 TVD: {cc_tvd_w:.4f})\n"
            )
        L.append(
            f"- **엔트로피 분해 (모드 붕괴 점검, 가중 vs 비가중)**:\n"
            f"  - 모델 조건부 엔트로피 (가중 $\\sum_i w_i H(P_i)$ / 비가중 $\\frac{{1}}{{N}}\\sum_i H(P_i)$): "
            f"**{cond_ent_w:.3f}bit** / **{cond_ent_unw:.3f}bit**\n"
            f"  - 모델 모집단 집계 엔트로피 (가중 $H(\\sum_i w_i P_i)$ / 비가중 $H(\\frac{{1}}{{N}}\\sum_i P_i)$): "
            f"**{pop_ent_w:.3f}bit** / **{pop_ent_unw:.3f}bit**\n"
            f"  - 페르소나 상호정보량 $I(Y; X)$ (가중 / 비가중): "
            f"**{mi_bits_w:.3f}bit** / **{mi_bits_unw:.3f}bit** "
            f"(실제 인간 응답 엔트로피: **{gold_ent:.3f}bit**)\n"
        )
        ex_rs = list(sub_r["existing_cell_rs"].values())
        ce_rs = list(sub_r["centered_cell_rs"].values())
        if ex_rs:
            L.append(
                f"- **subgroup r (성별×연령 {len(ex_rs)}셀 평균)**:\n"
                f"  - 기존 산식 (모델 미중심화): **{sub_r['existing_mean_r']:.3f}** "
                f"(범위 {min(ex_rs):.3f}~{max(ex_rs):.3f})\n"
            )
        if ce_rs:
            bat_ci_str = ""
            if pd.notna(sub_r.get("centered_battery_boot_ci_low")):
                bat_ci_str = (
                    f" / 배터리 클러스터({sub_r['n_clusters']}개) 부트스트랩 95% CI "
                    f"[{sub_r['centered_battery_boot_ci_low']:.3f}, {sub_r['centered_battery_boot_ci_high']:.3f}]"
                )
            L.append(
                f"  - 수정 산식 (동일 관측셀·FINALWT 가중 중심화): **{sub_r['centered_mean_r']:.3f}** "
                f"(범위 {min(ce_rs):.3f}~{max(ce_rs):.3f}, "
                f"문항 부트스트랩 95% CI [{sub_r['centered_boot_ci_low']:.3f}, {sub_r['centered_boot_ci_high']:.3f}]"
                f"{bat_ci_str})\n"
                f"  - 수정 산식 (동일 관측셀 비가중 중심화): **{sub_r['centered_unweighted_mean_r']:.3f}**\n"
                f"  - 수정 산식 (척도 폭 [0,1] 정규화 중심화): **{sub_r['scale_normalized_mean_r']:.3f}**\n"
                f"  - 문항 내 10셀 상관 평균: **{sub_r['within_item_mean_r']:.3f}**\n"
            )

    text = "".join(L)
    out_md = outdir / (
        "c0_validity_result.md"
        if getattr(args, "overwrite_legacy", False)
        else "c0_validity_recomputed.md"
    )
    out_md.write_text(text, encoding="utf-8")
    if all_item_rows:
        pd.DataFrame(all_item_rows).to_csv(
            outdir / "c0_validity_comparison.csv", index=False, encoding="utf-8-sig"
        )
    (outdir / "c0_validity_summary.json").write_text(
        json.dumps({"level": level, "models": summary_models}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(text)
    print(f"-> {out_md}")

    if getattr(args, "persona_matched", False):
        from reexperiment_20261009.score_persona_matched import CONDITIONS, run as run_persona_matched
        if outdir.name not in CONDITIONS:
            raise ValueError("페르소나 매칭 재채점은 명시된 clean 전수 조건 디렉터리가 필요합니다.")
        matched_out = outdir / "persona_matched_evaluation"
        run_persona_matched(argparse.Namespace(
            reexperiment_dir=str(outdir.parent), out=str(matched_out), inv=str(args.inv),
            classification=str(classification_path), year=args.year,
            conditions=[outdir.name], age_policies=["exact_age", "age_band"],
            min_valid_n=args.matched_min_valid_n, min_effective_n=args.matched_min_effective_n,
        ))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--level", choices=["c0", "adult", "demo", "full"], default="c0")
    b.add_argument("--inv", default="./inventory")
    b.add_argument("--worded", default="./selection/variables_worded.csv")
    b.add_argument("--year", type=int, default=2023)
    b.add_argument("--out", default="./c0_check")
    b.add_argument("--personas", default="./persona_sample/persona_sample.csv")
    b.add_argument("--n-personas", type=int, default=40)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--clean-prompts", action="store_true",
                   help="행정변수(SAMPLEAB) 제외, 결측 보기(모름/비해당) 필터링, 설문원문 NaN 라벨 대체 적용")
    s = sub.add_parser("score")
    s.add_argument("--out", default="./c0_check")
    s.add_argument("--inv", default="./inventory")
    s.add_argument("--classification", help="문항별 척도/정제 포함 규칙 CSV. 없으면 발표용이 아닌 레거시 진단으로 표시")
    s.add_argument("--persona-matched", action="store_true",
                   help="저장된 응답을 입력 페르소나와 같은 인간 집단으로 추가 재채점; 모집단 지표와 별도 저장")
    s.add_argument("--year", type=int, default=2023)
    s.add_argument("--matched-min-valid-n", type=int, default=5)
    s.add_argument("--matched-min-effective-n", type=float, default=5.0)
    s.add_argument("--use-code-positions", action="store_true",
                   help="비연속 선택지 코드일 때 순번(0..K-1) 대신 실제 코드값을 Wasserstein 좌표로 사용")
    s.add_argument("--overwrite-legacy", action="store_true",
                   help="기존 c0_validity_result.md를 덮어씀 (기본값: c0_validity_recomputed.md에 별도 저장)")
    args = ap.parse_args()
    {"build": build, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
