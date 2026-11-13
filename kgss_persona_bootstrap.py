r"""
KGSS 2023 FINALWT 가중 복원추출(bootstrap) 페르소나 표본 생성

목적
    실험 설계의 대조군 4수준(페르소나 없음/한국 성인/성별+연령/6변수) 중
    "6변수" 조건에 쓸 페르소나 표본을 만든다.

    개별 변수(성별, 연령, 학력...)를 따로따로 뽑으면 변수 간 결합분포가
    실제와 어긋난다(예: 고령×고학력처럼 실제론 드문 조합이 과다 생성됨).
    대신 실제 KGSS 2023 응답자 "행 전체"를 FINALWT 확률에 비례해
    복원추출하면, 그 응답자의 6개 변수 조합이 통째로 딸려오므로 결합분포가
    저절로 실제와 일치한다 — 서울 논문의 "연령대×성별만 쿼터 매칭"보다
    엄밀한 방식이다(CLAUDE.md 7절 "4. 실험 설계 확정" 참고).

    한 번 뽑아 고정한 표본을 대조군 4수준 전부에 재사용한다 — 6변수 조건은
    여기서 만든 페르소나 문장을 그대로 쓰고, "성별+연령" 조건은 같은
    응답자의 SEX/AGE만 뽑아 쓰면 된다(동일 응답자 집합이라 조건 간
    대응비교(within-persona)가 성립).

페르소나 변수 6개의 구성 (재구성, 확정 아님 — 주의)
    CLAUDE.md에 "페르소나 변수 6개 전부 사용"이라고만 적혀 있고 목록이 없어,
    다음 두 근거로 재구성했다:
      1. kgss_register_pilot.py의 PERSONA_* 템플릿이 실제 쓴 변수:
         나이·성별·학력·거주지역·혼인상태 (5개)
      2. CLAUDE.md의 "사후층화 변수는 성별·나이·지역·도시성"에서
         위 5개에 없는 도시성(URBAN)을 추가 (6개째)
    → SEX, AGE, EDUC, MARITAL, REGION, URBAN. INCOME/RINCOME은 결측률이
    각각 4.1%/36.3%로 높아 제외된 것으로 보이나, 이것도 추정이다.
    **팀 확인 후 다르면 --vars로 덮어쓸 것.**

사용법
    uv run kgss_persona_bootstrap.py --n 200 --seed 42 --out ./persona_sample
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

DEFAULT_VARS = ["SEX", "AGE", "EDUC", "MARITAL", "REGION", "URBAN"]

# 결측 취급: -1(비해당), -8(모름·무응답), 그리고 각 변수 값라벨에 남아있는
# 통합 이전 잔여 DK 코드(8/88/9/99 등)는 variables.csv의 value_labels에서
# 라벨이 "DK"/"Refusal"/"모르겠다" 류인 코드로 판별한다(CLAUDE.md 2절
# "값 라벨의 8, 88, 9, 99는 통합 이전 회차의 잔여물" 원칙 그대로 적용).
DK_LABEL_PAT = "DK|Refusal|모르겠|응답거부|무응답"


def load_value_labels(inv: Path) -> dict[str, dict[str, str]]:
    var = pd.read_csv(inv / "variables.csv").set_index("var")["value_labels"]
    out = {}
    for v, s in var.items():
        if isinstance(s, str):
            try:
                out[v] = json.loads(s)
            except Exception:
                out[v] = {}
    return out


def dk_codes(labels: dict[str, str]) -> set[float]:
    import re
    return {float(k) for k, lab in labels.items()
            if re.search(DK_LABEL_PAT, str(lab))}


def persona_sentence(row: pd.Series, labels: dict[str, dict[str, str]],
                     addressee: str, register: str) -> str:
    """kgss_register_pilot.py의 PERSONA_* 문체를 그대로 재사용하고, 5개 변수
    (나이·성별·학력·지역·혼인)로 된 원래 템플릿에 도시성(URBAN)만 괄호로
    덧붙인다 — 원래 있던 4문장(YOU_FORMAL/YOU_PLAIN/HE_FORMAL/HE_PLAIN)과
    자연스럽게 이어지도록, 새 절을 끼워넣지 않고 문장 끝에 추가했다.
    addressee: "당신" | "이사람"    register: "formal"(하십시오체) | "plain"(해라체)
    """
    age = int(row["AGE"])
    sex_lab = labels["SEX"].get(str(float(row["SEX"])), "").strip()
    educ_lab = labels["EDUC"].get(str(float(row["EDUC"])), "").strip()
    region_lab = labels["REGION"].get(str(float(row["REGION"])), "").strip()
    marital_lab = labels["MARITAL"].get(str(float(row["MARITAL"])), "").strip()
    urban_lab = labels["URBAN"].get(str(float(row["URBAN"])), "").strip()

    if addressee == "당신":
        subj = f"{age}세 {sex_lab}" + ("입니다" if register == "formal" else "이다")
        head = f"당신은 {subj}."
    else:
        subj = f"{age}세 {sex_lab}에 대한 설명" + ("입니다" if register == "formal" else "이다")
        head = f"다음은 {subj}."

    if register == "formal":
        tail = f"최종학력은 {educ_lab}이며 {region_lab}에 거주하며 {marital_lab}입니다."
    else:
        tail = f"최종학력은 {educ_lab}이며 {region_lab}에 거주하며 {marital_lab}이다."

    return f"{head} {tail} ({urban_lab} 지역 거주)"


def build(args):
    inv = Path(args.inv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inv / "kgss_clean.parquet")
    sub = df[df["YEAR"] == args.year].copy()
    print(f"[1/4] {args.year}년 표본 {len(sub)}명")

    labels = load_value_labels(inv)
    persona_vars = args.vars
    missing_vars = [v for v in persona_vars if v not in sub.columns]
    if missing_vars:
        raise SystemExit(f"변수 없음: {missing_vars}")

    # ---------- 결측(비해당/모름/DK) 응답자 제외
    mask = pd.Series(True, index=sub.index)
    for v in persona_vars:
        dk = dk_codes(labels.get(v, {})) | {-1.0, -8.0}
        n_before = mask.sum()
        mask &= ~sub[v].isin(dk) & sub[v].notna()
        n_dropped = n_before - mask.sum()
        if n_dropped:
            print(f"      {v}: 결측/DK로 {n_dropped}명 제외")
    elig = sub[mask].copy()
    print(f"[2/4] 6변수 전부 유효한 응답자 {len(elig)}명 "
         f"({len(elig)/len(sub)*100:.1f}%) — 이 중에서만 복원추출")
    if len(elig) < args.n:
        print(f"      경고: 유효 응답자({len(elig)})가 목표 n({args.n})보다 적음 "
             f"— 복원추출이라 상관없지만 중복이 많아질 것")

    # ---------- FINALWT 가중 복원추출
    w = elig["FINALWT"].to_numpy(dtype=float)
    w = w / w.sum()
    rng = np.random.default_rng(args.seed)
    chosen_pos = rng.choice(len(elig), size=args.n, replace=True, p=w)
    sample = elig.iloc[chosen_pos].reset_index(drop=True)
    sample.insert(0, "persona_id", range(len(sample)))
    print(f"[3/4] {args.n}명 복원추출 완료 (seed={args.seed})")

    # ---------- 진단: 재표집 표본의 주변분포가 가중 모집단과 일치하는지 확인
    print("      가중 모집단 대비 재표집 표본 주변분포 점검:")
    for v in persona_vars:
        pop = (elig.groupby(v)["FINALWT"].sum() / elig["FINALWT"].sum())
        smp = sample[v].value_counts(normalize=True).sort_index()
        both = pd.concat([pop.rename("모집단"), smp.rename("표본")], axis=1).fillna(0)
        maxdiff = (both["모집단"] - both["표본"]).abs().max()
        print(f"        {v}: 최대 주변분포 오차 {maxdiff*100:.1f}%p")

    # ---------- 페르소나 문장 생성
    rows = []
    for _, r in sample.iterrows():
        row = {"persona_id": int(r["persona_id"]), "RESPID": r.get("RESPID"),
              "FINALWT_원본": r["FINALWT"]}
        for v in persona_vars:
            row[v] = r[v]
            row[f"{v}_label"] = labels.get(v, {}).get(str(float(r[v])), "")
        for addressee in ["당신", "이사람"]:
            for register, tag in [("formal", "하십시오체"), ("plain", "해라체")]:
                try:
                    row[f"persona_{addressee}_{tag}"] = persona_sentence(
                        r, labels, addressee, register)
                except Exception as e:
                    row[f"persona_{addressee}_{tag}"] = f"[생성 실패: {e}]"
        rows.append(row)
    out = pd.DataFrame(rows)
    out_path = outdir / "persona_sample.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[4/4] 완료 -> {out_path} ({len(out)}행)")

    (outdir / "README.md").write_text(
        "# 6변수 페르소나 표본\n\n"
        f"KGSS {args.year}년 응답자를 FINALWT 가중치로 복원추출(n={args.n}, "
        f"seed={args.seed})해서 만든 표본입니다. `persona_id`로 4수준 대조군 "
        "전부(없음/한국성인/성별+연령/6변수)에서 재사용하십시오 — 같은 "
        "응답자 집합이어야 조건 간 대응비교(within-persona)가 성립합니다.\n\n"
        "- `성별+연령` 조건: 이 표본의 SEX/AGE 열만 사용\n"
        "- `6변수` 조건: `persona_당신_하십시오체` 등 열 사용\n"
        "- `페르소나 없음(C0)`/`한국 성인` 조건: 이 표본과 무관, 별도 프롬프트\n\n"
        "**주의**: 페르소나 변수 6개(SEX/AGE/EDUC/MARITAL/REGION/URBAN)는 "
        "스크립트 docstring에 적힌 근거로 재구성한 것이라 팀 확인이 필요합니다. "
        "다르면 `--vars`로 다시 생성하십시오.\n",
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", default="./inventory")
    ap.add_argument("--out", default="./persona_sample")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--n", type=int, default=200,
                    help="복원추출할 페르소나 수 (본실험 규모 확정 전 임시값)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vars", nargs="+", default=DEFAULT_VARS,
                    help="페르소나에 쓸 변수 목록 (기본값은 재구성치, 팀 확인 필요)")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
