"""246문항을 OpenAI Decisions API(gpt-6-luna)로 분류한다.

분류 항목 (문항당 요청 1회, 질문 3개):
  topic               choice     주제 영역 10개 중 하나
  socially_sensitive  predicate  사회적 바람직성 편향이 걸릴 만한 문항인가 (확률)
  desirable_end       choice     바람직하다고 여겨지는 쪽: 1번 쪽 / 마지막 번호 쪽 / 가운데 / 없음

척도 유형(이진·서열·명목)은 감사 때 코드로 이미 분류했으므로(items_246.csv의
scale_type) 여기서 다시 묻지 않는다.

실행:
  python3 classify_items_decisions.py --dry-run        # 요청 1건을 출력하고 토큰을 추정만 한다
  OPENAI_API_KEY=... python3 classify_items_decisions.py --run

출력:
  item_classification_decisions.csv  문항별 결과와 확률
  review_queue.csv                   사람이 검수할 문항(확신도 낮음, 민감 확률 0.3~0.7)

API 형식 출처: https://developers.openai.com/api/docs/guides/decisions
(2026-11-12 확인. 공개 베타라 형식과 요금이 바뀔 수 있다.)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITEMS = HERE / "items_246.csv"
OUT = HERE / "item_classification_decisions.csv"
REVIEW = HERE / "review_queue.csv"
RAW = HERE / "decisions_raw.jsonl"

ENDPOINT = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
PRICE_INPUT_PER_MTOK = 0.10  # 출력 무료. 2026-11-12 공식 가이드 기준

TOPICS = [
    ("politics_gov", "정치·정부·제도에 대한 태도와 신뢰"),
    ("economy_work_welfare", "경제·노동·복지·불평등"),
    ("family_gender", "가족·결혼·성역할·젠더"),
    ("migration_identity", "이민·다문화·민족 정체성·북한"),
    ("social_trust_participation", "대인 신뢰·사회적 관계·시민 참여"),
    ("religion_values", "종교·도덕·삶의 가치관"),
    ("health_wellbeing", "건강·행복·생활 만족·여가"),
    ("media_information", "미디어·정보·기술 이용"),
    ("discrimination_minority", "차별·혐오·소수자에 대한 태도"),
    ("personal_fact", "응답자 본인의 사실·행동·인구학 정보"),
]

QUESTIONS = [
    {
        "type": "choice",
        "name": "topic",
        "instructions": "이 설문 문항이 주로 다루는 주제 영역은 무엇인가?",
        "choices": [{"value": v, "description": d} for v, d in TOPICS],
    },
    {
        "type": "predicate",
        "name": "socially_sensitive",
        "instructions": (
            "한국 성인 응답자가 면접원 앞에서 이 문항에 답할 때, 사회적으로 바람직해 보이는 답이 "
            "분명히 존재해서 실제 생각과 다르게 답할 가능성이 큰 문항인가?"
        ),
    },
    {
        "type": "score",
        "name": "sensitivity_level",
        "instructions": (
            "한국 성인이 면접원 앞에서 이 문항에 솔직하게 답하기 얼마나 부담스러운가? "
            "답이 사회 규범·도덕·법·차별·성·소득·정신건강 등과 얽혀 체면이 걸리는 정도로 판단하라."
        ),
        "levels": [
            {"label": "0", "description": "부담 없음: 사실 확인이나 일상적 의견. 어느 답을 해도 체면이 걸리지 않는다."},
            {"label": "1", "description": "약간: 바람직한 답이 있긴 하지만 다르게 답해도 크게 문제 되지 않는다."},
            {"label": "2", "description": "상당함: 사회적으로 바람직한 답이 분명해서 일부는 실제와 다르게 답할 것이다."},
            {"label": "3", "description": "매우 큼: 법·도덕 위반, 성, 차별·혐오, 정신건강처럼 솔직한 답이 낙인이 될 수 있다."},
        ],
    },
    {
        "type": "choice",
        "name": "desirable_end",
        "instructions": (
            "선택지 중 한국 사회에서 사회적으로 더 바람직하다고 여겨지는 답은 어느 쪽에 있는가?"
        ),
        "choices": [
            {"value": "first", "description": "앞쪽 번호(1번 쪽) 선택지가 더 바람직하게 여겨진다."},
            {"value": "last", "description": "뒤쪽 번호(마지막 번호 쪽) 선택지가 더 바람직하게 여겨진다."},
            {"value": "middle", "description": "양 끝이 아니라 가운데 선택지가 더 바람직하게 여겨진다."},
            {"value": "none", "description": "뚜렷하게 더 바람직한 쪽이 없거나, 선택지가 순서가 없는 범주다."},
        ],
    },
]


def load_items() -> list[dict]:
    with ITEMS.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def item_input(row: dict) -> str:
    # 실험 프롬프트에서 마지막 지시문만 뺀 문항·선택지 원문을 그대로 쓴다.
    text = row["item_prompt"].replace("\n\n번호로만 답하십시오.", "").strip()
    return f"[변수명 {row['var']} · {row['label']}]\n{text}"


PROVIDERS = {
    # OpenAI 직접: model / input / questions(배열), predicate 사용
    "openai": {"endpoint": ENDPOINT, "key_env": "OPENAI_API_KEY", "models": [MODEL],
               "price": PRICE_INPUT_PER_MTOK},
    # OpenGateway: model / state / questions(맵), noul·choice의 criteria 사용
    # 형식 출처: https://opengateway.ai/docs/gateway/decisions (2026-11-12 확인)
    "opengateway": {"endpoint": "https://apis.opengateway.ai/v1/decisions", "key_env": "OPENGATEWAY_API_KEY",
                    "models": ["sionic-ai/jev-omni", "sionic-ai/decider-4b"],
                    "price": 0.0063},  # jev-omni 할인가. decider-4b는 0.0021
}


def opengateway_questions() -> dict:
    qs = {}
    for q in QUESTIONS:
        if q["type"] == "predicate":
            qs[q["name"]] = {"type": "noul", "instructions": q["instructions"] + " (`item` 기준)",
                             "criteria": {"true": "그런 문항이다", "false": "그런 문항이 아니다"}}
        elif q["type"] == "score":
            qs[q["name"]] = {"type": "score", "instructions": q["instructions"] + " (`item` 기준)",
                             "criteria": [lv["description"] for lv in q["levels"]]}
        else:
            qs[q["name"]] = {"type": "choice", "instructions": q["instructions"] + " (`item` 기준)",
                             "criteria": {c["value"]: c["description"] for c in q["choices"]}}
    return qs


def build_request(row: dict, provider: str = "openai", model: str = MODEL) -> dict:
    if provider == "opengateway":
        return {"model": model, "state": {"item": item_input(row)}, "questions": opengateway_questions()}
    return {"model": model, "input": item_input(row), "questions": QUESTIONS}


def normalize_answers(resp: dict) -> list[dict]:
    """OpenGateway의 answers 맵을 OpenAI 형식(배열)으로 바꾼다."""
    ans = resp.get("answers", [])
    if isinstance(ans, list):
        return ans
    out = []
    for name, a in ans.items():
        a = dict(a)
        a["name"] = name
        if "noul" in a:
            a["type"], a["probability"] = "predicate", a["noul"]
        elif "score" in a:
            a["type"] = "score"
        elif "choice" in a:
            a["type"] = "choice"
            probs = a.get("probabilities")
            if isinstance(probs, dict):
                a["probabilities"] = [{"value": k, "probability": v} for k, v in probs.items()]
        out.append(a)
    return out


def post(body: dict, api_key: str, endpoint: str = ENDPOINT, retries: int = 4) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(
            endpoint, data=data, method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise SystemExit(f"HTTP {e.code}: {msg[:500]}")
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise
    raise SystemExit("재시도 초과")


def parse(row: dict, resp: dict) -> dict:
    out = {"var": row["var"], "scale_type": row["scale_type"]}
    for a in normalize_answers(resp):
        name, typ = a.get("name"), a.get("type")
        if typ == "refusal":
            out[f"{name}"] = "REFUSAL"
            continue
        if typ == "choice":
            out[name] = a.get("choice")
            out[f"{name}_confidence"] = a.get("confidence")
            out[f"{name}_probs"] = json.dumps(
                {p["value"]: p["probability"] for p in a.get("probabilities", [])}, ensure_ascii=False)
        elif typ == "predicate":
            out[f"{name}_prob"] = a.get("probability")
        elif typ == "score":
            out[name] = a.get("score")
            out[f"{name}_confidence"] = a.get("confidence")
            out[f"{name}_probs"] = json.dumps(a.get("probabilities"), ensure_ascii=False)
    return out


def needs_review(r: dict) -> list[str]:
    why = []
    for name in ("topic", "desirable_end"):
        if r.get(name) == "REFUSAL":
            why.append(f"{name}_refusal")
        elif (r.get(f"{name}_confidence") or 0) < 0.7:
            why.append(f"{name}_low_conf")
    p = r.get("socially_sensitive_prob")
    if p is None:
        why.append("sensitive_missing")
    elif 0.3 <= p <= 0.7:
        why.append("sensitive_borderline")
    # 순서 없는 척도인데 한쪽 끝을 바람직하다고 고르면 모순
    if r.get("scale_type") == "nominal_unordered" and r.get("desirable_end") in ("first", "last"):
        why.append("nominal_with_direction")
    return why


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 n개만 (시험용)")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="opengateway")
    ap.add_argument("--model", default=None, help="기본: 제공자의 첫 모델 (opengateway → sionic-ai/jev-omni)")
    args = ap.parse_args()

    prov = PROVIDERS[args.provider]
    model = args.model or prov["models"][0]
    tag = model.split("/")[-1]
    global OUT, REVIEW, RAW
    OUT = HERE / f"item_classification_{tag}.csv"
    REVIEW = HERE / f"review_queue_{tag}.csv"
    RAW = HERE / f"decisions_raw_{tag}.jsonl"

    items = load_items()[: args.limit]
    chars = sum(len(json.dumps(build_request(r, args.provider, model), ensure_ascii=False)) for r in items)
    # 한국어 혼합 JSON은 대략 2자당 1토큰으로 넉넉히 잡는다.
    est_tok = chars / 2
    print(f"{args.provider} · {model} | 문항 {len(items)}개, 요청 {len(items)}건, 입력 약 {est_tok/1e3:.0f}K 토큰, "
          f"예상 비용 약 ${est_tok/1e6*prov['price']:.4f}")

    if args.dry_run:
        print(json.dumps(build_request(items[0], args.provider, model), ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get(prov["key_env"])
    if not api_key:
        sys.exit(f"{prov['key_env']}가 없습니다.")

    done = set()
    if RAW.exists():
        for line in RAW.open(encoding="utf-8"):
            done.add(json.loads(line)["var"])
    with RAW.open("a", encoding="utf-8") as raw:
        for i, row in enumerate(items, 1):
            if row["var"] in done:
                continue
            resp = post(build_request(row, args.provider, model), api_key, prov["endpoint"])
            raw.write(json.dumps({"var": row["var"], "response": resp}, ensure_ascii=False) + "\n")
            raw.flush()
            if i % 25 == 0:
                print(f"{i}/{len(items)}")

    by_var = {json.loads(l)["var"]: json.loads(l)["response"] for l in RAW.open(encoding="utf-8")}
    rows = [parse(r, by_var[r["var"]]) for r in items if r["var"] in by_var]
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "var", k))
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    review = [dict(var=r["var"], reasons=";".join(needs_review(r))) for r in rows if needs_review(r)]
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["var", "reasons"])
        w.writeheader()
        w.writerows(review)
    print(f"완료: {OUT.name} {len(rows)}행, 검수 대상 {len(review)}개 -> {REVIEW.name}")


if __name__ == "__main__":
    main()
