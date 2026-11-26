"""Verification 덱용 표 이미지 (Canva 표 셀은 API로 못 고쳐서 덱 표 스타일을 흉내 낸 PNG로 넣는다).
숫자 출처: experiments/20261126-haiku-think-original/out/*.csv, experiments/20261126-reasoning-concentration·reasoning-shift README(DeepSeek k=50),
smoke_haiku_cot/summary.txt·run.log, 덱 01 Review 표(Haiku low·medium 기존 스모크).
실행: uv run --no-project --with pillow python research/design_v2_20261112/smoke_haiku_cot/slides/make_tables.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
S = 2                                   # 2배 해상도
FONT = Path.home() / "Library/Fonts"
f = lambda w, size: ImageFont.truetype(str(FONT / f"Pretendard-{w}.otf"), size * S)
INK, RED, BLUE, LINE, HEAD, SIDE, WHITE = "#263035", "#ff3131", "#2f6fde", "#d6d9d8", "#e6e7e7", "#eeefef", "#ffffff"
W_TOTAL = 1740


def segs(x):
    """셀 내용: 문자열, 또는 [(텍스트, 색), ...]"""
    return [(x, INK)] if isinstance(x, str) else x


def wrap(segments, font, width, draw):
    """색 구간을 유지하며 줄바꿈. 반환: [[(텍스트, 색)], ...]"""
    lines, cur, cur_w = [], [], 0
    for text, color in segments:
        for tok in text.replace("\n", " \n ").split(" "):
            if tok == "\n":
                lines.append(cur); cur, cur_w = [], 0; continue
            if tok == "":
                continue
            piece = (tok if not cur else " " + tok)
            pw = draw.textlength(piece, font=font)
            if cur and cur_w + pw > width:
                lines.append(cur); cur, cur_w = [], 0
                piece, pw = tok, draw.textlength(tok, font=font)
            cur.append((piece, color)); cur_w += pw
    if cur:
        lines.append(cur)
    return lines


def table(name, cols, header, rows, caption=None, row_h=86, head_h=78, size=24, head_size=22,
          side=True, merge_first=None, cap_size=30, align_first="center"):
    """cols: 폭 비율. rows: 셀 리스트. merge_first: [(시작행, 행수)] 첫 열 병합."""
    total = sum(cols)
    widths = [c / total * W_TOTAL for c in cols]
    img = Image.new("RGB", (W_TOTAL * S, 2000 * S), "#f6f6f6")
    d = ImageDraw.Draw(img)
    fb, fh, fc = f("SemiBold", size), f("Bold", head_size), f("Bold", cap_size)
    # 행 높이(줄바꿈 반영)
    heights = []
    for r in rows:
        n = max(len(wrap(segs(c), fb, (w - 28) * S, d)) for c, w in zip(r, widths))
        heights.append(max(row_h, 24 + n * size * 1.45))
    y = 0
    xs = [sum(widths[:i]) for i in range(len(widths) + 1)]

    def cell(x0, x1, y0, y1, content, font, fill, align="center"):
        d.rectangle([x0 * S, y0 * S, x1 * S, y1 * S], fill=fill)
        lines = wrap(segs(content), font, (x1 - x0 - 28) * S, d)
        lh = font.size * 1.45
        ty = (y0 + y1) / 2 * S - lh * len(lines) / 2 + (lh - font.size) / 2 - font.size * 0.08
        for ln in lines:
            lw = sum(d.textlength(t, font=font) for t, _ in ln)
            tx = (x0 + 14) * S if align == "start" else ((x0 + x1) / 2 * S - lw / 2)
            for t, c in ln:
                d.text((tx, ty), t, font=font, fill=c)
                tx += d.textlength(t, font=font)
            ty += lh

    for i, h in enumerate(header):
        cell(xs[i], xs[i + 1], 0, head_h, h, fh, HEAD)
    y = head_h
    skip = set()
    ys = [head_h]
    for h in heights:
        ys.append(ys[-1] + h)
    merges = {s: n for s, n in (merge_first or [])}
    for ri, r in enumerate(rows):
        for ci, c in enumerate(r):
            if ci == 0 and ri in skip:
                continue
            y0, y1 = ys[ri], ys[ri + 1]
            if ci == 0 and ri in merges:
                y1 = ys[ri + merges[ri]]
                skip.update(range(ri + 1, ri + merges[ri]))
            fill = SIDE if (side and ci == 0) else WHITE
            cell(xs[ci], xs[ci + 1], y0, y1, c, fb, fill, align_first if ci == 0 else "center")
    bottom = ys[-1]
    # 선: 가로선(병합 칸 안쪽은 첫 열 건너뜀), 세로선, 바깥선
    for ri in range(len(rows) + 1):
        yy = ys[ri] if ri < len(ys) else bottom
        x_start = xs[1] if ri in skip else 0
        d.line([x_start * S, yy * S, W_TOTAL * S, yy * S], fill=LINE, width=2 * S)
    d.line([0, 0, W_TOTAL * S, 0], fill=LINE, width=2 * S)
    for x in xs:
        d.line([min(x, W_TOTAL - 1) * S, 0, min(x, W_TOTAL - 1) * S, bottom * S], fill=LINE, width=2 * S)
    d.line([0, bottom * S, W_TOTAL * S, bottom * S], fill=LINE, width=2 * S)
    H = bottom
    if caption:
        cy = bottom + 44
        for seg_line in caption:
            lines = wrap(segs(seg_line), fc, W_TOTAL * S, d)
            for ln in lines:
                lw = sum(d.textlength(t, font=fc) for t, _ in ln)
                tx = W_TOTAL / 2 * S - lw / 2
                for t, c in ln:
                    d.text((tx, cy * S), t, font=fc, fill=c)
                    tx += d.textlength(t, font=fc)
                cy += cap_size * 1.6
        H = cy
    img = img.crop((0, 0, W_TOTAL * S, int(H * S) + 4))
    img.save(OUT / f"{name}.png")
    print(name, img.size)


# 1. 피드백 1: 생각 지시 프롬프트
table("t1_feedback_prompt", [2.2, 2.6, 1.6, 1.6], ["설정", "시스템 프롬프트", "생각한 호출", "호출당 출력 평균"], [
    ["Haiku 추론 low", "기존 (숫자 하나만 출력)", [("1 / 300", RED)], "3.8토큰"],
    ["Haiku 추론 medium", "기존 (숫자 하나만 출력)", [("17 / 300", RED)], "10.1토큰"],
    ["Haiku 추론 low", "생각 지시", "38 / 50", "97토큰"],
    ["Haiku 추론 medium", "생각 지시", [("297 / 300", BLUE)], "142토큰"],
    ["Haiku 추론 high", "생각 지시", [("50 / 50", BLUE)], "182토큰"],
], caption=[[("'숫자 하나만 출력' 지시가 생각까지 막고 있었다", INK)],
            [("→ 생각을 지시하면 Claude도 거의 매번 추론한다", RED)]])

# 2. 피드백 2: DeepSeek 추론 수준별 (k=50, 197문항)
table("t2_feedback_high", [3.2, 1.3, 1.3, 1.3, 1.3], ["지표 (DeepSeek, 성별×연령)", "추론 끔", "low", "medium", "high"], [
    ["보정 후 사람 TVD", "0.284", "0.278", "0.277", "0.277"],
    ["최빈 보기 비율 (쏠림)", "0.813", "0.827", [("0.842", RED)], [("0.841", RED)]],
    ["직전 단계 대비 잡음 밖 흔들림", "–", "52%", "25%", [("3%", BLUE)]],
], caption=[[("변화는 medium까지, ", INK), ("medium과 high는 같은 분포", RED)],
            [("잡음만 있어도 5% · Haiku medium → high도 3~6%", INK)]], row_h=96)

# 3. 실험 설정
table("t3_setup", [1.5, 4.5], ["항목", "내용"], [
    ["기존 시스템 프롬프트", "보기 번호 숫자 하나만 출력하고 다른 말은 하지 마십시오."],
    ["생각 지시 프롬프트", [("답하기 전에 응답자의 입장에서 어떤 보기를 고를지 ", INK), ("충분히 생각하십시오.", RED),
                     (" 생각을 마친 뒤 최종 출력에는 보기 번호 숫자 하나만 적고 다른 말은 하지 마십시오.", INK)]],
    ["추론 설정", "Claude Haiku 5.5 · adaptive thinking · effort medium / high"],
    ["문항 · 페르소나", "test 제외 197문항 · 페르소나 없음 / 한국 성인 / 성별×연령 10셀"],
    ["반복 수", [("프롬프트당 k = 20", RED), ("  (DeepSeek 재표본: 197문항이면 k = 10부터 추론 효과 검출)", INK)]],
    ["수집", "94,560건 · 유효 99.9% · 생각 블록 99.5~100% · $4.91"],
], row_h=88, size=24)

# 4. 분석 방법
table("t4_method", [1.6, 2.2, 4.2], ["질문", "지표", "뜻"], [
    ["얼마나 바뀌나", "전후 TVD\n잡음 밖 흔들림", "같은 페르소나·같은 문항의 추론 전후 분포 거리\n잡음만으로 나올 수준을 넘는 비율 (분포가 같으면 5%)"],
    ["1위 보기", "최빈 보기 뒤바뀜\n사람 쪽 순변화", "1위 보기가 바뀐 비율\n사람 1위 보기와 맞는 문항이 늘어난 만큼"],
    ["쏠림", "최빈 보기 비율", "1위 보기에 몰린 응답 비율 (사람 0.54)"],
    ["사람과 가까워졌나", "보정 후 사람 TVD", "β 보정 후 사람 분포와의 거리 (validation 49문항)"],
], caption=[[("모든 설정 k = 20으로 맞춤 · 문항 부트스트랩 2,000회 95% 구간", INK)]], row_h=110, size=23)

# 5. 결과: 얼마나 바뀌나
table("t5_shift", [1.3, 1.6, 1.5, 1.5, 1.5, 1.8], ["모델", "비교", "페르소나 없음", "한국 성인", "성별×연령", "최빈 보기 뒤바뀜\n(성별×연령)"], [
    ["Haiku", "끔 → medium", [("0.361", RED), (" (49%)", INK)], [("0.347", RED), (" (44%)", INK)], [("0.329", RED), (" (43%)", INK)], [("32%", RED)]],
    ["", "medium → high", "0.079 (6%)", "0.049 (4%)", "0.051 (3%)", "3%"],
    ["DeepSeek", "끔 → medium", [("0.301", RED), (" (44%)", INK)], [("0.238", RED), (" (34%)", INK)], [("0.271", RED), (" (39%)", INK)], [("28%", RED)]],
    ["", "medium → high", "0.054 (2%)", "0.065 (2%)", "0.071 (2%)", "5%"],
], caption=[[("전후 TVD (잡음 밖 흔들림 비율) · 잡음 TVD 0.04~0.09", INK)],
            [("추론을 켜면 분포가 크게 바뀐다 → ", INK), ("medium과 high는 같은 분포", RED)]],
   merge_first=[(0, 2), (2, 2)], row_h=96, head_h=96)

# 6. 결과: 쏠림
table("t6_concentration", [1.3, 1.3, 1.6, 1.6, 1.6], ["모델", "설정", "페르소나 없음", "한국 성인", "성별×연령"], [
    ["Haiku", "추론 끔", "0.905", "0.898", "0.923"],
    ["", "medium", "0.877", [("0.929", RED)], "0.926"],
    ["", "high", "0.876", [("0.927", RED)], "0.921"],
    ["DeepSeek", "추론 끔", "0.841", "0.832", "0.819"],
    ["", "medium", [("0.896", RED)], [("0.864", RED)], [("0.845", RED)]],
    ["", "high", [("0.904", RED)], "0.856", [("0.845", RED)]],
    ["사람", "", "0.527", "0.527", "0.540"],
], caption=[[("최빈 보기 비율 · ", INK), ("빨강", RED), (": 추론 끔 대비 유의하게 증가 (95% 구간)", INK)],
            [("Haiku는 이미 강하게 쏠려 있고 더 쏠리지 않는다 · DeepSeek은 더 쏠린다", INK)]],
   merge_first=[(0, 3), (3, 3)], row_h=70, head_h=74, cap_size=28)

# 7. 결과: 사람 쪽인가
table("t7_toward_human", [1.3, 2.6, 1.5, 1.5, 1.5], ["모델", "지표 (추론 끔 → high)", "페르소나 없음", "한국 성인", "성별×연령"], [
    ["Haiku", "사람 쪽 순변화", "−0.017", "−0.021", "−0.029"],
    ["", "보정 후 사람 TVD 변화", [("−0.030", RED)], "−0.021", "−0.002"],
    ["DeepSeek", "사람 쪽 순변화", "+0.020", "+0.056", [("+0.079", RED)]],
    ["", "보정 후 사람 TVD 변화", "+0.009", "+0.007", "−0.007"],
], caption=[[("빨강", RED), (": 95% 구간이 0 밖 · 순변화 + = 사람 1위 보기와 맞는 문항 증가 · TVD − = 사람에 가까워짐", INK)],
            [("Haiku의 1위 보기는 3분의 1이 바뀌지만 사람 쪽도 반대쪽도 아니다", INK)]],
   merge_first=[(0, 2), (2, 2)], row_h=96, cap_size=28)
