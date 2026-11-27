"""미리보기 PDF에서 장·절·표·그림이 나오는 쪽을 찾아 toc_pages.json으로 쓴다."""
import sys, json, pymupdf, content as C
d = pymupdf.open(sys.argv[1]); texts = [p.get_text().replace("\n", " ") for p in d]
start = next(i for i, t in enumerate(texts) if i > 3 and "제 1장 서론" in t and "연구 배경" in t and "목 차" not in t)
pages = {}
def find(key, lab):
    for i in range(start, len(d)):
        if lab in texts[i]:
            pages[key] = i - start + 1; return
for b in C.T:
    if b[0] in ("ch", "sec"): find(b[1], b[1])
    if b[0] == "table": find("표 " + b[1], "[표 %s] %s" % (b[1], b[2][:8]))
    if b[0] == "fig": find("그림 " + b[1], "[그림 %s] %s" % (b[1], b[2][:8]))
for lab in ("참고문헌", "국내문헌", "국외문헌"):
    for i in range(len(d) - 1, start, -1):
        if lab in texts[i] and (lab != "참고문헌" or "국내문헌" in texts[i]):
            pages[lab] = i - start + 1; break
print(len(d), start, pages); json.dump(pages, open("toc_pages.json", "w"), ensure_ascii=False, indent=1)
