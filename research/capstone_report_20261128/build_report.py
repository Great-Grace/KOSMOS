# -*- coding: utf-8 -*-
"""DAT 캡스톤 보고서 양식(docx)에 content.py 원고를 끼워 넣는다.

사용: python build_report.py <양식.docx> <출력.docx> [toc_pages.json]
양식의 표지·속표지·요약·목차·본문 서식(글꼴 한국외대체, 크기, 쪽 번호 꼬리말)을 그대로 쓰고,
본문 문단은 양식의 장·절 문단을 본떠 새로 만든다.
"""
import json, re, shutil, sys, zipfile, pathlib, struct
from xml.sax.saxutils import escape

import content as C

HERE = pathlib.Path(__file__).parent
TPL, OUT = sys.argv[1], sys.argv[2]
PAGES = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}

z = zipfile.ZipFile(TPL)
files = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
doc = files["word/document.xml"].decode("utf-8")

head, body = doc[: doc.index("<w:body>") + len("<w:body>")], doc[doc.index("<w:body>") + len("<w:body>"):]
tail = body[body.rindex("</w:body>"):]
body = body[: body.rindex("</w:body>")]
paras = re.findall(r"<w:p[ >].*?</w:p>|<w:sectPr.*?</w:sectPr>", body, flags=re.S)
assert "".join(paras) == body, "문단 분해가 본문과 다르다"
assert len(paras) == 104, len(paras)

# ───────── 글자 단위 도우미 ─────────
def rpr(font, sz, bold=False, sup=None):
    b = "<w:b/><w:bCs/>" if bold else ""
    v = f'<w:vertAlign w:val="{sup}"/>' if sup else ""
    return (f'<w:rPr><w:rFonts w:ascii="{font}" w:eastAsia="{font}" w:hAnsi="{font}" w:cs="{font}" w:hint="eastAsia"/>'
            f'{b}<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>{v}</w:rPr>')

def run(text, font, sz, bold=False, sup=None):
    out = []
    for i, part in enumerate(text.split("\n")):
        if i: out.append(f"<w:r>{rpr(font, sz, bold, sup)}<w:br/></w:r>")
        if part:
            out.append(f'<w:r>{rpr(font, sz, bold, sup)}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return "".join(out)

TOK = re.compile(r"(\*\*.+?\*\*|\^\{[^}]*\}|_\{[^}]*\})")
def runs(text, font, sz, bold=False, boldfont=None):
    out = []
    for tok in TOK.split(text):
        if not tok: continue
        if tok.startswith("**"):
            out.append(run(tok[2:-2], boldfont or font, sz, bold=boldfont is None))
        elif tok.startswith("^{"):
            out.append(run(tok[2:-1], font, sz, bold, "superscript"))
        elif tok.startswith("_{"):
            out.append(run(tok[2:-1], font, sz, bold, "subscript"))
        else:
            out.append(run(tok, font, sz, bold))
    return "".join(out)

L, M, B = "한국외대체 L", "한국외대체 M", "한국외대체 B"
BASE = "<w:widowControl/><w:wordWrap/><w:autoSpaceDE/><w:autoSpaceDN/>"

def para(inner, ppr_extra="", font=L, sz=22):
    pre = ""
    for tag in ("<w:keepNext/>", "<w:pageBreakBefore/>"):  # 스키마 순서: keepNext, pageBreakBefore, widowControl …
        if tag in ppr_extra:
            pre += tag; ppr_extra = ppr_extra.replace(tag, "")
    return f"<w:p><w:pPr>{pre}{BASE}{ppr_extra}{rpr(font, sz)}</w:pPr>{inner}</w:p>"

def P_body(t):
    return para(runs(t, L, 22, boldfont=M),
                '<w:spacing w:after="120" w:line="384" w:lineRule="auto"/><w:ind w:firstLine="220"/><w:jc w:val="both"/>')

def P_li(t):
    return para(runs(t, L, 22, boldfont=M),
                '<w:spacing w:after="80" w:line="384" w:lineRule="auto"/><w:ind w:left="440" w:hanging="220"/><w:jc w:val="both"/>')

def P_eq(t):
    return para(runs(t, L, 22), '<w:spacing w:before="120" w:after="200" w:line="360" w:lineRule="auto"/><w:jc w:val="center"/>')

def P_ch(t, first):
    pb = "" if first else "<w:pageBreakBefore/>"
    return para(run(t, B, 32), f'{pb}<w:keepNext/><w:spacing w:after="360"/><w:outlineLvl w:val="0"/>', B, 32)

def P_sec(t):
    return para(run(t, M, 28), '<w:keepNext/><w:spacing w:before="360" w:after="200"/><w:outlineLvl w:val="1"/>', M, 28)

def P_h3(t):
    return para(run(t, M, 24), '<w:keepNext/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="2"/>', M, 24)

def P_caption(t, before=240, after=120, keep=True):
    k = "<w:keepNext/>" if keep else ""
    return para(runs(t, M, 20), f'{k}<w:spacing w:before="{before}" w:after="{after}"/><w:jc w:val="center"/>', M, 20)

def P_note(t):
    return para(runs(t, L, 17), '<w:spacing w:before="60" w:after="240" w:line="300" w:lineRule="auto"/><w:jc w:val="both"/>', L, 17)

# ───────── 표 ─────────
TEXT_W = 11906 - 1440 - 1440  # 9026 dxa

def table(header, rows, ratio):
    tot = sum(ratio)
    ws = [int(TEXT_W * r / tot) for r in ratio]
    ws[-1] = TEXT_W - sum(ws[:-1])
    def cell(t, w, hdr, first, keep):
        shade = '<w:shd w:val="clear" w:color="auto" w:fill="EDEDED"/>' if hdr else ""
        jc = "left" if (first or len(t) > 18) and not hdr else "center"
        inner = runs(t, M if hdr else L, 18)
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shade}<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr>{"<w:keepNext/>" if keep else ""}{BASE}<w:spacing w:before="40" w:after="40" w:line="264" w:lineRule="auto"/><w:jc w:val="{jc}"/>{rpr(L,18)}</w:pPr>{inner}</w:p></w:tc>')
    trs = []
    for ri, r in enumerate([header] + rows):
        hdr = ri == 0
        trpr = '<w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' if hdr else '<w:trPr><w:cantSplit/></w:trPr>'
        trs.append("<w:tr>" + trpr + "".join(cell(t, ws[ci], hdr, ci == 0, ri < len(rows)) for ci, t in enumerate(r)) + "</w:tr>")
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in ws)
    borders = ('<w:tblBorders><w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
               '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
               '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
               '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/></w:tblBorders>')
    return (f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_W}" w:type="dxa"/><w:jc w:val="center"/>{borders}'
            f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
            f'<w:tblLook w:val="04A0"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')

# ───────── 그림 ─────────
media, rels_add = [], []
def png_size(p):
    b = open(p, "rb").read(24)
    return struct.unpack(">II", b[16:24])

def figure(path, width_cm, idx):
    rid = f"rIdKosImg{idx}"
    name = f"media/kosmos_fig{idx}.png"
    media.append((f"word/{name}", (HERE / path).read_bytes()))
    rels_add.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{name}"/>')
    w, h = png_size(HERE / path)
    cx = int(width_cm * 360000); cy = int(cx * h / w)
    drawing = (f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
               f'<wp:docPr id="{900+idx}" name="그림 {idx}"/><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
               f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               f'<pic:nvPicPr><pic:cNvPr id="{900+idx}" name="kosmos_fig{idx}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
               f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
               f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
               f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>')
    return f'<w:p><w:pPr><w:keepNext/>{BASE}<w:spacing w:before="240" w:after="60"/><w:jc w:val="center"/></w:pPr><w:r>{drawing}</w:r></w:p>'

# ───────── 본문 조립 ─────────
out_body, toc, tlist, flist = [], [], [], []
first_ch, figi = True, 0
for blk in C.T:
    k = blk[0]
    if k == "ch":
        out_body.append(P_ch(blk[1], first_ch)); first_ch = False; toc.append(("ch", blk[1]))
    elif k == "sec":
        out_body.append(P_sec(blk[1])); toc.append(("sec", blk[1]))
    elif k == "h3":
        out_body.append(P_h3(blk[1]))
    elif k == "p":
        out_body.append(P_body(blk[1]))
    elif k == "li":
        out_body.append(P_li(blk[1]))
    elif k == "eq":
        out_body.append(P_eq(blk[1]))
    elif k == "table":
        _, no, cap, hdr, rows, ratio, note = blk
        out_body.append(P_caption(f"[표 {no}] {cap}"))
        out_body.append(table(hdr, rows, ratio))
        out_body.append(P_note(note) if note else para("", '<w:spacing w:after="200"/>'))
        tlist.append((no, cap))
    elif k == "fig":
        _, no, cap, path, wcm = blk
        figi += 1
        out_body.append(figure(path, wcm, figi))
        out_body.append(P_caption(f"[그림 {no}] {cap}", before=60, after=280, keep=False))
        flist.append((no, cap))

# 참고문헌 (양식 102번 문단 서식: 가운데 굵게 14pt, 새 쪽)
def P_refhead(t):
    return para(run(t, M, 24), '<w:keepNext/><w:spacing w:before="240" w:after="160"/>', M, 24)
def P_ref(t):
    return para(runs(t, L, 20), '<w:spacing w:after="100" w:line="336" w:lineRule="auto"/><w:ind w:left="440" w:hanging="440"/><w:jc w:val="left"/>', L, 20)
ref = [para(run("참고문헌", L, 28, bold=True), '<w:pageBreakBefore/><w:spacing w:after="240"/><w:jc w:val="center"/><w:outlineLvl w:val="0"/>', L, 28)]
ref.append(P_refhead("국내문헌")); ref += [P_ref(t) for t in C.REFS_KO]
ref.append(P_refhead("국외문헌")); ref += [P_ref(t) for t in C.REFS_EN]
toc += [("ch", "참고문헌"), ("sec", "국내문헌"), ("sec", "국외문헌")]

# ───────── 표지·요약의 자리 글 바꾸기 ─────────
def set_text(p, text):
    """문단의 첫 글자 run 서식을 살리고 글만 바꾼다."""
    rs = re.findall(r"<w:r[ >](?:(?!</w:r>).)*?<w:t[ >].*?</w:r>", p, flags=re.S)
    if not rs:
        return p
    first = rs[0]
    rp = re.search(r"<w:rPr>.*?</w:rPr>", first, flags=re.S)
    rp = rp.group(0) if rp else ""
    rp = rp.replace("<w:lastRenderedPageBreak/>", "")
    new = f'<w:r>{rp}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    p = p.replace(first, "\u0000", 1)
    for r_ in rs[1:]:
        p = p.replace(r_, "", 1)
    p = re.sub(r"<w:proofErr [^>]*/>", "", p)
    return p.replace("\u0000", new)

def clone_with(p, text):
    return set_text(p, text)

for i in (0, 21, 42):
    paras[i] = set_text(paras[i], f"{C.GEN} 캡스톤 프로젝트 보고서")
for i in (3, 24, 45):
    paras[i] = set_text(paras[i], C.TITLE1)
for i in (4, 25, 46):
    paras[i] = set_text(paras[i], C.TITLE2)
for i in (5, 26, 47):
    paras[i] = set_text(paras[i], C.SUBTITLE)
paras[28] = set_text(paras[28], C.EN1)
paras[29] = set_text(paras[29], C.EN2)
paras[36] = set_text(paras[36], C.DATE)
for i in (17, 38, 63):
    paras[i] = set_text(paras[i], C.GEN)
for i in (18, 39, 64):
    paras[i] = set_text(paras[i], C.TEAM)
for i in (19, 40, 65):
    paras[i] = set_text(paras[i], C.MEMBERS)

# 요약: 50번(국문) 서식으로 문단을 늘리고 빈 줄 51~54는 하나만 남긴다
def abstract_paras(proto, texts):
    out = []
    for t in texts:
        q = set_text(proto, t)
        q = q.replace('<w:ind w:left="800" w:firstLineChars="100" w:firstLine="180"/>',
                      '<w:spacing w:after="60" w:line="240" w:lineRule="auto"/><w:ind w:left="800" w:firstLineChars="100" w:firstLine="180"/><w:jc w:val="both"/>')
        out.append(q)
    return out
ko = abstract_paras(paras[50], C.ABSTRACT_KO)
en = abstract_paras(paras[57], C.ABSTRACT_EN)
paras[50] = "".join(ko); paras[51:55] = ["", "", "", ""]; paras[48] = ""; paras[62] = ""; paras[55] = para("", "<w:spacing w:after=\"120\"/>", L, 18)
paras[57] = "".join(en); paras[58:62] = ["", "", "", ""]

# 목차
paras[67] = set_text(paras[67], "목 차")
def toc_line(proto, label, page):
    q = set_text(proto, label)
    rp = re.search(r"<w:rPr>.*?</w:rPr>", re.search(r"<w:r>.*?</w:r>", q, flags=re.S).group(0), flags=re.S).group(0)
    return q.replace("</w:p>", f"<w:r>{rp}<w:tab/><w:t>{page}</w:t></w:r></w:p>")
toc_xml = []
for kind, label in toc:
    pg = PAGES.get(label, "")
    if kind == "ch":
        if toc_xml: toc_xml.append(paras[72])
        toc_xml.append(toc_line(paras[69], label, pg))
    else:
        toc_xml.append(toc_line(paras[70], label, pg))
paras[69] = "".join(toc_xml)
for i in range(70, 88):
    paras[i] = ""

# 표·그림 목차 (97번 문단에 구역 나눔이 있으므로 마지막 그림 항목에 옮긴다)
sect97 = re.search(r"<w:sectPr.*?</w:sectPr>", paras[97], flags=re.S).group(0)
short = lambda c: re.split(r" \(|\. ", c)[0]
tl = [toc_line(paras[90], f"[표 {no}] {short(cap)}", PAGES.get(f"표 {no}", "")) for no, cap in tlist]
paras[90] = "".join(tl)
paras[91] = paras[92] = paras[93] = ""
fl = [toc_line(paras[96], f"[그림 {no}] {short(cap)}", PAGES.get(f"그림 {no}", "")) for no, cap in flist]
fl[-1] = fl[-1].replace("<w:pPr>", "<w:pPr>", 1)
fl[-1] = re.sub(r"(<w:pPr>.*?)(<w:rPr>)", lambda m: m.group(1) + m.group(2), fl[-1], count=1, flags=re.S)
fl[-1] = fl[-1].replace("</w:rPr></w:pPr>", "</w:rPr>" + sect97 + "</w:pPr>", 1)
paras[96] = "".join(fl)
paras[97] = ""

# 본문: 98~102를 새 본문으로 바꾸고 마지막 구역 설정(103)은 그대로 둔다
paras[98] = "".join(out_body) + "".join(ref)
for i in (99, 100, 101, 102):
    paras[i] = ""

new_doc = head + "".join(paras) + tail
for ns, uri in [("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
                ("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")]:
    assert f'xmlns:{ns}="' in head, ns
files["word/document.xml"] = new_doc.encode("utf-8")

rels = files["word/_rels/document.xml.rels"].decode("utf-8")
files["word/_rels/document.xml.rels"] = rels.replace("</Relationships>", "".join(rels_add) + "</Relationships>").encode("utf-8")
for n, b in media:
    files[n] = b
ct = files["[Content_Types].xml"].decode("utf-8")
if 'Extension="png"' not in ct:
    ct = ct.replace("<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1)
files["[Content_Types].xml"] = ct.encode("utf-8")

# 문서 속성 제목
core = files["docProps/core.xml"].decode("utf-8")
core = re.sub(r"<dc:title>.*?</dc:title>|<dc:title/>", f"<dc:title>{escape(C.TITLE1 + ' ' + C.TITLE2)}</dc:title>", core)
files["docProps/core.xml"] = core.encode("utf-8")

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zo:
    for n in z.namelist():
        if n in files: zo.writestr(n, files[n])
    for n, b in media:
        if n not in z.namelist(): zo.writestr(n, b)
print("wrote", OUT, "| 장·절", len(toc), "| 표", len(tlist), "| 그림", len(flist))
