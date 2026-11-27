"""미리보기용: 한국외대체가 없는 환경에서 Apple SD Gothic Neo로 바꿔 PDF로 렌더링하고, 목차 쪽 번호를 뽑는다."""
import sys, zipfile, re, json, subprocess, pathlib
src, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
z = zipfile.ZipFile(src); tmp = outdir / "preview.docx"
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
    for n in z.namelist():
        b = z.read(n)
        if n.endswith(".xml"):
            for w in "LMB":
                b = b.replace(("한국외대체 " + w).encode(), f"AppleSDGothicNeo{w}00".encode())
        zo.writestr(n, b)
import os
prof = os.environ.get("LO_PROFILE")
extra = [f"-env:UserInstallation=file://{prof}"] if prof else []
subprocess.run(["soffice", *extra, "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(tmp)], capture_output=True)
