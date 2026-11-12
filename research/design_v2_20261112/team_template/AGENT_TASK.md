# 작업 지시서 (AI 에이전트용)

당신은 KOSMOS 연구팀원 **{name}**의 컴퓨터에서 K-EXAONE 2.5 API 호출 작업을 대신 수행한다. 이 문서대로만 진행하고, 끝나면 결과 zip 파일 하나를 만든다.

## 맡은 일

| 항목 | 값 |
|---|---|
| 담당 번호 | {member} |
| 표본 번호 | {lo}–{hi} (25개) |
| 프롬프트 | `prompts_v2.csv`의 phase 1 행 2,952개 |
| 총 호출 | {expected:,}건 |
| 예상 시간 | 20~37시간 (API 한도 초당 1회, 응답 지연에 따라 달라짐) |
| 비용 | 없음 (무료 API) |

## 지켜야 할 것

1. `prompts_v2.csv`, `run_exaone_team.py`, `config.json`, `finalize.py`를 **수정하지 않는다.** 실행기는 프롬프트 파일의 해시를 검사해서, 바뀌면 실행을 거부한다.
2. API 키를 화면이나 대화에 출력하지 않는다. 결과 zip, 로그, git 커밋에도 넣지 않는다. 키는 이 폴더의 `.env` 파일에만 둔다.
3. 실패한 호출은 실행기가 자동으로 재시도한다. 직접 호출 코드를 새로 쓰지 않는다.
4. 설치는 Python 3.9 이상과 `certifi` 패키지(선택)만 한다. 그 밖의 시스템 설정은 바꾸지 않는다.

## 순서

### 1. 환경 확인
- `python3 --version`(Windows는 `python --version`)으로 3.9 이상인지 확인한다.
- `pip install certifi`를 실행한다. 실패해도 계속 진행해도 된다.

### 2. API 키 넣기
- 사용자에게 **본인 계정의 K-EXAONE API 키**를 요청한다. 키 발급은 사용자가 직접 한다.
- 이 폴더에 `.env` 파일을 만들고 한 줄을 쓴다: `EXAONE_API_KEY=받은키`
- macOS/Linux에서는 `chmod 600 .env`로 권한을 막는다.

### 3. 시험 실행 (3건)
```
python3 run_exaone_team.py --test
```
- 화면에 `시험 1: ... → 답 '3' (유효 True, ok, ...)` 같은 줄이 3개 나오면 정상이다.
- `인증 실패`가 나오면 키를 다시 확인한다. `SSL 오류`나 `인증서 검증 실패`가 나오면 `pip install certifi` 후 다시 한다.

### 4. 본실행 (백그라운드)
대화 창을 닫아도 계속 돌도록 백그라운드로 실행한다. 로그는 실행기가 `exaone_m{member}_p1.log`에 직접 쓴다.

- **macOS**: `nohup caffeinate -i python3 run_exaone_team.py > run_stdout.txt 2>&1 &`
- **Linux**: `nohup python3 run_exaone_team.py > run_stdout.txt 2>&1 &`
- **Windows (PowerShell)**: `Start-Process -WindowStyle Minimized python -ArgumentList "run_exaone_team.py" -RedirectStandardOutput run_stdout.txt -RedirectStandardError run_stderr.txt`
  - 그다음 사용자에게 "전원 연결, 절전 모드 끄기"를 안내한다.

실행 직후 1~2분 뒤 진행 상황을 한 번 확인한다.

### 5. 진행 확인
```
python3 run_exaone_team.py --status
```
- `done`, `total`, `percent`, `complete`가 JSON으로 나온다. 호출은 하지 않는다.
- 사용자에게는 "현재 몇 %, 남은 시간 약 몇 시간" 정도로 짧게 보고한다. 남은 시간은 로그 마지막 줄에 있다.
- 프로세스가 멈춰 있고 `complete`가 false이면, 4단계 명령을 그대로 다시 실행한다. 이미 끝난 건은 건너뛴다.

### 6. 마무리
`complete`가 true가 되면 실행한다.
```
python3 finalize.py
```
- 점검을 통과하면 `KOSMOS_EXAONE_결과_m{member}_{name}.zip`이 생긴다.
- 사용자에게 이 파일 경로를 알려주고, **이 zip 하나만 태우에게 보내라**고 안내한다. 팀 공유 드라이브나 메신저 어느 쪽이든 된다.
- 점검에 실패하면 화면에 나온 이유대로 처리한다. 대개는 남은 건이 있는 경우이므로 4단계를 다시 실행한다.

## 사용자에게 보고할 형식

```
[KOSMOS EXAONE · 담당 {member} {name}]
상태: 진행 중 / 완료
진행: 00,000 / {expected:,} (00%)
유효 응답률: 00.0%
남은 시간: 약 00시간
결과 파일: (완료 시 zip 경로)
```

## 다시 이어서 할 때

사용자가 나중에 "AGENT_TASK.md 이어서 해줘"라고 하면 5단계부터 시작한다.
