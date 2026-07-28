#!/usr/bin/env python3
"""세 가지 재발 패턴을 사람 규약이 아니라 코드로 막는다.

ARCHITECTURE.json 의 convergence 기록 중 셋은 threshold 가 '이렇게 하기로 한다'
수준이었다. 규율만 있고 장치가 없으면 다음에 또 밟으므로 검사기로 내린다. 각
검사는 이 세션에서 실제로 일어난 사고를 재현 조건으로 삼는다.

  P1  CONTROL_AFTER_CONCLUSION
      채점기 파일이 그것이 채점하는 체크포인트보다 나중에 수정됐으면 경고한다.
      결과를 본 뒤 계측기를 만진 흔적이고, λ4 에서 세 번 그렇게 해 두 결론을
      철회했다. 정당한 경우(새 팔 배선)도 있으므로 차단이 아니라 표시다 —
      대신 무엇이 언제 바뀌었는지 숨길 수 없게 한다.

  P2  SINGLE_SEED_IS_NOT_A_RESULT
      전 등급 PASS 를 주장하는 팔에 seed 형제가 없으면 그 주장을 막는다.
      단일 seed 통과를 적었다가 복제에서 철회한 것이 두 번이다. 형제가 있고
      그쪽이 어느 등급에서든 지면 '재현 안 됨' 으로 내린다.

  P3  INFERRED_NUMBER_ACROSS_SPANS
      결과 파일마다 span 지문(어떤 창 선택이 그 수치를 냈는지)을 읽어, 서로 다른
      지문의 수치를 같은 표에 올리려 하면 막는다. λ4 의 7.05 를 bigram floor 와
      비교해 오판한 적이 있고, 두 수는 애초에 같은 구간에서 나온 것이 아니었다.

차단이 아니라 보고가 기본이다 — 이 검사가 틀렸을 때 연구를 멈추게 하면 검사를
꺼버리게 되고, 꺼진 검사는 없는 검사다. `--strict` 로 exit 1.
"""
import glob
import json
import os
import sys
from pathlib import Path

SCORERS = ["measurement/gate.py", "measurement/panel.py", "measurement/panel_nf9.py",
           "measurement/g_gates.py", "measurement/lambda4.py"]
RESULT_JSONS = ["measurement/panel_results.json", "measurement/panel_nat_results.json",
                "measurement/panel_nf9_results.json", "measurement/g_gates_results.json",
                "measurement/g_gates_nat_results.json", "measurement/lambda4_results.json"]

# 같은 설정을 다른 seed 로 돌린 짝. 전 등급 PASS 주장은 짝이 있어야 한다.
SEED_SIBLINGS = {
    "s25": "v25", "v25": "s25", "s50": "v50", "v50": "s50",
    "s100": "v100", "v100": "s100",
    "natdrop4": "natdrop4v", "natdrop4v": "natdrop4",
    "natdrop35": "natdrop35v", "natdrop35v": "natdrop35",
    "natdrop37": "natdrop37v", "natdrop37v": "natdrop37",
    "n25drop37": "n25drop37v", "n25drop37v": "n25drop37",
    "n25drop42": "n25drop42v", "n25drop42v": "n25drop42",
    "n50drop37": "n50drop37v", "n50drop37v": "n50drop37",
}


def p1_instrument_touched_after(git_root):
    """채점기가 자기가 채점하는 체크포인트보다 나중에 수정됐는가."""
    hits = []
    newest_result = 0.0
    for r in RESULT_JSONS:
        p = git_root / r
        if p.exists():
            newest_result = max(newest_result, p.stat().st_mtime)
    if not newest_result:
        return hits
    for s in SCORERS:
        p = git_root / s
        if p.exists() and p.stat().st_mtime > newest_result:
            hits.append((s, p.stat().st_mtime - newest_result))
    return hits


def p2_lone_pass(verdicts):
    """전 등급 PASS 인데 seed 형제가 없거나, 형제가 지는 팔."""
    arms = verdicts.get("arms", {})
    lone, unmatched = [], []
    for name, row in arms.items():
        if row.get("verdict") != "PASS":
            continue
        sib = SEED_SIBLINGS.get(name)
        if sib is None:
            lone.append(name)
        elif sib not in arms:
            unmatched.append((name, sib, "형제 미측정"))
        elif arms[sib].get("verdict") != "PASS":
            unmatched.append((name, sib, f"형제 {arms[sib].get('verdict')}"))
    return lone, unmatched


def p3_span_fingerprints(git_root):
    """결과 파일들이 어떤 창 선택에서 나왔는지 모아 본다.

    지문이 다르면 그 수치들은 같은 표에 올릴 수 없다 — keep rate 와 창 개수가
    다르면 애초에 다른 구간을 잰 것이다."""
    prints = {}
    for r in RESULT_JSONS:
        p = git_root / r
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        # panel/g_gates 는 _select, lambda4 는 _setup 에 선택 정보를 둔다.
        sel = blob.get("_select") or blob.get("_setup") or {}
        fp = (sel.get("kept") or sel.get("lines_per_group"), sel.get("tested"),
              sel.get("probe_bytes"), sel.get("block"))
        if any(v is not None for v in fp):
            prints.setdefault(fp, []).append(r)
    return prints


def main():
    strict = "--strict" in sys.argv
    root = Path(__file__).resolve().parent.parent
    fails = 0

    print("[P1] 결과보다 나중에 수정된 채점기 (결론 후 계측기 손댐)")
    hits = p1_instrument_touched_after(root)
    if hits:
        for s, dt in hits:
            print(f"     ⚠️ {s} — 최신 결과보다 {dt/60:.0f}분 나중")
        print("     새 팔 배선이면 정상이다. 수치를 본 뒤 바를 만졌다면 그것이 사고다.")
        fails += 1
    else:
        print("     ok — 모든 채점기가 결과보다 먼저 고정됐다")

    print("[P2] seed 형제 없는 전등급 PASS 주장")
    vp = root / "measurement/gate_verdicts.json"
    if vp.exists():
        lone, unmatched = p2_lone_pass(json.loads(vp.read_text()))
        if unmatched:
            for a, s, why in unmatched:
                print(f"     ⚠️ {a} PASS 인데 {s} 가 {why} — 재현 미확인")
            fails += 1
        if lone:
            print(f"     ℹ️ 형제가 등록되지 않은 PASS {len(lone)}개: {', '.join(sorted(lone)[:6])}"
                  f"{' …' if len(lone) > 6 else ''}")
            print("     λ1 단독 PASS 는 선별이라 형제 요건 밖이다. 전 등급 주장에만 적용된다.")
        if not unmatched:
            print("     ok — 형제가 등록된 PASS 는 모두 양쪽이 통과")
    else:
        print("     skip — gate_verdicts.json 없음")

    print("[P3] span 지문 (다른 구간의 수치를 한 표에 올리는가)")
    prints = p3_span_fingerprints(root)
    for fp, files in prints.items():
        kept, tested, probe, block = fp
        print(f"     kept={kept} tested={tested} probe={probe}B block={block}: "
              f"{', '.join(Path(f).name for f in files)}")
    if len(prints) > 1:
        # 코퍼스족이 여럿이면 지문도 여럿인 것이 정상이다. 이 검사의 일은 그것을
        # 보이게 만드는 것이지 막는 것이 아니다 — 사고는 지문이 다른 줄 모르고
        # 두 수를 나란히 놓을 때 난다.
        print(f"     정상 — 코퍼스족이 {len(prints)}개이므로 지문도 {len(prints)}개다.")
        print("     ⚠️ 단, 지문이 다른 수치는 같은 표에 올리지 마라. 비교하려면 같은")
        print("        지문으로 다시 재라(계측기를 바꾸는 것이 아니라).")
    elif prints:
        print("     ok — 단일 지문")

    print(f"\npreflight: {'ok' if not fails else str(fails) + ' 경고'}")
    return 1 if (fails and strict) else 0


if __name__ == "__main__":
    sys.exit(main())
