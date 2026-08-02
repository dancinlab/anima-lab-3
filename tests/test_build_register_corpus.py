import json

from measurement.build_register_corpus import build, normalized_lines, record_side


def test_record_split_is_stable():
    assert record_side("gongu:11257842") == record_side("gongu:11257842")
    assert {record_side(f"doc-{i}") for i in range(20)} == {"train", "fresh"}


def test_normalizer_filters_short_and_non_korean_lines():
    rows = list(normalized_lines("짧다.\nThis is a long English-only sentence that is filtered out.\n"
                                 "이 문장은 자연스러운 한국어 산문으로 충분히 길게 작성되었습니다."))
    assert rows == ["이 문장은 자연스러운 한국어 산문으로 충분히 길게 작성되었습니다.".encode()]


def test_build_keeps_documents_disjoint_and_deduplicates_globally(tmp_path):
    source = tmp_path / "source.jsonl"
    records = []
    shared = "공통으로 반복되는 문장이지만 두 출력에는 한 번만 들어가야 합니다."
    for i in range(100):
        text = f"문서 {i}의 서로 다른 한국어 문장으로 자연 문학 자료를 구성합니다.\n{shared}"
        records.append({"id": f"doc-{i}", "text": text})
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records))
    train, fresh = tmp_path / "train.txt", tmp_path / "fresh.txt"

    receipt = build(source, train, fresh, train_target_bytes=900,
                    fresh_target_bytes=700, train_fraction=0.6)

    train_lines = set(train.read_bytes().splitlines())
    fresh_lines = set(fresh.read_bytes().splitlines())
    assert train_lines
    assert fresh_lines
    assert train_lines.isdisjoint(fresh_lines)
    assert receipt["duplicate_lines_dropped"] > 0
    assert receipt["sizes"]["train"] >= 499
    assert receipt["sizes"]["fresh"] >= 299
