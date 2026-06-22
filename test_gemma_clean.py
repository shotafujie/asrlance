#!/usr/bin/env python3
"""
test_gemma_clean.py - Gemma 4 E2B 出力クリーンアップの単体テスト

Gemma 4 E2B は純粋な ASR ではなくマルチモーダル LLM のため、書き起こし本文の前に
前置き（"Here is the transcription:" / "書き起こし：" 等）や引用符・コードフェンスを
付けることがある。clean_gemma_transcription はこれらを除去して本文のみを返す。

実行: uv run --no-sync python test_gemma_clean.py
"""

from fileRecognizer import clean_gemma_transcription


CASES = [
    # (説明, 入力, 期待出力)
    ("素の本文はそのまま", "今日はいい天気ですね。", "今日はいい天気ですね。"),
    ("前後空白の除去", "  今日はいい天気ですね。  ", "今日はいい天気ですね。"),
    ("日本語インライン前置き", "書き起こし：今日はいい天気ですね。", "今日はいい天気ですね。"),
    ("日本語の前置き文（複数行）", "以下が書き起こしです。\n今日はいい天気ですね。", "今日はいい天気ですね。"),
    ("英語の前置き行", "Here is the transcription:\n今日はいい天気ですね。", "今日はいい天気ですね。"),
    ("コードフェンス除去", "```\n今日はいい天気ですね。\n```", "今日はいい天気ですね。"),
    ("鉤括弧の除去", "「今日はいい天気ですね。」", "今日はいい天気ですね。"),
    ("英語前置き＋引用符", 'Sure! The transcription is: "今日はいい天気ですね。"', "今日はいい天気ですね。"),
    ("空文字", "", ""),
    # 本文に「音声」等のキーワードが含まれても誤除去しない
    ("本文中のキーワードを誤除去しない", "音声認識は便利です。", "音声認識は便利です。"),
    ("コロンを含む本文を保持", "午後三時：会議を始めます。", "午後三時：会議を始めます。"),
]


def main() -> int:
    failed = 0
    for desc, src, expected in CASES:
        got = clean_gemma_transcription(src)
        ok = got == expected
        mark = "OK " if ok else "NG "
        print(f"[{mark}] {desc}: {src!r} -> {got!r}")
        if not ok:
            print(f"        expected: {expected!r}")
            failed += 1
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
