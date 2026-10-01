# 今後の追加候補モデル（2026-04 調査）

以下は調査済みで未実装。優先度順。

### S ランク（推奨）

1. ~~**IBM granite-4.0-1b-speech**~~ — 検証済み（上記テーブル）。自声20文では 17.25%。英単語の大小文字扱いが特徴的で、OpenASR の公称 WER とは別系統の CER ペナルティが効く。
2. **Voxtral-Mini-3B-2507** (Mistral) — 3B / Apache 2.0 / Audio-LLM（ASR + Q&A 同時）。Neosophie ベンチで WER 24%。
3. ~~**Cohere-transcribe-03-2026**~~ — 検証済み（上記テーブル / 「Cohere Transcribe の検証結果」）。自声20文では **11.81%**（Qwen3-ASR に次ぐ2位）。英語 OpenASR の SOTA(WER 5.42) は日本語1位には直結しなかったが実用圏。ライセンスは Apache 2.0、リポは gated（規約同意で即時）。

### A ランク

- **Phi-4-multimodal-instruct** (MS, 5.6B, MIT) — 日本語含む8言語学習。
- **Qwen3-Omni-30B-A3B** (Alibaba, MoE active 3B) — Audio/Video/Text フルマルチモーダル。ローカル実行は重い。
- **Meta Omnilingual ASR** (最大 7B) — 1,690 言語対応、fairseq2 必須。
- **litagin/anime-whisper** (Kotoba-Whisper 派生) — アニメ調音声 5,300時間で fine-tune。ドメイン特化事例。
- **sbintuitions/nest-ja-0.1b** — 日本語 35K 時間 SSL、ASR fine-tune のベースとして。

### B ランク（紹介のみ）

Meta SeamlessM4T-v2-Large（CC-BY-NC）、Meta MMS、ESPnet OWSM v4、Fun-ASR-Nano-2512、既存の Japanese HuBERT / wav2vec2 系。

※ Gemma 3n E2B/E4B は後継の **Gemma 4 E2B (audio)** として実装・検証済み（上記「対応モデル」「ベンチマーク実績」を参照）。
