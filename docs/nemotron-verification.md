# Nemotron CLI公開検証（2026-10-30）

公開main8a64c82から、Nemotron関連のコード・集計・手順だけを分離。他プロジェクトのコード／履歴、私的音声・参照・転記・入力hash、ユーザー固有パス・環境は追加しない。既存ローカル評価8416652を丸ごとcherry-pickしていない。

- `python -m unittest test_nemotron_runtime -v`: 11 tests PASS。通常一覧／両dispatch、prompt境界、必須設定、排他、model inventory/hash、数値出力／上書き拒否、推論失敗、音声形式、token cap、GPU cleanup失敗時のlock解放。
- 専用一時コピーの故障注入2件: cap判定を緩める、lock解放を外す。いずれも対応testが実際に失敗。
- 通常`batch_benchmark.py`から汎用合成「はい」1件（0.429875秒、公開音声なし）で両モデルsmoke完走。前後に共有lockを使用し、他アプリは停止しなかった。

| alias | load秒 | 無音2回暖機秒 | 合成1件推論秒 | 合成CER |
|---|---:|---:|---:|---:|
| nemotron-mlx8 | 15.150 | 0.598 | 0.158 | 50% |
| nemotron-stock | 31.829 | 10.422 | 0.958 | 150% |

短発話の誤認識があり、このsmokeは実行経路の検証に限る。モデル精度の合格や50件の再測定ではない。後続のcleanupをfinallyで確実にする変更は単体テストで確認し、推論計算は変更していない。

既存4構成50件の掲載値は集計CSVと元ローカルsummaryで照合。MLX8のraw/formattedは50件すべて一致していたため、公開CLIの後処理なしとの差を明示した。独立holdout、endpoint/Gijilog E2E、単語timestampは今回の検証範囲外。

共有quality gateはNOT_ADOPTED。正式submit PASSとは呼ばない。公開repoに既存GitHub Actions workflowはなく、今回もCIは新設せずローカル検証を記録する。復旧はこの公開変更のrevertで行い、force pushしない。本番アプリの変更はない。

独立review（QA/security/verifier/value）で公開差分を確認。変換時の監視例外でthreadだけ終了する問題を指摘され、runtimeと共通のfail-closed監視に修正し、観測例外テストを追加。公開smoke記録も確定。推移依存の完全lock・新規環境からの再変換は未実施で、その制限を残す。
