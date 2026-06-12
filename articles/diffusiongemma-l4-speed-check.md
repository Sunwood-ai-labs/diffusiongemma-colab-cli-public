---
title: "DiffusionGemmaは本当に速いのか？Colab L4で通常Gemma4と比較してわかったこと"
emoji: "🧪"
type: tech
topics:
  - gemma
  - diffusiongemma
  - colab
  - llamacpp
  - benchmark
published: false
---

## 結論

DiffusionGemmaの「速い」は、通常のLLMが1 tokenずつ左から生成する速度と、単純に同じ意味ではありません。

今回、Google Colab L4上でDiffusionGemmaの既存実測値と、通常Gemma4の正しい非会話baselineを比較しました。結果として、少なくとも今回のL4環境では「DiffusionGemmaが通常Gemma4より遅い」とは言えませんでした。

ただし、Googleが公式に掲げる「最大4倍」「H100で1000 tok/s超」の再現まではできていません。そこは次の実験で、vLLMまたは公式推奨構成に寄せて検証する必要があります。

## TL;DR

- DiffusionGemmaは256 tokenのcanvasを並列にdenoiseするため、通常のautoregressive decodeとは速度の意味が違う
- Googleの高速性主張は、低バッチ・単一GPU・専用アクセラレータでの並列生成が前提
- Colab L4 + GGUF + llama.cpp開発PRでは、公式のH100/RTX 5090級の速度は再現していない
- 今回の通常Gemma4 baselineは `llama-completion --jinja -no-cnv` が正解だった
- `llama-cli -no-cnv` は対話モードに入り、速度測定として破綻した
- 近いpromptでは、DiffusionGemmaが56.3 tok/s、通常Gemma4 completion decodeが53.01 tok/sだった
- 次回はvLLM/公式sampler条件で、DiffusionGemmaの本命速度を検証する

## 何を検証したかったのか

DiffusionGemmaは、Googleが公開した実験的なtext diffusionモデルです。通常のLLMのように1 tokenずつ生成するのではなく、token blockを並列に生成・修正します。

GoogleのDeveloper Guideでは、DiffusionGemmaはGPU上で最大4倍高速、H100では1000 tok/s超、RTX 5090では700 tok/s超と説明されています。

公式ドキュメント上の重要な前提は次の通りです。

- 256 token canvasを並列にdenoiseする
- 低バッチ・単一ユーザー・専用GPUで強い
- 高QPSのcloud servingでは、通常のautoregressive modelがbatchingで追いつく場合がある
- 高速性と引き換えに、品質ベンチマークでは通常Gemma4に劣る項目がある

つまり、単に「いつものllama.cppのtok/s」と同じ測り方をすると、比較軸を間違えます。

参考:

- https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- https://ai.google.dev/gemma/docs/diffusiongemma
- https://ai.google.dev/gemma/docs/diffusiongemma/model_card

## 実験環境

今回は、Google ColabのL4 GPUで検証しました。

重要なのは、Colab CLIだけで勝手にsessionを作ったのではなく、ブラウザ側でColab runtimeを作成し、そのブラウザ所有のruntimeをColab CLI側から採用したことです。

この運用は、以前のColab CLI検証の延長です。前回の記事では、Colab CLIで作ったruntimeを長時間維持しようとするとkeep-aliveが失敗し、runtimeがpruneされる一方で、ブラウザUIで起動したruntimeをCLIから一時的に掴む逆ルートに突破口があることを整理しました。

前提となる記事:

- Codexモバイル×Google Colab CLIで長時間実行の突破口が見えた  
  https://note.com/sunwood_ai_labs/n/n4d8214153375

今回のDiffusionGemma実験でも、この考え方を使っています。つまり、runtime保持の主体はColabブラウザUIに置き、CLIは環境構築・実行・結果回収の操作面として使います。

検証面は分けました。

- ブラウザUI: ColabがL4 runtimeに接続していること
- CLI: `colab sessions` にブラウザ所有のL4割当が見えること
- ランタイム内部: `torch` と `nvidia-smi` でNVIDIA L4を確認すること
- ローカル成果物: stdout、stderr、metadata、run reportを回収すること

この分離をしないと、「URLを開いた」「CLI sessionがある」「GPUで動いた」を混同します。

## llama-diffusion-cliの導入方法

DiffusionGemmaをGGUFで動かす場合、通常の `llama-cli` や `llama-server` では足りません。今回使ったのは、llama.cppのDiffusionGemma対応開発PRからビルドした `llama-diffusion-cli` です。

ここは重要です。リリース版のllama.cppをそのまま使ったのではなく、開発段階のPRをチェックアウトして試しています。

Colab L4/T4で再現する場合の最小手順は次の通りです。

```bash
apt-get update -y
apt-get install -y cmake ninja-build git git-lfs build-essential
python -m pip install -U "huggingface_hub[cli]"
```

llama.cppを取得し、DiffusionGemma対応PRをcheckoutします。

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
cd /content/llama.cpp

git fetch origin pull/24423/head
git checkout -B diffusiongemma FETCH_HEAD
```

CUDA有効で `llama-diffusion-cli` をビルドします。

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release --target llama-diffusion-cli
```

ビルド後の実行ファイルは次にできます。

```text
/content/llama.cpp/build/bin/llama-diffusion-cli
```

今回のL4/T4向け実験では、full precisionの `google/diffusiongemma-26B-A4B-it` ではなく、UnslothのGGUF量子化モデルを使いました。

```bash
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF \
  --local-dir /content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF \
  --include "*Q4_K_M*"
```

最小の実行確認は次の形です。

```bash
/content/llama.cpp/build/bin/llama-diffusion-cli \
  -m /content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -p "Write exactly three concise Japanese bullet points about DiffusionGemma." \
  -n 96 \
  --temp 0.2
```

今回のデモ動画で使ったように、生成過程をターミナル上で見せる場合は `--diffusion-visual` を付けます。

```bash
/content/llama.cpp/build/bin/llama-diffusion-cli \
  -m /content/models/unsloth__diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -p "Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe." \
  -n 96 \
  --temp 0.2 \
  --diffusion-visual \
  --diffusion-visual-progress \
  --diffusion-visual-interval 1 \
  --log-file llama_diffusion_visual.log
```

このリポジトリでは、上記の流れを手作業で打たなくても再現できるようにスクリプト化しています。

ローカルの公開リポジトリからColab CLIで流す場合は、L4またはT4のruntimeを用意したうえで次を実行します。

```bash
SESSION=diffusiongemma-l4-quant-smoke
colab new -s "$SESSION" --gpu L4
colab exec -s "$SESSION" --timeout 7200 -f scripts/run_diffusiongemma_gguf_probe.py
colab stop -s "$SESSION"
```

生成過程の可視化まで保存する場合は、visual capture用のスクリプトを実行します。

```bash
SESSION=diffusiongemma-l4-gguf-visual
colab new -s "$SESSION" --gpu L4
colab exec -s "$SESSION" --timeout 7200 -f scripts/run_diffusiongemma_gguf_visual_capture.py
colab stop -s "$SESSION"
```

すでにブラウザ側で作成したColab runtimeをCLIから採用する場合は、`colab sessions` で対象sessionを確認し、同じsession名に対して `colab exec` を流します。今回の記事で使った本筋はこのブラウザ所有runtimeをCLIから採用するルートです。

Colab runtime内部でスクリプトだけ直接実行する場合は次です。

```bash
python3 scripts/run_diffusiongemma_gguf_probe.py
python3 scripts/run_diffusiongemma_gguf_visual_capture.py
```

`run_diffusiongemma_gguf_probe.py` は通常のGGUF smoke test、`run_diffusiongemma_gguf_visual_capture.py` は `--diffusion-visual` のPTY出力を保存する実験です。

再現時に確認すべき成功条件は次の通りです。

```text
1. nvidia-smiでL4またはT4が見えている
2. git checkout後のbranchがdiffusiongemmaになっている
3. /content/llama.cpp/build/bin/llama-diffusion-cli が存在する
4. Q4_K_M GGUFが /content/models/... に保存されている
5. llama-diffusion-cli のreturn codeが0になる
6. 通常のGGUF smoke testでは generation_stdout.txt と generation_stderr.txt が残る
7. --diffusion-visual 実行では visual_terminal_capture.ansi、generation_terminal_tail.txt、llama_diffusion_visual.log が残る
```

GGUF系スクリプトの標準出力先は、Colab runtime内の `/content/diffusiongemma-colab-cli-public/<timestamp>/` です。Drive保存前提のfull Transformers probeとは出力先が違うため、結果確認時はここを混同しないようにします。

T4/L4でこのルートを使う理由は、公式Transformers経由のfull modelが60GB超のVRAMを要求するためです。L4/T4では、まずGGUF Q4_K_Mと `llama-diffusion-cli` の組み合わせで動作確認するのが現実的です。

## まずハマったこと: `llama-cli` では通常Gemma4 baselineにならない

最初は通常Gemma4を `llama-cli` で測ろうとしました。

しかし、このPR branchのGemma4では、`llama-cli -no-cnv` が非会話completionとして動かず、対話プロンプトに入りました。その結果、stdoutに `>` が出続け、測定はtimeoutしました。

さらに `-st` を付けると一応終了しましたが、出力には次の趣旨のメッセージが出ました。

```text
--no-conversation is not supported by llama-cli
please use llama-completion instead
```

つまり、通常Gemma4の非会話baselineとして使うべきなのは `llama-cli` ではなく `llama-completion` でした。

もう1つ罠がありました。Gemma4のcustom chat templateでは、`llama-completion` に `--jinja` が必要でした。付けない場合はtemplate処理で落ちます。

最終的に通った通常Gemma4 baselineはこれです。

```bash
/content/llama.cpp/build/bin/llama-completion \
  -m /content/models/ggml-org__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -p "Write a concise 3-bullet lab note about a robot barista calibrating espresso shots on a tiny Mars cafe." \
  -n 8 \
  --temp 0.2 \
  -ngl 99 \
  --no-warmup \
  --jinja \
  -no-cnv
```

## 通常Gemma4 baselineの結果

Colab L4での通常Gemma4 Q4_K_M GGUFの結果です。

```text
Prompt eval: 24 tokens, 13.35 ms/token, 74.91 tok/s
Decode eval: 7 runs, 18.86 ms/token, 53.01 tok/s
Total timing: 462.40 ms / 31 tokens
Command elapsed: 12.192 sec
Return code: 0
```

ここでのdecode速度は53.01 tok/sでした。

注意点として、`Command elapsed` にはmodel loadなども含まれます。一方、`Decode eval` はllama.cppの推論timingです。速度比較に使うなら、まずdecode evalを見ます。

証跡:

- `results/20260612T0208Z_browser_owned_l4_gemma4_completion_jinja_n8/metadata.json`
- `results/20260612T0208Z_browser_owned_l4_gemma4_completion_jinja_n8/stdout.txt`
- `results/20260612T0208Z_browser_owned_l4_gemma4_completion_jinja_n8/stderr.txt`
- `results/20260612T0208Z_browser_owned_l4_gemma4_completion_jinja_n8/run_report.md`

## DiffusionGemma側の既存L4結果

同じリポジトリ内には、DiffusionGemmaを `llama-diffusion-cli` で動かした既存のL4結果があります。

この実験の公開ポストはこちらです。

https://x.com/haru_maki_ch/status/2064990568846127528?s=46

代表値は次の通りです。

```text
Robot-barista prompt:
total time: 4551.07 ms
time per step: 216.72 ms
throughput: 56.3 tok/s
canvas: 256 tokens

Sunwood-note prompt:
total time: 4754.52 ms
time per step: 216.11 ms
throughput: 53.8 tok/s
canvas: 256 tokens

Earlier Japanese explainer prompt:
total time: 9289.70 ms
time per step: 442.37 ms
throughput: 27.6 tok/s
canvas: 256 tokens
```

最も近いrobot-barista promptでは、DiffusionGemmaは56.3 tok/sでした。

通常Gemma4のdecodeが53.01 tok/sだったため、少なくともこの比較では「DiffusionGemmaの方が遅い」とは言えません。

## ただし、これは完全な同条件比較ではない

ここが一番重要です。

DiffusionGemmaと通常Gemma4は、速度表示の意味が違います。

通常Gemma4は、基本的に1 tokenずつdecodeします。

一方、DiffusionGemmaは256 tokenのcanvasを用意し、複数stepで全体を修正しながら生成します。ログ上のthroughputは、canvas全体とdenoising stepをもとにした値です。

そのため、次の比較は雑に混ぜてはいけません。

- 通常Gemma4のdecode tok/s
- DiffusionGemmaのcanvas throughput
- wall-clock latency
- first token latency
- 256 token blockが完成するまでの時間
- quality benchmark
- batch sizeを上げたときのserver throughput

Googleが「速い」と言っているのは、主に低バッチ・単一GPUでのblock-parallel generationの話です。cloud servingで大量batchを組めるautoregressive modelとは、強い条件が違います。

## 今回わかったこと

今回の検証でわかったことは3つです。

1つ目。DiffusionGemmaの高速性は、通常のLLM decode速度と同じ土俵では語れません。256 token canvasを並列にdenoiseするモデルなので、比較には専用の指標が必要です。

2つ目。Colab L4 + GGUF + llama.cpp開発PRでも、DiffusionGemmaが通常Gemma4より明確に遅いという結果にはなりませんでした。近いpromptではDiffusionGemma 56.3 tok/s、通常Gemma4 53.01 tok/sです。

3つ目。公式の「最大4倍」「1000 tok/s超」を検証するには、今回の構成では足りません。Googleの推奨に近いvLLM構成、公式sampler、できればH100やRTX 5090クラスで再現する必要があります。

## 何を検証できて、何を検証できていないか

検証できたこと:

- Colab L4上で通常Gemma4の正しい非会話baselineを取れた
- `llama-cli` ではなく `llama-completion --jinja -no-cnv` が必要だと確認した
- DiffusionGemmaの既存L4結果と、通常Gemma4 baselineを同じ記事内で比較できる形に整理した
- 今回のL4結果では、DiffusionGemmaが通常Gemma4より遅いとは言えない

検証できていないこと:

- Google公式のH100/RTX 5090級の高速値の再現
- vLLM推奨設定でのDiffusionGemma速度
- 同一prompt、同一出力長、同一quality条件での厳密比較
- batch sizeを変えたときのthroughput curve
- quality劣化と速度のトレードオフの定量評価

## 次にやるべき本命実験

次は、公式の比較条件に寄せます。

やるべきことは明確です。

```text
1. DiffusionGemmaをvLLM推奨設定で起動する
2. canvas_length=256、Entropy-Bounded Denoising、adaptive stoppingを明示する
3. 通常Gemma4 26B A4B baselineも同じGPUで測る
4. 同一prompt、同一出力長、同一batch sizeで比較する
5. latency、throughput、qualityを別々に記録する
```

今回のL4実験は、公式速度の再現ではなく、比較の前提を整えるための検証でした。

そして、その意味では大きな収穫がありました。

`llama-cli` で通常Gemma4を測ると間違う。DiffusionGemmaのtok/sを通常decode tok/sと雑に混ぜると誤解する。速いかどうかを語るには、どの生成方式の、どのthroughputを見ているのかを先に固定する必要があります。

## まとめ

DiffusionGemmaは「通常Gemma4をそのまま置き換える高品質モデル」ではありません。

むしろ、低バッチ・単一GPU・ローカル対話用途で、GPUをより並列に使うための実験的モデルです。品質では通常Gemma4が強い場面があります。一方で、生成方式が根本的に違うため、条件が合えばtoken blockを高速に作れます。

今回のColab L4検証では、公式の4倍速はまだ再現していません。ただし、DiffusionGemmaが遅いという見方も支持されませんでした。

次はvLLM/公式設定で、DiffusionGemmaがどこまで速くなるのかを本命条件で確かめます。

## 参考リンク

DiffusionGemma Developer Guide

https://developers.googleblog.com/diffusiongemma-the-developer-guide/

DiffusionGemma model overview

https://ai.google.dev/gemma/docs/diffusiongemma

DiffusionGemma model card

https://ai.google.dev/gemma/docs/diffusiongemma/model_card

前提記事: Codexモバイル×Google Colab CLIで長時間実行の突破口が見えた

https://note.com/sunwood_ai_labs/n/n4d8214153375

今回のDiffusionGemma実験ポスト

https://x.com/haru_maki_ch/status/2064990568846127528?s=46

Google Colab CLI

https://github.com/googlecolab/google-colab-cli
