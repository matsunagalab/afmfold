# AFM-Fold 論文リバイズ作業ディレクトリ

## 概要
論文 **"AFM-Fold: Rapid Reconstruction of Protein Conformations from AFM Images"** (Kawai & Matsunaga) を *Biophysical Journal* へ投稿中のリビジョン作業を行うディレクトリ。

- **Manuscript ID**: BIOPHYSICAL-JOURNAL-D-25-00813
- **Editor**: Frauke Graeter
- **Decision**: Major revision (single revision allowed)
- **Revision deadline**: 2026-04-12
- **Today**: 2026-04-12（締切当日）

## ディレクトリ構成

### このディレクトリ (`paper/preparation/`) — リビジョン作業用
- `biophysj.tex` — 本文メインの LaTeX ソース（`\documentclass[lineno,biblatex]{biophys-new}`）
- `biophysj_SI.tex` — Supplementary Information の LaTeX ソース
- `biophys-new.cls` — Biophysical Journal 提供のクラスファイル
- `commands.tex` — カスタムマクロ定義
- `citation_biophysj.bib` — 文献データベース（biblatex + biber）
- `sections/` — 本文を章ごとに分割した `.tex` ファイル群
  - `abstract.tex`, `significance.tex`, `introduction.tex`, `methods.tex`,
    `results.tex`, `discussion.tex`, `endmatter.tex`
  - `af3.tex`, `afmfold.tex`, `cnn.tex`, `md.tex`, `related_work.tex` — methods の下位ファイル
  - `appendix.tex` — Supplementary 本体
- `images/` — 図ファイル（`figure-1.png` 〜 `figure-12.png`）
- `out/` — ビルド成果物（`biophysj.pdf`, `biophysj_SI.pdf` 等）
- `decision_letter.md` — エディタからの決定レター＋3 名の査読者コメント
- `responses_to_reviewers.docx` — 査読者への返信ドラフト（作成中）

### 関連ディレクトリ
- `../../round0/` — 最初の投稿ファイル（`manuscript.pdf`, `supplementary.pdf`, `cover_letter.pdf/docx`）。リビジョン前の snapshot として参照する。
- `../../build_main.sh`, `../../build_supplementary.sh` — LaTeX ビルドスクリプト（デフォルトは `paper/main.tex`, `paper/supplementary.tex` を想定しているため、このディレクトリのビルドには引数指定かスクリプト側の調整が必要）。
- `../../src/`, `../../scripts/`, `../../notebooks/` — AFM-Fold 実装とスクリプト・解析ノートブック（査読者の質問に対する追加解析はここで行う）。
- `../../figures/` — 図生成の元データ・スクリプト。

## ビルド
**SI → 本文の順でビルドすること**（`biophysj.tex` が `\externaldocument{biophysj_SI}` で SI のラベルを参照するため）。順序を間違えると `??` が大量に残る。

- `biophysj_SI.tex` は `fontspec` を使うため **xelatex** でビルド
- `biophysj.tex` は **pdflatex** でビルド
- 両方とも biblatex + biber

```bash
cd /Volumes/ssd/gdrive/work/afmfold/paper/preparation
# 1. SI を先にビルド (xelatex)
xelatex -interaction=nonstopmode biophysj_SI && biber biophysj_SI && xelatex -interaction=nonstopmode biophysj_SI && xelatex -interaction=nonstopmode biophysj_SI
# 2. 本文をビルド (pdflatex)
pdflatex -interaction=nonstopmode biophysj && biber biophysj && pdflatex -interaction=nonstopmode biophysj && pdflatex -interaction=nonstopmode biophysj
```

## 査読者対応の主要論点（decision letter 要約）

### エディタからの必須事項
- **Data Availability statement を追加**（現状欠落）。AFM データ・コード・学習済みモデルの入手方法を明記。

### Reviewer 1（全体像が分かりにくい）
- g-CNN の学習手順・pseudo-AFM 生成・orientation の選択などの説明不足。Figure 1 が不親切（学習ステップが抜けている）。
- ステップ・バイ・ステップのプロトコル追加を要望。
- 最終構造を AFM 像と比較する際の rigid-body fitting の手順と所要時間、cc 値への影響を議論せよ。
- pseudo-AFM の orientation 依存性、学習した蛋白質と異なる蛋白質への汎化、学習時間の扱いについて検討。
- 軽微: HS-AFM データの出典（Ref. 56?）と撮像条件を明記、Fig. 7/8 が本文から参照されていない、line 358 の "1 ms per frame" はタイポか、Ref 15/19 重複、line 253 の `??` 未解決参照。

### Reviewer 2（methodology / 実験応用 / 既存 flexible fitting との関係）
- **A. Methodology**
  1. g-CNN による CV 抽出の説明が不十分。tip-convolution や orientation による domain の見えにくさ、誤検出のバイアスを議論。実験 HS-AFM 像で CV の意味を可視化できないか。
  2. AK ベンチマークは鋭いチップを仮定。より現実的な tip shape での感度を検討。
  3. FlhAc では過去発表の tip-shape 再構成法を使わなかった理由。pseudo-AFM と実像の高さ分布フィッティングが cosine similarity の利用を正当化するためか、その必要性を議論。
- **B. FlhAc 応用**: Fig. 4 の pose が方向含め大きく異なり説得力に欠ける。cc 値の小数第2位差は rigid-body fitting 改善の根拠として弱い。ポーズに依らず予測できる点を強調し、Fig. 5a の AcD2-AcD4 距離 ~2 nm 変化のような、静的構造との差を示す別の提示を検討。Fig. 4 を残すならドメイン別に色分け。
- **C. Flexible fitting との関係**: NMFF-AFM に関する記述（harmonic fluctuations 限定）は誤解を招く。NMFF-AFM は iterative に linear response を当てて非線形挙動を扱い、ラップトップで tens of seconds / frame を実現している（Ref 17）。一方 AFM-Fold は CNN 学習に 2–3 日（48 GB GPU）・CV 定義の人手介入・AF3 guiding パラメータ調整が必要。実験者目線での実用性を含めたバランスある議論を追加。
- **軽微**: p.6 L200 "sub-Angstroms", p.8 L245 "adenylate"→"Adenylate", p.8 L253 `??`。

### Reviewer 3（概ね好意的、claim の調整）
1. CV から構造への degeneracy について、CV 誤差が構造多様性にどう伝播するかの sensitivity 解析 or 明示的な失敗モード記述を要望。
2. AK の pseudo-AFM は理想化された条件（小 tip, ノイズなし; Table S1）。sub-Å RMSD は "upper bound" と明記せよ。
3. FlhAc では cc の向上が控えめで、主に全体位置決めに依存。Abstract/Intro の "outperforms rigid-body fitting" を弱める方向で書き換え。
- 軽微: AK の "intermediate" 表現を離散状態と誤読されないよう明確化 / AFM の逐次取得・横方向分解能限界の議論を discussion に追加 / guided diffusion のハイパーパラメータ（restraint onset, strength など）を再現性のためにまとめる。

## 既知のタイポ・未解決参照（リビジョン中に必ず修正）
- line 253 付近の `as described in ??` — 未解決ラベル
- Ref 15 と Ref 19 が重複
- "sub-Angstroms" の表現見直し
- "adenylate" → "Adenylate"
- "1 ms per frame" → おそらく "1 s per frame"
- Fig. 7, Fig. 8 が本文未参照

## 作業上の注意
- **Manuscript の変更は視認できるように markup**（色付き/下線/highlight 等）すること（decision letter の指示）。
- リビジョン版ではフォーマットガイドラインに従う必要がある（Biophysical Journal Author Guidelines）。
- `round0/` は初版の参照用。書き換え対象にしないこと。
- 査読者コメントへの応答は `responses_to_reviewers.docx` に集約（作成中）。本文の変更と応答の番号を突き合わせて追えるようにする。
- **`responses_to_reviewers.docx` 内で「行頭が `>` で始まるブロック」および「青色文字の部分」は査読者コメントの原文**。いかなる場合も編集・修正・書き換え・整形・翻訳をしないこと（タイポや文法の修正も禁止）。返信は必ずその下に追記する形で行う。
  - `>` で始まるブロックは改行を挟んでも査読者コメントが続いていることがある。次の自分の応答が始まるまで（あるいは青色が終わるまで）すべて原文として扱う。
  - 判断に迷ったら色（青＝reviewer）を優先する。
- git status に多数のビルド成果物（`.aux`, `.bbl`, `.pdf` 等）が untracked で残っているが、通常はコミット対象外。
