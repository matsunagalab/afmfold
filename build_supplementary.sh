#!/bin/bash

# LaTeX PDF生成スクリプト (Supplementary用)
# Usage: ./build_supplementary.sh

set -e  # エラーで停止

# Supplementary TeXファイル
TEX_FILE="paper/supplementary.tex"

# ファイルが存在するかチェック
if [ ! -f "$TEX_FILE" ]; then
    echo "エラー: $TEX_FILE が見つかりません"
    exit 1
fi

# ファイル名とディレクトリを取得
TEX_DIR=$(dirname "$TEX_FILE")
TEX_BASENAME=$(basename "$TEX_FILE" .tex)

echo "=== LaTeX PDF生成開始 (Supplementary) ==="
echo "ファイル: $TEX_FILE"
echo "ディレクトリ: $TEX_DIR"
echo

# 作業ディレクトリに移動
cd "$TEX_DIR"

echo "1/4: 最初のpdflatexコンパイル..."
pdflatex "$TEX_BASENAME.tex"

echo
echo "2/4: biberによる文献処理..."
biber "$TEX_BASENAME"

echo
echo "3/4: 2回目のpdflatexコンパイル（文献リンク）..."
pdflatex "$TEX_BASENAME.tex"

echo
echo "4/4: 3回目のpdflatexコンパイル（相互参照解決）..."
pdflatex "$TEX_BASENAME.tex"

echo
echo "=== コンパイル完了 ==="

# 生成されたPDFファイルの情報を表示
if [ -f "$TEX_BASENAME.pdf" ]; then
    PDF_SIZE=$(ls -lh "$TEX_BASENAME.pdf" | awk '{print $5}')
    PDF_PAGES=$(pdfinfo "$TEX_BASENAME.pdf" 2>/dev/null | grep "Pages:" | awk '{print $2}' || echo "不明")
    echo "生成されたPDF: $TEX_BASENAME.pdf"
    echo "ファイルサイズ: $PDF_SIZE"
    echo "ページ数: $PDF_PAGES"
    echo "パス: $(pwd)/$TEX_BASENAME.pdf"
else
    echo "エラー: PDFファイルが生成されませんでした"
    exit 1
fi

echo
echo "=== 一時ファイルクリーンアップ ==="
# 一時ファイルを削除（オプション）
# コメントアウトを外すと一時ファイルも削除されます
# rm -f *.aux *.log *.out *.bbl *.bcf *.blg *.run.xml *.fdb_latexmk *.fls

echo "完了！"

