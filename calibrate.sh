#!/usr/bin/env bash
#
# Калибровочный снапшот судьи — три фазы, каждая пишет свой файл в
# data/calibration/ и ссылается на предыдущую через манифест.
#
#   answers   фаза A: прогнать RAG по golden-сету и заморозить ответы.
#             Самое дорогое: один вызов генератора на кейс.
#   verdicts  фаза B: судья поверх замороженных ответов.
#             Повторяется на каждую правку промпта судьи, генератор не трогается.
#   labels    фаза C: слепой файл для ручной разметки. Сети не требует.
#
# Источник для B и C по умолчанию — самый свежий подходящий файл; путь
# можно задать явно вторым аргументом.
#
# ВАЖНО: после фазы C файл вердиктов не открывать до конца разметки.
# Увиденный вердикт нельзя «развидеть», и κ по такому item будет завышена.

set -euo pipefail
cd "$(dirname "$0")"

DIR=data/calibration
PYTHON=python
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python
RUN="$PYTHON -m scripts.scratch_calibration"

latest() {
    local found
    found=$(ls -t "$DIR"/*_"$1".jsonl 2>/dev/null | head -1 || true)
    if [ -z "$found" ]; then
        echo "нет ни одного файла *_$1.jsonl в $DIR — сначала прогони предыдущую фазу" >&2
        exit 1
    fi
    echo "$found"
}

case "${1:-help}" in
    answers)
        # второй аргумент — limit для пробного прогона на N кейсах
        $RUN answers ${2:+--limit "$2"}
        ;;
    verdicts)
        src=${2:-$(latest answers)}
        echo "источник: $src"
        $RUN verdicts --source "$src"
        ;;
    labels)
        # второй аргумент — путь к вердиктам, третий — размер выборки
        src=${2:-$(latest verdicts)}
        echo "источник: $src"
        $RUN labels --source "$src" ${3:+--sample "$3"}
        ;;
    ls)
        ls -lt "$DIR"
        ;;
    *)
        sed -n '2,/^$/p' "$0" | sed -e 's/^#//' -e 's/^ //'
        echo
        echo "  ./calibrate.sh answers  [N]                 фаза A (N = limit, для пробы)"
        echo "  ./calibrate.sh verdicts [answers.jsonl]     фаза B"
        echo "  ./calibrate.sh labels   [verdicts.jsonl] [N] фаза C (N = размер выборки)"
        echo "  ./calibrate.sh ls                           что уже лежит в $DIR"
        ;;
esac
