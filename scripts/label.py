"""Разметка калибровочного набора: показывает claim и контекст, пишет метку.

Работает поверх слепого файла фазы C и НЕ открывает файл вердиктов — судью
ты видеть не должен, пока не разметишь всё.

    python -m scripts.label data/calibration/<run_id>_labels_blank.jsonl

Метки дописываются в <run_id>_labels_filled.jsonl по одной, сразу на диск.
Можно выйти на любом item и запустить снова — продолжится с того же места.
"""

import json
import sys
from pathlib import Path

from llm_eval_harness.provenance import read_snapshot

CRITERION = (
    "supported только если контекст это утверждает или ясно подразумевает; "
    "просто правдоподобное — НЕ supported"
)
KEYS = {"y": "yes", "n": "no", "u": "unsure"}


def label(blank_path: Path) -> None:
    manifest, items = read_snapshot(blank_path)
    out_path = blank_path.with_name(blank_path.name.replace("_labels_blank", "_labels_filled"))

    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["item_id"]] = row["label"]

    todo = [item for item in items if item["item_id"] not in done]
    print(f"\nвсего {len(items)}, размечено {len(done)}, осталось {len(todo)}")
    print(f"критерий: {CRITERION}")
    print("клавиши: y = supported, n = not supported, u = не знаю, q = выйти\n")

    with out_path.open("a", encoding="utf-8") as out:
        for number, item in enumerate(todo, start=1):
            print("=" * 72)
            print(f"[{len(done) + number}/{len(items)}]  {item['item_id']}")
            print("\nКОНТЕКСТ:")
            for i, chunk in enumerate(item["context"], start=1):
                print(f"  [{i}] {chunk}")
            print(f"\nCLAIM:  {item['claim']}")

            while True:
                answer = input("\n  y / n / u / q > ").strip().lower()
                if answer == "q":
                    print(f"\nсохранено в {out_path}")
                    return
                if answer in KEYS:
                    break
                print("  не понял — нужно y, n, u или q")

            out.write(
                json.dumps(
                    {"item_id": item["item_id"], "label": KEYS[answer]}, ensure_ascii=False
                )
                + "\n"
            )
            out.flush()

    print(f"\nготово, все {len(items)} размечены: {out_path}")


if __name__ == "__main__":
    label(Path(sys.argv[1]))
