"""Калибровочный снапшот: фаза A (генерация), B (судья), C (файл для разметки).

Почему это отдельная фаза, а не один проход вместе с судьёй: вызовы RAG-
пайплайна — самая дорогая и самая медленная часть, а промпты судьи будут
переписываться много раз. Ответы, замороженные в файл, делают каждую
следующую итерацию по судье (фаза B) ценой только вызовов судьи.

Скрипт одноразовый, артефакт — нет. Каждый запуск создаёт новый run_id и
новый файл. Снапшот, под который уже проставлены ручные метки, НЕ
перегенерируется: декомпозиция недетерминирована, список claims разъедется
и метки повиснут.

Запускать из корня репозитория (пути к данным относительные):

    python -m scripts.scratch_calibration answers --limit 3
    python -m scripts.scratch_calibration answers
    python -m scripts.scratch_calibration verdicts --source data/calibration/<run_id>_answers.jsonl
    python -m scripts.scratch_calibration labels   --source data/calibration/<run_id>_verdicts.jsonl

Именно `-m`: при `python scripts/scratch_calibration.py` в sys.path попадает
каталог скрипта, а не корень репозитория, и `game_lore_rag` не найдётся.
"""

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

from game_lore_rag.corpus import load_corpus
from game_lore_rag.generator import SYSTEM_PROMPT as GENERATOR_SYSTEM_PROMPT
from game_lore_rag.generator import AnyModelGenerator
from game_lore_rag.pipeline import RagPipeline, RagResult
from game_lore_rag.retriever import EmbeddingRetriever
from llm_eval_harness.datasets.eval_case import EvalCase
from llm_eval_harness.datasets.golden import load_golden_set
from llm_eval_harness.judge.judge import decompose_claims, describe_judge, verify_claims
from llm_eval_harness.provenance import (
    SnapshotWriter,
    _digest,
    _git_revision,
    read_snapshot,
)

GOLDEN_DATASET_PATH = "data/golden/npc_queries.yml"
CORPUS_DATA_PATH = "data/corpus/npc.yml"
OUTPUT_DIR = Path("data/calibration")

# Значения объявлены здесь, а не считаны с собранных объектов, потому что
# EmbeddingRetriever не хранит имя модели, а AnyModelGenerator не обязан его
# отдавать. Раз описать по факту нельзя — задаём явно и тем же значением
# конструируем пайплайн ниже. Манифест не может разойтись с тем, что реально
# работало, только если обе стороны читают одну константу.
GENERATOR_MODEL = "ds/deepseek-v4-flash"
RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def new_snapshot_path(run_id: str, suffix: str) -> Path:
    """Путь под новый снапшот; писать поверх существующего отказывается.

    run_id — с точностью до секунды, а фаза C отрабатывает мгновенно: два
    запуска подряд дали бы одно имя, и второй молча затёр бы первый. Пока
    метки не проставлены, это стоило бы перезапуска; после разметки —
    потери соответствия item_id тому, что человек размечал. Артефакт
    иммутабелен, так что лучше упасть.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{run_id}_{suffix}.jsonl"
    if path.exists():
        raise FileExistsError(f"снапшот уже существует, не перезаписываю: {path}")
    return path



def build_manifest(k: int) -> dict:
    """Манифест фазы A.

    Блока `judge` здесь нет намеренно: судья в этом прогоне не участвует, и
    манифест не должен описывать то, чего не было. Фаза B соберёт свой —
    с описанием судьи и ссылкой на run_id этого файла.

    Системный промпт генератора лежит внутри осознанно: ответы, которые
    потом будут размечаться, сформированы им, и его правка делает снапшот
    несравнимым с предыдущим.
    """
    now = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "phase": "answers",
        "run_id": now.strftime("%Y%m%dT%H%M%SZ"),
        "created_at": now.isoformat(),
        "code": _git_revision(),
        "pipeline": {
            "generator_model": GENERATOR_MODEL,
            "generator_system_prompt": GENERATOR_SYSTEM_PROMPT,
            "retriever": RETRIEVER_MODEL,
            "k": k,
        },
        "inputs": {
            "golden_set": {
                "path": GOLDEN_DATASET_PATH,
                "sha256": _digest(GOLDEN_DATASET_PATH),
            },
            "corpus": {
                "path": CORPUS_DATA_PATH,
                "sha256": _digest(CORPUS_DATA_PATH),
            },
        },
    }


def build_record(case: EvalCase, result: RagResult) -> dict:
    """Одна строка снапшота: всё, что фазе B и разметке понадобится о кейсе.

    RagResult и Document — датаклассы, в JSON напрямую не уходят, поэтому
    поля вытаскиваются руками. id чанка и его ранг сохраняются, хотя судье
    они не нужны: при разборе расхождений будет важно, какой именно документ
    судья видел и на каком месте.
    """
    return {
        "case_id": case.id,
        "status": "ok",
        "query": case.query,
        "answer": result.answer,
        "context": [
            {"id": document.id, "rank": rank, "score": score, "text": document.text}
            for rank, (document, score) in enumerate(result.retrieved, start=1)
        ],
    }


def dump_answers(limit: int | None = None) -> Path:
    golden_set = load_golden_set(GOLDEN_DATASET_PATH)
    corpus = load_corpus(CORPUS_DATA_PATH)

    # k берётся по ПОЛНОМУ golden-сету и только потом применяется срез:
    # пробный прогон на трёх кейсах должен идти с тем же k, что и полный,
    # иначе он проверяет не ту конфигурацию, которую ты собираешься запускать.
    k = max(len(case.relevant_ids) for case in golden_set)
    cases = golden_set[:limit] if limit else golden_set

    pipeline = RagPipeline(
        EmbeddingRetriever(RETRIEVER_MODEL),
        AnyModelGenerator(GENERATOR_MODEL),
    )
    pipeline.build(corpus)

    # Манифест строится до цикла: _git_revision и _digest могут упасть
    # (нет файла, не git-репозиторий). Пусть это случится за секунду до
    # старта, а не после семидесяти оплаченных вызовов генератора.
    manifest = build_manifest(k)
    path = new_snapshot_path(manifest["run_id"], "answers")

    failed = 0
    with SnapshotWriter(path, manifest) as writer:
        for number, case in enumerate(cases, start=1):
            try:
                result = pipeline.answer(case.query, k)
            except Exception as error:  # noqa: BLE001
                # Одна сетевая ошибка не должна стоить всего прогона. Но и
                # молча исчезнуть кейс не может: иначе число записей не
                # сойдётся с golden-сетом, и потом не отличить сбой модели
                # от потерянной строки.
                failed += 1
                writer.append(
                    {
                        "case_id": case.id,
                        "status": "generation_failed",
                        "query": case.query,
                        "error": repr(error),
                    }
                )
                print(f"[{number}/{len(cases)}] {case.id} — СБОЙ: {error}")
                continue

            writer.append(build_record(case, result))
            print(f"[{number}/{len(cases)}] {case.id}")

    print(f"\nснапшот: {path}")
    print(f"кейсов: {len(cases)}, сбоев генерации: {failed}")
    return path



# ── Фаза B: судья поверх замороженных ответов ─────────────────────────────


def build_verdicts_manifest(source_manifest: dict, source_path: Path) -> dict:
    """Манифест фазы B.

    run_id свой, а не унаследованный от источника. Это не формальность:
    смысл заморозки в том, что одни и те же ответы пересуживаются после
    каждой правки промпта. Наследуй run_id — второй прогон затёр бы
    первый, и сравнить две версии судьи было бы не с чем.

    Блока pipeline здесь нет: генератор в этом прогоне не работал. Чем
    порождены ответы, читается по ссылке source.

    describe_judge() кладётся целиком, без разбора по полям: добавится
    третий промпт — он приедет сам, и манифест не соврёт умолчанием.
    """
    now = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "phase": "verdicts",
        "run_id": now.strftime("%Y%m%dT%H%M%SZ"),
        "created_at": now.isoformat(),
        "code": _git_revision(),
        "judge": describe_judge(),
        "source": {
            "run_id": source_manifest["run_id"],
            "path": str(source_path),
            "sha256": _digest(str(source_path)),
        },
    }


def _stub_row(case_id: str, status: str, **extra) -> dict:
    """Строка для кейса, по которому вердиктов не будет.

    Кейс обязан присутствовать в файле при любом исходе, иначе множество
    item_id не свести с исходным golden-сетом и потом не отличить сбой
    судьи от потерянной строки. item_id здесь None: он адресует claim,
    а claim'а в этом исходе нет.
    """
    return {
        "item_id": None,
        "case_id": case_id,
        "claim_index": None,
        "status": status,
        **extra,
    }


def judge_record(record: dict) -> list[dict]:
    """Одна запись фазы A -> строки вердиктов (на кейс их обычно несколько)."""
    case_id = record["case_id"]

    if record["status"] != "ok":
        # Сбой генерации протаскивается дальше как есть: судить нечего,
        # но кейс между фазами исчезнуть не может.
        return [_stub_row(case_id, record["status"])]

    # Порядок рангов сохраняется осознанно: судья видит пронумерованные
    # пассажи и ссылается на номера, перестановка меняет его вход.
    context = [chunk["text"] for chunk in record["context"]]

    try:
        claims = decompose_claims(record["answer"],query=record["query"])
    except Exception as error:  # noqa: BLE001
        # Широкий except намеренно: JudgeError здесь не единственный исход.
        # Если модель вернула ответ без tool call, judge.py падает на
        # tool_calls[0] с TypeError, и ловить надо оба.
        return [_stub_row(case_id, "decompose_error", error=repr(error))]

    if not claims:
        return [_stub_row(case_id, "no_claims")]

    try:
        verdicts = verify_claims(claims, context)
    except Exception as error:  # noqa: BLE001
        return [_stub_row(case_id, "judge_error", error=repr(error), n_claims=len(claims))]

    rows = []
    # strict=True — дешёвая страховка: verify_claims уже проверяет, что
    # число вердиктов совпало с числом claims, но здесь это молчаливое
    # допущение, а молчаливые допущения стоит делать громкими.
    for index, (claim, verdict) in enumerate(zip(claims, verdicts, strict=True)):
        row = {
            "item_id": f"{case_id}#{index}",
            "case_id": case_id,
            "claim_index": index,
            "status": "ok",
            "claim": claim,
            "supported": verdict.supported,
            "reason": verdict.reason,
        }
        # Канонический текст claim — присланный судье, а не возвращённый им:
        # размечать человек будет именно присланный. Но если судья
        # перефразировал, это диагностика инструмента, и терять её нельзя.
        if verdict.claim.strip() != claim.strip():
            row["judge_claim"] = verdict.claim
        rows.append(row)

    return rows


def dump_verdicts(source_path: Path) -> Path:
    source_manifest, records = read_snapshot(source_path)

    # Запуск по файлу вердиктов вместо файла ответов даёт невнятную ошибку
    # где-то в середине прогона. Дешевле проверить на входе.
    if source_manifest.get("phase") != "answers":
        raise ValueError(
            f"ожидался снапшот фазы answers, получен {source_manifest.get('phase')!r}"
        )

    manifest = build_verdicts_manifest(source_manifest, source_path)
    path = new_snapshot_path(manifest["run_id"], "verdicts")

    counts: dict[str, int] = {}
    with SnapshotWriter(path, manifest) as writer:
        for number, record in enumerate(records, start=1):
            rows = judge_record(record)
            for row in rows:
                writer.append(row)
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(
                f"[{number}/{len(records)}] {record['case_id']} — "
                f"{rows[0]['status']}, строк: {len(rows)}"
            )

    print(f"\nвердикты: {path}")
    print("строк по статусам: " + ", ".join(f"{name}: {n}" for name, n in sorted(counts.items())))
    return path



# ── Фаза C: слепой файл для разметки ──────────────────────────────────────


def build_labels_manifest(
    verdicts_manifest: dict, answers_manifest: dict, seed: int, size: int
) -> dict:
    """Манифест фазы C.

    Ни судьи, ни пайплайна: этот прогон ничего не вызывал, он только
    перекладывал. Зато здесь сид и размер выборки — без них выборку не
    воспроизвести и размеченный набор потом не расширить.
    """
    now = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "phase": "labels_blank",
        "run_id": now.strftime("%Y%m%dT%H%M%SZ"),
        "created_at": now.isoformat(),
        "code": _git_revision(),
        "sample": {"seed": seed, "size": size},
        "source": {
            "verdicts_run_id": verdicts_manifest["run_id"],
            "answers_run_id": answers_manifest["run_id"],
        },
    }


def dump_labels(verdicts_path: Path, sample: int | None = None, seed: int = 0) -> Path:
    verdicts_manifest, verdict_rows = read_snapshot(verdicts_path)
    if verdicts_manifest.get("phase") != "verdicts":
        raise ValueError(
            f"ожидался снапшот фазы verdicts, получен {verdicts_manifest.get('phase')!r}"
        )

    # Путь к файлу ответов не нужно передавать руками: он записан в манифесте
    # фазы B. Цепочка провенанса для того и строилась.
    answers_path = Path(verdicts_manifest["source"]["path"])
    answers_manifest, answer_rows = read_snapshot(answers_path)
    context_by_case = {
        row["case_id"]: [chunk["text"] for chunk in row["context"]]
        for row in answer_rows
        if row["status"] == "ok"
    }

    items = [row for row in verdict_rows if row["status"] == "ok"]
    if sample is not None and sample < len(items):
        items = random.Random(seed).sample(items, sample)
        items.sort(key=lambda row: row["item_id"])

    manifest = build_labels_manifest(verdicts_manifest, answers_manifest, seed, len(items))
    path = new_snapshot_path(manifest["run_id"], "labels_blank")

    with SnapshotWriter(path, manifest) as writer:
        for item in items:
            # Вопроса здесь нет намеренно: verify_claims его тоже не видит,
            # и человек должен решать по тем же данным, что и судья.
            writer.append(
                {
                    "item_id": item["item_id"],
                    "claim": item["claim"],
                    "context": context_by_case[item["case_id"]],
                    "supported": None,
                }
            )

    print(f"\nдля разметки: {path}")
    print(f"items: {len(items)} из {len(verdict_rows)} строк вердиктов")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Калибровочный снапшот: фазы A и B.")
    subparsers = parser.add_subparsers(dest="phase", required=True)

    answers_parser = subparsers.add_parser("answers", help="фаза A: заморозить ответы RAG")
    answers_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="прогнать только первые N кейсов — для пробного запуска",
    )

    verdicts_parser = subparsers.add_parser(
        "verdicts", help="фаза B: судья поверх замороженных ответов"
    )
    verdicts_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="файл фазы A: data/calibration/<run_id>_answers.jsonl",
    )

    labels_parser = subparsers.add_parser(
        "labels", help="фаза C: слепой файл для ручной разметки"
    )
    labels_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="файл фазы B: data/calibration/<run_id>_verdicts.jsonl",
    )
    labels_parser.add_argument(
        "--sample", type=int, default=None, help="сколько items взять на разметку (по умолчанию все)"
    )
    labels_parser.add_argument("--seed", type=int, default=0, help="сид выборки")

    args = parser.parse_args()
    if args.phase == "answers":
        dump_answers(args.limit)
    elif args.phase == "verdicts":
        dump_verdicts(args.source)
    else:
        dump_labels(args.source, args.sample, args.seed)
