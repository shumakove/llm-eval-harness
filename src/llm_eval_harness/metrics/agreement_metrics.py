"""Согласие судьи с человеческой разметкой.

Слой доменно-независимый: функции принимают две выровненные
последовательности меток, а не пути к файлам. Джойн по item_id, отсев
неуверенных меток и чтение снапшотов — работа вызывающего скрипта.

Положительный класс — «claim НЕ подтверждён контекстом». Выбран так
намеренно: интерес в том, как часто судья пропускает неподтверждённый
claim, а это recall по положительному классу. Обратная конвенция сделала
бы главную цифру precision'ом по отрицательному классу и запутала бы
читателя отчёта.

`reference` — человеческая разметка (эталон), `predicted` — судья. Порядок
аргументов значим: kappa симметрична, precision и recall — нет.
"""

from pydantic.dataclasses import dataclass


@dataclass
class AgreementMatrix:
    """Матрица 2×2. True = claim не подтверждён контекстом.

    tp: человек и судья согласны, что claim не подтверждён.
    fn: человек сказал «не подтверждён», судья — «подтверждён». Судья
        пропустил дефект; faithfulness на таком кейсе завышен.
    fp: человек сказал «подтверждён», судья — «не подтверждён». Судья
        забраковал обоснованный claim; faithfulness занижен.
    tn: оба согласны, что claim подтверждён.
    """

    tp: int
    fn: int
    fp: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fn + self.fp + self.tn


def build_matrix(reference: list[bool], predicted: list[bool]) -> AgreementMatrix:
    if len(reference) != len(predicted):
        raise ValueError(
            f"последовательности разной длины: {len(reference)} и {len(predicted)}"
        )
    if not reference:
        raise ValueError("пустые последовательности — нечего сравнивать")

    tp = fn = fp = tn = 0
    for human, judge in zip(reference, predicted, strict=True):
        if human and judge:
            tp += 1
        elif human and not judge:
            fn += 1
        elif not human and judge:
            fp += 1
        else:
            tn += 1
    return AgreementMatrix(tp, fn, fp, tn)


def raw_agreement(matrix: AgreementMatrix) -> float:
    """Доля совпадений. На несбалансированных классах завышает: судья,
    всегда отвечающий «подтверждён», тоже наберёт немало. Считается
    именно затем, чтобы рядом с kappa было видно, почему kappa нужна.
    """
    return round((matrix.tp + matrix.tn) / matrix.n, 3)


def precision_unsupported(matrix: AgreementMatrix) -> float | None:
    """Когда судья бракует claim, насколько ему можно верить.

    None, если судья не забраковал ни одного claim: precision тогда не
    определён, а не равен нулю или единице.
    """
    predicted_positive = matrix.tp + matrix.fp
    if predicted_positive == 0:
        return None
    return round(matrix.tp / predicted_positive, 3)


def recall_unsupported(matrix: AgreementMatrix) -> float | None:
    """Какую долю неподтверждённых claim'ов судья поймал.

    Главная цифра для решения «годится ли судья как детектор»: остаток —
    это галлюцинации, прошедшие мимо метрики.

    None, если в эталоне нет ни одного неподтверждённого claim.
    """
    actual_positive = matrix.tp + matrix.fn
    if actual_positive == 0:
        return None
    return round(matrix.tp / actual_positive, 3)


def cohen_kappa(matrix: AgreementMatrix) -> float | None:
    """Cohen's kappa: согласие с поправкой на случайные совпадения.

        kappa = (p0 - pe) / (1 - pe)

    p0 — наблюдаемая доля совпадений.
    pe — ожидаемая при независимых оценщиках: сумма по классам
         произведений маргиналов ОБОИХ оценщиков. Не квадрат
         распространённости у одного из них — это даёт похожую на правду
         цифру и глазом не отлавливается.

    Возвращает None, когда pe == 1: kappa не определена, знаменатель ноль.
    Это ровно один случай — оба оценщика поставили всем items один и тот же
    класс. Если постоянен только один из них, pe < 1 и kappa считается
    (выйдет около нуля, что и правильно: согласия сверх случайного нет).
    """
    
    human_said_yes = (matrix.fp + matrix.tn) / matrix.n
    human_said_no = (matrix.tp + matrix.fn) / matrix.n
    judge_said_yes = (matrix.fn + matrix.tn) / matrix.n
    judge_said_no = (matrix.tp + matrix.fp) / matrix.n
    p0 = (matrix.tp + matrix.tn) / matrix.n
    pe = human_said_no * judge_said_no + human_said_yes * judge_said_yes
    if pe == 1: 
        return None
    return round((p0 - pe) / (1 - pe), 3) 


@dataclass
class AgreementReport:
    matrix: AgreementMatrix
    raw_agreement: float
    cohen_kappa: float | None
    precision_unsupported: float | None
    recall_unsupported: float | None


def summarize(reference: list[bool], predicted: list[bool]) -> AgreementReport:
    matrix = build_matrix(reference, predicted)
    return AgreementReport(
        matrix=matrix,
        raw_agreement=raw_agreement(matrix),
        cohen_kappa=cohen_kappa(matrix),
        precision_unsupported=precision_unsupported(matrix),
        recall_unsupported=recall_unsupported(matrix),
    )
