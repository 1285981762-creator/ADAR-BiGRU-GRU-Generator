from __future__ import annotations

from rapidfuzz.distance import Levenshtein


PAD, BOS, EOS, COPY, DELETE = range(5)
SUB_A, SUB_U, SUB_G, SUB_C = range(5, 9)
INS_A, INS_U, INS_G, INS_C = range(9, 13)
BASES = "AUGC"
SUB_ACTION = {base: SUB_A + index for index, base in enumerate(BASES)}
INS_ACTION = {base: INS_A + index for index, base in enumerate(BASES)}
ACTION_TO_BASE = {
    **{value: base for base, value in SUB_ACTION.items()},
    **{value: base for base, value in INS_ACTION.items()},
}
SOURCE_TOKEN = {"A": 3, "U": 4, "G": 5, "C": 6, "V": 7}
TEMPLATE_TOKEN = {"A": 3, "U": 4, "G": 5, "C": 6}


def normalize_rna(sequence: str) -> str:
    return sequence.strip().upper().replace("T", "U")


def reverse_complement(source: str) -> str:
    return normalize_rna(source).replace("V", "A").translate(str.maketrans("AUGC", "UACG"))[::-1]


def encode_source_and_template(target_with_v: str):
    normalized = normalize_rna(target_with_v)
    if normalized.count("V") != 1 or set(normalized) - set(SOURCE_TOKEN):
        raise ValueError("Target must contain exactly one V and only A/U/G/C/V")
    source = tuple(SOURCE_TOKEN[base] for base in normalized[::-1])
    template = reverse_complement(normalized)
    template_ids = tuple(TEMPLATE_TOKEN[base] for base in template)
    return source, template, template_ids


def action_script(template: str, observed_complement: str) -> tuple[int, ...]:
    observed = normalize_rna(observed_complement)
    if not observed or set(observed) - set(BASES):
        raise ValueError("Complement must be nonempty A/U/G/C sequence")
    editops = Levenshtein.editops(template, observed)
    insertions: dict[int, list[int]] = {}
    consuming: dict[int, int] = {}
    for edit in editops:
        if edit.tag == "insert":
            insertions.setdefault(edit.src_pos, []).append(INS_ACTION[observed[edit.dest_pos]])
        elif edit.tag == "delete":
            consuming[edit.src_pos] = DELETE
        elif edit.tag == "replace":
            consuming[edit.src_pos] = SUB_ACTION[observed[edit.dest_pos]]
        else:
            raise ValueError(f"Unexpected edit operation: {edit.tag}")
    actions = []
    for position in range(len(template) + 1):
        actions.extend(insertions.get(position, ()))
        if position < len(template):
            actions.append(consuming.get(position, COPY))
    actions.append(EOS)
    result = tuple(actions)
    if apply_actions(template, result) != observed:
        raise RuntimeError("Edit script does not reconstruct observed complement")
    return result


def apply_actions(template: str, actions) -> str:
    pointer = 0
    sequence = []
    for raw_action in actions:
        action = int(raw_action)
        if action == EOS:
            if pointer != len(template):
                raise ValueError("EOS before template was consumed")
            break
        if INS_A <= action <= INS_C:
            sequence.append(ACTION_TO_BASE[action])
            continue
        if pointer >= len(template):
            raise ValueError("Action consumed beyond template end")
        if action == COPY:
            sequence.append(template[pointer])
        elif action == DELETE:
            pass
        elif SUB_A <= action <= SUB_C:
            sequence.append(ACTION_TO_BASE[action])
        else:
            raise ValueError(f"Unknown action {action}")
        pointer += 1
    return "".join(sequence)


def decoder_inputs_and_pointers(actions: tuple[int, ...]):
    decoder_input = (BOS,) + actions[:-1]
    pointer = 0
    pointers = []
    for previous in decoder_input:
        if previous not in (BOS, PAD) and not (INS_A <= previous <= INS_C):
            pointer += 1
        pointers.append(pointer)
    return decoder_input, tuple(pointers)
