from adar_bigru.ops import apply_actions, encode_source_and_template, reverse_complement


def test_strict_complement_round_trip():
    target = "AUGCVACGU"
    _, template, _ = encode_source_and_template(target)
    assert template == reverse_complement(target)
    assert len(template) == len(target)
