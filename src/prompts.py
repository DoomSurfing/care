import re

# local file stem -> LLM-Adapters directory name is handled in the download script;
# here keys are our task names, values are file stems in data/test/
TASKS = {
    "boolq": "boolq", "piqa": "piqa", "social_iqa": "social_i_qa",
    "hellaswag": "hellaswag", "winogrande": "winogrande",
    "arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge", "openbookqa": "openbookqa",
}

def build_prompt(instruction: str, inp: str = "") -> str:
    """Template from context doc section 4. Newline placement is a literal reading
    (single newline after each header/body). Change ONLY here if it must differ."""
    p = f"### Instruction:\n{instruction}\n"
    if inp is not None and inp.strip():
        p += f"### Input:\n{inp}\n"
    p += "### Response:\n"
    return p

_FMT = re.compile(r"Answer format:\s*(.+)")

def parse_choices(instruction: str):
    """Valid answers parsed from the 'Answer format: a/b/c' line. None if absent."""
    m = _FMT.findall(instruction)
    if not m:
        return None
    ch = [c.strip() for c in m[-1].strip().split("/") if c.strip()]
    return ch or None