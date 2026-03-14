"""
prompt_templates.py

Prompt builders for MCAT question generation and prompt evaluation.

Design goals:
- Keep prompt construction centralized
- Make prompt versions explicit
- Enforce JSON-only outputs
- Support:
    1. science question generation
    2. CARS passage/question-set generation
    3. critique/evaluation during prompt optimization

Notes:
- The prompts are intentionally strict about output schema.
- Science prompts use a misconception-first distractor strategy.
- CARS prompts emphasize realistic wrong-answer patterns:
    - too extreme
    - unsupported inference
    - tone distortion
    - secondary point mistaken for thesis
"""

from __future__ import annotations

import json
from typing import Any


SCIENCE_PROMPT_VERSION = "science_v1"
CARS_PROMPT_VERSION = "cars_v1"
CRITIQUE_PROMPT_VERSION = "critique_v1"


# ---------------------------------------------------------------------------
# JSON schema reminders
# ---------------------------------------------------------------------------


SCIENCE_OUTPUT_SCHEMA = {
    "question_id": "string",
    "topic_id": "string",
    "mode": "science",
    "question": "string",
    "options": {
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string",
    },
    "correct_answer": "A|B|C|D",
    "explanation": "string",
    "difficulty": "easy|medium|hard",
    "model": "string",
    "prompt_version": "string",
}

SCIENCE_BATCH_OUTPUT_SCHEMA = {
    "items": [
        {
            "question_id": "string",
            "topic_id": "string",
            "mode": "science",
            "question": "string",
            "options": {
                "A": "string",
                "B": "string",
                "C": "string",
                "D": "string",
            },
            "correct_answer": "A|B|C|D",
            "explanation": "string",
            "difficulty": "easy|medium|hard",
            "model": "string",
            "prompt_version": "string",
        }
    ]
}


CARS_OUTPUT_SCHEMA = {
    "question_set_id": "string",
    "topic_id": "string",
    "mode": "cars",
    "passage": "string",
    "questions": [
        {
            "question": "string",
            "options": {
                "A": "string",
                "B": "string",
                "C": "string",
                "D": "string",
            },
            "correct_answer": "A|B|C|D",
            "explanation": "string",
        }
    ],
    "difficulty": "easy|medium|hard",
    "model": "string",
    "prompt_version": "string",
}

CRITIQUE_OUTPUT_SCHEMA = {
    "overall_score": "integer 1-10",
    "verdict": "keep|revise|reject",
    "strengths": ["string"],
    "weaknesses": ["string"],
    "schema_issues": ["string"],
    "content_issues": ["string"],
    "difficulty_assessment": "string",
    "realism_assessment": "string",
    "recommended_prompt_changes": ["string"],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def schema_to_pretty_json(schema: dict[str, Any]) -> str:
    """Render a schema reminder as pretty JSON."""
    return json.dumps(schema, ensure_ascii=False, indent=2)


def format_topic_block(topic: dict[str, Any]) -> str:
    """Format a canonical topic block for prompts."""
    return (
        f"topic_id: {topic.get('topic_id', '')}\n"
        f"category: {topic.get('category', '')}\n"
        f"subcategory: {topic.get('subcategory', '')}\n"
        f"title: {topic.get('title', '')}\n"
        f"content_to_test: {topic.get('content_to_test', '')}\n"
        f"tags: {json.dumps(topic.get('tags', []), ensure_ascii=False)}\n"
        f"adaptive_score: {topic.get('adaptive_score', 1.0)}"
    )


def build_json_only_instruction(schema: dict[str, Any]) -> str:
    """Standard JSON-only constraint block."""
    return (
        "Return valid JSON only.\n"
        "Do not include markdown fences.\n"
        "Do not include commentary before or after the JSON.\n"
        "Your output must match this schema reminder exactly in structure:\n"
        f"{schema_to_pretty_json(schema)}"
    )


# ---------------------------------------------------------------------------
# Science generation
# ---------------------------------------------------------------------------


def build_science_generation_prompt(
    topic: dict[str, Any],
    difficulty: str = "medium",
    num_questions: int = 5,
) -> str:
    """
    Build a science-generation prompt.

    This version requests multiple distinct science questions in one call
    to improve throughput during bulk generation.
    """
    topic_block = format_topic_block(topic)
    schema_block = build_json_only_instruction(SCIENCE_BATCH_OUTPUT_SCHEMA)

    return f"""
You are generating MCAT science multiple-choice questions.

Target topic:
{topic_block}

Task requirements:
- Mode: science
- Desired difficulty: {difficulty}
- Generate exactly {num_questions} DISTINCT standalone MCAT-style questions.
- Return them as a JSON object with a single key: "items"
- "items" must be a list of exactly {num_questions} question objects
- Each question must test a meaningfully different angle, mechanism, application, or misconception cluster
- Do not generate paraphrases of the same question
- The questions should test conceptual understanding, not just memorized facts
- Prefer mechanism, interpretation, comparison, or application over trivia

Very important generation strategy:
Use a misconception-first distractor design process for EACH question:
1. Identify the core concept from the topic
2. Identify realistic misconceptions or near-miss confusions a strong MCAT student could plausibly make
3. Build distractors from those misconceptions
4. Only after the misconceptions and distractors are clear, write the final stem so that the question is discriminative

Guidelines for good distractors:
- Distractors must be plausible, not silly
- Distractors should reflect common reasoning errors:
  - reversing causality
  - overgeneralizing a true rule
  - confusing related but distinct concepts
  - applying a concept in the wrong context
- Avoid obviously wrong answer choices
- Avoid repeated wording patterns that make the correct answer stand out

Question-writing constraints:
- Use four options only: A, B, C, D
- Exactly one option must be best
- Avoid ambiguity
- Avoid requiring external diagrams or tables
- Avoid “all of the above” and “none of the above”
- Avoid clues from length, tone, or specificity alone

Explanation constraints:
- State why the correct answer is correct
- Briefly say why the distractors are wrong or what misconception they reflect
- Keep it compact, but genuinely useful

Output requirements for EACH item:
- Set "topic_id" to the provided topic_id exactly
- Set "mode" to "science"
- Set "difficulty" to "{difficulty}" unless the final item clearly better matches easy or hard
- Set "model" to a placeholder string if unknown
- Set "prompt_version" to "{SCIENCE_PROMPT_VERSION}"
- If you do not know question_id, set it to an empty string and downstream code may fill it

Output diversity requirements:
- Vary the stem structure across questions
- Vary the misconception pattern across questions
- Do not produce duplicate or near-duplicate answer choices across items
- Do not ask the same underlying question in slightly different wording

{schema_block}
""".strip()



# ---------------------------------------------------------------------------
# CARS generation
# ---------------------------------------------------------------------------


def build_cars_generation_prompt(
    topic: dict[str, Any],
    difficulty: str = "hard",
    num_questions: int = 4,
    passage_word_target: int = 450,
) -> str:
    """
    Build a CARS-generation prompt.

    Produces:
    - an original passage
    - a set of MCAT-style reading questions
    - distractors based on realistic reading errors
    """
    topic_block = format_topic_block(topic)
    schema_block = build_json_only_instruction(CARS_OUTPUT_SCHEMA)

    return f"""
You are generating one MCAT CARS practice set.

Target topic:
{topic_block}

Task requirements:
- Mode: cars
- Desired difficulty: {difficulty}
- Write ONE original passage of roughly {passage_word_target} words.
- Then write exactly {num_questions} MCAT-style questions about that passage.
- The passage must be original and should feel like a humanities/social thought passage that rewards careful reasoning.
- The writing should support subtle inference, tone analysis, argument structure, and main-idea questions.

Passage-writing requirements:
- The passage should have a clear thesis or argumentative center, but not state every implication explicitly.
- Include nuance, contrast, and some rhetorical texture.
- Avoid highly technical outside knowledge.
- Avoid requiring specialized factual background.
- The passage should be rich enough that plausible misreadings are possible.

Question-writing requirements:
- Questions should be difficult and realistic.
- Test skills such as:
  - identifying thesis or central claim
  - inference from local context
  - reasoning beyond the text without going beyond what is supported
  - author's tone and attitude
  - function of a paragraph or sentence
  - distinction between primary and secondary claims
- Avoid trivial detail questions unless they support an inferential task.

Distractor requirements:
Build distractors from realistic CARS reading errors, such as:
- too extreme
- unsupported inference
- tone distortion
- secondary point mistaken for thesis
- partially true but not answering the question
- attractive paraphrase of a wrong idea

Important:
- Wrong answers should sound plausible to a rushed but capable student.
- Do not make distractors absurd or factually random.
- Exactly one answer should be best for each question.

Explanation requirements:
- For each question, explain why the correct answer is best.
- Briefly identify the reading error behind each wrong answer pattern where possible.

Output requirements:
- Set "topic_id" to the provided topic_id exactly.
- Set "mode" to "cars".
- Set "difficulty" to "{difficulty}" unless a different difficulty is more accurate.
- Set "model" to a placeholder string if unknown.
- Set "prompt_version" to "{CARS_PROMPT_VERSION}".
- If you do not know question_set_id, set it to an empty string and downstream code may fill it.

{schema_block}
""".strip()


# ---------------------------------------------------------------------------
# Critique / evaluation prompts
# ---------------------------------------------------------------------------


def build_generation_critique_prompt(
    topic: dict[str, Any],
    generated_object: dict[str, Any],
    mode: str,
) -> str:
    """
    Build a critique prompt for small-scale prompt optimization.

    This is meant for optimize_prompt.py, not large-scale generation.
    """
    topic_block = format_topic_block(topic)
    generated_block = json.dumps(generated_object, ensure_ascii=False, indent=2)
    schema_block = build_json_only_instruction(CRITIQUE_OUTPUT_SCHEMA)

    return f"""
You are evaluating a generated MCAT item for quality control during prompt optimization.

Topic:
{topic_block}

Mode:
{mode}

Generated output to critique:
{generated_block}

Your job:
Evaluate whether this item is good enough to keep as a model target for future bulk generation prompts.

Evaluate these dimensions:
1. Schema compliance
2. Conceptual correctness
3. MCAT realism
4. Distractor plausibility
5. Difficulty appropriateness
6. Clarity and lack of ambiguity
7. Explanation usefulness
8. Alignment to the provided topic

Scoring guidance:
- 9-10: strong item, very realistic, minimal revision needed
- 7-8: solid item but prompt could be improved
- 5-6: mixed quality, substantial revision needed
- 1-4: poor item, reject or heavily revise prompt

Verdict guidance:
- keep = strong enough prompt behavior
- revise = some promise but prompt should change
- reject = prompt is producing poor outputs for this kind of topic

Be concrete in your criticism:
- Point out if distractors are obviously weak
- Point out if the explanation is shallow
- Point out if the question is too easy, too factual, too vague, or not MCAT-like
- Point out if a CARS question relies on unsupported interpretation
- Point out if a science question does not really use misconception-first design

{schema_block}
""".strip()


def build_prompt_comparison_prompt(
    topic: dict[str, Any],
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    mode: str,
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    """
    Build a comparison prompt for evaluating two generations or two prompt variants.
    """
    topic_block = format_topic_block(topic)
    a_block = json.dumps(candidate_a, ensure_ascii=False, indent=2)
    b_block = json.dumps(candidate_b, ensure_ascii=False, indent=2)

    comparison_schema = {
        "winner": f"{label_a}|{label_b}|tie",
        "reasoning": ["string"],
        "better_on_realism": f"{label_a}|{label_b}|tie",
        "better_on_distractors": f"{label_a}|{label_b}|tie",
        "better_on_difficulty_targeting": f"{label_a}|{label_b}|tie",
        "recommended_changes": ["string"],
    }

    schema_block = build_json_only_instruction(comparison_schema)

    return f"""
You are comparing two candidate MCAT generations for the same topic.

Topic:
{topic_block}

Mode:
{mode}

Candidate {label_a}:
{a_block}

Candidate {label_b}:
{b_block}

Choose which candidate is better overall for future bulk generation calibration.

Criteria:
- MCAT realism
- conceptual correctness
- distractor plausibility
- difficulty targeting
- explanation usefulness
- alignment to topic
- lack of ambiguity

{schema_block}
""".strip()


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def get_prompt_version_for_mode(mode: str) -> str:
    """Return canonical prompt version for a generation mode."""
    mode = mode.strip().lower()
    if mode == "science":
        return SCIENCE_PROMPT_VERSION
    if mode == "cars":
        return CARS_PROMPT_VERSION
    raise ValueError(f"Unsupported mode: {mode}")


def build_generation_prompt(
    topic: dict[str, Any],
    mode: str,
    difficulty: str = "medium",
    num_questions: int = 4,
    passage_word_target: int = 450,
    science_num_questions: int = 5,
) -> str:
    """
    Unified prompt builder for generation code.
    """
    mode = mode.strip().lower()
    if mode == "science":
        return build_science_generation_prompt(
            topic=topic,
            difficulty=difficulty,
            num_questions=science_num_questions,
        )
    if mode == "cars":
        return build_cars_generation_prompt(
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions,
            passage_word_target=passage_word_target,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def get_schema_hint_for_mode(mode: str) -> str:
    """
    Return a schema hint string for the given mode.
    """
    mode = mode.strip().lower()
    if mode == "science":
        return schema_to_pretty_json(SCIENCE_BATCH_OUTPUT_SCHEMA)
    if mode == "cars":
        return schema_to_pretty_json(CARS_OUTPUT_SCHEMA)
    raise ValueError(f"Unsupported mode: {mode}")
