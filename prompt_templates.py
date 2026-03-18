"""
prompt_templates.py

Improved prompt builders for MCAT question generation, validation, and repair.

Key improvements over prior version:
- Stronger anti-duplication constraints
- More realistic MCAT science mix: concept, mechanism, experiment, data reasoning
- Better distractor instructions
- Separate validator prompts for science and CARS
- Dedicated repair prompts
- Richer schemas with skill tags, question types, validation score placeholders
- Tighter JSON-only enforcement
"""

from __future__ import annotations

import json
from typing import Any


SCIENCE_PROMPT_VERSION = "science_v2"
CARS_PROMPT_VERSION = "cars_v2"
SCIENCE_VALIDATOR_VERSION = "science_validator_v1"
CARS_VALIDATOR_VERSION = "cars_validator_v1"
SCIENCE_REPAIR_VERSION = "science_repair_v1"
CARS_REPAIR_VERSION = "cars_repair_v1"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

SCIENCE_ITEM_SCHEMA = {
    "question_id": "string",
    "topic_id": "string",
    "source_topic_title": "string",
    "mode": "science",
    "skill": "knowledge|reasoning|research|data",
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
    "tags": ["string"],
    "model": "string",
    "prompt_version": "string",
    "validation_score": "number",
}

SCIENCE_BATCH_SCHEMA = {
    "items": [SCIENCE_ITEM_SCHEMA]
}

CARS_QUESTION_SCHEMA = {
    "question": "string",
    "question_type": (
        "main_idea|detail|inference|function|tone|analogy|application|"
        "argument_structure|author_perspective"
    ),
    "options": {
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string",
    },
    "correct_answer": "A|B|C|D",
    "explanation": "string",
}

CARS_SET_SCHEMA = {
    "question_set_id": "string",
    "topic_id": "string",
    "source_topic_title": "string",
    "mode": "cars",
    "passage": "string",
    "questions": [CARS_QUESTION_SCHEMA],
    "difficulty": "easy|medium|hard",
    "tags": ["string"],
    "model": "string",
    "prompt_version": "string",
    "validation_score": "number",
}

SCIENCE_VALIDATION_SCHEMA = {
    "overall_score": "integer 1-10",
    "verdict": "keep|revise|reject",
    "schema_valid": "true|false",
    "strengths": ["string"],
    "schema_issues": ["string"],
    "content_issues": ["string"],
    "distractor_issues": ["string"],
    "difficulty_assessment": "string",
    "realism_assessment": "string",
    "duplication_risk": "low|medium|high",
    "required_fixes": ["string"],
}

CARS_VALIDATION_SCHEMA = {
    "overall_score": "integer 1-10",
    "verdict": "keep|revise|reject",
    "schema_valid": "true|false",
    "strengths": ["string"],
    "schema_issues": ["string"],
    "passage_issues": ["string"],
    "question_issues": ["string"],
    "distractor_issues": ["string"],
    "difficulty_assessment": "string",
    "realism_assessment": "string",
    "duplication_risk": "low|medium|high",
    "required_fixes": ["string"],
}

SCIENCE_REPAIR_OUTPUT_SCHEMA = SCIENCE_ITEM_SCHEMA
CARS_REPAIR_OUTPUT_SCHEMA = CARS_SET_SCHEMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def schema_to_pretty_json(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=2)


def format_topic_block(topic: dict[str, Any]) -> str:
    return (
        f"topic_id: {topic.get('topic_id', '')}\n"
        f"category: {topic.get('category', '')}\n"
        f"subcategory: {topic.get('subcategory', '')}\n"
        f"title: {topic.get('title', '')}\n"
        f"content_to_test: {topic.get('content_to_test', '')}\n"
        f"tags: {json.dumps(topic.get('tags', []), ensure_ascii=False)}\n"
        f"adaptive_score: {topic.get('adaptive_score', 1.0)}\n"
        f"last_seen: {topic.get('last_seen', 0.0)}"
    )


def build_json_only_instruction(schema: dict[str, Any]) -> str:
    return (
        "Return valid JSON only.\n"
        "Do not include markdown fences.\n"
        "Do not include commentary before or after the JSON.\n"
        "Do not include trailing commas.\n"
        "Do not omit required keys.\n"
        "Use double quotes for all JSON keys and string values.\n"
        "Your output must match this schema reminder exactly in structure:\n"
        f"{schema_to_pretty_json(schema)}"
    )


def build_batch_uniqueness_block(num_items: int) -> str:
    return f"""
Batch-level uniqueness requirements:
- You are generating a batch of {num_items} items.
- Every item must test a meaningfully different angle of the topic.
- Do not write paraphrases of the same underlying question.
- Avoid repeating the same misconception pattern across more than one item unless the stem/task is substantially different.
- Vary question style across the batch:
  - some direct concept questions
  - some mechanism questions
  - some application questions
  - some experiment or study interpretation questions
  - some data or graph/table-free reasoning questions
- Do not let the correct answer occupy the same letter position too often; distribute correct answers naturally.
""".strip()


def build_explanation_quality_block() -> str:
    return """
Explanation requirements:
- The explanation must identify why the correct answer is best.
- The explanation must briefly explain why EACH incorrect option is wrong.
- Keep explanations concise, but specific enough to teach.
- Do not just restate the correct answer.
""".strip()


# ---------------------------------------------------------------------------
# Science generation
# ---------------------------------------------------------------------------

def build_science_generation_prompt(
    topic: dict[str, Any],
    difficulty: str = "medium",
    num_questions: int = 5,
) -> str:
    topic_block = format_topic_block(topic)
    schema_block = build_json_only_instruction(SCIENCE_BATCH_SCHEMA)
    uniqueness_block = build_batch_uniqueness_block(num_questions)
    explanation_block = build_explanation_quality_block()

    return f"""
You are generating high-quality MCAT science multiple-choice questions.

Target topic:
{topic_block}

Task:
- Generate exactly {num_questions} DISTINCT standalone MCAT-style science questions.
- Return one JSON object with a single top-level key: "items".
- "items" must be a list of exactly {num_questions} question objects.
- Each item must align tightly to the provided topic and content_to_test.

Difficulty target:
- Desired difficulty: {difficulty}
- Use mostly {difficulty}-level items, but if one item is naturally a bit easier or harder, that is acceptable.

MCAT realism requirements:
- Questions should feel like realistic MCAT science practice, not classroom trivia.
- Prefer questions that test:
  - conceptual understanding
  - causal reasoning
  - mechanism
  - experimental interpretation
  - application of principles
  - distinction between closely related concepts
- Avoid pure fact-recall unless it is still discriminative.
- Avoid overly specialized facts not appropriate for the MCAT.

Skill mix:
Across the batch, include a natural mix of items with "skill" values such as:
- "knowledge" for direct conceptual understanding
- "reasoning" for multi-step inference
- "research" for interpreting an experiment or study design
- "data" for quantitative or result-based reasoning without needing an actual figure

Distractor design process:
For EACH question, silently follow this process before writing the final item:
1. Identify the core concept being tested.
2. Identify 2-4 realistic misconceptions or near-miss confusions.
3. Build answer choices from those misconceptions.
4. Write the stem so the best answer is uniquely supported.

Good distractors must:
- be plausible to a strong but imperfect MCAT student
- reflect realistic reasoning errors
- not be silly, irrelevant, or obviously false
- not differ from the correct answer only by tiny wording tricks
- not make the correct answer stand out by length, precision, or tone

Common wrong-answer patterns to use:
- confusing related but distinct concepts
- reversing cause and effect
- applying a valid rule in the wrong setting
- overgeneralizing a true principle
- choosing a partially true statement that does not answer the stem
- misreading what variable or condition is actually changing

Question-writing constraints:
- Use four options only: A, B, C, D
- Exactly one option must be best
- Avoid ambiguity
- Avoid “all of the above” and “none of the above”
- Avoid questions requiring an external figure, table, or diagram
- Avoid giving away the answer through wording asymmetry
- Avoid repeated stem openings across the batch

{uniqueness_block}

{explanation_block}

Output requirements for EACH item:
- Set "question_id" to an empty string if unknown
- Set "topic_id" to the provided topic_id exactly
- Set "source_topic_title" to the provided title exactly
- Set "mode" to "science"
- Set "difficulty" to "{difficulty}" unless another difficulty is clearly more accurate
- Set "tags" to a short list of relevant topic tags
- Set "model" to a placeholder string if unknown
- Set "prompt_version" to "{SCIENCE_PROMPT_VERSION}"
- Set "validation_score" to 0

Final quality checks before output:
- No duplicates or near-duplicates in the batch
- No item with more than one plausible best answer
- No weak distractors
- No trivia-only items
- No malformed JSON

{schema_block}
""".strip()


# ---------------------------------------------------------------------------
# CARS generation
# ---------------------------------------------------------------------------

def build_cars_generation_prompt(
    topic: dict[str, Any],
    difficulty: str = "hard",
    num_questions: int = 10,
    passage_word_target: int = 600,
) -> str:
    topic_block = format_topic_block(topic)
    schema_block = build_json_only_instruction(CARS_SET_SCHEMA)
    explanation_block = build_explanation_quality_block()

    return f"""
You are generating one high-quality MCAT CARS practice set.

Target domain / theme control:
{topic_block}

Task:
- Write ONE original passage of about {passage_word_target} words.
- Then write exactly {num_questions} MCAT CARS-style questions about that passage.
- Return a single JSON object.

Important note about topic usage:
- For CARS, the topic should guide the passage domain, theme, or intellectual setting.
- Do NOT turn the topic into factual recall.
- The passage should be answerable using reasoning from the text itself, not outside knowledge.

Passage requirements:
- The passage must be original.
- It should feel like a serious humanities or social-thought passage.
- It should contain a clear argumentative center, but not be simplistic.
- Include nuance, tension, contrast, qualification, or subtle shifts in tone.
- Make the passage rich enough that strong but rushed readers could make realistic mistakes.
- Avoid technical jargon requiring specialist knowledge.
- Avoid fake-scientific exposition; keep it appropriate for CARS.
- The author should have a discernible viewpoint, but not every implication should be explicit.

Question requirements:
Across the set, include a realistic spread of question types such as:
- main idea
- local inference
- author tone
- paragraph or sentence function
- argument structure
- author perspective
- analogy or application
- distinguishing thesis from supporting or secondary claims

Distractor requirements:
Wrong answers should be highly plausible and reflect realistic reading mistakes:
- too extreme
- unsupported inference
- tone distortion
- secondary point mistaken for thesis
- partly true but not responsive
- attractive paraphrase of a wrong interpretation
- broad claim not sufficiently supported by the passage

Question-writing constraints:
- Use four options only: A, B, C, D
- Exactly one option must be best
- Avoid trivial detail-only questions unless they support deeper reasoning
- Avoid questions whose answer depends on external knowledge
- Avoid making the correct answer obviously more nuanced or longer than the others
- Vary the style and focus of questions across the set

CARS-specific explanation requirements:
- Explain why the correct answer is best supported by the passage.
- Briefly explain the reading mistake behind each wrong answer.
- Keep the explanation concise but genuinely useful.

{explanation_block}

Output requirements:
- Set "question_set_id" to an empty string if unknown
- Set "topic_id" to the provided topic_id exactly
- Set "source_topic_title" to the provided title exactly
- Set "mode" to "cars"
- Set "difficulty" to "{difficulty}" unless another difficulty is more accurate
- Set "tags" to a short list of relevant passage-domain tags
- Set "model" to a placeholder string if unknown
- Set "prompt_version" to "{CARS_PROMPT_VERSION}"
- Set "validation_score" to 0

Final quality checks before output:
- Passage is coherent, nuanced, and readable
- Questions are not redundant
- No question has more than one plausible best answer
- Wrong answers are realistic
- No external knowledge is required
- JSON is valid

{schema_block}
""".strip()


# ---------------------------------------------------------------------------
# Science validator
# ---------------------------------------------------------------------------

def build_science_validator_prompt(
    topic: dict[str, Any],
    generated_item: dict[str, Any],
    prior_items: list[dict[str, Any]] | None = None,
) -> str:
    topic_block = format_topic_block(topic)
    generated_block = json.dumps(generated_item, ensure_ascii=False, indent=2)
    prior_block = json.dumps(prior_items or [], ensure_ascii=False, indent=2)
    schema_block = build_json_only_instruction(SCIENCE_VALIDATION_SCHEMA)

    return f"""
You are a strict MCAT science quality-control reviewer.

Target topic:
{topic_block}

Candidate item:
{generated_block}

Previously accepted items from this topic (for duplication checking):
{prior_block}

Your job:
Evaluate whether this item should be kept, revised once, or rejected.

You must judge all of the following:
1. Schema compliance
2. Scientific correctness
3. Alignment to the specific topic
4. MCAT realism
5. Distractor plausibility
6. Single-best-answer quality
7. Explanation usefulness
8. Duplication or near-duplication risk relative to prior items

Be strict.
Reject or revise items that have any of these problems:
- factual error
- ambiguous stem
- more than one plausible best answer
- distractors that are obviously weak or silly
- trivia-only question with little reasoning value
- explanation that does not explain why wrong answers are wrong
- poor alignment to the stated topic
- near-duplicate of an existing accepted item

Scoring guidance:
- 9-10: strong keep
- 7-8: revise only if there are limited fixable issues
- 5-6: borderline, likely revise or reject depending on issue severity
- 1-4: reject

Verdict guidance:
- keep = strong and usable as-is
- revise = fixable in one repair pass
- reject = fundamentally weak, incorrect, too duplicate, or not MCAT-like

When identifying issues:
- Put JSON/schema problems under "schema_issues"
- Put factual, conceptual, ambiguity, or topic problems under "content_issues"
- Put answer-choice weaknesses under "distractor_issues"
- Put only concrete repairable actions under "required_fixes"

{schema_block}
""".strip()


# ---------------------------------------------------------------------------
# CARS validator
# ---------------------------------------------------------------------------

def build_cars_validator_prompt(
    topic: dict[str, Any],
    generated_set: dict[str, Any],
    prior_sets: list[dict[str, Any]] | None = None,
) -> str:
    topic_block = format_topic_block(topic)
    generated_block = json.dumps(generated_set, ensure_ascii=False, indent=2)
    prior_block = json.dumps(prior_sets or [], ensure_ascii=False, indent=2)
    schema_block = build_json_only_instruction(CARS_VALIDATION_SCHEMA)

    return f"""
You are a strict MCAT CARS quality-control reviewer.

Target domain / theme control:
{topic_block}

Candidate CARS set:
{generated_block}

Previously accepted CARS sets from related topics (for duplication checking):
{prior_block}

Your job:
Evaluate whether this CARS set should be kept, revised once, or rejected.

You must judge all of the following:
1. Schema compliance
2. Passage quality and coherence
3. Passage suitability for CARS reasoning
4. Question realism
5. Distractor plausibility
6. Single-best-answer quality
7. Explanation usefulness
8. Duplication risk

Be strict.
Reject or revise if you see issues such as:
- passage is flat, generic, or too simple
- passage requires outside knowledge
- passage lacks enough nuance to support the questions
- questions are redundant
- questions are detail-only and not CARS-like
- wrong answers are too obvious
- more than one answer is plausible
- explanations are shallow
- passage or questions are near-duplicates of prior sets

Scoring guidance:
- 9-10: strong keep
- 7-8: revise if issues are limited and fixable
- 5-6: borderline, likely revise or reject
- 1-4: reject

Verdict guidance:
- keep = usable as-is
- revise = one repair pass could make it good
- reject = fundamentally weak or not CARS-like

When identifying issues:
- Put JSON/schema problems under "schema_issues"
- Put passage-level problems under "passage_issues"
- Put question-level problems under "question_issues"
- Put answer-choice weaknesses under "distractor_issues"
- Put only concrete repairable actions under "required_fixes"

{schema_block}
""".strip()


# ---------------------------------------------------------------------------
# Science repair
# ---------------------------------------------------------------------------

def build_science_repair_prompt(
    topic: dict[str, Any],
    generated_item: dict[str, Any],
    validator_feedback: dict[str, Any],
) -> str:
    topic_block = format_topic_block(topic)
    generated_block = json.dumps(generated_item, ensure_ascii=False, indent=2)
    feedback_block = json.dumps(validator_feedback, ensure_ascii=False, indent=2)
    schema_block = build_json_only_instruction(SCIENCE_REPAIR_OUTPUT_SCHEMA)

    return f"""
You are repairing an MCAT science item based on validator feedback.

Target topic:
{topic_block}

Original item:
{generated_block}

Validator feedback:
{feedback_block}

Your job:
Produce a corrected version of the item.

Repair rules:
- Preserve the same topic alignment.
- Preserve the same overall intent if possible.
- Fix only the issues identified by the validator.
- Ensure the repaired item has exactly one best answer.
- Strengthen weak distractors if needed.
- Correct any scientific inaccuracies.
- Improve the explanation so it states why the correct answer is right and why the incorrect answers are wrong.
- Preserve valid fields when possible.
- Keep the output in the same JSON structure.

If the item is fundamentally unsalvageable, still return your best repaired version rather than commentary.

Set:
- "prompt_version" to "{SCIENCE_REPAIR_VERSION}"
- "validation_score" to 0

{schema_block}
""".strip()


# ---------------------------------------------------------------------------
# CARS repair
# ---------------------------------------------------------------------------

def build_cars_repair_prompt(
    topic: dict[str, Any],
    generated_set: dict[str, Any],
    validator_feedback: dict[str, Any],
) -> str:
    topic_block = format_topic_block(topic)
    generated_block = json.dumps(generated_set, ensure_ascii=False, indent=2)
    feedback_block = json.dumps(validator_feedback, ensure_ascii=False, indent=2)
    schema_block = build_json_only_instruction(CARS_REPAIR_OUTPUT_SCHEMA)

    return f"""
You are repairing an MCAT CARS question set based on validator feedback.

Target domain / theme control:
{topic_block}

Original CARS set:
{generated_block}

Validator feedback:
{feedback_block}

Your job:
Produce a corrected version of the set.

Repair rules:
- Preserve the overall passage domain/theme alignment.
- Keep the set passage-based and reasoning-based.
- Fix only the issues identified by the validator.
- If the passage is weak, improve it enough to support the questions.
- If questions are weak or redundant, rewrite them.
- Ensure every question has exactly one best answer.
- Strengthen distractors so they reflect realistic reading mistakes.
- Improve explanations so they identify why the correct answer is best and why the wrong answers fail.
- Keep the output in the same JSON structure.

Set:
- "prompt_version" to "{CARS_REPAIR_VERSION}"
- "validation_score" to 0

{schema_block}
""".strip()


# ---------------------------------------------------------------------------
# Comparison prompt
# ---------------------------------------------------------------------------

def build_prompt_comparison_prompt(
    topic: dict[str, Any],
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    mode: str,
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    topic_block = format_topic_block(topic)
    a_block = json.dumps(candidate_a, ensure_ascii=False, indent=2)
    b_block = json.dumps(candidate_b, ensure_ascii=False, indent=2)

    comparison_schema = {
        "winner": f"{label_a}|{label_b}|tie",
        "reasoning": ["string"],
        "better_on_realism": f"{label_a}|{label_b}|tie",
        "better_on_distractors": f"{label_a}|{label_b}|tie",
        "better_on_difficulty_targeting": f"{label_a}|{label_b}|tie",
        "better_on_explanations": f"{label_a}|{label_b}|tie",
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

Choose which candidate is better overall.

Criteria:
- realism
- correctness
- distractor plausibility
- difficulty targeting
- explanation usefulness
- topic alignment
- lack of ambiguity
- uniqueness

{schema_block}
""".strip()


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def get_prompt_version_for_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode == "science":
        return SCIENCE_PROMPT_VERSION
    if mode == "cars":
        return CARS_PROMPT_VERSION
    raise ValueError(f"Unsupported mode: {mode}")


def get_validator_version_for_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode == "science":
        return SCIENCE_VALIDATOR_VERSION
    if mode == "cars":
        return CARS_VALIDATOR_VERSION
    raise ValueError(f"Unsupported mode: {mode}")


def build_generation_prompt(
    topic: dict[str, Any],
    mode: str,
    difficulty: str = "medium",
    num_questions: int = 10,
    passage_word_target: int = 600,
    science_num_questions: int = 5,
) -> str:
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


def build_validator_prompt(
    topic: dict[str, Any],
    generated_object: dict[str, Any],
    mode: str,
    prior_objects: list[dict[str, Any]] | None = None,
) -> str:
    mode = mode.strip().lower()
    if mode == "science":
        return build_science_validator_prompt(
            topic=topic,
            generated_item=generated_object,
            prior_items=prior_objects,
        )
    if mode == "cars":
        return build_cars_validator_prompt(
            topic=topic,
            generated_set=generated_object,
            prior_sets=prior_objects,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def build_repair_prompt(
    topic: dict[str, Any],
    generated_object: dict[str, Any],
    validator_feedback: dict[str, Any],
    mode: str,
) -> str:
    mode = mode.strip().lower()
    if mode == "science":
        return build_science_repair_prompt(
            topic=topic,
            generated_item=generated_object,
            validator_feedback=validator_feedback,
        )
    if mode == "cars":
        return build_cars_repair_prompt(
            topic=topic,
            generated_set=generated_object,
            validator_feedback=validator_feedback,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def get_schema_hint_for_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode == "science":
        return schema_to_pretty_json(SCIENCE_BATCH_SCHEMA)
    if mode == "cars":
        return schema_to_pretty_json(CARS_SET_SCHEMA)
    raise ValueError(f"Unsupported mode: {mode}")