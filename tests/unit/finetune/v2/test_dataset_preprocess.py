# tests/unit/finetune/v2/test_dataset_preprocess.py
"""Unit tests for dataset_preprocess format functions."""

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

QUESTION = "What is the total revenue in Q3?"
REASONING = "I see the revenue column. Q3 row shows 4.2M."
ANSWER = "The total revenue in Q3 is $4.2 million."
IMAGE_PATH = "/data/docs/report_q3.png"
TOOLS_JSON = '{"tools": ["extract_table"]}'


# ---------------------------------------------------------------------------
# Existing functions (regression guard)
# ---------------------------------------------------------------------------


class TestFormatVisionSft:
    def test_structure(self):
        from src.finetune.v2.dataset_preprocess import format_vision_sft

        result = format_vision_sft(IMAGE_PATH, QUESTION, ANSWER)
        assert "messages" in result
        msgs = result["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_image_token_in_user(self):
        from src.finetune.v2.dataset_preprocess import format_vision_sft

        result = format_vision_sft(IMAGE_PATH, QUESTION, ANSWER)
        user_content = result["messages"][1]["content"]
        assert IMAGE_PATH in user_content
        assert "<image>" in user_content

    def test_answer_in_assistant(self):
        from src.finetune.v2.dataset_preprocess import format_vision_sft

        result = format_vision_sft(IMAGE_PATH, QUESTION, ANSWER)
        assert result["messages"][2]["content"] == ANSWER

    def test_tools_in_system_when_provided(self):
        from src.finetune.v2.dataset_preprocess import format_vision_sft

        result = format_vision_sft(IMAGE_PATH, QUESTION, ANSWER, TOOLS_JSON)
        assert TOOLS_JSON in result["messages"][0]["content"]


class TestFormatNoToolSft:
    def test_structure(self):
        from src.finetune.v2.dataset_preprocess import format_no_tool_sft

        result = format_no_tool_sft(QUESTION, ANSWER)
        msgs = result["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_query_and_answer(self):
        from src.finetune.v2.dataset_preprocess import format_no_tool_sft

        result = format_no_tool_sft(QUESTION, ANSWER)
        assert result["messages"][1]["content"] == QUESTION
        assert result["messages"][2]["content"] == ANSWER


# ---------------------------------------------------------------------------
# format_cot_sft
# ---------------------------------------------------------------------------


class TestFormatCotSft:
    def test_returns_messages_dict(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        result = format_cot_sft(QUESTION, REASONING, ANSWER)
        assert isinstance(result, dict)
        assert "messages" in result

    def test_message_roles(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        msgs = format_cot_sft(QUESTION, REASONING, ANSWER)["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_think_tags_wrap_reasoning(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        content = format_cot_sft(QUESTION, REASONING, ANSWER)["messages"][2]["content"]
        assert "<think>" in content
        assert REASONING in content
        assert "</think>" in content

    def test_answer_follows_think_block(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        content = format_cot_sft(QUESTION, REASONING, ANSWER)["messages"][2]["content"]
        think_end = content.index("</think>")
        answer_pos = content.index(ANSWER)
        assert answer_pos > think_end

    def test_question_in_user_turn(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        msgs = format_cot_sft(QUESTION, REASONING, ANSWER)["messages"]
        assert QUESTION in msgs[1]["content"]

    def test_image_path_prepended_to_user_turn(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        msgs = format_cot_sft(QUESTION, REASONING, ANSWER, image_path=IMAGE_PATH)["messages"]
        user_content = msgs[1]["content"]
        assert f"<image>{IMAGE_PATH}</image>" in user_content
        assert QUESTION in user_content

    def test_no_image_path_no_image_tag(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        msgs = format_cot_sft(QUESTION, REASONING, ANSWER)["messages"]
        assert "<image>" not in msgs[1]["content"]

    def test_tools_json_in_system(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        msgs = format_cot_sft(QUESTION, REASONING, ANSWER, tools_json=TOOLS_JSON)["messages"]
        assert TOOLS_JSON in msgs[0]["content"]

    def test_no_tools_json_no_tools_in_system(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        msgs = format_cot_sft(QUESTION, REASONING, ANSWER)["messages"]
        assert "Available tools" not in msgs[0]["content"]

    def test_think_block_before_answer_in_assistant(self):
        from src.finetune.v2.dataset_preprocess import format_cot_sft

        content = format_cot_sft(QUESTION, REASONING, ANSWER)["messages"][2]["content"]
        assert content.index("<think>") < content.index(ANSWER)


# ---------------------------------------------------------------------------
# format_dpo_pair
# ---------------------------------------------------------------------------


class TestFormatDpoPair:
    CHOSEN_REASONING = "Carefully examined table. Row 3 col 2 = 4.2M."
    CHOSEN_ANSWER = "Q3 revenue is $4.2M."
    REJECTED_REASONING = "Looked at Q2 instead."
    REJECTED_ANSWER = "Q3 revenue is $3.1M."

    def _call(self, **kwargs):
        from src.finetune.v2.dataset_preprocess import format_dpo_pair

        return format_dpo_pair(
            QUESTION,
            self.CHOSEN_REASONING,
            self.CHOSEN_ANSWER,
            self.REJECTED_REASONING,
            self.REJECTED_ANSWER,
            **kwargs,
        )

    def test_top_level_keys(self):
        result = self._call()
        assert set(result.keys()) == {"prompt", "chosen", "rejected"}

    def test_prompt_is_list_of_two_messages(self):
        prompt = self._call()["prompt"]
        assert isinstance(prompt, list)
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"

    def test_question_in_user_message(self):
        prompt = self._call()["prompt"]
        assert QUESTION in prompt[1]["content"]

    def test_chosen_is_string_with_think(self):
        chosen = self._call()["chosen"]
        assert isinstance(chosen, str)
        assert "<think>" in chosen
        assert self.CHOSEN_REASONING in chosen
        assert "</think>" in chosen
        assert self.CHOSEN_ANSWER in chosen

    def test_rejected_is_string_with_think(self):
        rejected = self._call()["rejected"]
        assert isinstance(rejected, str)
        assert "<think>" in rejected
        assert self.REJECTED_REASONING in rejected
        assert "</think>" in rejected
        assert self.REJECTED_ANSWER in rejected

    def test_chosen_and_rejected_differ(self):
        result = self._call()
        assert result["chosen"] != result["rejected"]

    def test_image_path_in_user_turn(self):
        prompt = self._call(image_path=IMAGE_PATH)["prompt"]
        assert f"<image>{IMAGE_PATH}</image>" in prompt[1]["content"]

    def test_no_image_path_no_image_tag(self):
        prompt = self._call()["prompt"]
        assert "<image>" not in prompt[1]["content"]

    def test_tools_json_in_system(self):
        prompt = self._call(tools_json=TOOLS_JSON)["prompt"]
        assert TOOLS_JSON in prompt[0]["content"]

    def test_chosen_answer_follows_think_block(self):
        chosen = self._call()["chosen"]
        think_end = chosen.index("</think>")
        answer_pos = chosen.index(self.CHOSEN_ANSWER)
        assert answer_pos > think_end

    def test_rejected_answer_follows_think_block(self):
        rejected = self._call()["rejected"]
        think_end = rejected.index("</think>")
        answer_pos = rejected.index(self.REJECTED_ANSWER)
        assert answer_pos > think_end


# ---------------------------------------------------------------------------
# format_insight_sft
# ---------------------------------------------------------------------------


class TestFormatInsightSft:
    INSIGHT_CATEGORY = "anomaly"
    INSIGHT_TEXT = "Revenue in Q3 is 30% above the quarterly average — possible seasonal spike."
    VIZ_DIRECTIVE = "bar_chart: quarterly revenue comparison"

    def _call(self, **kwargs):
        from src.finetune.v2.dataset_preprocess import format_insight_sft

        return format_insight_sft(
            QUESTION,
            REASONING,
            self.INSIGHT_CATEGORY,
            self.INSIGHT_TEXT,
            ANSWER,
            **kwargs,
        )

    def test_returns_messages_dict(self):
        result = self._call()
        assert isinstance(result, dict)
        assert "messages" in result

    def test_message_roles(self):
        msgs = self._call()["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_think_block_present(self):
        content = self._call()["messages"][2]["content"]
        assert "<think>" in content
        assert REASONING in content
        assert "</think>" in content

    def test_insight_block_present(self):
        content = self._call()["messages"][2]["content"]
        assert '<insight category="anomaly">' in content
        assert self.INSIGHT_TEXT in content
        assert "</insight>" in content

    def test_answer_present(self):
        content = self._call()["messages"][2]["content"]
        assert ANSWER in content

    def test_order_think_then_insight_then_answer(self):
        content = self._call()["messages"][2]["content"]
        think_pos = content.index("<think>")
        insight_pos = content.index("<insight")
        answer_pos = content.index(ANSWER)
        assert think_pos < insight_pos < answer_pos

    def test_viz_directive_inserted_when_provided(self):
        content = self._call(viz_directive=self.VIZ_DIRECTIVE)["messages"][2]["content"]
        assert "<viz>" in content
        assert self.VIZ_DIRECTIVE in content
        assert "</viz>" in content

    def test_viz_block_between_insight_and_answer(self):
        content = self._call(viz_directive=self.VIZ_DIRECTIVE)["messages"][2]["content"]
        insight_end = content.index("</insight>")
        viz_pos = content.index("<viz>")
        answer_pos = content.index(ANSWER)
        assert insight_end < viz_pos < answer_pos

    def test_no_viz_directive_no_viz_tag(self):
        content = self._call()["messages"][2]["content"]
        assert "<viz>" not in content

    def test_tools_json_in_system(self):
        msgs = self._call(tools_json=TOOLS_JSON)["messages"]
        assert TOOLS_JSON in msgs[0]["content"]

    def test_insight_category_attribute_in_tag(self):
        content = self._call()["messages"][2]["content"]
        assert f'category="{self.INSIGHT_CATEGORY}"' in content

    def test_question_in_user_turn(self):
        msgs = self._call()["messages"]
        assert QUESTION in msgs[1]["content"]
