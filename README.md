# Agent Evaluation System

A correct final answer is only one part of a successful agent run. Did the agent identify the customer, check refund eligibility, select the right tools, and refund only the requested items?

This repository is a hands-on learning lab for answering those questions. It pairs a LangChain refund agent with **eight evaluation scenarios**, **seven metrics**, and small examples that isolate individual evaluation concepts. Start with the examples, follow the complete evaluation flow, then adapt it to your own agent.

![DeepEval report showing task completion, step efficiency, plan quality, and plan adherence](screenshots/trace_metrics.png)

## Contents

- [How the agent works](#how-the-agent-works)
- [Quick start](#quick-start)
- [Run evaluations](#run-evaluations)
- [Evaluation metrics](#evaluation-metrics)
- [Evaluation dataset](#evaluation-dataset)
- [Read the results](#read-the-results)
- [Try the chat demo](#try-the-chat-demo)
- [Explore and extend](#explore-and-extend)
- [Project map](#project-map)
- [Limitations and troubleshooting](#limitations-and-troubleshooting)

## How the agent works

The agent handles refund requests against the sample purchase records in [`agent/data.py`](agent/data.py). Its [system prompt](agent/prompts.py) requires it to identify the customer, check that purchases are no more than 30 days old and are not in the future, and refund only the items requested.

| Tool | Purpose |
|---|---|
| `get_user_purchase_history` | Look up a customer's purchases by username |
| `get_today_date` | Read the current date for eligibility checks |
| `calculate_right_refund` | Add the prices of the requested items |
| `issue_refund_ticket` | Return a simulated ticket confirmation for the refund amount |

The tools use in-memory sample data. Issuing a ticket returns a string; it does not contact a payment provider or create an external ticket. Eligibility checks are directed by the prompt rather than enforced inside the refund tools.

[`agent/agent.py`](agent/agent.py) creates the agent with an `InMemorySaver` checkpointer and the Pydantic [`Answer`](agent/objects.py) schema:

```python
class Answer(BaseModel):
    answer: str
    conversation: bool = True
```

`answer` contains the user-facing reply. `conversation` indicates whether the agent needs another user response. Evaluation cases each receive a new thread ID to keep their conversation histories separate.

```mermaid
flowchart TD
    Dataset["Eight refund scenarios"] --> Runner["evaluations/eval_all.py"]
    Runner --> Agent["LangChain refund agent"]
    Agent <--> Tools["Purchase history · date · refund calculation · ticket"]
    Agent --> Response["Structured answer + messages and tool calls"]
    Agent -. "CallbackHandler" .-> Trace["Execution trace"]
    Trace --> TraceMetrics["Task completion · step efficiency<br/>plan quality · plan adherence"]
    Response --> TestCase["LLMTestCase + expected answer and tools"]
    TestCase --> OutputMetrics["Argument correctness · tool correctness<br/>answer correctness"]
    TraceMetrics --> Report["Scores, pass rates, and reasons"]
    OutputMetrics --> Report
```

## Quick start

You need **Python 3.12+**, **uv**, and an **OpenAI API key** with access to the models configured in the code.

### 1. Get the project

```bash
git clone https://github.com/RuturajKotwal/agent-evaluation-system
cd agent-evaluation-system
uv sync --locked
```

### 2. Configure your API key

Copy [`.env.example`](.env.example) to `.env` in the repository root, then set:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
```

The `.env` file is ignored by Git. Keep your real key there.

The refund agent and its evaluator currently both use the model identifier `gpt-6-luna`. Check these two configuration files before running:

| Role | Configuration |
|---|---|
| Refund agent | [`agent/models.py`](agent/models.py) |
| DeepEval judge | [`evaluations/model.py`](evaluations/model.py) |

If that identifier is unavailable to your account, change each file to a model supported by its respective library and your API account. The standalone multiplication examples specify their own models inside their scripts.

### 3. Run the complete evaluation

Run commands from the repository root:

```bash
uv run --env-file .env python evaluations/eval_all.py
```

This runs the refund agent once per scenario, evaluates all seven metrics, and prints reports in the terminal. Agent generation and LLM-based judging make API calls and incur usage costs.

> **Check the dates before interpreting scores.** The purchase records use fixed dates, while `get_today_date` reads your machine's current date. The expected answers are static. Align the sample dates and expected outcomes for your experiment; see [Evaluation dataset](#evaluation-dataset).

## Run evaluations

### All seven metrics

[`evaluations/eval_all.py`](evaluations/eval_all.py) is the main entry point. It performs evaluation in two stages:

1. `dataset.evals_iterator(...)` runs four metrics against the execution trace captured by `CallbackHandler`.
2. The runner collects the same run's structured answer and tool calls into an `LLMTestCase`. `evaluate(...)` then applies the remaining three metrics.

Reusing the response avoids generating a different answer for each metric. The runner disables cache reads and writes and sets `inspect_after_run=False`, so it does not automatically launch the inspector. The terminal ends with:

```text
Completed all 7 metrics for 8 test cases.
```

Completion means the evaluation finished; individual metrics can still fail. The script produces two evaluation reports, one for each stage, rather than a single combined export.

### Study one metric at a time

For an individual metric applied to the refund agent:

```bash
uv run --env-file .env python evaluations/task_completion.py
uv run --env-file .env python evaluations/tool_correctness.py
uv run --env-file .env python evaluations/answer_correctness.py
```

Each individual script invokes the agent again. Its configuration can differ from the combined runner. In particular, `evaluations/argument_correctness.py` collects tool calls but omits them from its `LLMTestCase`; use the combined runner for refund argument evaluation.

## Evaluation metrics

These thresholds are defined in `build_metrics()` in [`evaluations/eval_all.py`](evaluations/eval_all.py). Scores run from 0 to 1; a score at or above the threshold passes.

| Metric | DeepEval implementation | What it checks | Evidence | Threshold |
|---|---|---|---|---:|
| Task completion | `TaskCompletionMetric` | Whether the agent achieved the task under the stated refund rules | Execution trace + task description | 0.80 |
| Step efficiency | `StepEfficiencyMetric` | Whether the agent took unnecessary or redundant steps | Execution trace | 0.80 |
| Plan quality | `PlanQualityMetric` | Whether the agent's plan is suitable for the task | Execution trace | 0.80 |
| Plan adherence | `PlanAdherenceMetric` | Whether execution follows the agent's plan | Execution trace | 0.80 |
| Argument correctness | `ArgumentCorrectnessMetric` | Whether tool inputs are appropriate for the request | User input + actual tool calls and arguments | 0.90 |
| Tool correctness | `ToolCorrectnessMetric` | Whether the agent called the expected tools | Actual tool calls + expected tools | 0.90 |
| Answer correctness | `GEval` | Whether the answer reaches the expected logical conclusion | Input + actual answer + expected answer | 0.80 |

The refund dataset specifies expected **tool names**, without reference arguments. The combined runner also leaves exact matching and tool ordering at their defaults. Its tool correctness score therefore does not establish that arguments or execution order were correct. Inspect those separately.

The `GEval` rubric checks the logical conclusion against the reference answer. Read the reasons alongside the score to see whether details such as the refund amount were assessed as intended.

## Evaluation dataset

[`evaluations/dataset.py`](evaluations/dataset.py) defines eight DeepEval `Golden` cases. A golden describes the desired behavior before the agent runs; an `LLMTestCase` adds the actual output and observed tool calls afterward.

| Scenario | Expected behavior encoded in the dataset |
|---|---|
| `user130`: refund headphones | Issue a $35 refund ticket |
| `user123`: refund book | Deny the refund because the purchase is over 30 days old |
| `user131`: refund only the mouse | Issue a $20 ticket for the mouse only |
| `user132`: refund paintbrush and canvas | Issue a $34 ticket for both requested items |
| `user129`: ask about returning the USB drive | Quote $16 without issuing a ticket |
| Username missing | Ask for the username |
| Unknown customer `user999` | Explain that no account was found; issue no ticket |
| `user131`: request an unpurchased keyboard | Explain that the item was not found; issue no ticket |

**These expected outcomes need a consistent date reference.** Most sample purchases are dated October 2026. For example, on October 15, 2026, the October 22 book purchase is in the future, while its reference answer calls it more than 30 days old. A correct policy decision can therefore disagree with the reference answer.

For a meaningful experiment, align [`agent/data.py`](agent/data.py) and the expected answers with the date you intend to test. For repeatable experiments, adapt the date tool to return a fixed reference date and record it with your results.

## Read the results

The screenshots below show an example run over eight cases. They illustrate how to diagnose failures; they are not a reproducible benchmark or a guarantee of current performance.

| Metric | Average score | Pass rate | Passed / total |
|---|---:|---:|---:|
| Task completion | 0.73 | 57.14% | 4 / 7 |
| Step efficiency | 1.00 | 100.00% | 7 / 7 |
| Plan quality | 0.43 | 14.29% | 1 / 7 |
| Plan adherence | 0.93 | 85.71% | 6 / 7 |
| Argument correctness | 0.70 | 42.86% | 3 / 7 |
| Tool correctness | 1.00 | 100.00% | 7 / 7 |
| Answer correctness | 0.99 | 100.00% | 7 / 7 |

![DeepEval aggregate scores for argument correctness, tool correctness, and answer correctness](screenshots/single_turn_metrics.png)

Average score and pass rate answer different questions. Argument correctness averages 0.70, but only three cases meet its 0.90 threshold. High step efficiency also does not establish that the agent made a good plan: inspect the plan quality failures and their traces together.

![An unknown-customer case that passes tool and answer correctness but fails argument correctness](screenshots/failed_argument_correctness.png)

In this case, the agent reports that the amount of $45 has been issued for the backpack, whereas the water_bottle is not eligible for refund. Tool correctness and answer correctness pass, while argument correctness fails. Read the captured arguments and the judge's reason before changing the agent: the failure may expose an agent mistake, missing evaluation evidence, or a rubric that needs refinement.

LLM judgments and agent responses can vary between runs. Keep the model choices, date reference, prompts, dataset, and thresholds with any results you compare.

## Explore and extend

A suggested reading order:

1. Explore [`agent/prompts.py`](agent/prompts.py), [`agent/tools.py`](agent/tools.py), and [`agent/agent.py`](agent/agent.py) to understand the refund workflow.
2. Read [`evaluations/dataset.py`](evaluations/dataset.py) to connect each business rule to an expected outcome.
3. Follow [`evaluations/eval_all.py`](evaluations/eval_all.py) from trace capture to test case construction and scoring.
4. Inspect a failed case, change one part of the system, and rerun with the same evaluation settings.

To evaluate your own agent, replace the sample data and tools, rewrite the system prompt, and define goldens for your task. Update `SUCCESSFUL_TASK_DESCRIPTION` and the `GEval` rubric in the combined runner so the judges assess your intended behavior.

Useful additions to the dataset include purchases exactly 30 days old, future purchases, mixed eligible and ineligible items, duplicate item requests, and attempts to bypass the refund policy. Add expected tool arguments when you want stronger reference checks, and enable order or exact-match checks only when those requirements are part of your task.

## Project map

| Path | Purpose |
|---|---|
| [`agent/agent.py`](agent/agent.py) | Four-tool agent, memory, structured response, and interactive conversation helper |
| [`agent/tools.py`](agent/tools.py) | Purchase lookup, date lookup, refund calculation, and simulated ticket tools |
| [`agent/data.py`](agent/data.py) | Sample purchases for ten customers |
| [`agent/prompts.py`](agent/prompts.py) | Refund policy and conversation instructions |
| [`agent/objects.py`](agent/objects.py) | `Answer` and `PurchaseItem` Pydantic schemas |
| [`agent/models.py`](agent/models.py) | Agent model configuration |
| [`evaluations/dataset.py`](evaluations/dataset.py) | Eight refund goldens with expected answers and tools |
| [`evaluations/eval_all.py`](evaluations/eval_all.py) | Combined seven-metric evaluation runner |
| [`evaluations/model.py`](evaluations/model.py) | Judge model configuration |
| [`evaluations/`](evaluations/) | Individual refund metric scripts |
| [`screenshots/`](screenshots/) | Example terminal evaluation reports |
| [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock) | Project dependencies and locked environment |
| [`.env.example`](.env.example) | API key template |

## Limitations and troubleshooting

This is an educational prototype with a small, single-turn evaluation dataset. The interactive agent has memory, but the eight goldens do not test complete multi-turn conversations. Refund eligibility is prompt-driven, and the tools do not provide production validation or persistent storage. Unknown users or items passed directly to the calculation tool can raise lookup errors.

| Symptom | What to check |
|---|---|
| Authentication or model access error | Set `OPENAI_API_KEY` in the root `.env` and verify both model configurations |
| Unexpected date-related failures | Align purchase dates, today's date, and the expected answers |
| Import errors when launching scripts | Run the documented file commands from the repository root; the scripts use local imports |
| Argument correctness has missing evidence | Use `evaluations/eval_all.py`, which passes `tools_called` into each test case |
| Different scores on a rerun | Compare model settings, dates, traces, and judge reasons before drawing conclusions |

Start with the documented Python script commands. The package entry point declared in `pyproject.toml` points to a `main` function that is not currently implemented.