# Homework 3: AI Orchestration with Kestra Answers

## Answers

1. AI Copilot has access to current Kestra plugin documentation.
2. Vague, generic, or fabricated - the model guesses from training data.
3. 60-100 tokens.
4. 2-5x more.
5. 2-4x more.
6. Use traditional task-based workflows for predictability and auditability.

## Run Evidence

Kestra was started from the course module:

```bash
cd 03-orchestration
docker compose up -d
```

The homework flows were imported into the `zoomcamp` namespace and executed locally.

### Q2: Non-RAG Response

Executed:

- `1_chat_without_rag`, execution `41NIL9Nneh18if4ZhZanxO`
- `2_chat_with_rag`, execution `68nPCklOY6zJUSt2c0UXjJ`

The non-RAG answer gave broad/generic feature claims such as event-driven flows, plugin-based authentication, namespace-level access control, flow templates, and enhanced UI. These were plausible but not the specific release-note-grounded features. The RAG answer returned release-specific items such as:

- New Filters
- No-Code Dashboard Editor
- Multi-Agent AI Systems
- Fix with AI
- Human Task
- Improved Air-Gapped Support
- Dozens of New Plugins

### Q3: Token Usage

Executed `4_simple_agent` twice:

- Short summary execution: `1fMrWuI3MaeCfPVRGdoz2d`
- Long summary execution: `5GERfxRLLkB66z83LYMn8`

Measured `multilingual_agent` token usage:

| Summary length | Input tokens | Output tokens | Total tokens |
| --- | ---: | ---: | ---: |
| short | 282 | 76 | 358 |
| long | 282 | 196 | 478 |

The short run used `76` output tokens, so the closest option is `60-100 tokens`.

### Q4: Long vs Short Output Tokens

The long run used about `196 / 76 = 2.58x` as many `multilingual_agent` output tokens as the short run, so the closest option is `2-5x more`.

### Q5: Changing `english_brevity` to 3 Sentences

Temporarily changed:

```yaml
Generate exactly 1 sentence English summary
```

to:

```yaml
Generate exactly 3 sentences English summary
```

Then reran `4_simple_agent` with `summary_length=long`, execution `6BBdF8cpFXVcywgf4sF3EL`.

Measured `english_brevity` output token usage:

| Version | Output tokens |
| --- | ---: |
| 1 sentence, long summary | 44 |
| 3 sentences, long summary | 84 |

The 3-sentence version used about `84 / 44 = 1.91x` as many output tokens, so the closest option is `2-4x more`.

### Additional Agent Runs

Executed `5_web_research_agent`, execution `6lmSC92JM8D8AUKBXi1Koi`.

The flow completed and saved `research_report.md`. The log reported that the agent made autonomous decisions about:

- which searches to perform
- how many searches were needed
- how to structure the report
- when the task was complete

Token usage: `6532` tokens.

Also executed `6_multi_agent_research`, execution `2NTg81puHWRfzubyMaFgHN`.

The main `analysis` agent called the research agent tool:

```text
Tool execution request: kestra_agent_tool
arguments: {"prompt":"kestra.io company information"}
```

The research agent gathered company information, and the main agent produced structured JSON that was parsed by the `parse_results` task.

### Q6: Best Practices

For strict compliance and deterministic repeatability, a traditional task-based workflow is preferred because the execution sequence is explicit, predictable, and easier to audit.
