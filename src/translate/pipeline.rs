use crate::error::{ProxyError, ProxyResult};
use crate::models::{anthropic, openai};
use crate::translate::core;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct TranslationPolicy {
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,
    pub model_map: BTreeMap<String, String>,
    pub ignore_terms: Vec<String>,
}

pub fn translate_request(
    req: anthropic::AnthropicRequest,
    policy: &TranslationPolicy,
) -> ProxyResult<openai::OpenAIRequest> {
    let model = select_model(&req, policy);

    let mut openai_messages = Vec::new();

    if let Some(system) = req.system {
        match system {
            anthropic::SystemPrompt::Single(text) => {
                openai_messages.push(openai::Message {
                    role: "system".to_string(),
                    content: Some(openai::MessageContent::Text(sanitize_prompt(
                        text,
                        &policy.ignore_terms,
                    ))),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            anthropic::SystemPrompt::Multiple(messages) => {
                for msg in messages {
                    openai_messages.push(openai::Message {
                        role: "system".to_string(),
                        content: Some(openai::MessageContent::Text(sanitize_prompt(
                            msg.text,
                            &policy.ignore_terms,
                        ))),
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
        }
    }

    for msg in req.messages {
        openai_messages.extend(core::translate_message(msg)?);
    }

    openai_messages = coalesce_system_messages(openai_messages);

    let tools = req.tools.and_then(|tools| {
        let filtered: Vec<_> = tools
            .into_iter()
            .filter(|t| !core::is_batch_tool(t))
            .map(core::translate_tool)
            .collect();

        if filtered.is_empty() {
            None
        } else {
            Some(filtered)
        }
    });

    Ok(openai::OpenAIRequest {
        model,
        messages: openai_messages,
        max_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences,
        stream: req.stream,
        stream_options: req.stream.and_then(|stream| {
            stream.then_some(openai::StreamOptions {
                include_usage: true,
            })
        }),
        tools,
        tool_choice: None,
        chat_template_kwargs: chat_template_kwargs_from_env(),
    })
}

// Build the upstream's chat_template_kwargs blob from environment variables.
// Right now only QWEN_DISABLE_THINKING is supported: when set to a truthy
// value, emits {"enable_thinking": false} so Qwen's chat template skips the
// reasoning preamble (saves tokens and latency for tool-calling workloads).
// Returns None when no relevant env var is set, so the field is omitted from
// the upstream request and behavior matches upstream's default.
fn chat_template_kwargs_from_env() -> Option<Value> {
    let mut kwargs = serde_json::Map::new();

    if let Ok(raw) = std::env::var("QWEN_DISABLE_THINKING") {
        if matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ) {
            kwargs.insert("enable_thinking".to_string(), json!(false));
        }
    }

    if kwargs.is_empty() {
        None
    } else {
        Some(Value::Object(kwargs))
    }
}

pub fn translate_response(
    resp: openai::OpenAIResponse,
    fallback_model: &str,
) -> ProxyResult<anthropic::AnthropicResponse> {
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| ProxyError::Transform("No choices in response".to_string()))?;

    let mut content = Vec::new();

    if let Some(text) = &choice.message.content {
        if !text.is_empty() {
            content.push(anthropic::ResponseContent::Text {
                content_type: "text".to_string(),
                text: text.clone(),
            });
        }
    }

    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            let input: Value =
                serde_json::from_str(&tool_call.function.arguments).unwrap_or_else(|_| json!({}));

            content.push(anthropic::ResponseContent::ToolUse {
                content_type: "tool_use".to_string(),
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
            });
        }
    }

    let stop_reason = choice
        .finish_reason
        .as_ref()
        .map(|r| match r.as_str() {
            "tool_calls" => "tool_use",
            "stop" => "end_turn",
            "length" => "max_tokens",
            _ => "end_turn",
        })
        .map(String::from);

    Ok(anthropic::AnthropicResponse {
        id: resp.id.unwrap_or_else(|| "msg_proxy".to_string()),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model.unwrap_or_else(|| fallback_model.to_string()),
        stop_reason,
        stop_sequence: None,
        usage: anthropic::Usage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens,
        },
    })
}

pub fn translate_models_list(resp: openai::ModelsListResponse) -> anthropic::ModelsListResponse {
    let data: Vec<_> = resp
        .data
        .into_iter()
        .map(|model| anthropic::ModelInfo {
            created_at: "1970-01-01T00:00:00Z".to_string(),
            display_name: model.id.clone(),
            id: model.id,
            model_type: "model".to_string(),
        })
        .collect();

    let first_id = data.first().map(|m| m.id.clone());
    let last_id = data.last().map(|m| m.id.clone());

    anthropic::ModelsListResponse {
        data,
        first_id,
        has_more: false,
        last_id,
    }
}

fn select_model(req: &anthropic::AnthropicRequest, policy: &TranslationPolicy) -> String {
    let has_thinking = req
        .extra
        .get("thinking")
        .and_then(|v| v.as_object())
        .map(|o| o.get("type").and_then(|t| t.as_str()) == Some("enabled"))
        .unwrap_or(false);

    let model = if has_thinking {
        policy
            .reasoning_model
            .clone()
            .unwrap_or_else(|| req.model.clone())
    } else {
        policy
            .completion_model
            .clone()
            .unwrap_or_else(|| req.model.clone())
    };

    policy.model_map.get(&model).cloned().unwrap_or(model)
}

// Many OpenAI-compatible servers (e.g. Swisscom AI Platform) reject requests
// where a `role: "system"` message appears after a non-system one, with
// "System message must be at the beginning." Claude Code v2.x sometimes
// injects skills/reminder text mid-conversation as `role: "system"` entries
// inside `messages[]`. Merge every system entry (regardless of original
// position) into a single system message at index 0 so the upstream stays
// happy without losing content.
fn coalesce_system_messages(messages: Vec<openai::Message>) -> Vec<openai::Message> {
    let has_system = messages.iter().any(|m| m.role == "system");
    if !has_system {
        return messages;
    }

    let mut system_parts: Vec<String> = Vec::new();
    let mut rest: Vec<openai::Message> = Vec::with_capacity(messages.len());

    for msg in messages {
        if msg.role == "system" {
            match msg.content {
                Some(openai::MessageContent::Text(text)) => {
                    if !text.is_empty() {
                        system_parts.push(text);
                    }
                }
                Some(openai::MessageContent::Parts(parts)) => {
                    for part in parts {
                        if let openai::ContentPart::Text { text } = part {
                            if !text.is_empty() {
                                system_parts.push(text);
                            }
                        }
                    }
                }
                None => {}
            }
        } else {
            rest.push(msg);
        }
    }

    if system_parts.is_empty() {
        return rest;
    }

    let merged = openai::Message {
        role: "system".to_string(),
        content: Some(openai::MessageContent::Text(system_parts.join("\n\n"))),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };

    let mut out = Vec::with_capacity(rest.len() + 1);
    out.push(merged);
    out.extend(rest);
    out
}

fn sanitize_prompt(text: String, terms: &[String]) -> String {
    let mut sanitized = text;
    let mut removed = Vec::new();

    for term in terms {
        let next = core::remove_term(&sanitized, term);
        if next != sanitized {
            sanitized = next;
            removed.push(term.clone());
        }
    }

    if !removed.is_empty() {
        tracing::debug!(
            "Removed configured system prompt terms for upstream compatibility: {}",
            removed.join("; ")
        );
    }

    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use serde_json::json;

    fn policy_from(config: &Config) -> TranslationPolicy {
        TranslationPolicy {
            reasoning_model: config.reasoning_model.clone(),
            completion_model: config.completion_model.clone(),
            model_map: config.model_map.clone(),
            ignore_terms: config.system_prompt_ignore_terms.clone(),
        }
    }

    fn default_policy() -> TranslationPolicy {
        policy_from(&Config::default())
    }

    #[test]
    fn applies_model_map_after_selection() {
        let req = anthropic::AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("pong".to_string()),
            }],
            max_tokens: 64,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(false),
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let policy = TranslationPolicy {
            model_map: [("claude-opus-4-6".to_string(), "openai/gpt-4.1".to_string())]
                .into_iter()
                .collect(),
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();
        assert_eq!(openai.model, "openai/gpt-4.1");
    }

    #[test]
    fn sanitizes_configured_system_prompt_terms() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("pong".to_string()),
            }],
            max_tokens: 64,
            system: Some(anthropic::SystemPrompt::Single(
                "Examples of risky actions: rm -rf.".to_string(),
            )),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(true),
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let policy = TranslationPolicy {
            ignore_terms: vec!["rm -rf".to_string()],
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();

        match &openai.messages[0].content {
            Some(openai::MessageContent::Text(text)) => {
                assert_eq!(text, "Examples of risky actions: .");
            }
            _ => panic!("expected sanitized system prompt"),
        }
    }

    #[test]
    fn streaming_request_includes_usage_stream_options() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(true),
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        assert_eq!(
            openai.stream_options.map(|options| options.include_usage),
            Some(true)
        );
    }

    #[test]
    fn qwen_disable_thinking_env_controls_chat_template_kwargs() {
        // Both branches live in one test to avoid env-var races with other
        // parallel tests touching QWEN_DISABLE_THINKING.
        let make_req = || anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let prev = std::env::var("QWEN_DISABLE_THINKING").ok();
        std::env::remove_var("QWEN_DISABLE_THINKING");

        let openai = translate_request(make_req(), &default_policy()).unwrap();
        assert!(
            openai.chat_template_kwargs.is_none(),
            "field must be omitted when env var is unset"
        );
        let serialized = serde_json::to_value(&openai).unwrap();
        assert!(
            serialized.get("chat_template_kwargs").is_none(),
            "serialized JSON must omit the field, got {serialized}"
        );

        std::env::set_var("QWEN_DISABLE_THINKING", "true");
        let openai = translate_request(make_req(), &default_policy()).unwrap();
        assert_eq!(
            openai.chat_template_kwargs,
            Some(json!({"enable_thinking": false}))
        );

        match prev {
            Some(v) => std::env::set_var("QWEN_DISABLE_THINKING", v),
            None => std::env::remove_var("QWEN_DISABLE_THINKING"),
        }
    }

    #[test]
    fn non_streaming_request_omits_usage_stream_options() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(false),
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        assert!(openai.stream_options.is_none());
    }

    #[test]
    fn converts_tool_definitions() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("use tool".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: Some(vec![anthropic::Tool {
                name: "read_file".to_string(),
                description: Some("Read a file".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
                tool_type: None,
            }]),
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        let tools = openai.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "read_file");
    }

    #[test]
    fn filters_batch_tools() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: Some(vec![anthropic::Tool {
                name: "batch_tool".to_string(),
                description: None,
                input_schema: json!({}),
                tool_type: Some("BatchTool".to_string()),
            }]),
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();
        assert!(openai.tools.is_none());
    }

    #[test]
    fn converts_image_content() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Blocks(vec![
                    anthropic::ContentBlock::Text {
                        text: "What is this?".to_string(),
                        cache_control: None,
                    },
                    anthropic::ContentBlock::Image {
                        source: anthropic::ImageSource {
                            source_type: "base64".to_string(),
                            media_type: "image/png".to_string(),
                            data: "iVBOR...".to_string(),
                        },
                    },
                ]),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        match &openai.messages[0].content {
            Some(openai::MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                match &parts[1] {
                    openai::ContentPart::ImageUrl { image_url } => {
                        assert!(image_url.url.starts_with("data:image/png;base64,"));
                    }
                    _ => panic!("expected image_url part"),
                }
            }
            _ => panic!("expected multi-part content"),
        }
    }

    #[test]
    fn converts_tool_use_and_tool_result() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                anthropic::Message {
                    role: "assistant".to_string(),
                    content: anthropic::MessageContent::Blocks(vec![
                        anthropic::ContentBlock::ToolUse {
                            id: "tool_1".to_string(),
                            name: "read_file".to_string(),
                            input: json!({"path": "/tmp"}),
                        },
                    ]),
                },
                anthropic::Message {
                    role: "user".to_string(),
                    content: anthropic::MessageContent::Blocks(vec![
                        anthropic::ContentBlock::ToolResult {
                            tool_use_id: "tool_1".to_string(),
                            content: anthropic::ToolResultContent::Text(
                                "file contents".to_string(),
                            ),
                            is_error: None,
                        },
                    ]),
                },
            ],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        let tool_calls = openai.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].id, "tool_1");
        assert_eq!(tool_calls[0].function.name, "read_file");

        assert_eq!(openai.messages[1].role, "tool");
        assert_eq!(openai.messages[1].tool_call_id, Some("tool_1".to_string()));
    }

    #[test]
    fn deserializes_tool_result_with_nested_content_blocks() {
        let body = json!({
            "model": "gpt-4o",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool_42",
                    "content": [
                        {"type": "text", "text": "first chunk"},
                        {"type": "text", "text": "second chunk"}
                    ]
                }]
            }]
        });

        let req: anthropic::AnthropicRequest = serde_json::from_value(body).unwrap();
        let openai = translate_request(req, &default_policy()).unwrap();

        let tool_msg = openai
            .messages
            .iter()
            .find(|m| m.role == "tool")
            .expect("expected a tool message");
        assert_eq!(tool_msg.tool_call_id, Some("tool_42".to_string()));
        match &tool_msg.content {
            Some(openai::MessageContent::Text(text)) => {
                assert_eq!(text, "first chunk\nsecond chunk");
            }
            other => panic!("expected flattened text content, got {:?}", other),
        }
    }

    #[test]
    fn converts_multiple_system_prompts() {
        let req = anthropic::AnthropicRequest {
            model: "gpt-4o".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("hi".to_string()),
            }],
            max_tokens: 100,
            system: Some(anthropic::SystemPrompt::Multiple(vec![
                anthropic::SystemMessage {
                    message_type: "text".to_string(),
                    text: "You are helpful.".to_string(),
                    cache_control: None,
                },
                anthropic::SystemMessage {
                    message_type: "text".to_string(),
                    text: "Be concise.".to_string(),
                    cache_control: None,
                },
            ])),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let openai = translate_request(req, &default_policy()).unwrap();

        // After the coalescer, multiple top-level system prompts get merged
        // into a single system message (with both bodies preserved) so that
        // strict OpenAI-compat servers don't reject the request.
        let system_msgs: Vec<_> = openai
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .collect();
        assert_eq!(system_msgs.len(), 1);
        match &system_msgs[0].content {
            Some(openai::MessageContent::Text(text)) => {
                assert!(text.contains("You are helpful."));
                assert!(text.contains("Be concise."));
            }
            other => panic!("expected merged text, got {:?}", other),
        }
    }

    #[test]
    fn coalesces_inline_system_messages_to_front() {
        // Claude Code sometimes sends role=system entries inside messages[]
        // mid-conversation. Strict OpenAI servers reject that. We must merge
        // every system entry into one and place it at index 0.
        let body = json!({
            "model": "gpt-4o",
            "max_tokens": 100,
            "system": "Top-level system prompt.",
            "messages": [
                {"role": "user", "content": "first user turn"},
                {"role": "system", "content": "Reminder: skill X is available."},
                {"role": "assistant", "content": "ok"},
                {"role": "system", "content": "Reminder: skill Y is available."},
                {"role": "user", "content": "do the thing"}
            ]
        });

        let req: anthropic::AnthropicRequest = serde_json::from_value(body).unwrap();
        let openai = translate_request(req, &default_policy()).unwrap();

        let system_indices: Vec<_> = openai
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "system")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(system_indices, vec![0]);

        match &openai.messages[0].content {
            Some(openai::MessageContent::Text(text)) => {
                assert!(text.contains("Top-level system prompt."));
                assert!(text.contains("Reminder: skill X is available."));
                assert!(text.contains("Reminder: skill Y is available."));
            }
            other => panic!("expected merged system text, got {:?}", other),
        }

        let rest_roles: Vec<_> = openai
            .messages
            .iter()
            .skip(1)
            .map(|m| m.role.as_str())
            .collect();
        assert_eq!(rest_roles, vec!["user", "assistant", "user"]);
    }

    #[test]
    fn uses_reasoning_model_when_thinking_enabled() {
        let req = anthropic::AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("think hard".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({"thinking": {"type": "enabled", "budget_tokens": 1000}}),
        };

        let policy = TranslationPolicy {
            reasoning_model: Some("gpt-4o-reasoning".to_string()),
            completion_model: Some("gpt-4o-mini".to_string()),
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();
        assert_eq!(openai.model, "gpt-4o-reasoning");
    }

    #[test]
    fn uses_completion_model_without_thinking() {
        let req = anthropic::AnthropicRequest {
            model: "claude-opus-4-6".to_string(),
            messages: vec![anthropic::Message {
                role: "user".to_string(),
                content: anthropic::MessageContent::Text("quick".to_string()),
            }],
            max_tokens: 100,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            metadata: None,
            extra: json!({}),
        };

        let policy = TranslationPolicy {
            reasoning_model: Some("gpt-4o-reasoning".to_string()),
            completion_model: Some("gpt-4o-mini".to_string()),
            ..default_policy()
        };

        let openai = translate_request(req, &policy).unwrap();
        assert_eq!(openai.model, "gpt-4o-mini");
    }

    #[test]
    fn response_with_all_fields_present() {
        let response = openai::OpenAIResponse {
            id: Some("chatcmpl-abc123".to_string()),
            object: Some("chat.completion".to_string()),
            created: Some(1700000000),
            model: Some("gpt-4o".to_string()),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some("hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "fallback-model").unwrap();
        assert_eq!(anthropic.id, "chatcmpl-abc123");
        assert_eq!(anthropic.model, "gpt-4o");
    }

    #[test]
    fn response_allows_missing_metadata() {
        let response = openai::OpenAIResponse {
            id: None,
            object: None,
            created: None,
            model: None,
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some("pong".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "openai/gpt-4o-mini").unwrap();
        assert_eq!(anthropic.id, "msg_proxy");
        assert_eq!(anthropic.model, "openai/gpt-4o-mini");
    }

    #[test]
    fn response_converts_tool_calls() {
        let response = openai::OpenAIResponse {
            id: Some("chatcmpl-1".to_string()),
            object: None,
            created: None,
            model: Some("gpt-4o".to_string()),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChoiceMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![openai::ToolCall {
                        id: "call_abc".to_string(),
                        call_type: "function".to_string(),
                        function: openai::FunctionCall {
                            name: "read_file".to_string(),
                            arguments: "{\"path\":\"/tmp\"}".to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            system_fingerprint: None,
        };

        let anthropic = translate_response(response, "fallback").unwrap();
        assert_eq!(anthropic.stop_reason, Some("tool_use".to_string()));
        assert!(!anthropic.content.is_empty());
    }

    #[test]
    fn models_list_translation() {
        let response = openai::ModelsListResponse {
            object: Some("list".to_string()),
            data: vec![
                openai::ModelInfo {
                    id: "gpt-4o-mini".to_string(),
                    object: Some("model".to_string()),
                    created: None,
                    owned_by: Some("azure".to_string()),
                },
                openai::ModelInfo {
                    id: "gpt-5-chat".to_string(),
                    object: Some("model".to_string()),
                    created: None,
                    owned_by: Some("azure".to_string()),
                },
            ],
        };

        let result = translate_models_list(response);
        assert_eq!(result.first_id.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(result.last_id.as_deref(), Some("gpt-5-chat"));
        assert!(!result.has_more);
    }

    #[test]
    fn empty_models_list() {
        let response = openai::ModelsListResponse {
            object: Some("list".to_string()),
            data: vec![],
        };
        let result = translate_models_list(response);
        assert!(result.data.is_empty());
        assert!(result.first_id.is_none());
    }
}
