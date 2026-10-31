export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // 1. Enforce API Endpoint Route
    if (request.method !== "POST" || !url.pathname.endsWith("/v1/messages")) {
      return new Response(JSON.stringify({ error: "Not Found" }), { status: 404 });
    }

    // 2. Authentication Check (Validates custom key set by user)
    const clientKey = request.headers.get("x-api-key") || 
                      request.headers.get("authorization")?.replace("Bearer ", "");

    if (!env.CUSTOM_PROXY_KEY || clientKey !== env.CUSTOM_PROXY_KEY) {
      return new Response(
        JSON.stringify({ error: { type: "authentication_error", message: "Invalid API Key" } }), 
        { status: 401, headers: { "Content-Type": "application/json" } }
      );
    }

    try {
      const anthropicBody = await request.json();

      // 3. Dynamic Model Selection Strategy
      // Map Claude models requested by Claude Code to Cloudflare Workers AI models
      let targetModel = env.DEFAULT_MODEL;

      const requestedModel = anthropicBody.model || "";
      if (requestedModel.includes("deepseek") || requestedModel.includes("r1")) {
        targetModel = "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b";
      } else if (requestedModel.includes("llama")) {
        targetModel = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
      } else if (requestedModel.startsWith("@cf/")) {
        // Direct pass-through if ANTHROPIC_MODEL was set to a Cloudflare identifier
        targetModel = requestedModel;
      }

      // 4. Translate Anthropic Message Payload to System/User Prompt Array
      const messages = [];
      if (anthropicBody.system) {
        messages.push({
          role: "system",
          content: typeof anthropicBody.system === "string" 
            ? anthropicBody.system 
            : JSON.stringify(anthropicBody.system)
        });
      }

      for (const msg of anthropicBody.messages || []) {
        let textContent = "";
        if (typeof msg.content === "string") {
          textContent = msg.content;
        } else if (Array.isArray(msg.content)) {
          textContent = msg.content
            .filter(item => item.type === "text")
            .map(item => item.text)
            .join("\n");
        }

        messages.push({
          role: msg.role === "assistant" ? "assistant" : "user",
          content: textContent
        });
      }

      // 5. Invoke Cloudflare Workers AI
      const aiResponse = await env.AI.run(targetModel, {
        messages: messages,
        stream: true,
        max_tokens: anthropicBody.max_tokens || 4096
      });

      // 6. Transform Stream to Anthropic-compliant Server-Sent Events (SSE)
      const { readable, writable } = new TransformStream();
      const writer = writable.getWriter();
      const encoder = new TextEncoder();

      (async () => {
        // Init event stream expected by Anthropic client
        await writer.write(encoder.encode(`event: message_start\ndata: ${JSON.stringify({
          type: "message_start",
          message: {
            id: "msg_" + Date.now(),
            type: "message",
            role: "assistant",
            model: targetModel,
            content: [],
            stop_reason: null,
            usage: { input_tokens: 0, output_tokens: 0 }
          }
        })}\n\n`));

        await writer.write(encoder.encode(`event: content_block_start\ndata: ${JSON.stringify({
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "" }
        })}\n\n`));

        // Read and format chunks from Cloudflare AI binding
        const reader = aiResponse.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop(); // Hold onto partial line

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.replace("data: ", "").trim();
              if (dataStr === "[DONE]") continue;

              try {
                const parsed = JSON.parse(dataStr);
                const textChunk = parsed.response || parsed.choices?.[0]?.delta?.content || "";
                
                if (textChunk) {
                  await writer.write(encoder.encode(`event: content_block_delta\ndata: ${JSON.stringify({
                    type: "content_block_delta",
                    index: 0,
                    delta: { type: "text_delta", text: textChunk }
                  })}\n\n`));
                }
              } catch (e) {
                // Ignore parse errors on incomplete chunk reads
              }
            }
          }
        }

        // Send closing SSE events
        await writer.write(encoder.encode(`event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n`));
        await writer.write(encoder.encode(`event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}\n\n`));
        await writer.write(encoder.encode(`event: message_stop\ndata: {"type":"message_stop"}\n\n`));

        await writer.close();
      })();

      return new Response(readable, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
          "anthropic-version": "2023-06-01"
        }
      });

    } catch (err) {
      return new Response(JSON.stringify({ error: { type: "api_error", message: err.message } }), { status: 500 });
    }
  }
};
