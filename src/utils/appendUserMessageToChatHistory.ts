import {ChatHistoryItem, ChatUserMessage} from "../types.js";
import type {LlamaImage} from "../evaluator/LlamaImage.js";

/**
 * Appends a user message to the chat history.
 * If the last message in the chat history is also a user message, the new message will be appended to it.
 */
export function appendUserMessageToChatHistory(
    chatHistory: readonly ChatHistoryItem[],
    message: string,
    images?: Array<string | Uint8Array | Buffer | LlamaImage>
) {
    const newChatHistory = chatHistory.slice();

    if (newChatHistory.length > 0 && newChatHistory[newChatHistory.length - 1]!.type === "user") {
        const lastUserMessage = newChatHistory[newChatHistory.length - 1]! as ChatUserMessage;

        newChatHistory[newChatHistory.length - 1] = {
            ...lastUserMessage,
            text: [lastUserMessage.text, message].join("\n\n"),
            images: images != null
                ? [...(lastUserMessage.images ?? []), ...images]
                : lastUserMessage.images
        };
    } else {
        newChatHistory.push({
            type: "user",
            text: message,
            images
        });
    }

    return newChatHistory;
}
