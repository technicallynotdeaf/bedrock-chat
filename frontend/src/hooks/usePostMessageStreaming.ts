import { fetchAuthSession } from 'aws-amplify/auth';
import { PostMessageRequest } from '../@types/conversation';
import { create } from 'zustand';
import i18next from 'i18next';
import { StreamingEvent } from './xstates/streaming';
import { PostStreamingStatus } from '../constants';

const WS_ENDPOINT: string = import.meta.env.VITE_APP_WS_ENDPOINT;
const CHUNK_SIZE = 32 * 1024; // 32KB per WebSocket message

const usePostMessageStreaming = create<{
  post: (params: {
    input: PostMessageRequest;
    hasKnowledge?: boolean;
    handleStreamingEvent: (event: StreamingEvent) => void;
  }) => Promise<void>;
  errorDetail: string | null;
  uploadProgress: number | null; // 0–100 during upload, null when idle
}>((set) => {
  return {
    errorDetail: null,
    uploadProgress: null,
    post: async ({ input, handleStreamingEvent }) => {
      set({ uploadProgress: null });
      handleStreamingEvent({ type: 'wakeup' });

      const token = (await fetchAuthSession()).tokens?.idToken?.toString();
      const payloadString = JSON.stringify({ ...input, token });

      // Pre-split payload into chunks for the WebSocket fallback path.
      const chunkedPayloads: string[] = [];
      const chunkCount = Math.ceil(payloadString.length / CHUNK_SIZE);
      for (let i = 0; i < chunkCount; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, payloadString.length);
        chunkedPayloads.push(payloadString.substring(start, end));
      }

      return new Promise<void>((resolve, reject) => {
        const ws = new WebSocket(WS_ENDPOINT);
        let uploadComplete = false;
        // For sequential chunk fallback
        let chunkIndex = 0;
        let chunkAckCount = 0;

        const sendNextChunk = () => {
          if (chunkIndex < chunkedPayloads.length) {
            ws.send(
              JSON.stringify({
                step: PostStreamingStatus.BODY,
                index: chunkIndex,
                part: chunkedPayloads[chunkIndex],
              })
            );
            chunkIndex++;
          }
        };

        const startSequentialChunking = () => {
          console.log(
            `[WS] Starting sequential chunk upload: ${chunkedPayloads.length} chunks`
          );
          chunkIndex = 0;
          chunkAckCount = 0;
          sendNextChunk();
        };

        ws.onopen = () => {
          ws.send(
            JSON.stringify({
              step: PostStreamingStatus.START,
              token: token,
            })
          );
        };

        ws.onmessage = (message) => {
          try {
            if (
              message.data === '' ||
              message.data === 'Message sent.' ||
              message.data.startsWith(
                '{"message": "Endpoint request timed out",'
              )
            ) {
              return;
            }

            // Backward compat: old backend returns plain string instead of
            // JSON with uploadUrl. Start chunking immediately.
            if (message.data === 'Session started.') {
              set({ uploadProgress: 0 });
              startSequentialChunking();
              return;
            }

            // Handle chunk ack — send next chunk sequentially
            if (message.data === 'Message part received.') {
              chunkAckCount++;
              set({
                uploadProgress: Math.round(
                  (chunkAckCount / chunkedPayloads.length) * 100
                ),
              });
              if (chunkAckCount === chunkedPayloads.length) {
                // All chunks uploaded — send END
                uploadComplete = true;
                set({ uploadProgress: null });
                ws.send(
                  JSON.stringify({
                    step: PostStreamingStatus.END,
                    token: token,
                  })
                );
              } else {
                sendNextChunk();
              }
              return;
            }

            // Try to parse as JSON
            let data;
            try {
              data = JSON.parse(message.data);
            } catch {
              if (!uploadComplete) {
                if (chunkIndex === 0 && chunkAckCount === 0) {
                  // Haven't started chunking — unknown START response.
                  // Start chunking as a safe fallback.
                  console.warn(
                    '[WS] Unrecognized START response, starting chunked upload:',
                    message.data
                  );
                  set({ uploadProgress: 0 });
                  startSequentialChunking();
                } else {
                  // Already chunking — retry the failed chunk
                  console.warn(
                    `[WS] Non-JSON during chunking, resending chunk ${chunkIndex - 1}:`,
                    message.data
                  );
                  if (chunkIndex > chunkAckCount) {
                    chunkIndex = chunkAckCount;
                    setTimeout(() => sendNextChunk(), 500);
                  }
                }
                return;
              }
              console.warn('[WS] Unexpected non-JSON message:', message.data);
              return;
            }

            // Handle session start with pre-signed upload URL
            if (data.uploadUrl && !uploadComplete) {
              set({ uploadProgress: 0 });
              // Try direct S3 upload with progress tracking
              const xhr = new XMLHttpRequest();
              xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                  set({ uploadProgress: Math.round((e.loaded / e.total) * 100) });
                }
              };
              xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                  uploadComplete = true;
                  set({ uploadProgress: null });
                  ws.send(
                    JSON.stringify({
                      step: PostStreamingStatus.END,
                      token: token,
                    })
                  );
                } else {
                  console.warn(
                    `[WS] S3 upload returned ${xhr.status}, falling back to chunks`
                  );
                  set({ uploadProgress: 0 });
                  startSequentialChunking();
                }
              };
              xhr.onerror = () => {
                // S3 upload failed (likely CORS) — fall back to sequential
                // WebSocket chunking. Slower but doesn't need CORS.
                console.warn(
                  '[WS] S3 direct upload failed, falling back to chunked upload'
                );
                set({ uploadProgress: 0 });
                startSequentialChunking();
              };
              xhr.open('PUT', data.uploadUrl);
              xhr.send(payloadString);
              return;
            }

            // Handle API Gateway error messages during chunking
            if (!uploadComplete && data.message && !data.status) {
              console.warn('[WS] API Gateway error during chunking:', data.message);
              // Resend the last unacked chunk
              if (chunkIndex > chunkAckCount) {
                chunkIndex = chunkAckCount;
                setTimeout(() => sendNextChunk(), 500);
              }
              return;
            }

            // Handle streaming status messages
            if (data.status) {
              switch (data.status) {
                case PostStreamingStatus.AGENT_THINKING: {
                  Object.entries(data.log).forEach(([toolUseId, toolInfo]) => {
                    const typedToolInfo = toolInfo as {
                      name: string;
                      input: { [key: string]: any }; // eslint-disable-line @typescript-eslint/no-explicit-any
                    };
                    handleStreamingEvent({
                      type: 'tool-use',
                      toolUseId: toolUseId,
                      name: typedToolInfo.name,
                      input: typedToolInfo.input,
                    });
                  });
                  break;
                }
                case PostStreamingStatus.AGENT_TOOL_RESULT:
                  handleStreamingEvent({
                    type: 'tool-result',
                    toolUseId: data.result.toolUseId,
                    status: data.result.status,
                  });
                  break;
                case PostStreamingStatus.AGENT_RELATED_DOCUMENT:
                  handleStreamingEvent({
                    type: 'related-document',
                    toolUseId: data.result.toolUseId,
                    relatedDocument: data.result.relatedDocument,
                  });
                  break;
                case PostStreamingStatus.REASONING:
                  handleStreamingEvent({
                    type: 'reasoning',
                    reasoning: data.completion,
                  });
                  break;
                case PostStreamingStatus.STREAMING:
                  handleStreamingEvent({
                    type: 'text',
                    text: data.completion,
                  });
                  break;
                case PostStreamingStatus.STREAMING_END:
                  handleStreamingEvent({ type: 'goodbye' });
                  ws.close();
                  break;
                case PostStreamingStatus.ERROR:
                  ws.close();
                  set({
                    errorDetail:
                      data.reason ||
                      i18next.t('error.predict.invalidResponse'),
                  });
                  throw new Error(
                    data.reason ||
                      i18next.t('error.predict.invalidResponse')
                  );
                default:
                  handleStreamingEvent({ type: 'reset' });
                  break;
              }
            }
          } catch (e) {
            console.error('[WS] Error:', e);
            reject(i18next.t('error.predict.general'));
          }
        };

        ws.onerror = () => {
          set({ uploadProgress: null });
          ws.close();
          reject(i18next.t('error.predict.general'));
        };
        ws.onclose = () => {
          set({ uploadProgress: null });
          resolve();
        };
      });
    },
  };
});

export default usePostMessageStreaming;
