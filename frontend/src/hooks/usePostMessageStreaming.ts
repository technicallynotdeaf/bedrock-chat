import { fetchAuthSession } from 'aws-amplify/auth';
import { PostMessageRequest } from '../@types/conversation';
import { create } from 'zustand';
import i18next from 'i18next';
import { StreamingEvent } from './xstates/streaming';
import { PostStreamingStatus } from '../constants';

const WS_ENDPOINT: string = import.meta.env.VITE_APP_WS_ENDPOINT;
// API Gateway WebSocket supports up to 128KB per message.
// Use 100KB chunks to leave room for the JSON wrapper (step, index fields).
const CHUNK_SIZE = 100 * 1024; // 100KB
// Max chunks to send in parallel before waiting for acks.
// Keeps concurrent Lambda invocations low to avoid throttling.
const CHUNK_BATCH_SIZE = 5;
// Number of times to retry a failed chunk before giving up.
const MAX_CHUNK_RETRIES = 3;
// Delay between retries in milliseconds.
const CHUNK_RETRY_DELAY_MS = 2000;

const usePostMessageStreaming = create<{
  post: (params: {
    input: PostMessageRequest;
    hasKnowledge?: boolean;
    handleStreamingEvent: (event: StreamingEvent) => void;
  }) => Promise<void>;
  errorDetail: string | null;
}>((set) => {
  return {
    errorDetail: null,
    post: async ({ input, handleStreamingEvent }) => {
      handleStreamingEvent({ type: 'wakeup' });

      const token = (await fetchAuthSession()).tokens?.idToken?.toString();
      const payloadString = JSON.stringify({
        ...input,
        token,
      });

      // chunking
      const chunkedPayloads: string[] = [];
      const chunkCount = Math.ceil(payloadString.length / CHUNK_SIZE);
      for (let i = 0; i < chunkCount; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, payloadString.length);
        chunkedPayloads.push(payloadString.substring(start, end));
      }

      // Track which chunk indices have been acknowledged.
      // This allows accurate retry logic without double-counting.
      const ackedChunks = new Set<number>();
      let nextBatchStart = 0;
      // true once all chunks are acked and END has been sent
      let chunkingComplete = false;
      // Track retry attempts per chunk index
      const chunkRetries = new Map<number, number>();

      const sendChunks = (ws: WebSocket, indices: number[]) => {
        console.log(
          `[FRONTEND_WS] Sending chunks [${indices.join(',')}] of ${chunkedPayloads.length}`
        );
        for (const i of indices) {
          ws.send(
            JSON.stringify({
              step: PostStreamingStatus.BODY,
              index: i,
              part: chunkedPayloads[i],
            })
          );
        }
      };

      const sendNextBatch = (ws: WebSocket) => {
        if (nextBatchStart >= chunkedPayloads.length) return;
        const batchEnd = Math.min(
          nextBatchStart + CHUNK_BATCH_SIZE,
          chunkedPayloads.length
        );
        const indices = [];
        for (let i = nextBatchStart; i < batchEnd; i++) {
          indices.push(i);
        }
        nextBatchStart = batchEnd;
        sendChunks(ws, indices);
      };

      const checkAllAcked = (ws: WebSocket) => {
        if (ackedChunks.size === chunkedPayloads.length) {
          chunkingComplete = true;
          console.log(
            `[FRONTEND_WS] All ${chunkedPayloads.length} chunks acknowledged — sending END`
          );
          ws.send(
            JSON.stringify({
              step: PostStreamingStatus.END,
              token: token,
            })
          );
          return true;
        }
        return false;
      };

      // Handle a successful ack for a specific chunk index
      const handleAck = (ws: WebSocket, index: number) => {
        ackedChunks.add(index);
        if (checkAllAcked(ws)) return;
        // If all chunks in the current batch are acked, send next batch
        if (ackedChunks.size >= nextBatchStart) {
          sendNextBatch(ws);
        }
      };

      // Handle a failed chunk: retry the specific index
      const handleChunkFailure = (ws: WebSocket, failedIndex: number, detail: string) => {
        const retries = (chunkRetries.get(failedIndex) || 0) + 1;
        chunkRetries.set(failedIndex, retries);
        console.warn(
          `[FRONTEND_WS] Chunk ${failedIndex} failed: ${detail} ` +
            `(retry ${retries}/${MAX_CHUNK_RETRIES}, ` +
            `${ackedChunks.size}/${chunkedPayloads.length} acked)`
        );
        if (retries > MAX_CHUNK_RETRIES) {
          throw new Error(
            'Failed to upload document after multiple retries. ' +
              'The file may be too large. Please try a smaller document.'
          );
        }
        setTimeout(() => {
          if (ws.readyState === WebSocket.OPEN) {
            sendChunks(ws, [failedIndex]);
          }
        }, CHUNK_RETRY_DELAY_MS);
      };

      // When we get an error but don't know which chunk it belongs to,
      // find the un-acked chunks in the current batch and retry them.
      let unknownErrorCount = 0;
      const handleUnknownChunkFailure = (ws: WebSocket, detail: string) => {
        unknownErrorCount++;
        // Find un-acked chunks that have been sent
        const unacked: number[] = [];
        for (let i = 0; i < nextBatchStart; i++) {
          if (!ackedChunks.has(i)) unacked.push(i);
        }
        console.warn(
          `[FRONTEND_WS] Unknown chunk failure: ${detail} ` +
            `(${unknownErrorCount} unknown errors, ` +
            `${unacked.length} un-acked chunks: [${unacked.join(',')}])`
        );
        // Wait a bit then retry all un-acked chunks
        if (unacked.length > 0 && unknownErrorCount <= MAX_CHUNK_RETRIES * unacked.length) {
          setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
              // Only retry chunks that still haven't been acked
              const stillUnacked = unacked.filter((i) => !ackedChunks.has(i));
              if (stillUnacked.length > 0) {
                sendChunks(ws, stillUnacked);
              } else if (!chunkingComplete) {
                checkAllAcked(ws);
              }
            }
          }, CHUNK_RETRY_DELAY_MS);
        } else if (unknownErrorCount > MAX_CHUNK_RETRIES * Math.max(unacked.length, 1)) {
          throw new Error(
            'Failed to upload document after multiple retries. ' +
              'The file may be too large. Please try a smaller document.'
          );
        }
      };

      return new Promise<void>((resolve, reject) => {
        const ws = new WebSocket(WS_ENDPOINT);

        ws.onopen = () => {
          console.log('[FRONTEND_WS] WebSocket connection opened');
          ws.send(
            JSON.stringify({
              step: PostStreamingStatus.START,
              token: token,
            })
          );
        };

        ws.onmessage = (message) => {
          try {
            console.log('[FRONTEND_WS] Received message:', message.data);
            if (
              message.data === '' ||
              message.data === 'Message sent.' ||
              // Ignore timeout message from api gateway
              message.data.startsWith(
                '{"message": "Endpoint request timed out",'
              )
            ) {
              return;
            } else if (message.data === 'Session started.') {
              sendNextBatch(ws);
              return;
            } else if (message.data === 'Message part received.') {
              // Legacy ack format (no index) — count it like before
              // This handles the case where backend hasn't been updated yet
              handleAck(ws, ackedChunks.size);
              return;
            } else if (message.data === 'Error.' && !chunkingComplete) {
              handleUnknownChunkFailure(ws, 'Lambda returned Error');
              return;
            }

            // Try to parse as JSON
            let data;
            try {
              data = JSON.parse(message.data);
            } catch {
              // Non-JSON message during chunking — treat as chunk error
              if (!chunkingComplete) {
                handleUnknownChunkFailure(ws, message.data);
                return;
              }
              throw new Error(i18next.t('error.predict.invalidResponse'));
            }

            // Handle indexed ack: {"ack": <index>}
            if (data.ack !== undefined && !chunkingComplete) {
              handleAck(ws, data.ack);
              return;
            }

            // Handle API Gateway error messages during chunk delivery
            if (!chunkingComplete && data.message && !data.status) {
              handleUnknownChunkFailure(ws, data.message);
              return;
            }

            console.log('[FRONTEND_WS] Parsed data:', data);

            if (data.status) {
              console.log('[FRONTEND_WS] Processing status:', data.status);
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
                  console.log(
                    '[FRONTEND_WS] Received STREAMING_END, ending thinking state'
                  );
                  try {
                    console.log(
                      '[FRONTEND_WS] Calling handleStreamingEvent goodbye'
                    );
                    handleStreamingEvent({
                      type: 'goodbye',
                    });
                    console.log(
                      '[FRONTEND_WS] handleStreamingEvent goodbye completed'
                    );

                    console.log('[FRONTEND_WS] Closing WebSocket');
                    ws.close();
                    console.log('[FRONTEND_WS] WebSocket closed successfully');
                  } catch (error) {
                    console.error(
                      '[FRONTEND_WS] Error in STREAMING_END handling:',
                      error
                    );
                    ws.close();
                  }
                  break;
                case PostStreamingStatus.ERROR:
                  ws.close();
                  console.error(data);
                  set({
                    errorDetail:
                      data.reason || i18next.t('error.predict.invalidResponse'),
                  });
                  throw new Error(
                    data.reason || i18next.t('error.predict.invalidResponse')
                  );
                default:
                  handleStreamingEvent({
                    type: 'reset',
                  });
                  break;
              }
            } else {
              ws.close();
              console.error(data);
              throw new Error(i18next.t('error.predict.invalidResponse'));
            }
          } catch (e) {
            console.error('[FRONTEND_WS] Error in onmessage handler:', e);
            console.error(
              '[FRONTEND_WS] Message data that caused error:',
              message.data
            );
            reject(i18next.t('error.predict.general'));
          }
        };

        ws.onerror = (e) => {
          console.error('[FRONTEND_WS] WebSocket error:', e);
          ws.close();
          reject(i18next.t('error.predict.general'));
        };
        ws.onclose = (event) => {
          console.log(
            '[FRONTEND_WS] WebSocket closed:',
            event.code,
            event.reason
          );
          resolve();
        };
      });
    },
  };
});

export default usePostMessageStreaming;
