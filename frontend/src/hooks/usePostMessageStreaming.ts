import { fetchAuthSession } from 'aws-amplify/auth';
import { PostMessageRequest } from '../@types/conversation';
import { create } from 'zustand';
import i18next from 'i18next';
import { StreamingEvent } from './xstates/streaming';
import { PostStreamingStatus } from '../constants';

const WS_ENDPOINT: string = import.meta.env.VITE_APP_WS_ENDPOINT;
const CHUNK_SIZE = 32 * 1024; //32KB
// Max chunks to send in parallel before waiting for acks.
// Keeps concurrent Lambda invocations low to avoid throttling.
const CHUNK_BATCH_SIZE = 10;

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

      let receivedCount = 0;
      let nextBatchStart = 0;

      const sendNextBatch = (ws: WebSocket) => {
        if (nextBatchStart >= chunkedPayloads.length) return;
        const batchEnd = Math.min(
          nextBatchStart + CHUNK_BATCH_SIZE,
          chunkedPayloads.length
        );
        console.log(
          `[FRONTEND_WS] Sending chunk batch ${nextBatchStart}-${batchEnd - 1} of ${chunkedPayloads.length}`
        );
        for (let i = nextBatchStart; i < batchEnd; i++) {
          ws.send(
            JSON.stringify({
              step: PostStreamingStatus.BODY,
              index: i,
              part: chunkedPayloads[i],
            })
          );
        }
        nextBatchStart = batchEnd;
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
              receivedCount++;
              if (receivedCount === chunkedPayloads.length) {
                // All chunks acknowledged — send END
                ws.send(
                  JSON.stringify({
                    step: PostStreamingStatus.END,
                    token: token,
                  })
                );
              } else if (receivedCount >= nextBatchStart) {
                // Current batch fully acked — send next batch
                sendNextBatch(ws);
              }
              return;
            }

            const data = JSON.parse(message.data);
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
