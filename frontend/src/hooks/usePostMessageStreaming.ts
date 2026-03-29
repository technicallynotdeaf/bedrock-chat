import { fetchAuthSession } from 'aws-amplify/auth';
import { PostMessageRequest } from '../@types/conversation';
import { create } from 'zustand';
import i18next from 'i18next';
import { StreamingEvent } from './xstates/streaming';
import { PostStreamingStatus } from '../constants';

const WS_ENDPOINT: string = import.meta.env.VITE_APP_WS_ENDPOINT;

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
      const payloadString = JSON.stringify({ ...input, token });

      return new Promise<void>((resolve, reject) => {
        const ws = new WebSocket(WS_ENDPOINT);
        let uploadStarted = false;

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
              message.data === 'Message part received.' ||
              message.data.startsWith(
                '{"message": "Endpoint request timed out",'
              )
            ) {
              return;
            }

            // Try to parse as JSON
            let data;
            try {
              data = JSON.parse(message.data);
            } catch {
              console.warn('[WS] Unexpected non-JSON message:', message.data);
              return;
            }

            // Handle session start with pre-signed upload URL
            if (data.uploadUrl && !uploadStarted) {
              uploadStarted = true;
              fetch(data.uploadUrl, {
                method: 'PUT',
                body: payloadString,
              })
                .then((resp) => {
                  if (!resp.ok) {
                    throw new Error(`Upload failed: ${resp.status}`);
                  }
                  ws.send(
                    JSON.stringify({
                      step: PostStreamingStatus.END,
                      token: token,
                    })
                  );
                })
                .catch((err) => {
                  console.error('[WS] S3 upload failed:', err);
                  set({
                    errorDetail:
                      'Failed to upload document. Please try again.',
                  });
                  ws.close();
                  reject(i18next.t('error.predict.general'));
                });
              return;
            }

            // Handle API Gateway error messages (e.g. throttle, internal error)
            if (data.message && !data.status) {
              console.warn('[WS] API Gateway error:', data.message);
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
          ws.close();
          reject(i18next.t('error.predict.general'));
        };
        ws.onclose = () => {
          resolve();
        };
      });
    },
  };
});

export default usePostMessageStreaming;
