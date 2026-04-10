from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, SimpleQueue
from threading import Thread
from typing import TYPE_CHECKING, BinaryIO, Literal, TypedDict

import boto3
from botocore.exceptions import ClientError
from app.auth import verify_token
from app.user import User

if TYPE_CHECKING:
    from app.agents.tools.agent_tool import ToolRunResult
    from app.repositories.conversation import RecordNotFoundError
    from app.routes.schemas.conversation import ChatInput
    from app.stream import OnStopInput, OnThinking

LARGE_PAYLOAD_SUPPORT_BUCKET = os.environ["LARGE_PAYLOAD_SUPPORT_BUCKET"]

s3_client = boto3.client("s3")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _chunk_prefix(connection_id: str) -> str:
    return f"ws-chunks/{connection_id}/"


def _session_key(connection_id: str) -> str:
    return f"{_chunk_prefix(connection_id)}session.json"


def _chunk_key(connection_id: str, index: int) -> str:
    # Zero-pad so lexicographic sort matches numeric order
    return f"{_chunk_prefix(connection_id)}{index:010d}"


class _NotifyCommand(TypedDict):
    type: Literal["notify"]
    payload: bytes | BinaryIO


class _StreamTokenCommand(TypedDict):
    type: Literal["stream_token"]
    token: str
    status: str  # "STREAMING" or "REASONING"


class _FinishCommand(TypedDict):
    type: Literal["finish"]


_Command = _NotifyCommand | _StreamTokenCommand | _FinishCommand


_STREAM_BATCH_INTERVAL = 0.05  # 50 ms
_STREAM_BATCH_MAX_TOKENS = 5


class NotificationSender:
    def __init__(self, endpoint_url: str, connection_id: str) -> None:
        self.commands = SimpleQueue[_Command]()
        self.endpoint_url = endpoint_url
        self.connection_id = connection_id

    def _send(self, gatewayapi, payload: bytes | BinaryIO) -> bool:
        """Send a single payload. Returns False if the connection is gone."""
        try:
            gatewayapi.post_to_connection(
                ConnectionId=self.connection_id,
                Data=payload,
            )
            return True
        except (
            gatewayapi.exceptions.GoneException,
            gatewayapi.exceptions.ForbiddenException,
        ) as e:
            logger.exception(
                f"Shutdown the notification sender due to an exception: {e}"
            )
            return False
        except Exception as e:
            logger.exception(f"Failed to send notification: {e}")
            return True

    def _flush_token_buffer(self, gatewayapi, buf: list[str], status: str) -> bool:
        """Flush accumulated streaming tokens as a single message."""
        if not buf:
            return True
        payload = json.dumps(
            dict(status=status, completion="".join(buf))
        ).encode("utf-8")
        buf.clear()
        return self._send(gatewayapi, payload)

    def run(self):
        import boto3

        gatewayapi = boto3.client(
            "apigatewaymanagementapi",
            endpoint_url=self.endpoint_url,
        )

        token_buf: list[str] = []
        buf_status: str = "STREAMING"
        last_flush = time.monotonic()

        while True:
            # If we have buffered tokens, use a short timeout so we flush
            # promptly; otherwise block until a command arrives.
            timeout = _STREAM_BATCH_INTERVAL if token_buf else None
            try:
                command = self.commands.get(timeout=timeout)
            except Empty:
                # Timeout — flush whatever we have and loop
                if not self._flush_token_buffer(gatewayapi, token_buf, buf_status):
                    break
                last_flush = time.monotonic()
                continue

            if command["type"] == "finish":
                self._flush_token_buffer(gatewayapi, token_buf, buf_status)
                break

            if command["type"] == "stream_token":
                # If the status changed (e.g. STREAMING → REASONING), flush first
                if token_buf and command["status"] != buf_status:
                    if not self._flush_token_buffer(gatewayapi, token_buf, buf_status):
                        break
                    last_flush = time.monotonic()
                buf_status = command["status"]
                token_buf.append(command["token"])
                now = time.monotonic()
                if (
                    len(token_buf) >= _STREAM_BATCH_MAX_TOKENS
                    or (now - last_flush) >= _STREAM_BATCH_INTERVAL
                ):
                    if not self._flush_token_buffer(gatewayapi, token_buf, buf_status):
                        break
                    last_flush = now
                continue

            # Regular (non-streaming) notification — flush tokens first
            if not self._flush_token_buffer(gatewayapi, token_buf, buf_status):
                break
            last_flush = time.monotonic()

            if not self._send(gatewayapi, command["payload"]):
                break

    def finish(self):
        self.commands.put(
            {
                "type": "finish",
            }
        )

    def notify(self, payload: bytes | BinaryIO):
        self.commands.put(
            {
                "type": "notify",
                "payload": payload,
            }
        )

    def on_stream(self, token: str):
        self.commands.put({"type": "stream_token", "token": token, "status": "STREAMING"})

    def on_stop(self, arg: OnStopInput):
        logger.debug(f"[WEBSOCKET_ON_STOP] WebSocket on_stop called with: {arg}")
        payload = json.dumps(
            dict(
                status="STREAMING_END",
                completion="",
                stop_reason=arg["stop_reason"],
                token_count=dict(
                    input=arg["input_token_count"],
                    output=arg["output_token_count"],
                    cache_read_input=arg["cache_read_input_count"],
                    cache_write_input=arg["cache_write_input_count"],
                ),
                price=arg["price"],
            )
        ).encode("utf-8")

        self.notify(payload=payload)

    def on_agent_thinking(self, tool_use: OnThinking):
        payload = json.dumps(
            dict(
                status="AGENT_THINKING",
                log={
                    tool_use["tool_use_id"]: {
                        "name": tool_use["name"],
                        "input": tool_use["input"],
                    },
                },
            )
        ).encode("utf-8")

        self.notify(payload=payload)

    def on_agent_tool_result(self, run_result: ToolRunResult):
        self.notify(
            payload=json.dumps(
                dict(
                    status="AGENT_TOOL_RESULT",
                    result={
                        "toolUseId": run_result["tool_use_id"],
                        "status": run_result["status"],
                    },
                )
            ).encode("utf-8")
        )

        for related_document in run_result["related_documents"]:
            self.notify(
                payload=json.dumps(
                    dict(
                        status="AGENT_RELATED_DOCUMENT",
                        result={
                            "toolUseId": run_result["tool_use_id"],
                            "relatedDocument": related_document.to_schema().model_dump(
                                by_alias=True
                            ),
                        },
                    )
                ).encode("utf-8")
            )

    def on_reasoning(self, token: str):
        self.commands.put({"type": "stream_token", "token": token, "status": "REASONING"})


def process_chat_input(
    user: User,
    chat_input: ChatInput,
    notificator: NotificationSender,
) -> dict:
    """Process chat input and send the message to the client."""
    # Deferred imports — these pull in bedrock, agents, vector_search, etc.
    # and are only needed when actually processing a chat message (END step).
    from app.repositories.conversation import RecordNotFoundError
    from app.usecases.chat import chat

    logger.info(
        f"Processing chat input for conversation: {chat_input.conversation_id}, "
        f"model: {chat_input.message.model}"
    )

    try:
        chat(
            user=user,
            chat_input=chat_input,
            on_stream=lambda token: notificator.on_stream(
                token=token,
            ),
            on_stop=lambda arg: notificator.on_stop(arg=arg),
            on_thinking=lambda tool_use: notificator.on_agent_thinking(
                tool_use=tool_use,
            ),
            on_tool_result=lambda run_result: notificator.on_agent_tool_result(
                run_result=run_result
            ),
            on_reasoning=lambda token: notificator.on_reasoning(
                token=token,
            ),
        )

        return {"statusCode": 200, "body": "Message sent."}

    except RecordNotFoundError:
        reason = (
            f"bot {chat_input.bot_id} not found."
            if chat_input.bot_id
            else "Invalid request."
        )
        notificator.notify(
            json.dumps(dict(status="ERROR", reason=reason)).encode("utf-8")
        )
        return {"statusCode": 200, "body": "Error."}

    except Exception as e:
        logger.exception(f"Failed to run stream handler: {e}")
        reason = f"Failed to run stream handler: {e}"
        # Send the error via post_to_connection so it reaches the client even
        # when the Lambda response is ignored after the API Gateway 29-second
        # integration timeout.
        notificator.notify(
            json.dumps(dict(status="ERROR", reason=reason)).encode("utf-8")
        )
        return {"statusCode": 200, "body": "Error."}


def _cleanup_s3_chunks(connection_id: str) -> None:
    """Delete all S3 objects for this connection. Best-effort; errors are logged only."""
    try:
        prefix = _chunk_prefix(connection_id)
        paginator = s3_client.get_paginator("list_objects_v2")
        objects_to_delete = []
        for page in paginator.paginate(Bucket=LARGE_PAYLOAD_SUPPORT_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects_to_delete.append({"Key": obj["Key"]})
        if objects_to_delete:
            s3_client.delete_objects(
                Bucket=LARGE_PAYLOAD_SUPPORT_BUCKET,
                Delete={"Objects": objects_to_delete},
            )
            logger.info(f"Cleaned up {len(objects_to_delete)} S3 chunks for {connection_id}")
    except Exception as e:
        logger.warning(f"Failed to clean up S3 chunks for {connection_id}: {e}")


def handler(event, context):
    logger.info(f"Received event: {event}")
    route_key = event["requestContext"]["routeKey"]

    if route_key == "$connect":
        return {"statusCode": 200, "body": "Connected."}
    elif route_key == "$disconnect":
        return {"statusCode": 200, "body": "Disconnected."}

    connection_id = event["requestContext"]["connectionId"]
    domain_name = event["requestContext"]["domainName"]
    stage = event["requestContext"]["stage"]
    endpoint_url = f"https://{domain_name}/{stage}"
    notificator = NotificationSender(
        endpoint_url=endpoint_url,
        connection_id=connection_id,
    )

    notification_thread = Thread(
        target=lambda: notificator.run(),
        daemon=True,
    )
    notification_thread.start()
    try:
        body = json.loads(event["body"])
        step = body.get("step")
        token = body.get("token")

        # Large-payload protocol:
        # 1. Client sends START  → Lambda returns a pre-signed S3 upload URL.
        # 2. Client PUTs payload directly to S3 via the pre-signed URL.
        # 3. Client sends END  → Lambda reads payload from S3,
        #                        calls Bedrock, streams response back.
        # (Legacy chunked BODY protocol is still supported as a fallback.)
        if step == "START":
            try:
                decoded = verify_token(token)
            except Exception as e:
                logger.exception(f"Invalid token: {e}")
                notificator.notify(
                    json.dumps(dict(status="ERROR", reason="Invalid token.")).encode("utf-8")
                )
                return {"statusCode": 200, "body": "Error."}

            user_id = decoded["sub"]

            s3_client.put_object(
                Bucket=LARGE_PAYLOAD_SUPPORT_BUCKET,
                Key=_session_key(connection_id),
                Body=json.dumps({"user_id": user_id}),
            )

            # Generate a pre-signed URL so the client can upload the full
            # payload directly to S3, bypassing WebSocket chunk limits.
            upload_key = f"{_chunk_prefix(connection_id)}payload"
            presigned_url = s3_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": LARGE_PAYLOAD_SUPPORT_BUCKET,
                    "Key": upload_key,
                },
                ExpiresIn=300,  # 5 minutes
            )

            return {
                "statusCode": 200,
                "body": json.dumps({"uploadUrl": presigned_url}),
            }

        elif step == "END":
            decoded = verify_token(token)
            user = User.from_decoded_token(decoded)

            # Try reading the direct S3 upload (new protocol) first.
            # Falls back to chunk assembly if the payload key doesn't exist.
            payload_key = f"{_chunk_prefix(connection_id)}payload"
            full_message = None
            try:
                payload_obj = s3_client.get_object(
                    Bucket=LARGE_PAYLOAD_SUPPORT_BUCKET, Key=payload_key
                )
                full_message = payload_obj["Body"].read().decode("utf-8")
                logger.info(
                    f"Read direct upload payload: {len(full_message):,} chars"
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    logger.info("No direct upload found, falling back to chunk assembly")
                else:
                    raise

            if full_message is None:
                # Fallback: assemble from chunked BODY uploads (old protocol)
                chunk_prefix = _chunk_prefix(connection_id)
                paginator = s3_client.get_paginator("list_objects_v2")
                all_objects = []
                for page in paginator.paginate(
                    Bucket=LARGE_PAYLOAD_SUPPORT_BUCKET, Prefix=chunk_prefix
                ):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        if not key.endswith("session.json") and not key.endswith("payload"):
                            all_objects.append(obj)
                chunk_objects = sorted(all_objects, key=lambda obj: obj["Key"])

                logger.info(f"Number of message chunks: {len(chunk_objects)}")

                if not chunk_objects:
                    raise ValueError("No message found — nothing was uploaded.")

                expected_prefix = _chunk_prefix(connection_id)
                received_indices = []
                for obj in chunk_objects:
                    suffix = obj["Key"][len(expected_prefix):]
                    try:
                        received_indices.append(int(suffix))
                    except ValueError:
                        logger.warning(f"Unexpected S3 key format: {obj['Key']}")
                expected_indices = list(range(len(chunk_objects)))
                if received_indices != expected_indices:
                    missing = set(expected_indices) - set(received_indices)
                    logger.error(
                        f"Chunk index mismatch: expected {expected_indices}, "
                        f"got {received_indices}, missing {missing}"
                    )
                    raise ValueError(
                        f"Upload incomplete: {len(missing)} chunk(s) missing. "
                        f"Please try uploading again."
                    )

                def _read_chunk(obj: dict) -> str:
                    resp = s3_client.get_object(
                        Bucket=LARGE_PAYLOAD_SUPPORT_BUCKET, Key=obj["Key"]
                    )
                    return resp["Body"].read().decode("utf-8")

                with ThreadPoolExecutor(max_workers=max(1, min(len(chunk_objects), 20))) as executor:
                    chunks = list(executor.map(_read_chunk, chunk_objects))

                full_message = "".join(chunks)

            from app.routes.schemas.conversation import ChatInput

            chat_input = ChatInput(**json.loads(full_message))

            # Process chat first, clean up S3 afterwards so the user gets
            # their first token as soon as possible.
            result = process_chat_input(
                user=user,
                chat_input=chat_input,
                notificator=notificator,
            )

            # Best-effort cleanup in a background thread — the 1-day lifecycle
            # rule on the ws-chunks/ prefix handles anything we miss.
            Thread(
                target=_cleanup_s3_chunks,
                args=(connection_id,),
                daemon=True,
            ).start()

            return result

        else:
            # BODY step — store this chunk as an S3 object
            part_index = body["index"]
            message_part = body["part"]

            # Validate chunk index to prevent abuse (e.g. huge indices filling S3)
            if not isinstance(part_index, int) or part_index < 0 or part_index > 10_000:
                logger.warning(f"Invalid chunk index: {part_index}")
                return {"statusCode": 200, "body": "Error."}

            s3_client.put_object(
                Bucket=LARGE_PAYLOAD_SUPPORT_BUCKET,
                Key=_chunk_key(connection_id, part_index),
                Body=message_part,
            )
            return {"statusCode": 200, "body": "Message part received."}

    except Exception as e:
        logger.exception(f"Operation failed: {e}")
        notificator.notify(
            json.dumps(dict(status="ERROR", reason=str(e))).encode("utf-8")
        )
        return {"statusCode": 200, "body": "Error."}

    finally:
        notificator.finish()
        notification_thread.join(timeout=60)
