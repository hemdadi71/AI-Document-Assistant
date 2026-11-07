
import uuid

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from database.models import Conversation, Message


class ConversationNotFound(Exception):
    pass


class ConversationService:

    def __init__(self, db: Session):
        self.db = db

    def get_or_create(
        self,
        user_id: uuid.UUID,
        project_id: str,
        mode: str,
        question: str,
        conversation_id: str | None = None,
    ) -> Conversation:

        if conversation_id:
            try:
                conversation_uuid = uuid.UUID(conversation_id)
            except ValueError as exc:
                raise ValueError(
                    "Invalid conversation ID."
                ) from exc

            conversation = self.db.scalar(
                select(Conversation)
                .where(
                    Conversation.id == conversation_uuid,
                    Conversation.user_id == user_id,
                    Conversation.project_id == project_id,
                )
                .with_for_update()
            )

            if conversation is None:
                raise ConversationNotFound()

            conversation.mode = mode
            return conversation

        conversation = Conversation(
            user_id=user_id,
            project_id=project_id,
            title=question[:200],
            mode=mode,
        )

        self.db.add(conversation)
        self.db.flush()

        return conversation

    def get_history(
        self,
        conversation_id: uuid.UUID,
    ) -> list[dict]:

        # Limit model context to the latest 40 messages (20 turns) while
        # returning them in chronological order.
        messages = list(self.db.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation_id
            )
            .order_by(
                Message.created_at.desc(),
                Message.id.desc(),
            )
            .limit(40)
        ).all())
        messages.reverse()

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
            if message.role in {"user", "assistant"}
        ]

    def save_turn(
        self,
        conversation: Conversation,
        question: str,
        answer: str,
        mode: str,
        sources: list | None = None,
    ) -> None:

        # Give the user message an earlier timestamp so ordering remains
        # deterministic even when both rows are inserted in one transaction.
        user_created_at = datetime.now(timezone.utc)
        assistant_created_at = user_created_at + timedelta(microseconds=1)

        self.db.add_all([
            Message(
                conversation_id=conversation.id,
                role="user",
                content=question,
                extra_data={"mode": mode},
                created_at=user_created_at,
            ),
            Message(
                conversation_id=conversation.id,
                role="assistant",
                content=answer,
                extra_data={
                    "mode": mode,
                    "sources": sources or [],
                },
                created_at=assistant_created_at,
            ),
        ])

        conversation.updated_at = assistant_created_at

    def list_conversations(
        self,
        user_id: uuid.UUID,
        project_id: str,
        limit: int = 50,
    ) -> list[dict]:

        conversations = self.db.scalars(
            select(Conversation)
            .options(
                selectinload(Conversation.messages)
            )
            .where(
                Conversation.user_id == user_id,
                Conversation.project_id == project_id,
            )
            .order_by(
                Conversation.updated_at.desc()
            )
            .limit(limit)
        ).all()

        return [
            {
                "id": str(item.id),
                "title": item.title,
                "mode": item.mode,
                "message_count": len(item.messages),
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in conversations
        ]

    def get_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        project_id: str,
    ) -> dict:

        conversation = self.db.scalar(
            select(Conversation)
            .options(
                selectinload(Conversation.messages)
            )
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.project_id == project_id,
            )
        )

        if conversation is None:
            raise ConversationNotFound()

        return {
            "id": str(conversation.id),
            "title": conversation.title,
            "mode": conversation.mode,
            "messages": [
                {
                    "id": str(message.id),
                    "role": message.role,
                    "content": message.content,
                    "extra_data": message.extra_data,
                    "created_at": message.created_at,
                }
                for message in conversation.messages
            ],
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
        }

    def delete_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        project_id: str,
    ) -> bool:

        conversation = self.db.scalar(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.project_id == project_id,
            )
        )

        if conversation is None:
            return False

        self.db.delete(conversation)
        self.db.flush()

        return True
