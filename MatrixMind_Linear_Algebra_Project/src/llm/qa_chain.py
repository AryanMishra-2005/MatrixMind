"""
RAG Question-Answering chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from src.llm.prompt_templates import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    CONVERSATIONAL_USER_TEMPLATE,
    FALLBACK_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
)
from src.retrieval.retriever import RetrievalResult, Retriever
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def is_basic_conversation(question: str) -> bool:
    """Return True only for short social/capability turns that should bypass RAG."""
    import re

    text = re.sub(r"[^a-z0-9' ]+", " ", (question or "").casefold())
    text = " ".join(text.split()).strip()
    if not text or len(text.split()) > 12:
        return False
    patterns = (
        r"^(hi|hello|hey|hii+|heya|good morning|good afternoon|good evening|good night)$",
        r"^(how are you|how are you doing|how is it going|what's up|whats up)[ ]*(today)?$",
        r"^(thanks|thank you|thank u|thx|many thanks|much appreciated)$",
        r"^(bye|goodbye|see you|see ya|talk to you later)$",
        r"^(who are you|what are you|what is your name|your name)$",
        r"^(what can you do|how can you help me|help)$",
        r"^(nice to meet you|pleased to meet you|good to see you)$",
    )
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def normalize_math_markdown(text: str) -> str:
    """Repair a few common LLM LaTeX typos without rewriting valid Markdown."""
    import re

    # Complex conjugates should not escape the asterisk.
    text = text.replace(r"^{\*}", "^*")
    text = re.sub(r"\\(alpha|beta|gamma|delta)\s*\{\\\*\}", r"\\\1^*", text)
    text = re.sub(r"\\(alpha|beta|gamma|delta)\s*\^\{\\\*\}", r"\\\1^*", text)

    # Fix common accidental double escaping of spacing commands.
    text = text.replace(r"\\;", r"\\, ")

    # Normalize 3+ backslashes used as a matrix row separator to LaTeX's \\\\.
    text = re.sub(r"(?<=\\\\)\\{1,}", r"\\\\", text)

    # Remove a dangling display-math closer after a vector expression followed by prose.
    # This specifically handles: ...\\end{pmatrix},$$ the inner product is ...
    text = re.sub(
        r"(\\end\{pmatrix\}\s*,?)\s*\$\$\s*(?=(?:the|and|where|here|then)\b)",
        r"\1 ", text, flags=re.IGNORECASE
    )
    return text


@dataclass
class QAResponse:

    question: str
    answer: str
    sources: List[str]
    retrieval_results: List[RetrievalResult]
    processing_time_ms: float
    model_used: str
    has_context: bool = True

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_scores": [
                {"source": r.source, "score": round(r.score, 4)}
                for r in self.retrieval_results
            ],
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_used": self.model_used,
            "has_context": self.has_context,
        }


class QAChain:

    def __init__(self, retriever: Retriever = None):
        self.retriever = retriever or Retriever()
        self._llm_client = None

    def _get_llm_client(self):
        """Lazy-initialize the Groq client on first use."""
        if self._llm_client is not None:
            return self._llm_client

        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "  1. Get a free key at https://console.groq.com\n"
                "  2. Add GROQ_API_KEY=your_key to your .env file"
            )

        try:
            from groq import Groq
            self._llm_client = Groq(api_key=settings.groq_api_key)
            log.info(f"✓ Groq client initialized (model: {settings.llm_model})")
        except ImportError:
            raise ImportError(
                "Groq package not found. Install it with:\n"
                "  pip install groq"
            )

        return self._llm_client

    def answer(
        self,
        question: str,
        top_k: int = None,
        threshold: float = None,
        use_graph: bool = True,
        attachments: Optional[List[tuple]] = None,
        conversation_history: Optional[List[dict]] = None,
        document_ids: list[str] | None = None,
    ) -> QAResponse:
        """`attachments` is a list of (filename, extracted_text) used as temporary context."""
        start_time = time.time()
        attachments = attachments or []

        # Social turns should not be judged against the document index.
        if is_basic_conversation(question) and not attachments:
            history = (conversation_history or [])[-settings.conversation_memory_messages:]
            history_text = "\n".join(
                f"{m.get('role', 'user')}: {str(m.get('content', ''))[:settings.conversation_context_chars]}"
                for m in history if m.get("role") in {"user", "assistant"}
            )
            prompt = CONVERSATIONAL_USER_TEMPLATE.format(question=question, history=history_text or "No previous conversation.")
            answer_text = normalize_math_markdown(self._generate(CONVERSATIONAL_SYSTEM_PROMPT, prompt))
            return QAResponse(question, answer_text, [], [], (time.time() - start_time) * 1000,
                              self._active_model(), has_context=False)

        # User-supplied attachments are the explicit subject of this turn.
        # Skip unrelated vector retrieval so old topics cannot contaminate summaries.
        prior_questions = [m.get("content", "") for m in (conversation_history or [])
                           if m.get("role") == "user"][-3:]
        retrieval_query = question + (("\nPrior user context: " + " | ".join(prior_questions)) if prior_questions else "")
        results = [] if attachments else self.retriever.retrieve(
            query=retrieval_query,
            top_k=top_k or settings.top_k_results,
            threshold=settings.similarity_threshold if threshold is None else threshold,
            use_graph=use_graph,
            document_ids=document_ids,
        )

        # Step 2: Build prompt
        has_context = len(results) > 0 or len(attachments) > 0

        if has_context:
            context = self.retriever.format_context(results)
            if attachments:
                attached = "\n\n".join(f"[Attachment: {name}]\n{text}" for name, text in attachments)
                context = (context + "\n\n" + attached).strip()
            history = (conversation_history or [])[-settings.conversation_memory_messages:]
            history_text = "\\n".join(
                f"{m.get('role', 'user')}: {str(m.get('content', ''))[:settings.conversation_context_chars]}"
                for m in history if m.get("role") in {"user", "assistant"}
            )
            if attachments:
                user_message = (
                    "USER REQUEST: " + question + "\n\n"
                    "ATTACHED MATERIAL (extracted text; preserve its order and labels):\n" + context + "\n\n"
                    "Write a complete, faithful summary of the attached material. Identify its main "
                    "message, key definitions, methods, equations, examples, and important details. Do not "
                    "introduce unrelated topics. If any text is missing or unclear, "
                    "state that limitation rather than guessing."
                )
            else:
                user_message = RAG_USER_TEMPLATE.format(context=context, question=question)
            if history_text and not attachments:
                user_message = "Conversation history (for resolving references only; source facts must come from retrieved context):\n" + history_text + "\n\n" + user_message
            if attachments:
                system_prompt = (
                    "You are a careful multimodal document and image analyst. The attached material "
                    "is the primary and authoritative subject of this request. Summarize only what is "
                    "visible or explicitly extracted from the attachment. Do not answer from unrelated "
                    "knowledge-base passages or previous topics. Preserve headings, dates, labels, "
                    "numbers, sequence, and relationships. If text is unreadable, say so. Give a complete "
                    "structured summary and do not invent missing details. Format with clean Markdown."
                )
            else:
                system_prompt = RAG_SYSTEM_PROMPT
        else:
            user_message = FALLBACK_PROMPT.format(question=question)
            system_prompt = "You are a helpful linear algebra tutor."

        # Step 3: Generate answer via Groq
        answer_text = normalize_math_markdown(self._generate(system_prompt, user_message))

        # Step 4: Build response
        sources = list({r.source for r in results} | {name for name, _ in attachments})
        elapsed_ms = (time.time() - start_time) * 1000

        return QAResponse(
            question=question,
            answer=answer_text,
            sources=sources,
            retrieval_results=results,
            processing_time_ms=elapsed_ms,
            model_used=self._active_model(),
            has_context=has_context,
        )

    def stream_answer(self, question: str, top_k: int = None, threshold: float = None,
                      use_graph: bool = True, conversation_history: Optional[List[dict]] = None,
                      document_ids: list[str] | None = None):
        """Yield text deltas from Groq; social turns bypass document retrieval."""
        if is_basic_conversation(question):
            history = (conversation_history or [])[-settings.conversation_memory_messages:]
            history_text = "\n".join(
                f"{m.get('role', 'user')}: {str(m.get('content', ''))[:settings.conversation_context_chars]}"
                for m in history if m.get("role") in {"user", "assistant"}
            )
            prompt = CONVERSATIONAL_USER_TEMPLATE.format(question=question, history=history_text or "No previous conversation.")
            system = CONVERSATIONAL_SYSTEM_PROMPT
        else:
            prior_questions = [m.get("content", "") for m in (conversation_history or [])
                               if m.get("role") == "user"][-3:]
            retrieval_query = question + (("\nPrior user context: " + " | ".join(prior_questions)) if prior_questions else "")
            results = self.retriever.retrieve(retrieval_query, top_k or settings.top_k_results,
                settings.similarity_threshold if threshold is None else threshold,
                use_graph=use_graph, document_ids=document_ids)
            if not results:
                prompt = FALLBACK_PROMPT.format(question=question)
                system = "You are a helpful linear algebra tutor."
            else:
                context = self.retriever.format_context(results)
                prompt = RAG_USER_TEMPLATE.format(context=context, question=question)
                history = (conversation_history or [])[-settings.conversation_memory_messages:]
                if history:
                    prior = "\n".join(f"{m.get('role')}: {str(m.get('content',''))[:settings.conversation_context_chars]}" for m in history
                                        if m.get("role") in {"user", "assistant"})
                    prompt = "Conversation history for reference resolution only:\n" + prior + "\n\n" + prompt
                system = RAG_SYSTEM_PROMPT
        if not settings.groq_api_key:
            response = self.answer(question, top_k=top_k, threshold=threshold,
                                   use_graph=use_graph, conversation_history=conversation_history,
                                   document_ids=document_ids)
            yield response.answer
            return
        client = self._get_llm_client()
        stream = client.chat.completions.create(model=settings.llm_model,
            max_tokens=max(settings.max_tokens, 8192), temperature=settings.temperature,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            stream=True)
        full_answer = ""
        continuation_count = 0
        try:
            while True:
                finish_reason = None
                for event in stream:
                    if not event.choices:
                        continue
                    choice = event.choices[0]
                    content = getattr(choice.delta, "content", None)
                    if content:
                        full_answer += content
                        yield content
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                if finish_reason != "length" or continuation_count >= 2:
                    break
                continuation_count += 1
                log.warning("LLM output reached token limit; requesting continuation %d", continuation_count)
                close_current = getattr(stream, "close", None)
                if close_current:
                    close_current()
                stream = client.chat.completions.create(
                    model=settings.llm_model, max_tokens=max(settings.max_tokens, 8192),
                    temperature=settings.temperature, stream=True,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": full_answer},
                        {"role": "user", "content": (
                            "Continue the answer from exactly where it stopped. Do not repeat any "
                            "previous text. Finish all remaining sections, steps, equations and the "
                            "conclusion. Keep the same source-grounding and Markdown rules."
                        )},
                    ],
                )
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

    def _active_model(self) -> str:
        return settings.llm_model if settings.groq_api_key else "Local RAG fallback (add GROQ_API_KEY for AI generation)"

    @staticmethod
    def _offline_generate(system_prompt: str, user_message: str) -> str:
        """Useful grounded demo mode when a Groq API key has not been configured."""
        import re
        lowered = user_message.casefold()
        if any(word in lowered for word in ("hello", "hi", "hey", "good morning", "good evening")) and ("student message" in lowered or "user message" in lowered):
            return "Hello! I’m MatrixMind, your linear algebra study assistant. Ask me about matrices, systems of equations, vector spaces, eigenvalues, or upload your class notes."
        if "thank" in lowered:
            return "You’re welcome! Keep practising, and ask whenever you want a concept or calculation explained step by step."
        if "who are you" in lowered or "what can you do" in lowered:
            return "I’m MatrixMind. I can explain linear algebra concepts, retrieve relevant passages from the indexed course notes, connect related topics through the knowledge graph, and calculate common matrix operations."

        context = ""
        if "RETRIEVED LINEAR-ALGEBRA NOTES:" in user_message:
            context = user_message.split("RETRIEVED LINEAR-ALGEBRA NOTES:", 1)[1].split("STUDENT QUESTION:", 1)[0].strip()
        elif "CONTEXT FROM KNOWLEDGE BASE:" in user_message:
            context = user_message.split("CONTEXT FROM KNOWLEDGE BASE:", 1)[1].split("---", 1)[0].strip()
        elif "ATTACHED MATERIAL" in user_message:
            context = user_message.split("ATTACHED MATERIAL", 1)[1].strip()
        elif "[Source:" in user_message:
            context = user_message

        question_match = re.search(r"(?:STUDENT QUESTION|Student message|User message):\s*(.*?)(?:\n|$)", user_message, re.I)
        question = question_match.group(1).strip() if question_match else user_message[-400:].strip()
        if context and context not in {"", "No previous conversation."}:
            blocks = re.split(r"(?=\[Source:)", context)
            cleaned = []
            sources = []
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                source = re.search(r"\[Source:\s*([^\]]+)\]", block)
                if source:
                    sources.append(source.group(1))
                    block = block[source.end():].strip()
                if block:
                    cleaned.append(block)
            excerpt = "\n\n".join(cleaned[:3])
            excerpt = excerpt[:2600]
            source_line = ", ".join(dict.fromkeys(sources)) or "your supplied notes"
            return (
                "**Relevant material found in the MatrixMind knowledge base**\n\n"
                "The local demo mode has retrieved the following course material. It is showing the matched notes directly; "
                "configure a Groq API key in `.env` to enable generated, step-by-step AI tutoring.\n\n"
                + excerpt + "\n\n**Sources:** " + source_line
            )
        return (
            "I couldn’t find a sufficiently close passage in the indexed linear-algebra notes for that question. "
            "Try asking about matrices, determinants, Gaussian elimination, vector spaces, rank, eigenvalues, "
            "orthogonality, or upload your lecture notes in the Document Library.\n\n"
            "**AI mode:** Add your Groq API key as `GROQ_API_KEY` in the project’s `.env` file to enable generated answers."
        )

    def _generate(self, system_prompt: str, user_message: str) -> str:
        if not settings.groq_api_key:
            return self._offline_generate(system_prompt, user_message)
        client = self._get_llm_client()

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            parts = []
            for continuation in range(3):
                response = client.chat.completions.create(
                    model=settings.llm_model, max_tokens=max(settings.max_tokens, 8192),
                    temperature=settings.temperature, messages=messages,
                )
                choice = response.choices[0]
                chunk = choice.message.content or ""
                parts.append(chunk)
                if getattr(choice, "finish_reason", None) != "length" or continuation == 2:
                    break
                log.warning("LLM output reached token limit; requesting continuation %d", continuation + 1)
                messages.extend([
                    {"role": "assistant", "content": chunk},
                    {"role": "user", "content": (
                        "Continue exactly where you stopped without repeating prior text. "
                        "Complete all requested sections, steps, equations and conclusion."
                    )},
                ])
            return "\n\n".join(part for part in parts if part)

        except Exception as e:
            log.error(f"Groq generation failed: {e}")
            raise RuntimeError(f"LLM generation error: {e}") from e