"""RAG Chatbot using Ollama and ChromaDB."""
from typing import List, Tuple, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import re

from config import LLM_MODEL, TOP_K_RESULTS
from ingest import load_vector_store
from utils.metrics import get_tracker, estimate_tokens

# Language templates
PROMPT_TEMPLATES = {
    "en": """
You are a helpful medical information assistant. Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so honestly.
Always cite your sources when possible.

Context:
{context}

Question: {question}

Instructions:
- Provide accurate, helpful information based on the context
- If information comes from specific sources, mention them
- If you're unsure or the context doesn't cover the question, be honest about it
- Keep responses clear and easy to understand
- For medical topics, remind users to consult healthcare professionals for personal advice
- Respond in English

Answer:
""",
    "de": """
Sie sind ein hilfreicher medizinischer Informationsassistent. Beantworten Sie die Frage des Benutzers basierend auf dem bereitgestellten Kontext.
Wenn der Kontext keine relevanten Informationen enthält, sagen Sie dies ehrlich.
Zitieren Sie immer Ihre Quellen, wenn möglich.

Kontext:
{context}

Frage: {question}

Anweisungen:
- Geben Sie genaue, hilfreiche Informationen basierend auf dem Kontext
- Wenn Informationen aus bestimmten Quellen stammen, erwähnen Sie diese
- Wenn Sie unsicher sind oder der Kontext die Frage nicht abdeckt, seien Sie ehrlich
- Halten Sie die Antworten klar und verständlich
- Bei medizinischen Themen erinnern Sie die Benutzer daran, medizinisches Fachpersonal zu konsultieren
- Antworten Sie auf Deutsch

Antwort:
"""
}

SUGGESTION_PROMPTS = {
    "en": """Based on this Q&A, suggest 3 short follow-up questions the user might ask.
Question: {question}
Answer: {answer}

Return ONLY 3 questions, one per line, no numbering or bullets:""",
    "de": """Basierend auf diesem Frage-Antwort-Paar, schlagen Sie 3 kurze Folgefragen vor, die der Benutzer stellen könnte.
Frage: {question}
Antwort: {answer}

Geben Sie NUR 3 Fragen zurück, eine pro Zeile, ohne Nummerierung oder Aufzählungszeichen:"""
}


class MedicalChatbot:
    """RAG-based medical chatbot."""
    
    def __init__(self):
        """Initialize the chatbot."""
        self.vector_store = None
        self.llm = None
        self.language = "en"
        self._initialize()
    
    def _initialize(self):
        """Load models and vector store."""
        print("Loading vector store...")
        self.vector_store = load_vector_store()
        
        print(f"Initializing LLM ({LLM_MODEL})...")
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.3,
            num_ctx=4096,  # Context window for Mistral 7B
        )
    
    def set_language(self, language: str):
        """Set response language (en/de)."""
        self.language = language if language in ["en", "de"] else "en"
    
    def retrieve_context(self, query: str) -> Tuple[str, List[Document], float]:
        """Retrieve relevant documents for a query with confidence score."""
        # Use similarity_search_with_score for confidence calculation
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=TOP_K_RESULTS)
        
        # Calculate confidence (lower distance = higher confidence)
        # ChromaDB returns L2 distance, convert to confidence percentage
        if docs_with_scores:
            avg_distance = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
            # Convert distance to confidence (0-100%)
            # Typical L2 distances range from 0 to ~2, we cap at 1.5 for scaling
            confidence = max(0, min(100, (1.5 - avg_distance) / 1.5 * 100))
        else:
            confidence = 0
        
        docs = [doc for doc, _ in docs_with_scores]
        
        # Format context
        context_parts = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '')
            source_info = f"{source}" + (f" (Page {page})" if page else "")
            context_parts.append(f"[Source {i}: {source_info}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context, docs, confidence
    
    def generate_suggestions(self, question: str, answer: str) -> List[str]:
        """Generate follow-up question suggestions."""
        try:
            prompt = SUGGESTION_PROMPTS[self.language].format(
                question=question,
                answer=answer[:500]  # Limit answer length
            )
            response = self.llm.invoke(prompt)
            
            # Parse suggestions (one per line)
            suggestions = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                # Remove common prefixes
                line = re.sub(r'^[\d\-\*\.\)]+\s*', '', line)
                if line and len(line) > 5:
                    suggestions.append(line)
            
            return suggestions[:3]  # Return max 3 suggestions
        except:
            return []
    
    def chat(self, question: str, language: Optional[str] = None) -> Dict:
        """Process a question and return response with sources and confidence."""
        if language:
            self.set_language(language)
        
        # Retrieve context with confidence
        context, source_docs, confidence = self.retrieve_context(question)
        
        # Generate response using language-specific template
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATES[self.language])
        prompt = prompt_template.format(
            context=context,
            question=question
        )
        
        response = self.llm.invoke(prompt)
        answer = response.content
        
        # Generate follow-up suggestions
        suggestions = self.generate_suggestions(question, answer)
        
        # Format sources
        sources = []
        for doc in source_docs:
            source_info = {
                "source": doc.metadata.get('source', 'Unknown'),
                "type": doc.metadata.get('type', 'unknown'),
            }
            if doc.metadata.get('page'):
                source_info["page"] = doc.metadata.get('page')
            if doc.metadata.get('title'):
                source_info["title"] = doc.metadata.get('title')
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 1),
            "suggestions": suggestions,
            "language": self.language
        }
    
    def chat_with_history(self, question: str, history: List[Tuple[str, str]], 
                          language: str = "en") -> Dict:
        """Chat with conversation history (for Gradio)."""
        self.set_language(language)
        
        # Add history context to the question if relevant
        if history:
            history_context = "\n".join([
                f"User: {h[0]}\nAssistant: {h[1]}" 
                for h in history[-3:]  # Last 3 exchanges
            ])
            enhanced_question = f"Previous conversation:\n{history_context}\n\nCurrent question: {question}"
        else:
            enhanced_question = question
        
        tracker = get_tracker()
        
        with tracker.track(model=LLM_MODEL) as metrics:
            # Use self.chat() instead of self.chain.invoke()
            result = self.chat(enhanced_question, language)
            
            # Estimate tokens after getting result
            metrics.prompt_tokens = estimate_tokens(enhanced_question)
            metrics.completion_tokens = estimate_tokens(result.get("answer", ""))
            metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
        
        # Print metrics to console
        print(f"📈 {tracker.get_last()}")
        
        # Add metrics to result so UI can display it
        result["metrics"] = str(tracker.get_last())
        
        # Format response with sources and confidence
        confidence = result["confidence"]
        conf_emoji = "🟢" if confidence >= 70 else "🟡" if confidence >= 40 else "🔴"
        
        answer = result["answer"]
        answer += f"\n\n{conf_emoji} **{'Confidence' if language == 'en' else 'Konfidenz'}:** {confidence}%"
        
        if result["sources"]:
            answer += f"\n\n📚 **{'Sources' if language == 'en' else 'Quellen'}:**\n"
            seen_sources = set()
            for src in result["sources"]:
                source_str = src['source']
                if src.get('page'):
                    source_str += f" (Page {src['page']})"
                if source_str not in seen_sources:
                    answer += f"- {source_str}\n"
                    seen_sources.add(source_str)
        
        return {
            "answer": answer,
            "suggestions": result.get("suggestions", []),
            "metrics": result.get("metrics", "")
        }


# Global instance
_chatbot = None

def get_chatbot() -> MedicalChatbot:
    """Get or create chatbot instance."""
    global _chatbot
    if _chatbot is None:
        _chatbot = MedicalChatbot()
    return _chatbot


if __name__ == "__main__":
    # Test the chatbot
    bot = get_chatbot()
    
    print("\nMedical Chatbot Ready!")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
            
        result = bot.chat(question)
        print(f"\nAssistant: {result['answer']}")
        print(f"\nSources: {result['sources']}\n")
