"""Gradio web interface for the Medical Chatbot with voice input and suggestions."""
import gradio as gr
from chatbot import get_chatbot
from ingest import ingest_all
import os
import json
import datetime
from config import CHROMA_DB_DIR
import time
from utils.metrics import get_tracker

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

# Global state
current_suggestions = []

def save_conversation(history):
    """Save the current conversation to a JSON file."""
    if not history:
        return "No conversation to save."
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    
    # Ensure directory exists
    os.makedirs("saved_chats", exist_ok=True)
    filepath = os.path.join("saved_chats", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    return f"✅ Conversation saved to {filepath}"


def get_saved_chats_list():
    """Get list of saved chat files for the dropdown."""
    if not os.path.exists("saved_chats"):
        return []
    files = [f for f in os.listdir("saved_chats") if f.endswith(".json")]
    return sorted(files, reverse=True)  # Newest first


def load_chat_history(filename: str):
    """Load a chat history from the selected file."""
    if not filename:
        return None
    
    filepath = os.path.join("saved_chats", filename)
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                history = json.load(f)
            return history
    except Exception as e:
        print(f"Error loading chat: {e}")
        return None
    return None


def refresh_chat_list():
    """Refresh the choices in the dropdown."""
    return gr.update(choices=get_saved_chats_list())


def inspect_database(query: str):
    """Debug function to inspect raw database chunks and metadata."""
    if not query:
        return "Please enter a search term."
    
    try:
        # Get chatbot instance to access the vector store
        bot = get_chatbot()
        
        # Try to access vector store directly to get raw documents
        # Common attribute names for LangChain objects
        db = getattr(bot, "vector_store", None) or getattr(bot, "db", None)
        
        if db:
            # Increased k to 10 to find data more reliably
            docs = db.similarity_search(query, k=10)
            output = []
            for i, doc in enumerate(docs):
                meta = doc.metadata
                source = meta.get("source", "Unknown")
                # specific check for structured data (likely has distinct metadata)
                is_structured = "json" in source or "dict" in str(meta)
                
                info = f"--- Chunk {i+1} ---\n"
                info += f"📂 Source: {source}\n"
                info += f"🏷️ Type: {'Structured Data' if is_structured else 'Document/Text'}\n"
                info += f"ℹ️ Metadata: {json.dumps(meta, default=str)}\n"
                info += f"📝 Content Preview: {doc.page_content[:400]}...\n"
                output.append(info)
            
            return "\n".join(output) if output else "No matching chunks found in database."
        else:
            return "⚠️ Could not access vector store directly. Ensure 'get_chatbot()' returns an object with 'vector_store' attribute."
            
    except Exception as e:
        return f"❌ Error inspecting database: {str(e)}"


def check_knowledge_base_exists() -> bool:
    """Check if the knowledge base has been created."""
    return os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR)


def check_doctors_file_exists() -> bool:
    """Check if doctors.jsonl exists."""
    return os.path.exists("doctors.jsonl")


def transcribe_audio(audio_path, language="en"):
    """Transcribe audio to text using SpeechRecognition."""
    if audio_path is None or not SPEECH_AVAILABLE:
        return None
    
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        
        # Use Google's free speech recognition
        # Language codes: en-US, de-DE
        lang_code = "de-DE" if language == "de" else "en-US"
        text = recognizer.recognize_google(audio_data, language=lang_code)
        return text
    except sr.UnknownValueError:
        return None  # Could not understand audio
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None
    except Exception as e:
        print(f"Error transcribing: {e}")
        return None


def messages_to_pairs(msgs):
    """Convert Gradio messages format to list of [user, assistant] pairs."""
    pairs = []
    pending_user = None
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append([pending_user, content])
            pending_user = None
    return pairs

def pairs_to_messages(pairs):
    """Convert list of [user, assistant] pairs to Gradio messages format."""
    messages = []
    for u, a in pairs:
        messages.append({"role": "user", "content": u})
        # Only add assistant message if it exists (handles pending state)
        if a is not None:
            messages.append({"role": "assistant", "content": a})
    return messages


def respond(message: str, history: list, language: str):
    """Generate response for the chat interface."""
    global current_suggestions
    
    if not message or not message.strip():
        # Yield empty update to clear state if necessary
        yield history, "", gr.update(choices=[]), gr.update(visible=False)
        return
    
    try:
        # 1. IMMEDIATE UPDATE: Show user message in chat, clear input
        pairs_history = messages_to_pairs(history)
        
        # Add user message to history with "Thinking..." placeholder
        temp_pairs = pairs_history + [[message, "⏳ Thinking..."]]
        temp_ui_history = pairs_to_messages(temp_pairs)
        
        # Yield immediate feedback so question is visible
        yield temp_ui_history, "", gr.update(visible=False), gr.update(visible=False)
        
        # 2. PROCESS: Call chatbot
        chatbot = get_chatbot()
        # Note: We pass original history to backend
        result = chatbot.chat_with_history(message, pairs_history, language)
        
        answer = result["answer"]
        suggestions = result.get("suggestions", [])
        current_suggestions = suggestions
        
        # 3. FINAL UPDATE: Show actual answer
        final_pairs = pairs_history + [[message, answer]]
        final_ui_history = pairs_to_messages(final_pairs)
        
        suggestions_visible = len(suggestions) > 0
        
        yield (
            final_ui_history,
            "",  # Ensure input stays cleared
            gr.update(choices=suggestions, visible=suggestions_visible),
            gr.update(visible=suggestions_visible)
        )
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nMake sure the knowledge base is initialized and Ollama is running."
        pairs_history = messages_to_pairs(history)
        pairs_history = pairs_history + [[message, error_msg]]
        yield pairs_to_messages(pairs_history), "", gr.update(choices=[], visible=False), gr.update(visible=False)


def use_suggestion(suggestion: str, history: list, language: str):
    """Use a suggested question."""
    if suggestion:
        yield from respond(suggestion, history, language)
    else:
        yield history, "", gr.update(), gr.update()


def clear_chat():
    """Clear chat history and suggestions."""
    global current_suggestions
    current_suggestions = []
    return [], "", gr.update(choices=[], visible=False), gr.update(visible=False)


def initialize_knowledge_base(include_web: bool, incremental: bool) -> str:
    """Initialize or refresh the knowledge base."""
    try:
        mode = "incremental" if incremental else "full"
        ingest_all(include_web=include_web, incremental=incremental)
        return f"✅ Knowledge base updated successfully! (Mode: {mode})"
    except Exception as e:
        return f"❌ Error initializing knowledge base: {str(e)}"


def create_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for polished UI
    custom_css = """
    .chat-window { height: 600px !important; }
    .input-row { background-color: var(--background-fill-secondary); padding: 10px; border-radius: 12px; }
    .sidebar-col { border-right: 1px solid var(--border-color-primary); padding-right: 10px; }
    """
    
    # Use Soft theme for better UI
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    )
    
    with gr.Blocks(title="Medical Chatbot - Platform4AI", theme=theme, css=custom_css) as demo:
        
        with gr.Row():
            gr.Markdown("""
            # 🏥 Medical Information Chatbot
            ### Trusted medical answers from verified sources
            """)
        
        with gr.Tab("💬 Chat"):
            with gr.Row():
                # --- LEFT SIDEBAR: History ---
                with gr.Column(scale=1, elem_classes=["sidebar-col"], variant="panel"):
                    gr.Markdown("### 🗄️ History")
                    
                    history_dropdown = gr.Dropdown(
                        label="Saved Conversations",
                        choices=get_saved_chats_list(),
                        interactive=True,
                        allow_custom_value=False
                    )
                    
                    with gr.Row():
                        load_history_btn = gr.Button("📂 Load", size="sm", variant="secondary")
                        refresh_list_btn = gr.Button("🔄 Refresh", size="sm")

                    gr.Markdown("---")
                    gr.Markdown("**Manage:**")
                    clear_btn = gr.Button("🗑️ New Chat", size="sm", variant="stop")
                    save_btn = gr.Button("💾 Save Current", size="sm")
                    status_msg = gr.Markdown("", visible=True)

                # --- RIGHT COLUMN: Chat Interface ---
                with gr.Column(scale=4):
                    # 1. Language Selection (Top)
                    with gr.Row():
                        language = gr.Radio(
                            choices=[("🇬🇧 English", "en"), ("🇩🇪 Deutsch", "de")],
                            value="en",
                            label="Language / Sprache",
                            interactive=True
                        )

                    # 2. Chat Display (Full Width)
                    chatbot_display = gr.Chatbot(
                        label="Conversation",
                        height=600,
                        elem_classes=["chat-window"],
                        render_markdown=True,
                        avatar_images=(None, "🤖")
                    )
                    
                    # Suggested questions (Hidden)
                    suggestions_label = gr.Markdown("### 💡 Suggested questions:", visible=False)
                    suggestions_radio = gr.Radio(
                        choices=[],
                        label="",
                        visible=False,
                        interactive=True
                    )
                    
                    # 3. Input Area Group (Full Width)
                    with gr.Group():
                        with gr.Row(elem_classes=["input-row"]): 
                            msg_input = gr.Textbox(
                                placeholder="Type your question here... / Geben Sie Ihre Frage ein...",
                                label="Your Question",
                                scale=8,
                                show_label=False,
                                container=False,
                                autofocus=True
                            )
                            submit_btn = gr.Button("📨 Send", variant="primary", scale=1, min_width=80)
                            voice_input = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                show_label=False,
                                container=False,
                                scale=1,
                                min_width=50
                            )

                    # 4. Examples (Bottom)
                    gr.Markdown("### 📝 Quick Ask / Beispiele")
                    with gr.Row():
                        ex1 = gr.Button("Flu Symptoms", size="sm")
                        ex2 = gr.Button("Doctor in Oberhausen", size="sm")
                        ex3 = gr.Button("Was ist Diabetes?", size="sm")

            # Event handlers
            submit_btn.click(
                fn=respond,
                inputs=[msg_input, chatbot_display, language],
                outputs=[chatbot_display, msg_input, suggestions_radio, suggestions_label]
            )
            
            msg_input.submit(
                fn=respond,
                inputs=[msg_input, chatbot_display, language],
                outputs=[chatbot_display, msg_input, suggestions_radio, suggestions_label]
            )
            
            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot_display, msg_input, suggestions_radio, suggestions_label]
            )

            save_btn.click(
                fn=save_conversation,
                inputs=[chatbot_display],
                outputs=[status_msg]
            )
            
            # History Load/Refresh Handlers
            load_history_btn.click(
                fn=load_chat_history,
                inputs=[history_dropdown],
                outputs=[chatbot_display]
            )
            
            refresh_list_btn.click(
                fn=refresh_chat_list,
                outputs=[history_dropdown]
            )

            # Suggestion click handler
            suggestions_radio.change(
                fn=use_suggestion,
                inputs=[suggestions_radio, chatbot_display, language],
                outputs=[chatbot_display, msg_input, suggestions_radio, suggestions_label]
            )
            
            # Voice input handler - transcribe and submit
            def handle_voice(audio_path, history, lang):
                """Handle voice input by transcribing and responding."""
                if audio_path is None:
                    yield history, "", gr.update(), gr.update()
                    return
                
                # Transcribe audio
                transcribed = transcribe_audio(audio_path, lang)
                
                if transcribed:
                    yield from respond(transcribed, history, lang)
                else:
                    error_msg = "Could not understand audio. Please try again or type your question." if lang == "en" else "Audio nicht verstanden. Bitte versuchen Sie es erneut oder geben Sie Ihre Frage ein."
                    pairs_history = messages_to_pairs(history)
                    pairs_history = pairs_history + [["[Voice input]", error_msg]]
                    yield pairs_to_messages(pairs_history), "", gr.update(visible=False), gr.update(visible=False)
            
            voice_input.stop_recording(
                fn=handle_voice,
                inputs=[voice_input, chatbot_display, language],
                outputs=[chatbot_display, msg_input, suggestions_radio, suggestions_label]
            )
            
            # Example button handlers (Fixed generator compatibility)
            def example_handler_1(h, l):
                yield from respond("What are common symptoms of the flu?", h, l)

            def example_handler_2(h, l):
                yield from respond("How can I find a general practitioner in Oberhausen?", h, l)
                
            def example_handler_3(h, l):
                yield from respond("Was ist Diabetes?", h, l)

            ex1.click(
                fn=example_handler_1,
                inputs=[chatbot_display, language],
                outputs=[chatbot_display, msg_input, suggestions_radio, suggestions_label]
            )
            ex2.click(
                fn=example_handler_2,
                inputs=[chatbot_display, language],
                outputs=[chatbot_display, msg_input, suggestions_radio, suggestions_label]
            )
            ex3.click(
                fn=example_handler_3,
                inputs=[chatbot_display, language],
                outputs=[chatbot_display, msg_input, suggestions_radio, suggestions_label]
            )
        
        with gr.Tab("⚙️ Admin"):
            gr.Markdown("### Knowledge Base Management")
            
            with gr.Row():
                kb_status = gr.Textbox(
                    label="Validation Status",
                    value="Ready" if check_knowledge_base_exists() else "Knowledge base not initialized",
                    interactive=False,
                    scale=2
                )
                
                file_status_text = "✅ Found doctors.jsonl" if check_doctors_file_exists() else "❌ doctors.jsonl Missing"
                file_status = gr.Textbox(
                    label="Data File Status",
                    value=file_status_text,
                    interactive=False, 
                    scale=1
                )
            
            with gr.Row():
                include_web_checkbox = gr.Checkbox(
                    label="Include web crawling",
                    value=True
                )
                incremental_checkbox = gr.Checkbox(
                    label="Incremental update (faster - only new/changed content)",
                    value=False
                )
            
            with gr.Row():
                init_btn = gr.Button("🔄 Full Refresh", variant="primary")
                update_btn = gr.Button("⚡ Quick Update (Incremental)", variant="secondary")
            
            # Disable buttons and stream status while long updates are running
            def make_update_handler(incremental_mode: bool):
                def _handler(include_web_opt):
                    status_msg = "⏳ Updating knowledge base... Large datasets (~6000 chunks) can take several minutes."
                    yield status_msg, gr.update(interactive=False), gr.update(interactive=False)
                    progress = gr.Progress(track_tqdm=True)
                    chunk_counter = {"count": 0}
                    start_ts = time.perf_counter()
                    callback_used = True

                    def progress_callback(step: int = 1, total_chunks=None):
                        chunk_counter["count"] += step
                        elapsed = time.perf_counter() - start_ts
                        cps = chunk_counter["count"] / elapsed if elapsed else 0
                        desc = f"Chunks processed: {chunk_counter['count']} ({cps:.2f} chunks/sec)"
                        progress(chunk_counter["count"], total=total_chunks, desc=desc)

                    try:
                        ingest_all(include_web=include_web_opt, incremental=incremental_mode, progress_callback=progress_callback)
                    except TypeError:
                        callback_used = False
                        ingest_all(include_web=include_web_opt, incremental=incremental_mode)

                    elapsed = time.perf_counter() - start_ts
                    final_rate = (chunk_counter["count"] / elapsed) if (elapsed and chunk_counter["count"]) else 0
                    rate_text = f"{final_rate:.2f} chunks/sec" if (callback_used and chunk_counter["count"] > 0) else "chunks/sec not available"
                    result_msg = f"✅ Knowledge base updated successfully! (Mode: {'incremental' if incremental_mode else 'full'}) - {rate_text}"
                    yield result_msg, gr.update(interactive=True), gr.update(interactive=True)
                return _handler
            
            init_btn.click(
                fn=make_update_handler(False),
                inputs=[include_web_checkbox],
                outputs=[kb_status, init_btn, update_btn]
            )
            
            update_btn.click(
                fn=make_update_handler(True),
                inputs=[include_web_checkbox],
                outputs=[kb_status, init_btn, update_btn]
            )
            
            # --- NEW: Database Inspector Section ---
            gr.Markdown("---")
            gr.Markdown("### 🔍 Database Inspector (Debug)")
            gr.Markdown("Use this to verify if structured data (doctors) or websites were crawled properly.")
            
            with gr.Row():
                inspect_input = gr.Textbox(
                    label="Search Database", 
                    placeholder="Enter a doctor's name (e.g., 'Dachwitz'), city, or term...",
                    scale=4
                )
                inspect_btn = gr.Button("🔎 Inspect Chunks", variant="secondary", scale=1)
            
            inspect_output = gr.Code(label="Raw Database Chunks & Metadata", language="json", lines=10)
            
            inspect_btn.click(
                fn=inspect_database,
                inputs=[inspect_input],
                outputs=[inspect_output]
            )
            # ---------------------------------------
            
            # --- Metrics Display Section ---
            gr.Markdown("---")
            gr.Markdown("### 📈 Performance Metrics")
            
            def get_metrics_display():
                tracker = get_tracker()
                last = tracker.get_last()
                summary = tracker.get_summary()
                
                if not last:
                    return "No LLM calls tracked yet. Ask a question first!"
                
                return f"""
**Last Call:**
{last}

**Session Summary:**
- Total Calls: {summary['total_calls']}
- Total Latency: {summary['total_latency_seconds']}s
- Avg Latency: {summary['avg_latency_seconds']}s
- Total Tokens: {summary['total_tokens']}
- Avg Tokens/Call: {summary.get('avg_tokens_per_call', 'N/A')}
"""
            
            metrics_display = gr.Markdown("No metrics yet.")
            refresh_metrics_btn = gr.Button("🔄 Refresh Metrics", size="sm")
            
            refresh_metrics_btn.click(
                fn=get_metrics_display,
                outputs=[metrics_display]
            )
            # -------------------------------
            
            gr.Markdown("> Processing large datasets (e.g., ~6000 chunks) can take a while. Use quick incremental updates when possible.")
            gr.Markdown("""
            ### Instructions
            
            1. **First time setup:** Click "🔄 Full Refresh" to process PDFs and crawl all websites
            2. **Regular updates:** Click "⚡ Quick Update" - only processes new/changed content (much faster!)
            3. **Web crawling:** Disable to only use PDF content
            
            ### How Incremental Updates Work
            
            - Stores a hash of each crawled page
            - On update, skips pages that haven't changed
            - Only re-processes modified content
            - Cache stored in `crawl_cache.json`
            
            ### Requirements
            
            - Ollama must be running (`ollama serve`)
            - Models required: `llama3.2`, `nomic-embed-text`
            """)
            gr.Markdown("> Doctors directory data from `doctors.jsonl` is included in ingestion. Keep the file in the project root before running updates.")
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### About This Chatbot
            
            This medical information chatbot uses **Retrieval-Augmented Generation (RAG)** to provide
            accurate answers based on trusted medical sources.
            
            **Technology Stack:**
            - 🤖 **LLM:** Llama 3.2 (via Ollama)
            - 📊 **Embeddings:** nomic-embed-text
            - 🗄️ **Vector Store:** ChromaDB
            - 🌐 **Interface:** Gradio
            - 🔗 **Framework:** LangChain
            
            **Data Sources:**
            - Local PDF medical documents
            - **arzt-auskunft.de** (German doctor directory)
            - **gesundheitsinformation.de** (German health information)
            - **gesund.bund.de** (Federal Ministry of Health portal)
            - **kvno.de** (Association of Statutory Health Insurance Physicians North Rhine)
            
            ---
            
            *Built for Platform4AI*
            """)
    
    return demo


if __name__ == "__main__":
    # Check prerequisites
    if not check_knowledge_base_exists():
        print("⚠️  Knowledge base not found. Please initialize it from the Settings tab.")
    
    if not check_doctors_file_exists():
        print("⚠️  doctors.jsonl not found. Structured data will be missing from ingestion.")

    # Launch the app
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
