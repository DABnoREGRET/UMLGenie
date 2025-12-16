import streamlit as st
from PIL import Image
import uml_utils
import re

class Feature:
    def __init__(self, ai_service, rag_engine):
        self.ai_service = ai_service
        self.rag_engine = rag_engine

    def render(self):
        raise NotImplementedError

class ScannerFeature(Feature):
    def render(self):
        st.header("🔍 Smart UML Scanner")
        st.warning("⚠️ Note: This feature requires an AI model with Vision capabilities (e.g., Gemini 1.5 Pro/Flash, GPT-4o).")
        uploaded_file = st.file_uploader("Upload UML Diagram", type=["png", "jpg", "jpeg", "webp"])
        
        if uploaded_file and self.ai_service.is_configured():
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Diagram", use_container_width=True)
            
            if st.button("Analyze Diagram"):
                with st.spinner("Analyzing..."):
                    prompt = "Analyze this UML diagram. Explain the relationships, identify patterns, and spot any potential design issues."
                    response_text = self.ai_service.analyze_image(prompt, image)
                    st.markdown(response_text)

class EditorFeature(Feature):

    def render(self):
        st.header("✏️ Visual Editor")
        
        if "editor_code" not in st.session_state:
            st.session_state.editor_code = "classDiagram\n    class User {\n        +String username\n        +String password\n        +login()\n    }"
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Code")
            code = st.text_area("Input UML", value=st.session_state.editor_code, height=600, key="editor_area")
            st.session_state.editor_code = code
            
            # Show Explanation from last fix if available
            if "last_fix_explanation" in st.session_state and st.session_state.last_fix_explanation:
                with st.expander("🤖 AI Fix Explanation", expanded=True):
                    st.info(st.session_state.last_fix_explanation)
                    if st.button("Clear Explanation"):
                        st.session_state.last_fix_explanation = None
                        st.rerun()
            
            if st.button("Magic Fix (AI)"):
                 if self.ai_service.is_configured():
                    with st.status("🧙‍♂️ AI Magic Fix Running...", expanded=True) as status:
                        ctx = ""
                        if st.session_state.get('precision_mode', False):
                            status.write("🔍 Searching RAG Index for 'syntax error fix'...")
                            results = self.rag_engine.query("syntax error fix")
                            status.write(f"✅ Found {len(results)} context chunks.")
                            ctx = "\n".join(results)
                        else:
                            status.write("ℹ️ Standard Mode active.")
                        
                        status.write("🧠 Generating Fix...")
                        response = self.ai_service.generate_uml(f"Fix this UML code to be valid:\n{code}", ctx)
                        
                        if response:
                             # Separate Code from Explanation
                             import re
                             code_match = re.search(r'```(?:\w+)?\s*(.*?)```', response, re.DOTALL)
                             
                             extracted_code = ""
                             explanation = ""
                             
                             if code_match:
                                 extracted_code = code_match.group(1).strip()
                                 # Remove the code block from response to get explanation
                                 explanation = response.replace(code_match.group(0), "").strip()
                             else:
                                 # Fallback: Assume all is code if no block, or maybe just text?
                                 # Let's try raw parsing if it starts with known keywords
                                 if any(x in response for x in ["@startuml", "graph ", "sequenceDiagram", "classDiagram"]):
                                     extracted_code = response
                                 else:
                                     explanation = response

                             if extracted_code:
                                 # Clean "mermaid" or "plantuml" from start if it crept in inside the block (rare but happens)
                                 if extracted_code.lower().startswith("mermaid"): extracted_code = extracted_code[7:].strip()
                                 if extracted_code.lower().startswith("plantuml"): extracted_code = extracted_code[8:].strip()
                                 
                                 st.session_state.editor_code = extracted_code
                                 st.session_state.last_fix_explanation = explanation # Store explanation
                                 status.update(label="✨ Fix Applied!", state="complete", expanded=False)
                                 st.rerun()
                             else:
                                 st.error("Could not extract valid code from AI response.")
                                 st.caption("Response: " + response)
        
        with col2:
            st.subheader("Preview")
            if "@startuml" in code:
                url = uml_utils.get_plantuml_url(code)
                st.image(url, caption="Preview", use_container_width=True)
            elif "classDiagram" in code or "sequenceDiagram" in code or "graph" in code:
                 # Assume Mermaid - FORCE LOCAL
                 import streamlit.components.v1 as components
                 html_code = uml_utils.render_mermaid_html(code)
                 components.html(html_code, height=600, scrolling=True)
            else:
                 st.info("Enter PlantUML (@startuml) or Mermaid code to preview.")
class NLGenFeature(Feature):
    def render(self):
        # Header with Clear Button (for visibility)
        c_head, c_btn = st.columns([6, 1])
        with c_head:
            st.header("💬 UML Chat & Architect")
        with c_btn:
             if st.button("🗑️ Clear", key="clear_top", help="Clear chat history"):
                st.session_state.messages = []
                st.rerun()        
        
        # Initialize preview code state
        if "preview_code" not in st.session_state:
            st.session_state.preview_code = None

        # Language toggle
        # Language toggle
        # Configuration UI
        c1, c2 = st.columns([1, 1])
        with c1:
             language_pref = st.radio("Output Language", ["Auto", "PlantUML", "Mermaid"], horizontal=False, index=0)
        with c2:
             use_web_search = st.toggle("🌍 Use Web Search", value=False, help="Allow AI to use general knowledge")

        # Determine language arg
        pref_arg = "auto"
        if "PlantUML" in language_pref: pref_arg = "PlantUML"
        elif "Mermaid" in language_pref: pref_arg = "Mermaid"

        # --- Chat History with Inline Preview ---
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Render content
                self._parse_and_render_msg(msg["content"])

        # --- Chat Input & Controls ---
        # Clear button ALSO in sidebar for "pinned" access
        with st.sidebar:
            st.divider()
            if st.button("🗑️ Clear Chat", key="clear_sidebar", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        prompt = st.chat_input("Ask a question...")
        
        if prompt:
            if not self.ai_service.is_configured():
                st.error("Please configure API Key first.")
                return
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        # Handle AI Response
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
           with st.chat_message("assistant"):
                with st.status("Thinking...", expanded=True) as status:
                    # ... [Existing RAG Logic] ...
                    context = ""
                    user_prompt = st.session_state.messages[-1]["content"]
                    
                    if use_web_search:
                         status.write("🌍 Web Search: Querying DuckDuckGo...")
                         web_results = uml_utils.perform_web_search(user_prompt)
                         status.write(f"✅ Web: Found results.")
                         context += f"\n\nHigh Priority Web Search Results:\n{web_results}\n\n"

                    # Force RAG for Mermaid
                    use_rag = st.session_state.get('precision_mode', False) or pref_arg == "Mermaid"

                    if use_rag:
                        status.write("🔍 RAG: Searching docs...")
                        # Determine branch from language preference
                        target_branch = None
                        if pref_arg == "PlantUML": target_branch = "plantuml"
                        elif pref_arg == "Mermaid": target_branch = "mermaid"
                        
                        results = self.rag_engine.query(user_prompt, branch=target_branch)
                        status.write(f"✅ RAG: Found {len(results)} chunks.")
                        context += "\n".join(results)
                        
                        # [Logging code...]
                        print(f"RAG Query: {user_prompt}")
                    else:
                        status.write("ℹ️ Standard generation.")
                    
                    status.write("📝 Drafting UML...")
                    response_text = self.ai_service.generate_uml(f"Generate a UML diagram for: {user_prompt}", context, preferred_language=pref_arg)
                    status.update(label="Done!", state="complete", expanded=False)
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Auto-render inline for new message
                st.rerun()

    def _parse_and_render_msg(self, content):
        # Try to find code block first
        code_content = None
        is_plantuml = False
        
        # 1. Try Regex for blocks
        code_match = re.search(r'```(?:\w+)?\s*(.*?)```', content, re.DOTALL)
        if code_match:
            code_content = code_match.group(1).strip()
            # Clean identifiers
            if code_content.lower().startswith("plantuml"):
                code_content = code_content[8:].strip()
            elif code_content.lower().startswith("mermaid"):
                code_content = code_content[7:].strip()
                
        # 2. Fallback: Check for raw PlantUML
        elif "@startuml" in content:
            # Extract from start to end if embedded in text, or take whole if clean
            start = content.find("@startuml")
            end = content.find("@enduml")
            if end != -1:
                code_content = content[start:end+7].strip()
            else:
                code_content = content[start:].strip()
                
        # 3. Fallback: Check for raw Mermaid keywords using heuristic
        # Also supports Directives (%%) and Frontmatter (---)
        elif any(content.lower().strip().startswith(k) for k in ["graph ", "sequencediagram", "classdiagram", "statediagram", "erdiagram", "gantt", "pie", "flowchart", "journey", "%%", "---"]):
             code_content = content.strip()

        if code_content:
            if "@startuml" in code_content:
                # PlantUML still needs cloud for now unless local server, but user asked to remove cloud PREVIEW option. 
                # We will render the image but remove the link.
                url = uml_utils.get_plantuml_url(code_content)
                st.image(url, use_container_width=True)
            else:
                # Assume Mermaid - FORCE LOCAL
                import streamlit.components.v1 as components
                html_code = uml_utils.render_mermaid_html(code_content)
                components.html(html_code, height=400, scrolling=True)



class DBFeature(Feature):
    def render(self):
        st.header("🗄️ Database to UML")
        
        db_uri = st.text_input("Database URI (SQLAlchemy format)", placeholder="sqlite:///sqlite.db")
        
        if st.button("Generate from DB"):
             if self.ai_service.is_configured():
                with st.spinner("Inspecting DB & Generating..."):
                    schema = uml_utils.get_database_schema(db_uri)
                    if "Error" in schema:
                        st.error(schema)
                    else:
                        st.subheader("Detected Schema")
                        st.code(schema)
                        
                        response = self.ai_service.generate_uml(f"Create a UML Class Diagram (Mermaid) based on this schema:\n{schema}")
                        st.markdown(response)
