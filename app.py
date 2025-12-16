import streamlit as st
from rag_engine import UMLRAG
from ai_service import AIService
from features import ScannerFeature, EditorFeature, NLGenFeature, DBFeature

class UMLGenieApp:
    def __init__(self):
        st.set_page_config(page_title="UMLGenie", page_icon="🧞", layout="wide")
        self.init_session_state()
        
        self.rag_engine = UMLRAG()
        self.ai_service = AIService()
        
        self.features = {
            "Scanner & Explainer": ScannerFeature(self.ai_service, self.rag_engine),
            "Visual Editor": EditorFeature(self.ai_service, self.rag_engine),
            "Natural Language Gen": NLGenFeature(self.ai_service, self.rag_engine),
            "Database-to-UML": DBFeature(self.ai_service, self.rag_engine)
        }

    def init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "api_key_configured" not in st.session_state:
            st.session_state.api_key_configured = False

    def render_sidebar(self):
        with st.sidebar:
            st.title("🧞 UMLGenie")
            self.configure_ai()
            st.divider()
            mode = st.radio("Mode", list(self.features.keys()))
            self.configure_rag()
            st.info("Supported: PlantUML, Mermaid")
            return mode

    def save_api_key(self, key_name, key_value):
        import os
        secrets_path = ".streamlit/secrets.toml"
        os.makedirs(".streamlit", exist_ok=True)
        
        lines = []
        if os.path.exists(secrets_path):
             with open(secrets_path, "r") as f:
                 lines = f.readlines()
        
        new_lines = []
        found = False
        for line in lines:
            if line.strip().startswith(f"{key_name} =") or line.strip().startswith(f"{key_name}="):
                new_lines.append(f'{key_name} = "{key_value}"\n')
                found = True
            elif line.strip().startswith(f"# {key_name} =") or line.strip().startswith(f"#{key_name} =") or line.strip().startswith(f"#{key_name}="):
                 new_lines.append(f'{key_name} = "{key_value}"\n')
                 found = True
            else:
                new_lines.append(line)
        
        if not found:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines.append("\n")
            new_lines.append(f'{key_name} = "{key_value}"\n')
            
        with open(secrets_path, "w") as f:
            f.writelines(new_lines)

    def configure_ai(self):
        # 1. Load Provider from secrets
        default_in = 0
        if "AI_PROVIDER" in st.secrets:
            saved_provider = st.secrets["AI_PROVIDER"]
            if saved_provider == "Google Gemini": default_in = 0
            elif saved_provider == "OpenRouter": default_in = 1

        provider = st.radio("AI Provider", ["Google Gemini", "OpenRouter"], index=default_in)
        
        api_key = ""
        model_name = "gemini-2.0-flash"
        
        if provider == "Google Gemini":
            api_key = st.text_input("Gemini API Key", type="password")
            if not api_key and "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
                st.success("Gemini Key from Secrets")
            elif api_key:
                if st.button("Save Gemini Config"):
                    self.save_api_key("GOOGLE_API_KEY", api_key)
                    self.save_api_key("AI_PROVIDER", "Google Gemini")
                    st.success("Saved to secrets.toml!")
                    st.rerun()
                st.success("Gemini Configured!")
        else:
            api_key = st.text_input("OpenRouter API Key", type="password")
            
            # Load default model from secrets
            default_model = "google/gemini-2.0-flash-001"
            if "OPENROUTER_MODEL" in st.secrets:
                default_model = st.secrets["OPENROUTER_MODEL"]
                
            model_name = st.text_input("Model Name", value=default_model)
            
            if not api_key and "OPENROUTER_API_KEY" in st.secrets:
                 api_key = st.secrets["OPENROUTER_API_KEY"]
                 st.success("OpenRouter Key from Secrets")
 
            # Allow saving if key is present (either typed or from secrets)
            if api_key:
                if st.button("Save OpenRouter Config"):
                    self.save_api_key("OPENROUTER_API_KEY", api_key)
                    self.save_api_key("OPENROUTER_MODEL", model_name)
                    self.save_api_key("AI_PROVIDER", "OpenRouter")
                    st.success("Saved to secrets.toml!")
                    st.rerun()
                st.success("OpenRouter Configured!")
                
        if api_key:
            self.ai_service.configure(provider, api_key, model_name)
            st.session_state.api_key_configured = True

    def configure_rag(self):
        precision_mode = st.toggle("Precision Mode (RAG)", value=False, help="Use scraped documentation for better accuracy")
        st.session_state.precision_mode = precision_mode
        
        if precision_mode:
            st.success("✅ Precision Mode (RAG) is ACTIVE")
            if not self.rag_engine.load():
                st.warning("Index not found. Running ingestion...")
                with st.spinner("Ingesting documentation..."):
                    import ingest_docs
                    ingest_docs.main()
                    if self.rag_engine.load():
                        st.success("Index Loaded!")
                    else:
                        st.error("Index Creation Failed")
        else:
             st.caption("Standard Mode (No Context)")


    def run(self):
        mode = self.render_sidebar()
        feature = self.features.get(mode)
        if feature:
            feature.render()

if __name__ == "__main__":
    app = UMLGenieApp()
    app.run()
