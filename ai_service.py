import google.generativeai as genai
from openai import OpenAI
import io
import base64

class AIService:
    def __init__(self):
        self.api_key = None
        self.provider = None
        self.model_name = None
        self.openrouter_client = None

    def configure(self, provider, api_key, model_name="gemini-2.0-flash"):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
        elif provider == "OpenRouter":
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            
    def is_configured(self):
        return bool(self.api_key)

    def _image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_uml(self, prompt, context="", preferred_language="auto", history=None):
        if history is None:
            history = []
            
        # Build instructions based on language
        lang_instruction = "Generate VALID Mermaid JS code (` ```mermaid `) or PlantUML code (`@startuml`...`@enduml`)."
        if preferred_language == "PlantUML":
            lang_instruction = "Generate **PlantUML** code ONLY (`@startuml` ... `@enduml`). Do NOT use Mermaid."
        elif preferred_language == "Mermaid":
            lang_instruction = "Generate **Mermaid JS** code ONLY (inside ```mermaid block). Do NOT use PlantUML."

        system_instruction = f"""You are a UML & Software Architecture Expert.
        
        Instructions:
        1. {lang_instruction}
           - **Mermaid**: Start DIRECTLY with the diagram type (e.g. `sequenceDiagram`, `graph TD`, `classDiagram`) OR a Directive/Frontmatter (e.g. `%%{{init: ...}}%%`, `---`). Do **NOT** wrap in markdown blocks (` ``` `). Just provide the code.
           - **PlantUML**: Start DIRECTLY with `@startuml`. Do **NOT** wrap in markdown blocks.
        2. If the user asks a question about UML concepts, explain it.
        3. Can mix text with code, but keep the code clean and strictly valid.
        4. **Language Preferences (Auto Mode)**:
           - **Use Mermaid** for: Sequence, Flowchart, Class, State, ERD, User Journey, Gantt, Pie.
           - **Use PlantUML** for: Use Case, Component, Deployment, Object, Timing, Network/Architecture.
           - If unsure, default to PlantUML.
        5. **Simplicity Level**: REQUIRED: Keep diagrams **SIMPLE, HIGH-LEVEL, and MINIMAL** by default. Do not add excessive details, attributes, or methods unless the user explicitly requests a "detailed" or "complex" diagram. Focus on the core relationships.
        
        STRICT FORMATTING RULES:
        - **NO** inline enums like `category: enum {{Pizza, Drink}}`. Use standard class definitions or notes for Enums.
        - **Enum Workaround**:
          ```
          class Status {{
            <<enumeration>>
            Pending
            Active
          }}
          ```
        - Mermaid Example:
          graph TD
            A-->B
        
        - Mermaid Config Example:
          graph TD
            A-->B
        
        - PlantUML Example:
          @startuml
          class A
          @enduml
        """
        
        # User input block with RAG context
        user_block = f"""User Request: {prompt}
        
        Context from Docs (RAG):
        {context}
        """

        if self.provider == "Google Gemini":
            # For Gemini, we simulate history by appending it to the prompt text (since we use generate_content)
            # or we could use start_chat. Stateless approach: Formatting history as string.
            history_text = ""
            for msg in history:
                role = "User" if msg['role'] == 'user' else "Model"
                history_text += f"{role}: {msg['content']}\n\n"
            
            final_prompt = f"{system_instruction}\n\nChat History:\n{history_text}\n\n{user_block}"
            
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(final_prompt)
            return response.text
            
        elif self.provider == "OpenRouter":
            if not self.openrouter_client:
                return "Error: OpenRouter Client not initialized."
            
            # Construct messages list
            messages = [{"role": "system", "content": system_instruction}]
            # Add History
            for msg in history:
                # OpenRouter expects 'user' or 'assistant'
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current request
            messages.append({"role": "user", "content": user_block})
            
            response = self.openrouter_client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        return "Error: Invalid Provider"

    def analyze_media(self, prompt, file_data, mime_type):
        """
        Analyzes media (Image, PDF, Text) using the configured provider.
        file_data: bytes of the file
        mime_type: string (e.g. 'image/png', 'application/pdf')
        """
        if self.provider == "Google Gemini":
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            # Gemini accepts parts with mime_type and data
            content_part = {
                "mime_type": mime_type,
                "data": file_data
            }
            
            response = model.generate_content([prompt, content_part])
            return response.text
            
        elif self.provider == "OpenRouter":
            if not self.openrouter_client:
                return "Error: OpenRouter Client not initialized."

            # OpenRouter (mostly Vision/Text)
            # If it's an image, use standard vision payload
            if mime_type.startswith("image/"):
                base64_image = base64.b64encode(file_data).decode("utf-8")
                response = self.openrouter_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                return response.choices[0].message.content
            
            # If text/code, decode and pass as text
            elif mime_type.startswith("text/") or mime_type in ["application/json", "application/javascript"]:
                try:
                    text_content = file_data.decode("utf-8")
                    full_prompt = f"{prompt}\n\nAttached Content ({mime_type}):\n{text_content}"
                    response = self.openrouter_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    return f"Error processing text file: {str(e)}"
            
            else:
                 return f"Error: OpenRouter provider currently only supports Images and Text files. PDF/Other not supported via direct upload yet (Provider: {self.provider})."

        return "Error: Invalid Provider"
