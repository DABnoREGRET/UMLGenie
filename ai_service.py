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

    def generate_uml(self, prompt, context="", preferred_language="auto"):
        # Build instructions based on language
        lang_instruction = "Generate VALID Mermaid JS code (` ```mermaid `) or PlantUML code (`@startuml`...`@enduml`)."
        if preferred_language == "PlantUML":
            lang_instruction = "Generate **PlantUML** code ONLY (`@startuml` ... `@enduml`). Do NOT use Mermaid."
        elif preferred_language == "Mermaid":
            lang_instruction = "Generate **Mermaid JS** code ONLY (inside ```mermaid block). Do NOT use PlantUML."

        full_prompt = f"""You are a UML & Software Architecture Expert.
        
        User Request: {prompt}
        
        Context from Docs (RAG):
        {context}
        
        Instructions:
        1. {lang_instruction}
           - **Mermaid**: Start DIRECTLY with the diagram type (e.g. `sequenceDiagram`, `graph TD`, `classDiagram`) OR a Directive/Frontmatter (e.g. `%%{{init: ...}}%%`, `---`). Do **NOT** wrap in markdown blocks (` ``` `). Just provide the code.
           - **PlantUML**: Start DIRECTLY with `@startuml`. Do **NOT** wrap in markdown blocks.
        2. If the user asks a question about UML concepts, explain it.
        3. Can mix text with code, but keep the code clean and strictly valid.
        4. PREFER Mermaid JS for sequence diagrams and flowcharts. PREFER PlantUML for component, deployment, and complex class diagrams (Unless user specified a language).
        
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
        
        if self.provider == "Google Gemini":
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(full_prompt)
            return response.text
        elif self.provider == "OpenRouter":
            if not self.openrouter_client:
                return "Error: OpenRouter Client not initialized."
            
            response = self.openrouter_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.choices[0].message.content
        return "Error: Invalid Provider"

    def analyze_image(self, prompt, image):
        if self.provider == "Google Gemini":
            model = genai.GenerativeModel("gemini-2.0-flash") # Vision capable
            response = model.generate_content([prompt, image])
            return response.text
        elif self.provider == "OpenRouter":
            if not self.openrouter_client:
                return "Error: OpenRouter Client not initialized."
                
            base64_image = self._image_to_base64(image)
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
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        return "Error: Invalid Provider"
