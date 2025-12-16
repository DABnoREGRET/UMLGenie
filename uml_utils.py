import zlib
import base64
import requests
from sqlalchemy import create_engine, inspect

# --- PlantUML Utilities ---

def deflate_and_encode(text):
    """
    Compresses and encodes text for PlantUML URL.
    Algorithm: Deflate -> Base64 (with custom mapping)
    """
    zlibbed = zlib.compress(text.encode('utf-8'))
    compressed = zlibbed[2:-4]
    return encode64(compressed)

def encode64(data):
    # PlantUML uses a custom base64 mapping
    mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    res = ""
    for i in range(0, len(data), 3):
        b1 = data[i]
        b2 = data[i+1] if i+1 < len(data) else 0
        b3 = data[i+2] if i+2 < len(data) else 0
        
        c1 = b1 >> 2
        c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
        c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
        c4 = b3 & 0x3F
        
        if i+2 < len(data):
            res += mapping[c1] + mapping[c2] + mapping[c3] + mapping[c4]
        elif i+1 < len(data):
            res += mapping[c1] + mapping[c2] + mapping[c3]
        else:
            res += mapping[c1] + mapping[c2]
            
    return res

def get_plantuml_url(code):
    encoded = deflate_and_encode(code)
    return f"http://www.plantuml.com/plantuml/svg/{encoded}"

# --- Database Utilities ---

def get_database_schema(connection_string):
    """
    Connects to a database and returns a text description of tables and columns.
    """
    try:
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        
        schema_text = []
        table_names = inspector.get_table_names()
        
        for table_name in table_names:
            columns = inspector.get_columns(table_name)
            pks = inspector.get_pk_constraint(table_name).get('constrained_columns', [])
            fks = inspector.get_foreign_keys(table_name)
            
            schema_text.append(f"Table: {table_name}")
            for col in columns:
                pk_str = " (PK)" if col['name'] in pks else ""
                schema_text.append(f"  - {col['name']}: {col['type']}{pk_str}")
            
            for fk in fks:
                schema_text.append(f"  - FK: {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
            schema_text.append("")
            
        return "\n".join(schema_text)
    except Exception as e:
        return f"Error inspecting database: {str(e)}"

# --- Mermaid Utilities ---

def render_mermaid(code):
    # Server-side rendering via mermaid.ink (Best for static images)
    import base64
    code_bytes = code.encode('ascii')
    base64_bytes = base64.b64encode(code_bytes)
    base64_string = base64_bytes.decode('ascii')
    return f"https://mermaid.ink/svg/{base64_string}"

def render_mermaid_html(code):
    """
    Returns HTML string to render Mermaid diagram locally using JavaScript.
    Includes Zoom, Pan, and Theme Toggle.
    """
    # Escape backticks if present to strictly avoid breaking the template string
    code = code.replace("`", "") 

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; font-family: sans-serif; overflow: hidden; }}
            #controls {{
                position: fixed; top: 10px; right: 10px; z-index: 100;
                background: rgba(255, 255, 255, 0.8); padding: 5px; border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2); display: flex; gap: 5px;
            }}
            button {{
                border: 1px solid #ccc; background: white; cursor: pointer;
                padding: 5px 10px; border-radius: 4px; font-size: 14px;
            }}
            button:hover {{ background: #f0f0f0; }}
            #diagram-container {{ width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center; }}
            svg {{ width: 100%; height: 100%; }}
        </style>
    </head>
    <body>
        <div id="controls">
            <button onclick="toggleTheme()" title="Toggle Theme">🌓</button>
            <button onclick="zoomIn()" title="Zoom In">➕</button>
            <button onclick="zoomOut()" title="Zoom Out">➖</button>
            <button onclick="resetZoom()" title="Reset">🔄</button>
        </div>
        
        <div id="diagram-container" class="mermaid">
            {code}
        </div>

        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
        
        <script>
            let panZoomInstance = null;
            let currentTheme = 'default';
            const code = `{code}`;

            function initMermaid(theme) {{
                // Dispose old
                const container = document.getElementById('diagram-container');
                container.removeAttribute('data-processed');
                container.innerHTML = code;
                
                // Config
                mermaid.initialize({{ 
                    startOnLoad: false, 
                    theme: theme,
                    securityLevel: 'loose'
                }});

                // Render
                mermaid.run({{
                    querySelector: '.mermaid',
                    postRenderCallback: (id) => {{
                        setupPanZoom(id);
                    }}
                }});
            }}

            function setupPanZoom(id) {{
                const svg = document.querySelector('#diagram-container svg');
                if (!svg) return;
                
                // Ensure SVG takes full space
                svg.style.width = '100%';
                svg.style.height = '100%';

                panZoomInstance = svgPanZoom(svg, {{
                    zoomEnabled: true,
                    controlIconsEnabled: false,
                    fit: true,
                    center: true,
                    minZoom: 0.1,
                    maxZoom: 10
                }});
            }}

            function toggleTheme() {{
                currentTheme = currentTheme === 'default' ? 'dark' : 'default';
                initMermaid(currentTheme);
                
                // Update button text/icon if needed
                const btn = document.querySelector('button[title="Toggle Theme"]');
                btn.textContent = currentTheme === 'default' ? '🌓' : '☀️';
                
                // Update controls bg for contrast
                document.getElementById('controls').style.background = 
                    currentTheme === 'dark' ? 'rgba(50, 50, 50, 0.8)' : 'rgba(255, 255, 255, 0.8)';
                document.body.style.background = 
                    currentTheme === 'dark' ? '#1e1e1e' : 'white';
            }}

            function zoomIn() {{ if(panZoomInstance) panZoomInstance.zoomIn(); }}
            function zoomOut() {{ if(panZoomInstance) panZoomInstance.zoomOut(); }}
            function resetZoom() {{ if(panZoomInstance) panZoomInstance.resetZoom(); }}

            // First Init
            initMermaid('default');
        </script>
    </body>
    </html>
    """

# --- Web Search Utilities ---

def perform_web_search(query, max_results=3):
    """
    Performs a DuckDuckGo search and returns formatted results.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r['title']}\nURL: {r['href']}\nBody: {r['body']}\n")
        
        if not results:
            return "No web results found."
            
        return "\n---\n".join(results)
    except ImportError:
        return "Error: duckduckgo-search library not installed."
    except Exception as e:
        return f"Error searching web: {str(e)}"
