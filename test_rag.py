from rag_engine import UMLRAG

def test_rag():
    rag = UMLRAG()
    success = rag.load()
    
    if not success:
        print("Failed to load RAG.")
        return

    print("\n--- Test 1: Mermaid Query ---")
    results = rag.query("How to make a sequence diagram?", branch="mermaid")
    print(f"Results: {len(results)}")
    if results and "mermaid" in results[0].lower():
        print("✅ Mermaid context found.")
    else:
        print("⚠️ Check Mermaid context.")

    print("\n--- Test 2: PlantUML Query ---")
    results = rag.query("component diagram syntax", branch="plantuml")
    print(f"Results: {len(results)}")
    if results:
        print("✅ PlantUML context found.")

    print("\n--- Test 3: Routing (Auto) ---")
    # Should route to PlantUML
    plant_branch = rag.route_query("I want a PlantUML component diagram")
    print(f"Query: 'PlantUML component' -> Routed to: {plant_branch}")
    assert plant_branch == "plantuml"

    # Should route to Mermaid
    mermaid_branch = rag.route_query("Mermaid flowchart example")
    print(f"Query: 'Mermaid flowchart' -> Routed to: {mermaid_branch}")
    assert mermaid_branch == "mermaid"

    print("\n✅ Verification Complete")

if __name__ == "__main__":
    test_rag()
