import json


def save_registry(chunk_metadata):

    registry = {}

    for item in chunk_metadata:

        pdf = item["pdf"]

        if pdf not in registry:
            registry[pdf] = {
                "pages": set()
            }

        registry[pdf]["pages"].add(item["page"])

    output = []

    for pdf, info in registry.items():

        output.append({
            "name": pdf,
            "pages": len(info["pages"])
        })

    with open("indexes/documents.json", "w") as file:
        json.dump(output, file, indent=4)