import json

import requests

DIAL_EMBEDDINGS = "https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings"


class DialEmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, deployment_name: str, api_key: str):
        if not api_key or api_key == "":
            raise ValueError("No API key")

        self._api_key = api_key
        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)

    def _from_data(self, data: list[dict]) -> dict[int, list[float]]:
        return {
            embedding_obj["index"]: embedding_obj["embedding"] for embedding_obj in data
        }

    def get_embeddings(
        self, dimensions: int, inputs: str | list[str]
    ) -> dict[int, list[float]]:
        headers = {"api-key": self._api_key, "Content-Type": "application/json"}
        request_data = {"input": inputs, "dimensions": dimensions}

        print("=" * 100)
        print(f"\nsearching for embedding: '{inputs}'. Dimension is: {dimensions}\n")

        response = requests.post(
            url=self._endpoint,
            headers=headers,
            json=request_data,
            timeout=60,
        )

        if response.status_code == 200:
            res_json = response.json()
            data = res_json.get("data", [])

            # print("===== Response =====")
            # print(json.dumps(res_json, indent=2))
            print("=" * 100)

            return self._from_data(data)
        raise Exception(f"HTTP {response.status_code}: {response.text}")
