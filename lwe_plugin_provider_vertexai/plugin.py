from langchain_community.llms.vertexai import VertexAI

from lwe.core.provider import Provider, PresetValue

class ProviderVertexai(Provider):
    """
    Access to Vertex AI models
    """

    @property
    def capabilities(self):
        return {
            'validate_models': True,
            'models': {
                'text-bison': {
                    'max_tokens': 8192,
                },
                'code-bison': {
                    'max_tokens': 6144,
                },
                'code-gecko': {
                    'max_tokens': 2048,
                },
            },
        }

    @property
    def default_model(self):
        return 'text-bison'

    def llm_factory(self):
        return VertexAI

    def customization_config(self):
        return {
            'model_name': PresetValue(str, options=self.available_models),
            'temperature': PresetValue(float, min_value=0.0, max_value=1.0),
            'max_output_tokens': PresetValue(int, min_value=1, max_value=2048, include_none=True),
            'top_k': PresetValue(int, min_value=1, max_value=40),
            'top_p': PresetValue(float, min_value=0.0, max_value=1.0),
            'project': PresetValue(str, include_none=True),
            'location': PresetValue(str),
            'request_parallelism': PresetValue(int, min_value=1),
            'max_retries': PresetValue(int, min_value=1),
        }
